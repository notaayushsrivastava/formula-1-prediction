"""M2: a structured neural ranker (PyTorch, CPU only).

One score per driver (higher = faster), built from three parts:
    score = linear(features)                                 # the interpretable part
          + driver_head(driver_emb) + team_head(team_emb)    # learned fingerprints
          + MLP([non-monotone features, embeddings])         # 1 hidden layer, ReLU, dropout
- Embeddings: every driver and team gets a small learned vector. Id 0 = "unknown"; in
  training 10% of ids are hidden as 0, so 0 learns what a typical newcomer looks like.
- Guaranteed monotonicity: grid, quali/practice/teammate gaps and car_mean/skill_mean
  enter ONLY the linear part, with weight = -softplus(raw) <= 0, and never the MLP.
- Loss: Plackett-Luce over finishers (as in B1/M1) + lambda_dnf x DNF cross-entropy.
- Early stopping on TIME: train on the older 85% of races, watch the newest 15%, stop
  when they stop improving; then retrain on everything for that many epochs.
- Seed ensemble: n_seeds networks; scores and DNF chances are averaged, and the seeds'
  disagreement becomes score_sd (model uncertainty) in the race simulation.
"""
from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._optional import require
from ..strength import StrengthParams
from ..strength_features import STRENGTH_FEATURES, StrengthFeatures
from .base import (
    N_SIMS, design_matrix, features_of, finish_tensor, fit_scale, moment_of, predict_from_scores,
)

MONOTONE_FEATURES = ("grid", "quali_gap_pct", "practice_pace_gap_pct",
                     "teammate_quali_gap_pct", "car_mean", "skill_mean", "pace_pred")
VALIDATION_SHARE = 0.15
EMBEDDING_DROPOUT = 0.10
UNKNOWN = 0
TORCH_HINT = ("For a CPU-only PyTorch (much smaller), first run: "
              "pip install torch --index-url https://download.pytorch.org/whl/cpu")


def pl_loglik_torch(torch: Any, u: Any, mask: Any) -> Any:
    """Plackett-Luce log-likelihood in torch (same maths as base.pl_loglik_and_grad).

    u[r, k] = score of the driver who finished k-th in race r; mask marks real finishers.
    Empty places get -1e9 instead of -inf, which would turn the gradients into NaN.
    """
    filled = torch.where(mask, u, torch.full_like(u, -1e9))
    rest = torch.flip(torch.logcumsumexp(torch.flip(filled, dims=[1]), dim=1), dims=[1])
    return ((filled - rest) * mask).sum()


def validation_races(rows: pd.DataFrame, share: float = VALIDATION_SHARE) -> np.ndarray:
    """RaceIds of the newest `share` of races by date (time-based, never random)."""
    races = rows.drop_duplicates("RaceId").sort_values(["Date", "RaceId"])["RaceId"].to_numpy()
    count = int(round(share * len(races)))
    if count < 1 or len(races) - count < 2:
        return races[:0]
    return races[-count:]


@contextlib.contextmanager
def deterministic(torch: Any) -> Iterator[None]:
    """One CPU thread and deterministic algorithms: same seed -> identical numbers."""
    threads = torch.get_num_threads()
    was_on = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(was_on)


@dataclass
class Batch:
    """Rows as tensors. order/mask: finishers per race in finishing order (for the loss)."""

    free: Any
    mono: Any
    drivers: Any
    teams: Any
    order: Any
    mask: Any
    dnf: Any


@functools.lru_cache(maxsize=None)
def _rank_net_class(torch: Any) -> Any:
    """The network class (built only once torch has been imported)."""
    nn = torch.nn

    class RankNet(nn.Module):
        def __init__(self, n_free: int, n_mono: int, n_drivers: int, n_teams: int,
                     emb_dim: int, hidden: int, dropout: float, use_emb: bool,
                     use_mlp: bool) -> None:
            super().__init__()
            self.use_emb, self.use_mlp = use_emb, use_mlp
            self.w_free = nn.Parameter(torch.zeros(n_free))
            self.w_mono_raw = nn.Parameter(torch.zeros(n_mono))  # weight = -softplus(raw)
            extra = 0
            if use_emb:
                self.driver_emb = nn.Embedding(n_drivers + 1, emb_dim)  # row 0 = unknown
                self.team_emb = nn.Embedding(n_teams + 1, emb_dim)
                nn.init.normal_(self.driver_emb.weight, std=0.01)
                nn.init.normal_(self.team_emb.weight, std=0.01)
                self.driver_head = nn.Linear(emb_dim, 1, bias=False)
                self.team_head = nn.Linear(emb_dim, 1, bias=False)
                extra = 2 * emb_dim
            if use_mlp:
                self.mlp = nn.Sequential(nn.Linear(n_free + extra, hidden), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(hidden, 1))
            self.dnf = nn.Linear(n_free + n_mono + extra, 1)

        def mono_weights(self) -> Any:
            return -nn.functional.softplus(self.w_mono_raw)  # always <= 0

        def forward(self, free: Any, mono: Any, drivers: Any, teams: Any) -> tuple[Any, Any]:
            score = free @ self.w_free + mono @ self.mono_weights()
            extras = []
            if self.use_emb:
                d, t = self.driver_emb(drivers), self.team_emb(teams)
                score = score + self.driver_head(d).squeeze(-1) + self.team_head(t).squeeze(-1)
                extras = [d, t]
            if self.use_mlp:  # never sees the monotone features
                score = score + self.mlp(torch.cat([free, *extras], dim=1)).squeeze(-1)
            dnf_logit = self.dnf(torch.cat([free, mono, *extras], dim=1)).squeeze(-1)
            return score, dnf_logit

    return RankNet


class NeuralRanker:
    """M2. Defaults are small and sensible; every setting is logged by params()."""

    def __init__(self, use_strength: bool = True, use_embeddings: bool = True,
                 use_mlp: bool = True, hidden: int = 32, emb_dim: int = 4,
                 dropout: float = 0.2, lr: float = 0.01, weight_decay: float = 1e-3,
                 max_epochs: int = 300, patience: int = 30, lambda_dnf: float = 0.5,
                 n_seeds: int = 10, seed: int = 0, n_sims: int = N_SIMS,
                 strength_params: StrengthParams | None = None, strength_maxiter: int = 60,
                 combine_pace_sd: bool = True, name: str | None = None) -> None:
        self.use_strength = use_strength
        self.use_embeddings = use_embeddings
        self.use_mlp = use_mlp
        self.hidden = hidden
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.lambda_dnf = lambda_dnf
        self.n_seeds = n_seeds
        self.seed = seed
        self.n_sims = n_sims
        self.strength_params = strength_params
        self.strength_maxiter = strength_maxiter
        self.combine_pace_sd = combine_pace_sd
        self.name = name or self._auto_name()

    def _auto_name(self) -> str:
        if self.use_embeddings and self.use_mlp:
            base = "M2-full"
        elif not self.use_embeddings and not self.use_mlp:
            base = "M2-linear"
        else:
            base = "M2-no-MLP" if self.use_embeddings else "M2-no-emb"
        return base if self.use_strength else f"{base} (no strength)"

    def params(self) -> dict[str, Any]:
        """Every setting, plus the best epoch per seed and the temperature once fitted."""
        out: dict[str, Any] = {
            "use_strength": self.use_strength, "use_embeddings": self.use_embeddings,
            "use_mlp": self.use_mlp, "hidden": self.hidden, "emb_dim": self.emb_dim,
            "dropout": self.dropout, "lr": self.lr, "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs, "patience": self.patience,
            "lambda_dnf": self.lambda_dnf, "n_seeds": self.n_seeds, "seed": self.seed,
            "n_sims": self.n_sims, "combine_pace_sd": self.combine_pace_sd}
        if hasattr(self, "best_epochs_"):
            out["best_epochs"] = list(self.best_epochs_)
            out["scale"] = round(float(self.scale_), 6)
        return out

    # ---------- fit ----------
    def fit(self, train: pd.DataFrame) -> NeuralRanker:
        """Standardise, map ids, early-stop on the newest races, refit, average seeds."""
        torch = self._torch()
        self.moment_ = moment_of(train)
        if self.use_strength:
            self.strength_ = StrengthFeatures(self.strength_params, self.strength_maxiter)
            rows = self.strength_.fit(train)
        else:
            rows = train.reset_index(drop=True)
        self.features_ = features_of(train) + (STRENGTH_FEATURES if self.use_strength else [])
        self.monotone_ = [f for f in self.features_ if f in MONOTONE_FEATURES]
        self.free_ = [f for f in self.features_ if f not in MONOTONE_FEATURES]
        self.free_idx_ = np.array([self.features_.index(f) for f in self.free_], dtype=int)
        self.mono_idx_ = np.array([self.features_.index(f) for f in self.monotone_], dtype=int)
        raw = design_matrix(rows, self.features_)
        self.mean_ = raw.mean(axis=0)
        spread = raw.std(axis=0)
        self.std_ = np.where(spread > 0, spread, 1.0)
        self.driver_ids_ = _id_map(rows["DriverId"])
        self.team_ids_ = _id_map(rows["TeamId"])
        latest = rows.sort_values("Date").drop_duplicates("DriverId", keep="last")
        self.driver_info_ = latest.set_index("DriverId")[["Driver", "TeamId"]]
        self.validation_races_ = validation_races(rows)
        held = rows["RaceId"].isin(self.validation_races_).to_numpy()
        val_scores: list[np.ndarray] = []
        with deterministic(torch):
            everything = self._batch(torch, rows)
            older = self._batch(torch, rows[~held].reset_index(drop=True))
            newest = self._batch(torch, rows[held].reset_index(drop=True)) if held.any() else None
            self.nets_, self.best_epochs_ = [], []
            for k in range(self.n_seeds):
                epochs = self.max_epochs
                if newest is not None:
                    net, epochs = self._train(torch, older, self.max_epochs, self.seed + k, newest)
                    val_scores.append(self._forward(torch, net, newest)[0])
                final, _ = self._train(torch, everything, epochs, self.seed + k)
                self.nets_.append(final)
                self.best_epochs_.append(int(epochs))
        if val_scores:
            val_rows = rows[held]
            self.scale_ = fit_scale(np.mean(val_scores, axis=0), val_rows["RaceId"].to_numpy(),
                                    val_rows["Finish"].to_numpy(float))
        else:
            self.scale_ = 1.0
        return self

    # ---------- predict ----------
    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        """Average the seeds; their spread (and M1's pace_sd, if used) becomes score_sd."""
        if moment_of(race) != self.moment_:
            raise ValueError(f"{self.name} was fitted on '{self.moment_}' rows.")
        rows = self._prepare(race)
        scores, p_dnf = self._ensemble(rows)
        spread = scores.std(axis=0)
        if self.use_strength and self.combine_pace_sd:
            spread = np.sqrt(spread**2 + self._strength_var(rows))
        return predict_from_scores(race, scores.mean(axis=0), p_dnf.mean(axis=0), self.scale_,
                                   self.n_sims, self.seed, spread)

    def seed_scores(self, race: pd.DataFrame) -> np.ndarray:
        """Raw scores (no temperature) of every seed network: shape (n_seeds, drivers)."""
        return self._ensemble(self._prepare(race))[0]

    def encode_drivers(self, race: pd.DataFrame) -> np.ndarray:
        """Embedding ids of the race's drivers (0 = never seen in training)."""
        return np.array([self.driver_ids_.get(str(d), UNKNOWN) for d in race["DriverId"]])

    # ---------- explanations ----------
    def coefficients(self) -> pd.Series:
        """Linear weights per standardised feature, averaged over seeds (monotone ones <= 0)."""
        weights = [np.concatenate([net.w_free.detach().numpy(),
                                   net.mono_weights().detach().numpy()]) for net in self.nets_]
        series = pd.Series(np.mean(weights, axis=0), index=self.free_ + self.monotone_)
        return series.reindex(series.abs().sort_values(ascending=False).index)

    def driver_contributions(self) -> pd.Series:
        """Score added by each driver's embedding (+ = faster), averaged over seeds."""
        torch = self._torch()
        self._need_embeddings()
        with torch.no_grad():
            values = [net.driver_head(net.driver_emb.weight).squeeze(-1).numpy()
                      for net in self.nets_]
        return pd.Series(np.mean(values, axis=0), index=self._driver_names())

    def embeddings(self) -> pd.DataFrame:
        """Seed 0's driver embeddings (seeds differ by a rotation, so they are not averaged)."""
        self._need_embeddings()
        weights = self.nets_[0].driver_emb.weight.detach().numpy()[1:]
        names = self._driver_names()[1:]
        table = pd.DataFrame(weights, columns=[f"emb_{k}" for k in range(weights.shape[1])])
        info = self.driver_info_.reindex(names)
        table.insert(0, "DriverId", names)
        table.insert(1, "Driver", info["Driver"].to_numpy())
        table.insert(2, "TeamId", info["TeamId"].to_numpy())
        return table

    def integrated_gradients(self, race: pd.DataFrame, steps: int = 50) -> pd.DataFrame:
        """Per driver, per feature: how much each input moved the score away from a
        driver with average (training-mean) features. Rows sum to score - baseline score."""
        torch = self._torch()
        rows = self._prepare(race)
        batch = self._batch(torch, rows, labels=False)
        x = torch.cat([batch.free, batch.mono], dim=1)
        n_free = batch.free.shape[1]
        total = torch.zeros_like(x)
        with deterministic(torch):
            for net in self.nets_:
                net.eval()
                for k in range(steps):  # midpoint rule along the straight path 0 -> x
                    point = (x * ((k + 0.5) / steps)).requires_grad_(True)
                    score, _ = net(point[:, :n_free], point[:, n_free:], batch.drivers,
                                   batch.teams)
                    total += torch.autograd.grad(score.sum(), point)[0]
        attribution = (x * total / (steps * len(self.nets_))).detach().numpy()
        return pd.DataFrame(attribution, index=rows["Driver"].to_numpy(),
                            columns=self.free_ + self.monotone_)

    # ---------- training helpers ----------
    def _train(self, torch: Any, batch: Batch, epochs: int, seed: int,
               val: Batch | None = None) -> tuple[Any, int]:
        """Full-batch AdamW. With `val`: early stopping, returns the best epoch's network."""
        torch.manual_seed(seed)
        net = _rank_net_class(torch)(len(self.free_), len(self.monotone_),
                                     len(self.driver_ids_), len(self.team_ids_), self.emb_dim,
                                     self.hidden, self.dropout, self.use_embeddings,
                                     self.use_mlp)
        optimiser = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        generator = torch.Generator().manual_seed(seed)
        best_loss, best_epoch, best_state, waited = math.inf, epochs, None, 0
        for epoch in range(1, epochs + 1):
            net.train()
            drivers = _hide_ids(torch, batch.drivers, generator)
            teams = _hide_ids(torch, batch.teams, generator)
            score, logit = net(batch.free, batch.mono, drivers, teams)
            loss = self._loss(torch, score, logit, batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if val is None:
                continue
            current = self._val_loss(torch, net, val)
            if current < best_loss - 1e-6:
                best_loss, best_epoch, waited = current, epoch, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best_epoch

    def _loss(self, torch: Any, score: Any, logit: Any, batch: Batch) -> Any:
        """Plackett-Luce NLL per race + lambda_dnf x DNF binary cross-entropy."""
        dnf = torch.nn.functional.binary_cross_entropy_with_logits(logit, batch.dnf)
        if batch.order.numel() == 0:
            return self.lambda_dnf * dnf
        races = batch.order.shape[0]
        ranking = -pl_loglik_torch(torch, score[batch.order], batch.mask) / races
        return ranking + self.lambda_dnf * dnf

    def _val_loss(self, torch: Any, net: Any, val: Batch) -> float:
        net.eval()
        with torch.no_grad():
            score, logit = net(val.free, val.mono, val.drivers, val.teams)
            return float(self._loss(torch, score, logit, val))

    def _forward(self, torch: Any, net: Any, batch: Batch) -> tuple[np.ndarray, np.ndarray]:
        """(scores, DNF probabilities) of one network, in evaluation mode."""
        net.eval()
        with torch.no_grad():
            score, logit = net(batch.free, batch.mono, batch.drivers, batch.teams)
            return score.double().numpy(), torch.sigmoid(logit).double().numpy()

    def _ensemble(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch()
        batch = self._batch(torch, rows, labels=False)
        with deterministic(torch):
            outputs = [self._forward(torch, net, batch) for net in self.nets_]
        return np.array([s for s, _ in outputs]), np.array([p for _, p in outputs])

    def _batch(self, torch: Any, rows: pd.DataFrame, labels: bool = True) -> Batch:
        """Standardised features, ids, finishing order and DNF labels as tensors."""
        x = (design_matrix(rows, self.features_) - self.mean_) / self.std_
        finish = rows["Finish"].to_numpy(float) if labels else np.full(len(rows), np.nan)
        tensor, mask = finish_tensor(np.arange(len(rows)), rows["RaceId"].to_numpy(), finish)
        dnf = rows["DNF"].to_numpy(float) if labels else np.zeros(len(rows))
        return Batch(
            free=torch.tensor(x[:, self.free_idx_], dtype=torch.float32),
            mono=torch.tensor(x[:, self.mono_idx_], dtype=torch.float32),
            drivers=torch.tensor(self.encode_drivers(rows), dtype=torch.long),
            teams=torch.tensor([self.team_ids_.get(str(t), UNKNOWN) for t in rows["TeamId"]],
                               dtype=torch.long),
            order=torch.tensor(tensor[..., 0].astype(np.int64)),
            mask=torch.tensor(mask),
            dnf=torch.tensor(dnf, dtype=torch.float32))

    def _prepare(self, race: pd.DataFrame) -> pd.DataFrame:
        return self.strength_.for_race(race) if self.use_strength else race.reset_index(drop=True)

    def _strength_var(self, rows: pd.DataFrame) -> np.ndarray:
        """Score variance caused by M1's car/skill uncertainty (through the linear weights)."""
        weights = self.coefficients()
        a = weights["car_mean"] / self.std_[self.features_.index("car_mean")]
        b = weights["skill_mean"] / self.std_[self.features_.index("skill_mean")]
        var = (a**2 * rows["car_sd"] ** 2 + b**2 * rows["skill_sd"] ** 2
               + 2 * a * b * rows["car_skill_cov"])
        return np.maximum(var.to_numpy(float), 0.0)

    def _driver_names(self) -> list[str]:
        names = ["<unknown>"] * (len(self.driver_ids_) + 1)
        for driver, index in self.driver_ids_.items():
            names[index] = driver
        return names

    def _need_embeddings(self) -> None:
        if not self.use_embeddings:
            raise ValueError(f"{self.name} has no embeddings.")

    def _torch(self) -> Any:
        return require("torch", f"M2 {type(self).__name__}", extra="nn", hint=TORCH_HINT)


def _id_map(values: pd.Series) -> dict[str, int]:
    """Ids from TRAINING data only: 1, 2, ... (0 is reserved for 'unknown')."""
    return {v: i for i, v in enumerate(sorted(values.astype(str).unique()), start=1)}


def _hide_ids(torch: Any, ids: Any, generator: Any) -> Any:
    """Embedding dropout: replace ~10% of ids with 0 ('unknown') for this epoch."""
    hide = torch.rand(ids.shape, generator=generator) < EMBEDDING_DROPOUT
    return ids.masked_fill(hide, UNKNOWN)
