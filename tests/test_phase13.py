"""Phase 13: the LaTeX race-preview PDF. Offline; compiling is skipped without LuaLaTeX."""
import datetime as dt
import shutil
import subprocess
from pathlib import Path

import pytest

import fake_world as fw
from pitwall import cli
from pitwall.report import build_race_report, escape_latex, pdf_pages, render
from pitwall.report.latex import pct

pytest.importorskip("jinja2")
H = dt.timedelta(hours=1)
SECTIONS = ["In one minute", "The full field", "Probabilities", "Teams and strategy",
            "Safety car", "Why: what drives the prediction", "Championship impact",
            "Track record", "Methodology", "Sources and licences", "Disclaimer",
            "Not betting advice", "Not affiliated with Formula 1", "CC BY 4.0",
            "CC BY-NC-SA 4.0"]


def lualatex_works():
    if shutil.which("lualatex") is None:
        return False
    probe = Path(__file__).with_name("_probe_dir")
    probe.mkdir(exist_ok=True)
    try:
        (probe / "t.tex").write_text(
            "\\documentclass{article}\\usepackage{fontspec}\\begin{document}x\\end{document}")
        done = subprocess.run(["lualatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
                              cwd=probe, capture_output=True, timeout=180)
        return done.returncode == 0
    finally:
        shutil.rmtree(probe, ignore_errors=True)


def test_escaping():
    assert escape_latex("A&B_C 50% #1 $x {y} ~ ^ \\") == (
        r"A\&B\_C 50\% \#1 \$x \{y\} \textasciitilde{} \textasciicircum{} \textbackslash{}")
    assert escape_latex(None) == ""
    assert pct(0.1234) == r"12.3\%" and pct(None) == "--" and pct(float("nan")) == "--"
    assert pct(0.0004).startswith("$<$")


def _build(tmp_path, monkeypatch, **kw):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pitwall.predict.service.HISTORY_PREDICTIONS", tmp_path / "none.pq")
    s = fw.sources(quali_upto=6)
    when = fw.at(2026, 6, "Qualifying", 3 * H)
    return build_race_report(2026, "Desert", "quali", s, now=lambda: when, first_year=2024, **kw)


def test_tex_has_every_section_and_saves_the_record(tmp_path, monkeypatch):
    tex = _build(tmp_path, monkeypatch, tex_only=True)
    text = tex.read_text()
    for section in SECTIONS:
        assert section in text, section
    assert tex.name == "2026-06-quali.tex"
    assert Path("predictions/2026-06-quali.json").exists()  # the report also makes the record
    assert "ALP" in text and "Desert-Loop Grand Prix" in text
    assert "\\VAR{" not in text and "\\BLOCK{" not in text
    assert "penalties" in text  # the grid-penalty note at the quali moment


def test_special_characters_in_data_are_escaped():
    context = {"event": "Rock & Roll_GP 100%"}
    from pitwall.report.latex import environment
    out = environment().from_string(r"\VAR{event|e}").render(**context)
    assert out == r"Rock \& Roll\_GP 100\%"


def test_render_fails_loudly_on_missing_values():
    with pytest.raises(Exception):
        render("race_report.tex", {})


@pytest.mark.skipif(not lualatex_works(), reason="LuaLaTeX with fontspec is not installed")
def test_pdf_compiles_within_eight_pages(tmp_path, monkeypatch):
    pdf = _build(tmp_path, monkeypatch, out=tmp_path / "report.pdf")
    assert pdf.exists() and pdf.read_bytes()[:4] == b"%PDF"
    assert 1 <= pdf_pages(pdf) <= 8


def test_cli_report_tex_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "make_sources",
                        lambda openf1=True, weather=True: fw.sources(5, 5, weather, openf1))
    monkeypatch.setattr(cli, "utcnow", lambda: fw.at(2026, 6, "Practice 1", -10 * H))
    monkeypatch.setattr("pitwall.predict.service.HISTORY_PREDICTIONS", tmp_path / "none.pq")
    import pitwall.report.build as build
    real = build.build_live_rows
    monkeypatch.setattr(build, "build_live_rows",
                        lambda *a, **k: real(*a, **{**k, "first_year": 2024,
                                                    "history_from": 2024}))
    import pitwall.predict.service as service
    real_service = service.build_live_rows
    monkeypatch.setattr(service, "build_live_rows",
                        lambda *a, **k: real_service(*a, **{**k, "first_year": 2024,
                                                            "history_from": 2024}))
    assert cli.main(["report", "next", "--moment", "pre", "--tex-only",
                     "--out", "r.pdf"]) == 0
    assert "In one minute" in Path("r.tex").read_text()
    assert cli.main(["report", "next", "--moment", "pre", "--within-days", "1",
                     "--tex-only"]) == 0  # the race is further away: nothing to do
    assert cli.main(["report", "next", "--moment", "quali",
                     "--skip-if-unavailable", "--tex-only"]) == 0
