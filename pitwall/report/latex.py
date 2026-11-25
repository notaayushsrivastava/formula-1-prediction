"""LaTeX helpers: escaping, a Jinja2 environment with LaTeX-safe delimiters, compiling.

Jinja's usual {{ }} and {% %} clash with LaTeX braces, so the template uses
\\VAR{...} for values, \\BLOCK{...} for logic and \\#{...} for comments. Every value
from data goes through `escape_latex` (the `e` filter): a driver or event name with
'&', '_' or '%' would otherwise break the document.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any

from ..errors import PitwallError

ENGINE = "lualatex"
_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}"}
_PATTERN = re.compile("|".join(re.escape(k) for k in _SPECIAL))


class ReportError(PitwallError):
    """The report could not be rendered or compiled."""


def escape_latex(value: Any) -> str:
    """Text safe to put in a LaTeX document (None -> '')."""
    if value is None:
        return ""
    return _PATTERN.sub(lambda m: _SPECIAL[m.group(0)], str(value))


def pct(value: Any, digits: int = 1) -> str:
    """0.1234 -> '12.3\\%' (escaped for LaTeX); '<0.1\\%' for tiny values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if number != number:  # NaN
        return "--"
    if 0 < number < 0.001:
        return r"$<$0.1\%"
    return f"{100 * number:.{digits}f}\\%"


def environment() -> Any:
    """The Jinja2 environment (Jinja2 is imported lazily: pip install pitwall-f1[report])."""
    try:
        import jinja2
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ReportError("The race report needs Jinja2: pip install 'pitwall-f1[report]'") \
            from exc
    env = jinja2.Environment(
        block_start_string=r"\BLOCK{", block_end_string="}",
        variable_start_string=r"\VAR{", variable_end_string="}",
        comment_start_string=r"\#{", comment_end_string="}",
        line_statement_prefix="%%", line_comment_prefix="%#",
        trim_blocks=True, lstrip_blocks=True, autoescape=False,
        undefined=jinja2.StrictUndefined,
        loader=jinja2.FunctionLoader(_load_template))
    env.filters["e"] = escape_latex
    env.filters["pct"] = pct
    return env


def _load_template(name: str) -> str:
    return resources.files("pitwall.report").joinpath("templates", name).read_text("utf-8")


def render(template: str, context: dict[str, Any]) -> str:
    """Render a packaged template with the given values."""
    return str(environment().get_template(template).render(**context))


def have_engine(engine: str = ENGINE) -> bool:
    """True if the TeX engine is installed."""
    return shutil.which(engine) is not None


def compile_pdf(tex: str, out: Path | str, figures: dict[str, Path] | None = None,
                engine: str = ENGINE, runs: int = 2, timeout: float = 240.0) -> Path:
    """Compile LaTeX source to a PDF at `out` (in a temporary folder; figures are copied
    next to the .tex by name). Raises ReportError with the end of the log on failure."""
    if not have_engine(engine):
        raise ReportError(f"'{engine}' was not found. Install a TeX distribution with "
                          "LuaLaTeX (e.g. TeX Live), or use --tex-only.")
    target = Path(out)
    with tempfile.TemporaryDirectory(prefix="pitwall-report-") as tmp:
        folder = Path(tmp)
        for name, path in (figures or {}).items():
            shutil.copyfile(path, folder / name)
        (folder / "report.tex").write_text(tex, encoding="utf-8")
        for _ in range(runs):  # twice: page numbers ("page 2 of 6") need a second pass
            done = subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error",
                                   "report.tex"], cwd=folder, capture_output=True,
                                  text=True, timeout=timeout)
            if done.returncode != 0:
                log = (folder / "report.log")
                tail = log.read_text("utf-8", "replace")[-3000:] if log.exists() else done.stdout
                raise ReportError(f"{engine} failed:\n{tail}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / "report.pdf", target)
    return target


def pdf_pages(path: Path | str) -> int:
    """Number of pages of a PDF (counts page objects; enough for our own PDFs)."""
    data = Path(path).read_bytes()
    return len(re.findall(rb"/Type\s*/Page[^s]", data))
