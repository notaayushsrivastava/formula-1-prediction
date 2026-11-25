"""pitwall.report: the LaTeX race-preview PDF (Phase 13).

    pitwall-f1 report 2026 Qatar --moment quali --out report.pdf

Needs Jinja2 (pip install 'pitwall-f1[report]') and LuaLaTeX (a TeX distribution).
"""
from .build import build_race_report, report_context
from .latex import ReportError, compile_pdf, escape_latex, have_engine, pdf_pages, render

__all__ = ["build_race_report", "report_context", "ReportError", "compile_pdf",
           "escape_latex", "have_engine", "pdf_pages", "render"]
