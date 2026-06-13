"""Human-facing report generation for completed VPA jobs and workflows."""

from .collect import collect_report_model
from .models import ReportModel
from .preflight import validate_final_report_contract
from .render import render_report_html, resolve_report_output_path, write_report

__all__ = [
    "ReportModel",
    "collect_report_model",
    "render_report_html",
    "resolve_report_output_path",
    "validate_final_report_contract",
    "write_report",
]
