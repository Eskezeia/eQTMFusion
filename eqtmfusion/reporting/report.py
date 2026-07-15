"""
eqtmfusion.reporting
========================
Automated report generation: builds a single self-contained HTML report
(methods, data summary, QC results, feature selection results, model
parameters, validation metrics, embedded plots, biomarker summary) via
Jinja2. PDF export is supported if `weasyprint` is installed (optional --
HTML-to-PDF rendering has non-Python system dependencies on some platforms,
so it is not a hard requirement).
"""

import os
import base64
import warnings
from datetime import datetime
from jinja2 import Template


REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<style>
  body { font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 960px; margin: 40px auto; color: #222; }
  h1 { border-bottom: 3px solid #2c3e50; padding-bottom: 8px; }
  h2 { color: #2c3e50; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 14px; }
  th { background: #f4f6f7; }
  .meta { color: #666; font-size: 13px; }
  .fig { margin: 20px 0; text-align: center; }
  .fig img { max-width: 100%; border: 1px solid #eee; }
  .fig-caption { font-size: 13px; color: #555; margin-top: 6px; }
  code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
</style>
</head>
<body>
  <h1>{{ title }}</h1>
  <p class="meta">Generated {{ generated_at }} by eQTMFusion v{{ version }}</p>

  <h2>1. Methods</h2>
  <div>{{ methods_html | safe }}</div>

  <h2>2. Data Summary</h2>
  {% for name, table_html in data_summary_tables.items() %}
    <h3>{{ name }}</h3>
    {{ table_html | safe }}
  {% endfor %}

  <h2>3. QC Results</h2>
  {% for name, table_html in qc_tables.items() %}
    <h3>{{ name }}</h3>
    {{ table_html | safe }}
  {% endfor %}

  <h2>4. Feature Selection Results</h2>
  {% for name, table_html in feature_selection_tables.items() %}
    <h3>{{ name }}</h3>
    {{ table_html | safe }}
  {% endfor %}

  <h2>5. Model Parameters</h2>
  <pre>{{ model_params }}</pre>

  <h2>6. Validation Metrics</h2>
  {% for name, table_html in metrics_tables.items() %}
    <h3>{{ name }}</h3>
    {{ table_html | safe }}
  {% endfor %}

  <h2>7. Visualizations</h2>
  {% for fig in figures %}
    <div class="fig">
      <img src="data:image/png;base64,{{ fig.b64 }}" />
      <div class="fig-caption">{{ fig.caption }}</div>
    </div>
  {% endfor %}

  <h2>8. Biomarker Summary</h2>
  {% for name, table_html in biomarker_tables.items() %}
    <h3>{{ name }}</h3>
    {{ table_html | safe }}
  {% endfor %}

  <h2>9. Interpretation</h2>
  <div>{{ interpretation_html | safe }}</div>

</body>
</html>
"""

DEFAULT_METHODS_TEXT = """
<p>Data were preprocessed using k-nearest-neighbor imputation and standard
scaling (eqtmfusion.preprocessing). cis-/trans-eQTM associations were tested
via linear models adjusting for specified covariates, with Benjamini-Hochberg
FDR correction (eqtmfusion.eqtm). Feature selection combined an eQTM-derived
biological prior (Tier 1) with variance/MAD-based (Tier 2) and supervised
ML-based (Tier 3) selection (eqtmfusion.feature_selection). Multi-omics
modalities were integrated via early/intermediate/late/graph-based fusion
(eqtmfusion.fusion). Outcome models were validated via k-fold cross-validation
and, where available, an independent external cohort (eqtmfusion.utilities.validation).</p>
"""


def _fig_to_b64(fig) -> str:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_html_report(
    output_path: str,
    title: str = "eQTMFusion Analysis Report",
    methods_html: str = None,
    data_summary_tables: dict = None,
    qc_tables: dict = None,
    feature_selection_tables: dict = None,
    model_params: str = "",
    metrics_tables: dict = None,
    figures: list = None,
    biomarker_tables: dict = None,
    interpretation_html: str = "<p>Add interpretation notes here.</p>",
) -> str:
    """
    figures: list of dicts {"figure": matplotlib Figure, "caption": str}
    *_tables: dict of {section_name: pandas.DataFrame}, converted to HTML internally
    """
    from eqtmfusion import __version__

    def _tables_to_html(tables: dict) -> dict:
        if not tables:
            return {}
        return {name: df.to_html(classes="table", border=0) for name, df in tables.items()}

    fig_payload = []
    for fig_entry in (figures or []):
        fig_payload.append({
            "b64": _fig_to_b64(fig_entry["figure"]),
            "caption": fig_entry.get("caption", ""),
        })

    template = Template(REPORT_TEMPLATE)
    html = template.render(
        title=title,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        version=__version__,
        methods_html=methods_html or DEFAULT_METHODS_TEXT,
        data_summary_tables=_tables_to_html(data_summary_tables),
        qc_tables=_tables_to_html(qc_tables),
        feature_selection_tables=_tables_to_html(feature_selection_tables),
        model_params=model_params,
        metrics_tables=_tables_to_html(metrics_tables),
        figures=fig_payload,
        biomarker_tables=_tables_to_html(biomarker_tables),
        interpretation_html=interpretation_html,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    return output_path


def generate_pdf_report(html_path: str, pdf_path: str) -> str:
    """
    Convert an HTML report to PDF. Requires `pip install weasyprint`
    (which itself requires system libraries: Pango, Cairo, GDK-PixBuf --
    see https://doc.courtbouillon.org/weasyprint/stable/first_steps.html).
    Falls back to a clear error message if unavailable rather than failing
    silently.
    """
    try:
        from weasyprint import HTML
    except ImportError:
        raise ImportError(
            "weasyprint not installed or its system dependencies are missing. "
            "Install with `pip install weasyprint` and see "
            "https://doc.courtbouillon.org/weasyprint/stable/first_steps.html "
            "for platform-specific system libraries. The HTML report at "
            f"'{html_path}' is available as a fallback."
        )
    HTML(html_path).write_pdf(pdf_path)
    return pdf_path
