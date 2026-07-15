"""
eqtmfusion.reporting.tables
===============================
Publication-standard summary tables:

  - table1_cohort_characteristics: the conventional clinical-paper "Table 1"
    (demographics/characteristics, optionally stratified by group, with
    between-group test p-values)
  - table_model_performance_summary: mean +/- SD across CV folds, formatted
    as publication-ready strings, alongside the raw numeric columns
  - table_top_biomarkers: ranked, modality-annotated feature importance table

All functions return a pandas DataFrame; use save_table() to write CSV/XLSX,
or dataframe_to_image() to render a table as a PNG for supplementary figures.
"""

import numpy as np
import pandas as pd
from scipy import stats


def _is_continuous(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and series.nunique() > 10


def table1_cohort_characteristics(
    clinical_df: pd.DataFrame,
    group_col: str = None,
    continuous_cols: list = None,
    categorical_cols: list = None,
    decimals: int = 1,
) -> pd.DataFrame:
    """
    Build a conventional "Table 1": continuous variables as mean (SD),
    categorical variables as n (%), optionally split by group_col with a
    between-group test p-value (Welch's t-test for 2-group continuous,
    ANOVA for >2-group continuous, chi-square for categorical).

    If continuous_cols / categorical_cols are None, columns are auto-typed.
    """
    df = clinical_df.copy()
    exclude = {group_col} if group_col else set()
    if continuous_cols is None:
        continuous_cols = [c for c in df.columns if c not in exclude and _is_continuous(df[c])]
    if categorical_cols is None:
        categorical_cols = [c for c in df.columns if c not in exclude and c not in continuous_cols]

    groups = sorted(df[group_col].dropna().unique()) if group_col else [None]
    rows = []

    for col in continuous_cols:
        row = {"Variable": col}
        overall = df[col].dropna()
        row["Overall"] = f"{overall.mean():.{decimals}f} ({overall.std():.{decimals}f})"
        group_vals = []
        for g in groups:
            if group_col:
                vals = df.loc[df[group_col] == g, col].dropna()
                row[str(g)] = f"{vals.mean():.{decimals}f} ({vals.std():.{decimals}f})"
                group_vals.append(vals)
        if group_col and len(groups) >= 2:
            group_vals = [g for g in group_vals if len(g) > 1]
            if len(group_vals) == 2:
                _, p = stats.ttest_ind(*group_vals, equal_var=False)
            elif len(group_vals) > 2:
                _, p = stats.f_oneway(*group_vals)
            else:
                p = np.nan
            row["p-value"] = f"{p:.3g}" if pd.notna(p) else "NA"
        rows.append(row)

    for col in categorical_cols:
        overall_counts = df[col].value_counts(dropna=True)
        n_total = overall_counts.sum()
        header_row = {"Variable": f"{col}, n (%)", "Overall": ""}
        if group_col:
            for g in groups:
                header_row[str(g)] = ""
        rows.append(header_row)

        for category in overall_counts.index:
            row = {"Variable": f"  {category}"}
            n_overall = overall_counts[category]
            row["Overall"] = f"{n_overall} ({100 * n_overall / n_total:.1f}%)"
            if group_col:
                for g in groups:
                    sub = df.loc[df[group_col] == g, col]
                    n_g = (sub == category).sum()
                    denom = sub.notna().sum()
                    row[str(g)] = f"{n_g} ({100 * n_g / denom:.1f}%)" if denom > 0 else "0 (0.0%)"
            rows.append(row)

        if group_col and len(groups) >= 2:
            try:
                contingency = pd.crosstab(df[col], df[group_col])
                chi2, p, _, _ = stats.chi2_contingency(contingency)
                rows[-1]["p-value"] = f"{p:.3g}"  # attach to last category row (convention)
            except (ValueError, KeyError):
                pass

    result = pd.DataFrame(rows)
    n_row = {"Variable": "N", "Overall": str(len(df))}
    if group_col:
        for g in groups:
            n_row[str(g)] = str((df[group_col] == g).sum())
    result = pd.concat([pd.DataFrame([n_row]), result], ignore_index=True)
    return result.fillna("")


def table_model_performance_summary(
    cv_metrics_df: pd.DataFrame, decimals: int = 3,
) -> pd.DataFrame:
    """
    Format per-fold CV metrics into a publication-ready summary table:
    one row per metric, mean +/- SD as a formatted string plus raw
    mean/std/min/max/n_folds columns for downstream use.
    """
    rows = []
    for metric in cv_metrics_df.columns:
        vals = cv_metrics_df[metric].dropna()
        rows.append({
            "Metric": metric,
            "Mean \u00b1 SD": f"{vals.mean():.{decimals}f} \u00b1 {vals.std():.{decimals}f}",
            "Min": round(vals.min(), decimals),
            "Max": round(vals.max(), decimals),
            "N folds": len(vals),
        })
    return pd.DataFrame(rows)


def table_top_biomarkers(
    importance_df: pd.DataFrame, value_col: str = None, top_n: int = 25,
) -> pd.DataFrame:
    """
    Ranked, modality-annotated biomarker table for supplementary reporting.
    Splits "modality__feature" index naming into separate Modality/Feature
    columns and adds a Rank column.
    """
    if value_col is None:
        value_col = importance_df.columns[0]
    top = importance_df.sort_values(value_col, ascending=False).head(top_n).copy()

    parsed = top.index.to_series().str.extract(r"^(?:([^_]+)__)?(.+)$")
    modality = parsed[0].fillna("(unspecified)")
    feature = parsed[1]

    result = pd.DataFrame({
        "Rank": range(1, len(top) + 1),
        "Modality": modality.values,
        "Feature": feature.values,
        value_col: top[value_col].round(6).values,
    })
    return result


def save_table(df: pd.DataFrame, filepath_no_ext: str, formats=("csv", "xlsx")) -> dict:
    """Save a table in one or more formats. Returns dict of {format: path}."""
    import os
    os.makedirs(os.path.dirname(filepath_no_ext) or ".", exist_ok=True)
    saved = {}
    for fmt in formats:
        path = f"{filepath_no_ext}.{fmt}"
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "xlsx":
            df.to_excel(path, index=False)
        elif fmt == "tex":
            with open(path, "w") as f:
                f.write(df.to_latex(index=False))
        else:
            raise ValueError(f"Unsupported table format: {fmt}")
        saved[fmt] = path
    return saved


def dataframe_to_image(
    df: pd.DataFrame, filepath: str, title: str = None,
    col_widths: list = None, dpi: int = 300, fontsize: int = 8,
):
    """
    Render a dataframe as a publication-style table image (PNG), for cases
    where a table needs to be embedded as a figure (e.g. a supplementary
    "Table S1" panel) rather than a separate CSV/XLSX file.
    """
    import matplotlib.pyplot as plt
    from eqtmfusion.visualization.style import publication_style

    with publication_style():
        n_rows, n_cols = df.shape
        fig_height = 0.35 * (n_rows + 1) + (0.4 if title else 0)
        fig, ax = plt.subplots(figsize=(min(12, 1.4 * n_cols), fig_height))
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=fontsize + 1, fontweight="bold", pad=10)

        table = ax.table(
            cellText=df.values, colLabels=df.columns, cellLoc="left",
            loc="center", colWidths=col_widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.scale(1, 1.4)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#dddddd")
            if row == 0:
                cell.set_facecolor("#1f3a52")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f7f9fa")

        fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return filepath
