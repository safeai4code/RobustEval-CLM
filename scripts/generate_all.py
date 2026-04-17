"""Generate all result tables and figures from outputs_public.

Combines:
  - collect_passat1.py     → passat1.tex
  - collect_passat1_noise.py → passat1_noise.tex, noise_results.csv
  - collect_rrs.py         → rrs.tex
  - plot_noise_figure.py   → noise_robustness_figure.{pdf,png}
"""

import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.metadata import ATTACKS, DATASETS, MODELS, NOISE_LEVELS, NOISE_TYPES, QUANTIZED_TYPES

OUTPUTS_DIR = ROOT / "outputs_public"
RESULTS_DIR = ROOT / "statistic_results"

# ── Shared labels ───────────────────────────────────────────────────────────────

DATASET_LABELS = {"mbpp": "MBPP", "humaneval": "HumanEval", "canitedit": "CanItEdit"}
QUANT_LABELS = {"base": "FP", "bnb8": "8-bit", "bnb4": "4-bit"}
NOISE_TYPE_LABELS = {"gaussian": "G", "uniform": "U"}

# ── Shared helpers ──────────────────────────────────────────────────────────────


def _read_pass_rates(folder: Path) -> Optional[tuple[float, float]]:
    """Load (base, plus) pass@1 floats from pass_rates.json, or None."""
    result_path = folder / "pass_rates.json"
    if not result_path.exists():
        return None
    try:
        with open(result_path) as fh:
            data = json.load(fh)
        base_val = data.get("base")
        plus_val = data.get("plus")
        if base_val is None or plus_val is None:
            return None
        return float(base_val), float(plus_val)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _fmt_pass_pair(folder: Path) -> str:
    """Return 'base / plus' percentage string, or 'x' if missing."""
    pair = _read_pass_rates(folder)
    if pair is None:
        return "x"
    return f"{pair[0] * 100:.1f} / {pair[1] * 100:.1f}"


def _adversarial_folder(dataset: str, org: str, model_name: str, quant_type: str,
                        attack: Optional[str] = None) -> Path:
    """Build output folder path for adversarial/clean results."""
    base = OUTPUTS_DIR / dataset / org / model_name
    if attack is None:
        return base / "noise" / "gaussian" / "0.0" / quant_type
    return base / attack / quant_type


def _noise_folder(dataset: str, org: str, model_name: str, quant_type: str,
                  noise_type: str, level: str) -> Path:
    """Build output folder path for noise results."""
    return OUTPUTS_DIR / dataset / org / model_name / "noise" / noise_type / level / quant_type


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Pass@1 adversarial table  (passat1.tex)
# ═══════════════════════════════════════════════════════════════════════════════


def _build_passat1_table(dataset: str) -> str:
    """Build a LaTeX table for one dataset's adversarial pass@1 results."""
    ds_label = DATASET_LABELS.get(dataset, dataset.capitalize())

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\begin{tabular}{lll cccc}",
        r"\toprule",
        f"\\multirow{{2}}{{*}}{{\\textbf{{Family}}}} & "
        f"\\multirow{{2}}{{*}}{{\\textbf{{Model}}}} & "
        f"\\multirow{{2}}{{*}}{{\\textbf{{Prec.}}}} & "
        f"\\multicolumn{{4}}{{c}}{{\\textbf{{{ds_label} (Base / Plus)}}}} \\\\",
        r"\cmidrule(lr){4-7}",
        r"& & & Clean & Char & Synonym & Translate \\",
        r"\midrule",
    ]

    for family_idx, (family, models) in enumerate(MODELS.items()):
        quant_types = QUANTIZED_TYPES['moe'] if family == 'MOE' else QUANTIZED_TYPES['adversarial']
        n_family_rows = len(models) * len(quant_types)
        lines.append(f"% {'=' * 20} {family} Family {'=' * 20}")
        lines.append(f"\\multirow{{{n_family_rows}}}{{*}}{{\\textbf{{{family}}}}}")

        for model_idx, model_path in enumerate(models):
            org, model_name = model_path.split("/", 1)
            n_quant_rows = len(quant_types)

            for quant_idx, quant_type in enumerate(quant_types):
                quant_label = QUANT_LABELS.get(quant_type, quant_type)
                clean = _fmt_pass_pair(_adversarial_folder(dataset, org, model_name, quant_type))
                adv_cells = [
                    _fmt_pass_pair(_adversarial_folder(dataset, org, model_name, quant_type, atk))
                    for atk in ATTACKS
                ]
                cells_str = " & ".join([clean] + adv_cells)

                if quant_idx == 0:
                    model_col = f"\\multirow{{{n_quant_rows}}}{{*}}{{{model_name}}}"
                    lines.append(f"& {model_col} & {quant_label} & {cells_str} \\\\")
                else:
                    pad = " " * (len(model_name) + 20)
                    lines.append(f"& {pad} & {quant_label} & {cells_str} \\\\")

            is_last_model = model_idx == len(models) - 1
            is_last_family = family_idx == len(MODELS) - 1
            if not is_last_model:
                lines.append(r"\cmidrule{2-7}")
            elif not is_last_family:
                lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2mm}",
        f"\\caption{{Detailed pass@1 results for Clean performance and three Adversarial attacks "
        f"on the \\textbf{{{ds_label}}} dataset. "
        f"Each cell reports the performance on the Base and Plus test sets respectively (Base / Plus).}}",
        f"\\label{{tab:adv_{dataset}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def generate_passat1_tex() -> None:
    """Write passat1.tex with one adversarial table per dataset."""
    output_path = RESULTS_DIR / "passat1.tex"
    tables = [_build_passat1_table(ds) for ds in DATASETS]
    with open(output_path, "w") as f:
        f.write("\n\n".join(tables))
    print(f"Written to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Pass@1 noise tables + CSV  (passat1_noise.tex, noise_results.csv)
# ═══════════════════════════════════════════════════════════════════════════════


def _build_noise_family_table(family: str, model_paths: list[str]) -> str:
    """Build a LaTeX table for one model family's noise results."""
    n_levels = len(NOISE_LEVELS)
    last_col = 4 + n_levels
    quant_types = QUANTIZED_TYPES['moe'] if family == 'MOE' else QUANTIZED_TYPES['noise']

    col_spec = "ll ll" + " c" * n_levels
    level_headers = " & ".join(NOISE_LEVELS)

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\setlength{\tabcolsep}{3pt}",
        r"\footnotesize",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"\\textbf{{Model}} & \\textbf{{Dataset}} & \\textbf{{Prec.}} & \\textbf{{Noise}}"
        f" & \\multicolumn{{{n_levels}}}{{c}}{{\\textbf{{Pass@1 (Base / Plus)}}}} \\\\",
        f"\\cmidrule(lr){{5-{last_col}}}",
        f" & & & & {level_headers} \\\\",
        r"\midrule",
    ]

    for model_idx, model_path in enumerate(model_paths):
        org, model_name = model_path.split("/", 1)
        n_model_rows = len(DATASETS) * len(quant_types) * len(NOISE_TYPES)

        for ds_idx, ds in enumerate(DATASETS):
            ds_label = DATASET_LABELS.get(ds, ds.capitalize())
            n_ds_rows = len(quant_types) * len(NOISE_TYPES)

            for quant_idx, quant_type in enumerate(quant_types):
                quant_label = QUANT_LABELS.get(quant_type, quant_type)
                n_quant_rows = len(NOISE_TYPES)

                for noise_idx, noise_type in enumerate(NOISE_TYPES):
                    noise_label = NOISE_TYPE_LABELS.get(noise_type, noise_type)
                    level_cells = [
                        _fmt_pass_pair(_noise_folder(ds, org, model_name, quant_type, noise_type, lv))
                        for lv in NOISE_LEVELS
                    ]
                    cells_str = " & ".join(level_cells)

                    if ds_idx == 0 and quant_idx == 0 and noise_idx == 0:
                        model_col = f"\\multirow{{{n_model_rows}}}{{*}}{{{model_name}}}"
                    else:
                        model_col = ""
                    if quant_idx == 0 and noise_idx == 0:
                        ds_col = f"\\multirow{{{n_ds_rows}}}{{*}}{{{ds_label}}}"
                    else:
                        ds_col = ""
                    if noise_idx == 0:
                        quant_col = f"\\multirow{{{n_quant_rows}}}{{*}}{{{quant_label}}}"
                    else:
                        quant_col = ""

                    lines.append(f"{model_col} & {ds_col} & {quant_col} & {noise_label} & {cells_str} \\\\")

            if ds_idx < len(DATASETS) - 1:
                lines.append(f"\\cmidrule{{2-{last_col}}}")
        if model_idx < len(model_paths) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2mm}",
        f"\\caption{{Pass@1 results under weight noise perturbations for the "
        f"\\textbf{{{family}}} family. "
        f"G = Gaussian, U = Uniform. "
        f"Each cell reports Base / Plus performance.}}",
        f"\\label{{tab:noise_{family.lower()}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def _collect_noise_csv_rows() -> list[dict]:
    """Collect all noise pass@1 results into row dicts for CSV output."""
    rows = []
    for dataset in DATASETS:
        for family, models in MODELS.items():
            quant_types = QUANTIZED_TYPES['moe'] if family == 'MOE' else QUANTIZED_TYPES['noise']
            for model_path in models:
                org, model_name = model_path.split("/", 1)
                for quant_type in quant_types:
                    quant_label = QUANT_LABELS.get(quant_type, quant_type)
                    for noise_type in NOISE_TYPES:
                        row: dict = {
                            "dataset": dataset,
                            "family": family,
                            "model": model_name,
                            "prec": quant_label,
                            "noise_type": noise_type,
                        }
                        for level in NOISE_LEVELS:
                            row[level] = _fmt_pass_pair(
                                _noise_folder(dataset, org, model_name, quant_type, noise_type, level)
                            )
                        rows.append(row)
    return rows


def generate_noise_tex_csv() -> None:
    """Write passat1_noise.tex and noise_results.csv."""
    output_tex = RESULTS_DIR / "passat1_noise.tex"
    tables = [_build_noise_family_table(fam, paths) for fam, paths in MODELS.items()]
    with open(output_tex, "w") as f:
        f.write("\n\n".join(tables))
    print(f"Written to {output_tex}")

    output_csv = RESULTS_DIR / "noise_results.csv"
    rows = _collect_noise_csv_rows()
    fieldnames = ["dataset", "family", "model", "prec", "noise_type"] + NOISE_LEVELS
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written to {output_csv}")


# ═══════════════════════════════════════════════════════════════════════════════
#  3. RRS table  (rrs.tex)
# ═══════════════════════════════════════════════════════════════════════════════

_SIZE_PATTERN = re.compile(r"(\d+(?:\.\d+)?[bBmMkK])", re.IGNORECASE)
_MOE_SIZE_LABELS = {
    "Qwen3-Coder-30B-A3B-Instruct": "Q-30B-A3B",
    "DeepSeek-Coder-V2-Lite-Base": "DS-V2-L",
    "Codestral-22B-v0.1": "CS-22B",
    "Mistral-Small-3.2-24B-Instruct-2506": "MS-24B",
}


def _extract_size_label(model_name: str) -> str:
    """Extract a human-readable size label from a model name."""
    if model_name in _MOE_SIZE_LABELS:
        return _MOE_SIZE_LABELS[model_name]
    match = _SIZE_PATTERN.search(model_name)
    if match:
        raw = match.group(1)
        suffix = raw[-1].upper()
        number_str = raw[:-1]
        try:
            number = float(number_str)
            if number == int(number):
                return f"{int(number)}{suffix}"
            return f"{number}{suffix}"
        except ValueError:
            return raw.upper()
    return model_name


def _compute_rrs(before_base: float, after_base: float,
                 before_bnb8: float, after_bnb8: float) -> float:
    """Compute Robustness Ratio Score."""
    numerator = abs(before_base - after_base)
    denominator = abs(before_bnb8 - after_bnb8)
    if denominator == 0.0:
        return float("inf")
    return numerator / denominator


def _rrs_pair(dataset: str, org: str, model_name: str, attack: str,
              base_idx: int, plus_idx: int,
              quant_type: str = "bnb8") -> tuple[str, list[float]]:
    """Compute a 'base / plus' RRS cell string and the raw RRS values."""
    before_fp = _read_pass_rates(_adversarial_folder(dataset, org, model_name, "base"))
    after_fp = _read_pass_rates(_adversarial_folder(dataset, org, model_name, "base", attack))
    before_q = _read_pass_rates(_adversarial_folder(dataset, org, model_name, quant_type))
    after_q = _read_pass_rates(_adversarial_folder(dataset, org, model_name, quant_type, attack))

    if any(v is None for v in [before_fp, after_fp, before_q, after_q]):
        return "--/--", []

    raw_values: list[float] = []

    def _fmt(idx: int) -> str:
        rrs = _compute_rrs(before_fp[idx], after_fp[idx], before_q[idx], after_q[idx])
        raw_values.append(rrs)
        if rrs == float("inf"):
            val_str = r"$\infty$"
        else:
            val_str = f"{rrs:.2f}"
        eps = 1e-9
        if rrs > 1.0 + eps:
            return rf"\dred{{{val_str}}}"
        if rrs < 1.0 - eps:
            return rf"\dgreen{{{val_str}}}"
        return rf"\black{{{val_str}}}"

    cell = f"{_fmt(base_idx)} / {_fmt(plus_idx)}"
    return cell, raw_values


def _build_rrs_table() -> tuple[str, int, int, int]:
    """Build a single LaTeX RRS table combining all families, datasets, attacks."""
    family_rows: list[tuple[str, list[tuple[str, list[str], str]]]] = []
    red_count = green_count = black_count = 0
    col_red = [0] * 9
    col_total = [0] * 9
    eps = 1e-9

    ds_configs = [
        ("mbpp", 0, 1),
        ("humaneval", 0, 1),
        ("canitedit", 1, 0),  # CanItEdit-L = plus (1), CanItEdit-D = base (0)
    ]

    for family, model_paths in MODELS.items():
        model_data: list[tuple[str, list[str], str]] = []
        for model_path in model_paths:
            org, model_name = model_path.split("/", 1)
            size_label = _extract_size_label(model_name)
            quant = "bnb4" if family == "MOE" else "bnb8"

            cells: list[str] = []
            row_total = row_red = 0
            col_idx = 0

            for ds, bi, pi in ds_configs:
                for attack in ATTACKS:
                    cell, raws = _rrs_pair(ds, org, model_name, attack, bi, pi, quant)
                    cells.append(cell)
                    n_red = sum(1 for v in raws if v > 1.0 + eps)
                    row_total += len(raws)
                    row_red += n_red
                    col_red[col_idx] += n_red
                    col_total[col_idx] += len(raws)
                    col_idx += 1
                    red_count += n_red
                    green_count += sum(1 for v in raws if v < 1.0 - eps)
                    black_count += sum(1 for v in raws if 1.0 - eps <= v <= 1.0 + eps)

            pct_str = f"{row_red / row_total * 100:.0f}\\%" if row_total > 0 else "--"
            model_data.append((size_label, cells, pct_str))
        family_rows.append((family, model_data))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Robustness Comparison (RRS) of Original and Quantized LLMs Under"
        r" Adversarial Attacks. Ch/W/S = character/word/sentence-level. Each cell shows"
        r" \textit{base\,/\,plus} results. RRS${>}1$ indicates quantized model is more"
        r" robust; $\uparrow$/$\downarrow$ denote infinite-ratio cases.}",
        r"\label{tab:rq1}",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{0.90}",
        r"\footnotesize",
        r"\begin{tabular}{ll ccc ccc ccc c}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Family}} &",
        r"\multirow{2}{*}{\textbf{Size}} &",
        r"\multicolumn{3}{c}{\textbf{MBPP / MBPP+}} &",
        r"\multicolumn{3}{c}{\textbf{HumanEval / HumanEval+}} &",
        r"\multicolumn{3}{c}{\textbf{CanItEdit-L / CanItEdit-D}} & \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"  & & \textbf{Ch} & \textbf{W} & \textbf{S}"
        r"    & \textbf{Ch} & \textbf{W} & \textbf{S}"
        r"    & \textbf{Ch} & \textbf{W} & \textbf{S}"
        r"    & \textbf{\%${>}1$} \\",
        r"\midrule",
    ]

    for fam_idx, (family, model_data) in enumerate(family_rows):
        n_rows = len(model_data)
        lines.append(f"\\multirow{{{n_rows}}}{{*}}{{{family}}}")
        for size_label, cells, pct_str in model_data:
            cells_str = " & ".join(cells)
            lines.append(f"  & {size_label} & {cells_str} & {pct_str} \\\\")
        if fam_idx < len(family_rows) - 1:
            lines.append(r"\midrule")

    col_summary_cells = []
    for i in range(9):
        if col_total[i] > 0:
            col_summary_cells.append(f"{col_red[i] / col_total[i] * 100:.0f}\\%")
        else:
            col_summary_cells.append("--")
    all_red = sum(col_red)
    all_total = sum(col_total)
    overall_pct = f"{all_red / all_total * 100:.0f}\\%" if all_total > 0 else "--"
    col_summary_str = " & ".join(col_summary_cells)
    lines.append(r"\midrule")
    lines.append(
        f"  \\multicolumn{{2}}{{l}}{{\\textbf{{\\%${{{'>'}}}1$}}}}"
        f" & {col_summary_str} & \\\\"
    )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    return "\n".join(lines), red_count, green_count, black_count


def generate_rrs_tex() -> None:
    """Write rrs.tex with the combined RRS table."""
    output_tex = RESULTS_DIR / "rrs.tex"
    table, red_count, green_count, black_count = _build_rrs_table()
    with open(output_tex, "w") as fh:
        fh.write(table + "\n")
    print(f"Written to {output_tex}")
    total_count = red_count + green_count + black_count
    if total_count > 0:
        print(f"Total cells: {total_count}")
        print(f"Red   (RRS > 1): {red_count} ({red_count / total_count * 100:.2f}%)")
        print(f"Green (RRS < 1): {green_count} ({green_count / total_count * 100:.2f}%)")
        print(f"Black (RRS = 1): {black_count} ({black_count / total_count * 100:.2f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Noise robustness figure  (noise_robustness_figure.{pdf,png})
# ═══════════════════════════════════════════════════════════════════════════════

FAMILY_ORDER = ["DeepSeek", "LLaMA", "CodeGen", "StarCoder", "NextCoder", "Gemma", "MOE"]
DATASET_ORDER = ["mbpp", "humaneval", "canitedit"]
NOISE_ORDER = ["gaussian", "uniform"]
NOISE_LABELS_LONG = {"gaussian": "Gaussian", "uniform": "Uniform"}
QUANT_STYLES = {"base": "-", "bnb8": "--"}
QUANT_STYLES_MOE = {"base": "-", "bnb4": "--"}
COLORS_PER_FAMILY = {
    "DeepSeek": ["#1f77b4", "#ff7f0e", "#2ca02c"],
    "LLaMA": ["#1f77b4", "#ff7f0e", "#2ca02c"],
    "CodeGen": ["#1f77b4", "#ff7f0e", "#2ca02c"],
    "StarCoder": ["#1f77b4", "#ff7f0e", "#2ca02c"],
    "NextCoder": ["#1f77b4", "#ff7f0e", "#2ca02c"],
    "Gemma": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    "MOE": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
}


def _load_plus_pass_at_1(dataset: str, model: str, noise_type: str,
                         noise_level: str, quant: str) -> Optional[float]:
    """Load plus pass@1 for the noise figure (uses org/model_name path)."""
    fpath = OUTPUTS_DIR / dataset / model / "noise" / noise_type / noise_level / quant / "pass_rates.json"
    if not fpath.exists():
        return None
    try:
        with open(fpath) as f:
            data = json.load(f)
        return data["plus"]
    except (KeyError, json.JSONDecodeError):
        return None


def _short_model_name(full_name: str) -> str:
    return full_name.split("/")[-1]


def generate_noise_figure() -> None:
    """Generate the noise robustness figure as PDF and PNG."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5,
        "lines.linewidth": 0.7,
        "lines.markersize": 1,
    })

    n_rows = len(FAMILY_ORDER)
    n_cols = len(DATASET_ORDER) * len(NOISE_ORDER)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 9), sharex=True)
    fig.subplots_adjust(left=0.06, right=0.82, top=0.93, bottom=0.07, hspace=0.2, wspace=0.06)

    x_positions = np.arange(len(NOISE_LEVELS))
    x_labels = NOISE_LEVELS

    for row_idx, family in enumerate(FAMILY_ORDER):
        models = MODELS[family]
        colors = COLORS_PER_FAMILY[family]
        row_vals = []

        for col_idx, (ds, nt) in enumerate(
            [(ds, nt) for ds in DATASET_ORDER for nt in NOISE_ORDER]
        ):
            ax = axes[row_idx, col_idx]
            for m_idx, model in enumerate(models):
                color = colors[m_idx]
                short_name = _short_model_name(model)
                quant_map = QUANT_STYLES_MOE if family == "MOE" else QUANT_STYLES

                for quant, ls in quant_map.items():
                    vals = []
                    xs = []
                    for nl_idx, nl in enumerate(NOISE_LEVELS):
                        v = _load_plus_pass_at_1(ds, model, nt, nl, quant)
                        if v is not None:
                            vals.append(v)
                            xs.append(nl_idx)
                            row_vals.append(v)
                    if not vals:
                        continue
                    label_suffix = {"bnb8": " (8bit)", "bnb4": " (4bit)"}.get(quant, "")
                    ax.plot(xs, vals, linestyle=ls, color=color, marker="o",
                            label=f"{short_name}{label_suffix}")

            ax.set_xlim(-0.3, len(NOISE_LEVELS) - 0.7)
            if row_idx == n_rows - 1:
                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_labels, rotation=45, ha="right")
            else:
                ax.set_xticks(x_positions)
                ax.set_xticklabels([])
            if col_idx == 0:
                ax.set_ylabel(f"{family}\npass@1")
            ax.grid(True, alpha=0.3, linewidth=0.5)

        y_min = min(row_vals) if row_vals else 0
        y_max = max(row_vals) if row_vals else 1
        margin = (y_max - y_min) * 0.08 if y_max > y_min else 0.05
        shared_ylim = (y_min - margin, y_max + margin)
        for col_idx in range(n_cols):
            axes[row_idx, col_idx].set_ylim(shared_ylim)
            if col_idx > 0:
                axes[row_idx, col_idx].set_yticklabels([])

    for col_idx, (ds, nt) in enumerate(
        [(ds, nt) for ds in DATASET_ORDER for nt in NOISE_ORDER]
    ):
        axes[0, col_idx].set_title(f"{NOISE_LABELS_LONG[nt]}", pad=8)

    fig.canvas.draw()
    for ds_idx, ds in enumerate(DATASET_ORDER):
        left_col = ds_idx * 2
        right_col = ds_idx * 2 + 1
        left_pos = axes[0, left_col].get_position()
        right_pos = axes[0, right_col].get_position()
        center_x = (left_pos.x0 + right_pos.x1) / 2
        top_y = left_pos.y1
        fig.text(center_x, top_y + 0.04, DATASET_LABELS[ds], ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    for row_idx, family in enumerate(FAMILY_ORDER):
        handles_dict = {}
        for col_idx in range(n_cols):
            h, l = axes[row_idx, col_idx].get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in handles_dict:
                    handles_dict[label] = handle
        if handles_dict:
            handles = list(handles_dict.values())
            labels = list(handles_dict.keys())
            axes[row_idx, -1].legend(
                handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                frameon=False, handlelength=1.5,
            )

    fig.supxlabel("Noise Level", fontsize=8)

    out_pdf = RESULTS_DIR / "noise_robustness_figure.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved to {out_pdf} and {out_png}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("1/4  Generating passat1.tex ...")
    generate_passat1_tex()

    print("=" * 60)
    print("2/4  Generating passat1_noise.tex + noise_results.csv ...")
    generate_noise_tex_csv()

    print("=" * 60)
    print("3/4  Generating rrs.tex ...")
    generate_rrs_tex()

    print("=" * 60)
    print("4/4  Generating noise_robustness_figure ...")
    generate_noise_figure()

    print("=" * 60)
    print("All done.")


if __name__ == "__main__":
    main()
