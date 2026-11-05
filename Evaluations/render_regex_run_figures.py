#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Multilingual Medical Diagnosis contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Render publication-style figures for regex run metrics."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CACHE_ROOT = (REPO_ROOT / "Evaluations" / ".cache").resolve()
LOCAL_MPLCONFIGDIR = (REPO_ROOT / "Evaluations" / ".mplconfig").resolve()
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MPLCONFIGDIR))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


DEFAULT_METRICS_DIR = REPO_ROOT / "results" / "evaluation"
DEFAULT_OUTPUT_DIR = DEFAULT_METRICS_DIR / "figures"

MODEL_ORDER = [
    "Qwen 3.5",
    "Llama 3.3 70B",
    "Gemini 3.1 Flash Lite",
    "Gemini 3.5 Flash",
]
LANGUAGE_ORDER = ["english", "spanish", "mandarin", "hindi", "clinical"]
LANGUAGE_LABELS = {
    "english": "English",
    "spanish": "Spanish",
    "mandarin": "Mandarin",
    "hindi": "Hindi",
    "clinical": "Clinical",
}
SYSTEM_LABELS = {
    "sys": "With System Prompt",
    "nosys": "Without System Prompt",
}
BAR_STYLE = {
    "sys": {"color": "#2C7FB8", "hatch": "///"},
    "nosys": {"color": "#F28E2B", "hatch": "\\\\\\"},
}
CLINICAL_STYLE = {
    ("lay_seed", "nosys"): {"label": "Lay Seed, No Sys", "color": "#E07A5F", "hatch": ""},
    ("lay_seed", "sys"): {"label": "Lay Seed, Sys", "color": "#3D405B", "hatch": "///"},
    ("clinical_seed", "nosys"): {"label": "Clinical Seed, No Sys", "color": "#81B29A", "hatch": ""},
    ("clinical_seed", "sys"): {"label": "Clinical Seed, Sys", "color": "#4D908E", "hatch": "xxx"},
}
QWEN_UNIFORM_STYLE = {
    ("lay_seed", "nosys"): {"label": "Lay Seed, No Sys", "color": "#A7C7E7", "hatch": ""},
    ("lay_seed", "sys"): {"label": "Lay Seed, Sys", "color": "#2C5A8A", "hatch": ""},
    ("clinical_seed", "nosys"): {"label": "Clinical Seed, No Sys", "color": "#A7C7E7", "hatch": "///"},
    ("clinical_seed", "sys"): {"label": "Clinical Seed, Sys", "color": "#2C5A8A", "hatch": "///"},
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
}
CONDITION_ORDER = [
    "malar rash",
    "iga vasculitis",
    "juvenile dermatomyositis",
    "dermatomyositis",
    "morphea",
    "normal",
]
CONDITION_LABELS = {
    "lupus": "Lupus",
    "malar rash": "Malar Rash",
    "iga vasculitis": "IgA Vasculitis",
    "juvenile dermatomyositis": "JDM",
    "dermatomyositis": "Dermatomyositis",
    "morphea": "Morphea",
    "normal": "Negative",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("row_count", "positive_count", "negative_count", "true_positive", "false_positive", "false_negative", "true_negative"):
            if key in row:
                row[key] = int(row[key])
        for key in ("accuracy", "precision", "recall", "f1"):
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_publication_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 16,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "figure.titlesize": 19,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.25,
            "savefig.dpi": 300,
            "figure.dpi": 200,
        }
    )


def metric_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        lookup[(row["model"], row["system_prompt"], row["language"], row["prompt_group"])] = row
    return lookup


def condition_accuracy_summary(
    row_level_rows: list[dict[str, Any]],
    *,
    model: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in row_level_rows:
        if row.get("evaluation_set") != "main":
            continue
        if row.get("model") != model:
            continue
        report_language = row.get("report_language")
        prompt_group = row.get("prompt_group")
        if prompt_group != "lay_seed":
            continue
        if report_language in {"english", "spanish", "mandarin", "hindi"}:
            cohort = "lay"
        elif report_language == "clinical":
            cohort = "clinical"
        else:
            continue
        key = (row["canonical_condition"], cohort, row["system_prompt"])
        grouped[key]["total"] += 1
        if str(row.get("is_correct", "")).lower() == "true":
            grouped[key]["correct"] += 1

    summary: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        for cohort in ("lay", "clinical"):
            for system_prompt in ("nosys", "sys"):
                totals = grouped.get((condition, cohort, system_prompt))
                if not totals or totals["total"] == 0:
                    continue
                summary.append(
                    {
                        "condition": condition,
                        "cohort": cohort,
                        "system_prompt": system_prompt,
                        "row_count": totals["total"],
                        "accuracy": totals["correct"] / totals["total"],
                    }
                )
    return summary


def condition_seed_accuracy_summary(
    row_level_rows: list[dict[str, Any]],
    *,
    model: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in row_level_rows:
        if row.get("evaluation_set") != "main":
            continue
        if row.get("model") != model:
            continue
        if row.get("report_language") != "clinical":
            continue
        if row.get("prompt_group") != "clinical_seed":
            continue
        key = (row["canonical_condition"], row["system_prompt"])
        grouped[key]["total"] += 1
        if str(row.get("is_correct", "")).lower() == "true":
            grouped[key]["correct"] += 1

    summary: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        for system_prompt in ("nosys", "sys"):
            totals = grouped.get((condition, system_prompt))
            if not totals or totals["total"] == 0:
                continue
            summary.append(
                {
                    "condition": condition,
                    "system_prompt": system_prompt,
                    "row_count": totals["total"],
                    "accuracy": totals["correct"] / totals["total"],
                }
            )
    return summary


def weighted_model_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["system_prompt"])].append(row)

    summary: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for system_prompt in ("sys", "nosys"):
            group = grouped.get((model, system_prompt), [])
            if not group:
                continue
            total_rows = sum(row["row_count"] for row in group)
            entry = {
                "model": model,
                "system_prompt": system_prompt,
                "row_count": total_rows,
            }
            for metric in ("accuracy", "precision", "recall", "f1"):
                entry[metric] = sum(row["row_count"] * row[metric] for row in group) / total_rows
            summary.append(entry)
    return summary


def weighted_distractor_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["system_prompt"])].append(row)

    summary: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for system_prompt in ("sys", "nosys"):
            group = grouped.get((model, system_prompt), [])
            if not group:
                continue
            total_rows = sum(row["row_count"] for row in group)
            summary.append(
                {
                    "model": model,
                    "system_prompt": system_prompt,
                    "row_count": total_rows,
                    "accuracy": sum(row["row_count"] * row["accuracy"] for row in group) / total_rows,
                    "f1": sum(row["row_count"] * row["f1"] for row in group) / total_rows,
                }
            )
    return summary


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def render_language_metric_panels(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    metric: str,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    axes_flat = axes.flatten()

    base_positions = [0, 1, 2, 3, 4.4]
    main_offsets = {"nosys": -0.16, "sys": 0.16}
    clinical_offsets = {
        ("lay_seed", "nosys"): -0.30,
        ("lay_seed", "sys"): -0.10,
        ("clinical_seed", "nosys"): 0.10,
        ("clinical_seed", "sys"): 0.30,
    }
    bar_width_main = 0.28
    bar_width_clinical = 0.18

    for index, model in enumerate(MODEL_ORDER):
        ax = axes_flat[index]
        for language_index, language in enumerate(LANGUAGE_ORDER):
            x_base = base_positions[language_index]
            if language != "clinical":
                for system_prompt in ("nosys", "sys"):
                    row = lookup.get((model, system_prompt, language, "lay_seed"))
                    if row is None:
                        continue
                    style = BAR_STYLE[system_prompt]
                    ax.bar(
                        x_base + main_offsets[system_prompt],
                        row[metric],
                        width=bar_width_main,
                        color=style["color"],
                        hatch=style["hatch"],
                        edgecolor="black",
                        linewidth=0.8,
                        zorder=3,
                    )
            else:
                for key, offset in clinical_offsets.items():
                    prompt_group, system_prompt = key
                    row = lookup.get((model, system_prompt, language, prompt_group))
                    if row is None:
                        continue
                    style = CLINICAL_STYLE[key]
                    ax.bar(
                        x_base + offset,
                        row[metric],
                        width=bar_width_clinical,
                        color=style["color"],
                        hatch=style["hatch"],
                        edgecolor="black",
                        linewidth=0.8,
                        zorder=3,
                    )

        ax.set_title(model)
        ax.set_xticks(base_positions)
        ax.set_xticklabels([LANGUAGE_LABELS[language] for language in LANGUAGE_ORDER],
                           rotation=20, ha="right", fontsize=12)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(axis="y", zorder=0)

    legend_handles = [
        Patch(
            facecolor=BAR_STYLE["nosys"]["color"],
            hatch=BAR_STYLE["nosys"]["hatch"],
            edgecolor="black",
            label="No Sys",
        ),
        Patch(
            facecolor=BAR_STYLE["sys"]["color"],
            hatch=BAR_STYLE["sys"]["hatch"],
            edgecolor="black",
            label="Sys",
        ),
        Patch(
            facecolor=CLINICAL_STYLE[("lay_seed", "nosys")]["color"],
            edgecolor="black",
            label="Clinical: Lay Seed, No Sys",
        ),
        Patch(
            facecolor=CLINICAL_STYLE[("lay_seed", "sys")]["color"],
            hatch=CLINICAL_STYLE[("lay_seed", "sys")]["hatch"],
            edgecolor="black",
            label="Clinical: Lay Seed, Sys",
        ),
        Patch(
            facecolor=CLINICAL_STYLE[("clinical_seed", "nosys")]["color"],
            edgecolor="black",
            label="Clinical: Clinical Seed, No Sys",
        ),
        Patch(
            facecolor=CLINICAL_STYLE[("clinical_seed", "sys")]["color"],
            hatch=CLINICAL_STYLE[("clinical_seed", "sys")]["hatch"],
            edgecolor="black",
            label="Clinical: Clinical Seed, Sys",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.01, 1.05),
    )
    fig.suptitle(f"Regex {METRIC_LABELS[metric]} by Language and Prompt Condition", y=1.10)
    fig.subplots_adjust(top=0.72, bottom=0.14, wspace=0.24, hspace=0.55)
    save_figure(fig, output_dir, f"language_{metric}_panels")


def render_overall_summary(summary_rows: list[dict[str, Any]], *, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    metrics = ["accuracy", "f1"]
    x_positions = list(range(len(MODEL_ORDER)))
    width = 0.34
    offsets = {"nosys": -width / 2, "sys": width / 2}

    for ax, metric in zip(axes, metrics, strict=True):
        for system_prompt in ("nosys", "sys"):
            values = []
            for model in MODEL_ORDER:
                matching = next(
                    (row for row in summary_rows if row["model"] == model and row["system_prompt"] == system_prompt),
                    None,
                )
                values.append(matching[metric] if matching is not None else 0.0)
            style = BAR_STYLE[system_prompt]
            ax.bar(
                [x + offsets[system_prompt] for x in x_positions],
                values,
                width=width,
                color=style["color"],
                hatch=style["hatch"],
                edgecolor="black",
                linewidth=0.8,
                label=SYSTEM_LABELS[system_prompt],
                zorder=3,
            )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(MODEL_ORDER, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"Weighted {METRIC_LABELS[metric]}")
        ax.grid(axis="y", zorder=0)

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Overall Regex Performance by Model", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "overall_weighted_summary")


def render_distractor_summary(summary_rows: list[dict[str, Any]], *, output_dir: Path) -> None:
    available_models = [model for model in MODEL_ORDER if any(row["model"] == model for row in summary_rows)]
    if not available_models:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    metrics = ["accuracy", "f1"]
    x_positions = list(range(len(available_models)))
    width = 0.34
    offsets = {"nosys": -width / 2, "sys": width / 2}

    for ax, metric in zip(axes, metrics, strict=True):
        for system_prompt in ("nosys", "sys"):
            values = []
            for model in available_models:
                matching = next(
                    (row for row in summary_rows if row["model"] == model and row["system_prompt"] == system_prompt),
                    None,
                )
                values.append(matching[metric] if matching is not None else float("nan"))
            style = BAR_STYLE[system_prompt]
            ax.bar(
                [x + offsets[system_prompt] for x in x_positions],
                values,
                width=width,
                color=style["color"],
                hatch=style["hatch"],
                edgecolor="black",
                linewidth=0.8,
                label=SYSTEM_LABELS[system_prompt],
                zorder=3,
            )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(available_models, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"Distractor Weighted {METRIC_LABELS[metric]}")
        ax.grid(axis="y", zorder=0)

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Distractor Regex Performance by Model", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "distractor_weighted_summary")


def render_clinical_detail(lookup: dict[tuple[str, str, str, str], dict[str, Any]], *, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    metrics = ["accuracy", "f1"]
    clinical_keys = [
        ("lay_seed", "nosys"),
        ("lay_seed", "sys"),
        ("clinical_seed", "nosys"),
        ("clinical_seed", "sys"),
    ]
    x_positions = list(range(len(MODEL_ORDER)))
    width = 0.16
    offsets = [-0.24, -0.08, 0.08, 0.24]

    for ax, metric in zip(axes, metrics, strict=True):
        for offset, key in zip(offsets, clinical_keys, strict=True):
            prompt_group, system_prompt = key
            values = []
            for model in MODEL_ORDER:
                row = lookup.get((model, system_prompt, "clinical", prompt_group))
                values.append(row[metric] if row is not None else 0.0)
            style = CLINICAL_STYLE[key]
            ax.bar(
                [x + offset for x in x_positions],
                values,
                width=width,
                color=style["color"],
                hatch=style["hatch"],
                edgecolor="black",
                linewidth=0.8,
                label=style["label"],
                zorder=3,
            )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(MODEL_ORDER, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"Clinical {METRIC_LABELS[metric]}")
        ax.grid(axis="y", zorder=0)

    handles = [
        Patch(facecolor=CLINICAL_STYLE[key]["color"], hatch=CLINICAL_STYLE[key]["hatch"], edgecolor="black", label=CLINICAL_STYLE[key]["label"])
        for key in clinical_keys
    ]
    fig.legend(handles=handles, loc="upper left", ncol=1, frameon=False, bbox_to_anchor=(0.01, 1.10))
    fig.suptitle("Clinical Prompt Breakdown", y=1.15)
    fig.tight_layout()
    fig.subplots_adjust(top=0.66)
    save_figure(fig, output_dir, "clinical_detail_summary")


def render_qwen_language_accuracy(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    model = "Qwen 3.5"
    base_positions = [0, 1, 2, 3, 4.4]
    main_offsets = {"nosys": -0.16, "sys": 0.16}
    clinical_offsets = {
        ("lay_seed", "nosys"): -0.30,
        ("lay_seed", "sys"): -0.10,
        ("clinical_seed", "nosys"): 0.10,
        ("clinical_seed", "sys"): 0.30,
    }
    bar_width_main = 0.28
    bar_width_clinical = 0.18

    for language_index, language in enumerate(LANGUAGE_ORDER):
        x_base = base_positions[language_index]
        if language != "clinical":
            for system_prompt in ("nosys", "sys"):
                row = lookup.get((model, system_prompt, language, "lay_seed"))
                if row is None:
                    continue
                style = QWEN_UNIFORM_STYLE[("lay_seed", system_prompt)]
                ax.bar(
                    x_base + main_offsets[system_prompt],
                    row["accuracy"],
                    width=bar_width_main,
                    color=style["color"],
                    hatch=style["hatch"],
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=3,
                )
        else:
            for key, offset in clinical_offsets.items():
                prompt_group, system_prompt = key
                row = lookup.get((model, system_prompt, language, prompt_group))
                if row is None:
                    continue
                style = QWEN_UNIFORM_STYLE[key]
                ax.bar(
                    x_base + offset,
                    row["accuracy"],
                    width=bar_width_clinical,
                    color=style["color"],
                    hatch=style["hatch"],
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=3,
                )

    ax.set_xticks(base_positions)
    ax.set_xticklabels([LANGUAGE_LABELS[language] for language in LANGUAGE_ORDER])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", zorder=0)

    legend_handles = [
        Patch(
            facecolor=QWEN_UNIFORM_STYLE[("lay_seed", "nosys")]["color"],
            edgecolor="black",
            label=QWEN_UNIFORM_STYLE[("lay_seed", "nosys")]["label"],
        ),
        Patch(
            facecolor=QWEN_UNIFORM_STYLE[("lay_seed", "sys")]["color"],
            edgecolor="black",
            label=QWEN_UNIFORM_STYLE[("lay_seed", "sys")]["label"],
        ),
        Patch(
            facecolor=QWEN_UNIFORM_STYLE[("clinical_seed", "nosys")]["color"],
            hatch=QWEN_UNIFORM_STYLE[("clinical_seed", "nosys")]["hatch"],
            edgecolor="black",
            label=QWEN_UNIFORM_STYLE[("clinical_seed", "nosys")]["label"],
        ),
        Patch(
            facecolor=QWEN_UNIFORM_STYLE[("clinical_seed", "sys")]["color"],
            hatch=QWEN_UNIFORM_STYLE[("clinical_seed", "sys")]["hatch"],
            edgecolor="black",
            label=QWEN_UNIFORM_STYLE[("clinical_seed", "sys")]["label"],
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True, framealpha=0.95)
    fig.tight_layout()
    save_figure(fig, output_dir, "qwen_language_accuracy")


def render_qwen_condition_accuracy(
    summary_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12.6, 5.2))
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    x_positions = list(range(len(CONDITION_ORDER)))
    width = 0.18
    bar_keys = [
        ("lay", "nosys"),
        ("lay", "sys"),
        ("clinical", "nosys"),
        ("clinical", "sys"),
    ]
    offsets = {
        ("lay", "nosys"): -0.27,
        ("lay", "sys"): -0.09,
        ("clinical", "nosys"): 0.09,
        ("clinical", "sys"): 0.27,
    }
    cohort_style = {
        ("lay", "nosys"): {"label": "Lay, No Sys", "color": "#A7C7E7", "hatch": ""},
        ("lay", "sys"): {"label": "Lay, Sys", "color": "#2C5A8A", "hatch": ""},
        ("clinical", "nosys"): {"label": "Clinical, No Sys", "color": "#A7C7E7", "hatch": "///"},
        ("clinical", "sys"): {"label": "Clinical, Sys", "color": "#2C5A8A", "hatch": "///"},
    }

    for key in bar_keys:
        values: list[float] = []
        mask: list[bool] = []
        for condition in CONDITION_ORDER:
            row = next(
                (
                    item
                    for item in summary_rows
                    if item["condition"] == condition
                    and item["cohort"] == key[0]
                    and item["system_prompt"] == key[1]
                ),
                None,
            )
            if row is None:
                values.append(0.0)
                mask.append(False)
            else:
                values.append(row["accuracy"])
                mask.append(True)
        style = cohort_style[key]
        xs = [x + offsets[key] for x, present in zip(x_positions, mask, strict=True) if present]
        ys = [y for y, present in zip(values, mask, strict=True) if present]
        ax.bar(
            xs,
            ys,
            width=width,
            color=style["color"],
            hatch=style["hatch"],
            edgecolor="black",
            linewidth=0.8,
            label=style["label"],
            zorder=3,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([CONDITION_LABELS[condition] for condition in CONDITION_ORDER], rotation=18, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Accuracy", fontsize=20)
    ax.grid(axis="y", zorder=0)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.95,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_figure(fig, output_dir, "qwen_condition_accuracy")


def render_qwen_clinical_seed_accuracy(
    summary_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 4.8))
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    condition_rows = [condition for condition in CONDITION_ORDER if any(row["condition"] == condition for row in summary_rows)]
    x_positions = list(range(len(condition_rows)))
    width = 0.34
    offsets = {"nosys": -width / 2, "sys": width / 2}

    for system_prompt in ("nosys", "sys"):
        values = []
        for condition in condition_rows:
            row = next(
                (item for item in summary_rows if item["condition"] == condition and item["system_prompt"] == system_prompt),
                None,
            )
            values.append(row["accuracy"] if row is not None else 0.0)
        style = {
            "nosys": {"label": "Clinical Seed, No Sys", "color": "#A7C7E7", "hatch": "///"},
            "sys": {"label": "Clinical Seed, Sys", "color": "#2C5A8A", "hatch": "///"},
        }[system_prompt]
        ax.bar(
            [x + offsets[system_prompt] for x in x_positions],
            values,
            width=width,
            color=style["color"],
            hatch=style["hatch"],
            edgecolor="black",
            linewidth=0.8,
            label=style["label"],
            zorder=3,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([CONDITION_LABELS[condition] for condition in condition_rows], rotation=18, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Accuracy", fontsize=20)
    ax.grid(axis="y", zorder=0)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.95,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    save_figure(fig, output_dir, "qwen_clinical_seed_condition_accuracy")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    metrics_dir = args.metrics_dir.resolve()
    output_dir = ensure_output_dir(args.output_dir.resolve())
    set_publication_style()

    main_rows = load_csv(metrics_dir / "main_metrics.csv")
    distractor_rows = load_csv(metrics_dir / "distractor_metrics.csv")
    row_level_rows = load_csv(metrics_dir / "row_level_results.csv")
    main_lookup = metric_lookup(main_rows)
    overall_summary = weighted_model_summary(main_rows)
    distractor_summary = weighted_distractor_summary(distractor_rows)
    qwen_condition_summary = condition_accuracy_summary(row_level_rows, model="Qwen 3.5")
    qwen_clinical_seed_summary = condition_seed_accuracy_summary(row_level_rows, model="Qwen 3.5")

    render_language_metric_panels(main_lookup, metric="accuracy", output_dir=output_dir)
    render_language_metric_panels(main_lookup, metric="f1", output_dir=output_dir)
    render_overall_summary(overall_summary, output_dir=output_dir)
    render_distractor_summary(distractor_summary, output_dir=output_dir)
    render_clinical_detail(main_lookup, output_dir=output_dir)
    render_qwen_language_accuracy(main_lookup, output_dir=output_dir)
    render_qwen_condition_accuracy(qwen_condition_summary, output_dir=output_dir)
    render_qwen_clinical_seed_accuracy(qwen_clinical_seed_summary, output_dir=output_dir)

    write_summary_csv(output_dir / "overall_weighted_summary.csv", overall_summary)
    write_summary_csv(output_dir / "distractor_weighted_summary.csv", distractor_summary)
    write_summary_csv(output_dir / "qwen_condition_accuracy_summary.csv", qwen_condition_summary)
    write_summary_csv(output_dir / "qwen_clinical_seed_condition_accuracy_summary.csv", qwen_clinical_seed_summary)

    print(f"figures: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
