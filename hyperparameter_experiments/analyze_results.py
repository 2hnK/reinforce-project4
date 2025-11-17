from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class GroupEntry:
    label: str
    experiment: str
    value: float | str | None


@dataclass
class GroupConfig:
    group_id: str
    title: str
    parameter: str
    entries: List[GroupEntry]


ROOT = Path(__file__).resolve().parent
STABILITY_DIR = ROOT / "stability"
COMBINATION_DIR = ROOT / "combination"
SINGLE_RESULTS = STABILITY_DIR / "results" / "hyperparameter_experiments_final.json"
COMB_RESULTS = COMBINATION_DIR / "results" / "combination_experiments_final.json"
SINGLE_VIZ_DIR = SINGLE_RESULTS.parent / "visualizations" / "summary"
COMB_VIZ_DIR = COMB_RESULTS.parent / "visualizations"
TABLE_OUTPUT = SINGLE_RESULTS.parent / "SUMMARY_TABLES.md"

SINGLE_GROUPS: List[GroupConfig] = [
    GroupConfig(
        group_id="clip_param",
        title="Clip Parameter Sweep",
        parameter="clip_param",
        entries=[
            GroupEntry("clip=0.10 (conservative)", "clip_conservative", 0.10),
            GroupEntry("clip=0.20 (baseline)", "baseline", 0.20),
            GroupEntry("clip=0.30 (aggressive)", "clip_aggressive", 0.30),
        ],
    ),
    GroupConfig(
        group_id="entropy_coeff",
        title="Entropy Coefficient Sweep",
        parameter="entropy_coeff",
        entries=[
            GroupEntry("entropy=0.000 (baseline)", "baseline", 0.0),
            GroupEntry("entropy=0.001", "entropy_minimal", 0.001),
            GroupEntry("entropy=0.010", "entropy_medium", 0.01),
            GroupEntry("entropy=0.050", "entropy_high", 0.05),
        ],
    ),
    GroupConfig(
        group_id="gamma",
        title="Discount Factor Sweep",
        parameter="gamma",
        entries=[
            GroupEntry("gamma=0.95 (short)", "gamma_short", 0.95),
            GroupEntry("gamma=0.99 (baseline)", "baseline", 0.99),
            GroupEntry("gamma=0.995 (long)", "gamma_long", 0.995),
        ],
    ),
    GroupConfig(
        group_id="grad_clip",
        title="Gradient Clipping Sweep",
        parameter="grad_clip",
        entries=[
            GroupEntry("no clip (baseline)", "baseline", "None"),
            GroupEntry("grad_clip=0.5", "grad_clip_tight", 0.5),
            GroupEntry("grad_clip=1.0", "grad_clip_loose", 1.0),
        ],
    ),
    GroupConfig(
        group_id="vf_clip",
        title="Value Function Clip Sweep",
        parameter="vf_clip_param",
        entries=[
            GroupEntry("vf_clip=1.0", "vf_clip_tight", 1.0),
            GroupEntry("vf_clip=10.0 (baseline)", "baseline", 10.0),
            GroupEntry("vf_clip=100.0", "vf_clip_loose", 100.0),
        ],
    ),
    GroupConfig(
        group_id="kl_coeff",
        title="KL Constraint Sweep",
        parameter="kl_coeff",
        entries=[
            GroupEntry("KL disabled", "kl_disabled", "off"),
            GroupEntry("kl_coeff=0.2 (baseline)", "baseline", 0.2),
            GroupEntry("kl_coeff=0.1", "kl_weak", 0.1),
            GroupEntry("kl_coeff=0.5", "kl_strong", 0.5),
        ],
    ),
]


def load_experiment_map(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    data = json.loads(path.read_text())
    return {exp["name"]: exp for exp in data["experiments"]}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_markdown_table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    headers = list(headers)
    divider = ["---" for _ in headers]
    body_lines = [" | ".join(headers), " | ".join(divider)]
    for row in rows:
        body_lines.append(" | ".join(row))
    return "\n".join(body_lines)


def collect_stats(exp: dict) -> dict:
    stats = exp["statistics"]
    return {
        "mean": stats["final_reward_mean"],
        "std": stats["final_reward_std"],
        "cv": stats.get("final_reward_cv"),
    }


def plot_group(group: GroupConfig, rows: List[dict], out_dir: Path) -> None:
    ensure_dir(out_dir)
    labels = [row["label"] for row in rows]
    means = [row["mean"] for row in rows]
    stds = [row["std"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color="#4C72B0")
    ax.set_ylabel("Final reward (mean ± std)")
    ax.set_title(group.title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean, f"{mean:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / f"{group.group_id}_final_reward.png", dpi=150)
    plt.close(fig)


def summarize_single(single_map: Dict[str, dict]) -> str:
    ensure_dir(SINGLE_VIZ_DIR)
    sections: List[str] = ["# 단일 하이퍼파라미터 실험 테이블"]

    for group in SINGLE_GROUPS:
        rows = []
        md_rows = []
        for entry in group.entries:
            exp = single_map.get(entry.experiment)
            if exp is None:
                raise KeyError(f"Experiment '{entry.experiment}' not found in single-run results")
            stats = collect_stats(exp)
            row = {
                "label": entry.label,
                "value": entry.value,
                "mean": stats["mean"],
                "std": stats["std"],
                "cv": stats["cv"],
            }
            rows.append(row)
            md_rows.append(
                [
                    entry.label,
                    str(entry.value),
                    f"{stats['mean']:.2f}",
                    f"{stats['std']:.2f}",
                    f"{abs(stats['cv']):.3f}" if stats["cv"] is not None else "-",
                ]
            )
        sections.append(f"\n## {group.title}\n")
        sections.append(
            format_markdown_table(
                ["설정", group.parameter, "최종 보상 평균", "표준편차", "CV"],
                md_rows,
            )
        )
        plot_group(group, rows, SINGLE_VIZ_DIR)

    return "\n".join(sections)


def summarize_combination(comb_map: Dict[str, dict]) -> str:
    ensure_dir(COMB_VIZ_DIR)
    sections: List[str] = ["# 조합 실험 테이블"]
    rows = []
    for exp in comb_map.values():
        stats = exp.get("statistics", {})
        rows.append(
            {
                "name": exp["name"],
                "description": exp.get("description", ""),
                "mean": stats.get("final_reward_mean", float("nan")),
                "std": stats.get("final_reward_std", float("nan")),
            }
        )

    rows.sort(key=lambda r: r["mean"], reverse=True)
    sections.append(
        format_markdown_table(
            ["실험", "설명", "최종 보상 평균", "표준편차"],
            [
                [
                    row["name"],
                    row["description"],
                    f"{row['mean']:.2f}",
                    f"{row['std']:.2f}",
                ]
                for row in rows
            ],
        )
    )

    labels = [row["name"] for row in rows]
    means = [row["mean"] for row in rows]
    stds = [row["std"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color="#55A868")
    ax.set_ylabel("Final reward (mean ± std)")
    ax.set_title("Combination Experiments")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean, f"{mean:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(COMB_VIZ_DIR / "combination_final_reward.png", dpi=150)
    plt.close(fig)

    return "\n".join(sections)


def main() -> None:
    single_map = load_experiment_map(SINGLE_RESULTS)
    combination_map = load_experiment_map(COMB_RESULTS)

    single_md = summarize_single(single_map)
    combo_md = summarize_combination(combination_map)

    TABLE_OUTPUT.write_text(single_md + "\n\n" + combo_md)
    print(f"Markdown tables exported to {TABLE_OUTPUT}")
    print(f"Single-run plots saved to {SINGLE_VIZ_DIR}")
    print(f"Combination plots saved to {COMB_VIZ_DIR}")


if __name__ == "__main__":
    main()
