import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Relations grouped by dataset (used by plot and latex modes)
DATASET_RELATIONS = {
    "Hertziana": ["context", "iconography", "object", "photography"],
    "Wikidataset": [
        "artist education",
        "artist materials and techniques",
        "style historical context",
        "style influences and precursors",
    ],
    "SemArt": ["content", "context", "description", "form"],
}

# Shortened labels for Wikidataset x-axis
MAPPING_REDUCED = {
    "artist education": "education",
    "artist materials and techniques": "material",
    "style historical context": "context",
    "style influences and precursors": "influence",
}

MODEL_NAMES = {
    "sagenet": "ArtSAGENet",
    "clip": "CLIP",
    "clipft" : "CLIP_ft",
    "siglipft": "SigLIP_ft",
    "colpali": "ColPali",
    "colqwen2": "ColQwen",
    "msc": "MSC",
    "siglip": "SigLIP",
    "sheafclip": "CANVAS",
}

DATASET_DISPLAY = {
    "SemArt": "SemArt",
    "HertzianaDP": "HertzianaDP",
    "WikiArtPlus": "WikiArt+",
}

COLOR_MAP = {
    "ArtSAGENet": ("#1f77b4", "/"),
    "CLIP":       ("#9467bd", "+"),
    "ColPali":    ("#2ca02c", "*"),
    "ColQwen":    ("#17becf", "|"),
    "MSC":        ("#ff7f0e", "."),
    "SigLIP":     ("#d62728", "-"),
    "CANVAS":  ("#e377c2", "x"),
}

ORDERED_MODELS = ["sagenet", "clip", "colpali", "colqwen2", "msc", "siglip", "sheafclip", "clipft", "siglipft"]

DATASET_ORDER = ["HertzianaDP", "SemArt", "WikiArtPlus"]

TASKS = {"retrieval"}



def parse_results_relations(metric: str, k: int, json_dir: str) -> dict:
    """Parse JSON files and compute per-(dataset, model, relation) mean for plot/latex."""
    raw = defaultdict(list)

    for filename in os.listdir(json_dir):
        if not filename.endswith(".json"):
            continue
        if "sheafclip" in filename and "b" not in filename:
            continue

        name = filename.replace("_metrics.json", "")
        parts = name.split("_")

        if len(parts) < 4:
            continue

        model, dataset = parts[0], parts[1]
        if dataset not in DATASET_RELATIONS:
            continue

        with open(os.path.join(json_dir, filename), "r") as f:
            metrics = json.load(f)

        # Collect metric@k per relation (handles both space and underscore keys)
        for relation in DATASET_RELATIONS[dataset]:
            for key in [
                f"test_{relation}_t2i_{metric}@{k}",
                f"test_{relation.replace(' ', '_')}_t2i_{metric}@{k}",
            ]:
                if key in metrics:
                    raw[(dataset, model, relation)].append(metrics[key])

    summary = defaultdict(lambda: defaultdict(dict))
    for (dataset, model, relation), values in raw.items():
        summary[dataset][model][relation] = {"mean": np.array(values).mean()}
    return summary


def parse_results_table(json_dir: str) -> dict:
    """Parse JSON files and collect all numeric metrics for table generation."""
    raw = defaultdict(lambda: defaultdict(list))

    for filename in os.listdir(json_dir):
        if not filename.endswith(".json"):
            continue

        name = filename.replace("_metrics.json", "")
        parts = name.split("_")

        # Locate task token by name
        try:
            task_idx = next(i for i, p in enumerate(parts) if p in TASKS)
        except StopIteration:
            continue
        if task_idx < 2 or task_idx + 1 >= len(parts):
            continue

        model = parts[0]
        dataset = "_".join(parts[1:task_idx])
        task = parts[task_idx]
        loss = parts[task_idx + 2] if task_idx + 2 < len(parts) else "unknown"

        # Skip sheafclip variants without 'b' in loss
        if "sheafclip" in name and "b" not in str(loss):
            continue

        with open(os.path.join(json_dir, filename), "r") as f:
            metrics = json.load(f)

        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                raw[(model, dataset, task)][metric_name].append(value)

    # Aggregate mean and std
    summary = {}
    for key, metrics in raw.items():
        summary[key] = {}
        for metric_name, values in metrics.items():
            values = np.array(values, dtype=float)
            summary[key][metric_name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    return summary


# ===== GENERATION =====

def generate_plot(summary: dict, metric: str, k: int) -> None:
    """Generate a grouped bar chart (one subplot per dataset) and save as PDF."""
    plt.rcParams.update({"font.size": 18})

    datasets_to_plot = sorted(d for d in DATASET_RELATIONS if d in summary)
    fig, ax_list = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
    handles, labels = None, None

    for idx, dataset in enumerate(datasets_to_plot):
        ax = ax_list[idx]
        relations = DATASET_RELATIONS[dataset]
        models = [m for m in ORDERED_MODELS if m in summary[dataset]]

        x = np.arange(len(relations))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            means = [
                summary[dataset][model].get(rel, {}).get("mean", 0)
                for rel in relations
            ]

            color, hatch = COLOR_MAP[MODEL_NAMES[model]]
            ax.bar(x + i * width, means, width=width,
                   color=color, hatch=hatch, label=MODEL_NAMES[model])

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        if dataset == "Wikidataset":
            ax.set_xticklabels([MAPPING_REDUCED[r] for r in relations], rotation=30)
        else:
            ax.set_xticklabels(relations, rotation=30)
        ax.set_title(DATASET_DISPLAY.get(dataset, dataset))

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    ax_list[0].set_ylabel(f"{metric.upper()}@{k}")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True)
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/all_datasets_histogram_{metric}@{k}.pdf", format="pdf")
    print(f"Saved images/all_datasets_histogram_{metric}@{k}.pdf")


def generate_latex(summary: dict, metric: str, k: int) -> None:
    """Export per-(dataset, model) pgfplots coordinate macros to a .tex file."""
    datasets_to_export = sorted(d for d in DATASET_RELATIONS if d in summary)
    os.makedirs("images", exist_ok=True)

    with open(f"images/pgfplotsdata_{metric}@{k}.tex", "w") as f:
        for dataset in datasets_to_export:
            relations = DATASET_RELATIONS[dataset]
            models = [m for m in ORDERED_MODELS if m in summary[dataset]]
            ds_clean = dataset.replace(" ", "")

            for model in models:
                macro = f"data{ds_clean}{model.replace(' ', '')}"
                coords = " ".join(
                    f"({i},{summary[dataset][model].get(rel, {}).get('mean', 0):.4f})"
                    for i, rel in enumerate(relations)
                )
                f.write(f"\\def\\{macro}{{{coords}}}\n")
        f.write("\n")

    print(f"Exported images/pgfplotsdata_{metric}@{k}.tex")


def generate_table(metric: str, json_dir: str) -> None:
    """Generate LaTeX table rows for classification and retrieval metrics."""
    summary = parse_results_table(json_dir)

    # Classification metrics per dataset (metric@1)
    classification_metrics = {
        "HertzianaDP": [
            f"test_artist_i2t_{metric}@1",
            f"test_acquisition period_i2t_{metric}@1",
        ],
        "WikiArtPlus": [
            f"test_artist_i2t_{metric}@1",
            f"test_date_i2t_{metric}@1",
            f"test_genre_i2t_{metric}@1",
            f"test_artwork_style_i2t_{metric}@1",
        ],
        "SemArt": [
            f"test_author_i2t_{metric}@1",
            f"test_genre_i2t_{metric}@1",
            f"test_school_i2t_{metric}@1",
            f"test_timeframe_i2t_{metric}@1",
            f"test_material_i2t_{metric}@1",
        ],
    }

    # Retrieval metrics per dataset (metric@5 and metric@10, both directions)
    retrieval_cols = [
        f"test_t2i_{metric}@5",
        f"test_t2i_{metric}@10",
        f"test_i2t_{metric}@5",
        f"test_i2t_{metric}@10",
    ]
    retrieval_metrics = {ds: retrieval_cols for ds in DATASET_ORDER}

    def get_cell(model, dataset, task, m):
        stats = summary.get((model, dataset, task), {}).get(m)
        return f"{stats['mean']:.3f}" if stats else "-"

    models = [m for m in ORDERED_MODELS if m in {k[0] for k in summary}]

    os.makedirs("images", exist_ok=True)
    outpath = f"images/tables_{metric}.tex"

    with open(outpath, "w") as f:
        # Classification table
        f.write(f"% Classification table ({metric}@1)\n")
        for model in models:
            row = [MODEL_NAMES[model]]
            for ds in DATASET_ORDER:
                for m in classification_metrics[ds]:
                    row.append(get_cell(model, ds, "retrieval", m))
            f.write(" & ".join(row) + " \\\\\n")

        f.write("\n")

        # Retrieval table
        f.write(f"% Retrieval table ({metric}@5/10)\n")
        for model in models:
            row = [MODEL_NAMES[model]]
            for ds in DATASET_ORDER:
                for m in retrieval_metrics[ds]:
                    row.append(get_cell(model, ds, "retrieval", m))
            f.write(" & ".join(row) + " \\\\\n")

    print(f"Exported {outpath}")


# ===== CLI =====

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate recall/precision/ndcg visualisations from JSON metric files."
    )
    parser.add_argument(
        "--mode",
        choices=["plot", "latex", "table", "all"],
        default="all",
        help="Output mode (default: all)",
    )
    parser.add_argument(
        "--metric",
        choices=["recall", "precision", "ndcg"],
        default="recall",
        help="Metric type (default: recall)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="k value for plot/latex modes (default: 10)",
    )
    parser.add_argument(
        "--dir",
        default=".",
        help="Directory containing JSON metric files (default: .)",
    )
    args = parser.parse_args()

    if args.mode in ("plot", "latex", "all"):
        summary = parse_results_relations(args.metric, args.k, args.dir)

    if args.mode in ("plot", "all"):
        generate_plot(summary, args.metric, args.k)
    if args.mode in ("latex", "all"):
        generate_latex(summary, args.metric, args.k)
    if args.mode in ("table", "all"):
        generate_table(args.metric, args.dir)
