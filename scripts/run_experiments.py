"""
Automated Experiment Runner for Chapter 5 (Evaluation).
Executes benchmark queries across all 3 domains, records metrics,
and exports directly to JSON and LaTeX tabular format.
"""
import os
import json
from src.data_loader import data_loader
from src.algorithms.benchmark import run_benchmark_suite
from src.api.routes import execute_internal_ranking

OUTPUT_DIR = "experiments/quantitative"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_all_benchmarks():
    dataset_ids = ["nlp", "medical", "astrophysics"]
    all_results = {}

    print("========================================================")
    print("STARTING AUTOMATED EXPERIMENTAL BENCHMARK EVALUATION")
    print("========================================================")

    for d_id in dataset_ids:
        print(f"\n[+] Running benchmark for domain: '{d_id}'...")
        dataset = data_loader.get_dataset(d_id)

        # Search execution adapter
        def search_adapter(d, q, mode):
            res, _, _ = execute_internal_ranking(d.dataset_id, q, mode)
            return [item.node for item in res]

        metrics = run_benchmark_suite(dataset, search_adapter)
        all_results[d_id] = [m.model_dump() for m in metrics]

    # 1. Export Raw Results to JSON
    json_path = os.path.join(OUTPUT_DIR, "benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[✔] Raw JSON benchmark metrics exported to: {json_path}")

    # 2. Export Directly to LaTeX Table (.tex) for Overleaf
    latex_path = os.path.join(OUTPUT_DIR, "benchmark_table.tex")
    export_to_latex(all_results, latex_path)
    print(f"[✔] Publication-ready LaTeX table generated at: {latex_path}")


def export_to_latex(results_dict, output_path):
    """Generates an IEEE/Springer compliant LaTeX table code."""
    latex_code = [
        "% Auto-generated benchmark table for Chapter 5",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Quantitative Evaluation of Retrieval Paradigms Across Benchmark Datasets}",
        "\\label{tab:benchmark_results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\hline",
        "\\textbf{Domain} & \\textbf{Retrieval Paradigm} & \\textbf{P@1} & \\textbf{MRR} & \\textbf{NDCG@3} & \\textbf{Latency (ms)} \\\\",
        "\\hline"
    ]

    domain_names = {
        "nlp": "NLP \\& LLMs",
        "medical": "Medical AI",
        "astrophysics": "Astrophysics"
    }

    for d_id, metrics in results_dict.items():
        domain_title = domain_names.get(d_id, d_id)
        first_row = True
        for m in metrics:
            dom_col = f"\\multirow{{4}}{{*}}{{{domain_title}}}" if first_row else ""
            p1 = f"{m['precision_at_1'] * 100:.1f}\\%"
            mrr = f"\\textbf{{{m['mrr']:.3f}}}" if m['mrr'] >= 0.9 else f"{m['mrr']:.3f}"
            ndcg = f"\\textbf{{{m['ndcg_at_3']:.3f}}}" if m['ndcg_at_3'] >= 0.9 else f"{m['ndcg_at_3']:.3f}"
            lat = f"{m['latency_ms']:.1f}"

            latex_code.append(
                f"{dom_col} & {m['algorithm_name']} & {p1} & {mrr} & {ndcg} & {lat} \\\\"
            )
            first_row = False
        latex_code.append("\\hline")

    latex_code.extend([
        "\\end{tabular}",
        "}",
        "\\end{table*}"
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_code))


if __name__ == "__main__":
    run_all_benchmarks()