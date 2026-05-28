from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name == "src" else THIS_FILE.parent
INPUT_FILE = ROOT / "data" / "processed" / "lda_input_hanlp.txt"
JIEBA_INPUT_FILE = ROOT / "data" / "processed" / "lda_input_jieba.txt"
RESULTS_DIR = ROOT / "results"
JIEBA_RESULTS_DIR = ROOT / "results_jieba"

TOPIC_COUNTS = [3, 5, 10]
NUM_WORDS = 10
RANDOM_STATE = 42
PASSES = 30
ITERATIONS = 400


def load_documents(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [[tok for tok in line.strip().split() if tok] for line in lines if line.strip()]


def save_topics(model: LdaModel, output_path: Path, k: int) -> None:
    rows: list[str] = [f"K={k} 主题词（每个主题前 {NUM_WORDS} 个）\n"]
    for topic_id, topic_words in model.show_topics(num_topics=k, num_words=NUM_WORDS, formatted=False):
        words = " ".join([w for w, _ in topic_words])
        rows.append(f"Topic {topic_id}: {words}")
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def save_metrics_csv(metrics: Iterable[dict[str, float]], output_path: Path) -> None:
    fields = ["k", "coherence_c_v", "perplexity"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in metrics:
            writer.writerow(m)


def plot_metric(x: list[int], y: list[float], title: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, marker="o")
    for xi, yi in zip(x, y):
        plt.text(xi, yi, f"{yi:.4f}", fontsize=9, ha="left", va="bottom")
    plt.title(title)
    plt.xlabel("Number of Topics (K)")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def try_export_ldavis(model: LdaModel, corpus, dictionary: Dictionary, output_path: Path) -> tuple[bool, str]:
    if importlib.util.find_spec("pyLDAvis") is None:
        return False, "未安装 pyLDAvis"
    if importlib.util.find_spec("pyLDAvis.gensim_models") is None:
        return False, "未安装 pyLDAvis.gensim_models"

    pyldavis = importlib.import_module("pyLDAvis")
    gensimvis = importlib.import_module("pyLDAvis.gensim_models")

    try:
        vis_data = gensimvis.prepare(model, corpus, dictionary)
        html = pyldavis.prepared_data_to_html(vis_data)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def run_experiment(input_file: Path, results_dir: Path, label: str) -> list[dict[str, float]]:
    results_dir.mkdir(parents=True, exist_ok=True)
    docs = load_documents(input_file)
    if not docs:
        raise RuntimeError(f"输入文件为空或无可用文档: {input_file}")

    dictionary = Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    metrics: list[dict[str, float]] = []
    coherence_values: list[float] = []
    perplexities: list[float] = []

    print(f"实验语料: {label}")
    print(f"输入文件: {input_file}")
    print(f"输出目录: {results_dir}")
    print(f"输入文档数: {len(docs)}")
    print(f"词典大小: {len(dictionary)}")

    for k in TOPIC_COUNTS:
        print(f"\n=== 训练 LDA ({label}): K={k} ===")
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=RANDOM_STATE,
            passes=PASSES,
            iterations=ITERATIONS,
            alpha="auto",
            eta="auto",
        )

        coherence = CoherenceModel(model=model, texts=docs, dictionary=dictionary, coherence="c_v").get_coherence()
        perplexity = model.log_perplexity(corpus)

        coherence_values.append(coherence)
        perplexities.append(perplexity)
        metrics.append({"k": k, "coherence_c_v": coherence, "perplexity": perplexity})

        topics_path = results_dir / f"topics_K{k}.txt"
        save_topics(model, topics_path, k)

        ldavis_path = results_dir / f"lda_vis_K{k}.html"
        ok, msg = try_export_ldavis(model, corpus, dictionary, ldavis_path)
        if ok:
            print(f"pyLDAvis 已生成: {ldavis_path}")
        else:
            print(f"pyLDAvis 生成失败(K={k}): {msg}")

        print(f"K={k} coherence(c_v): {coherence:.6f}")
        print(f"K={k} perplexity(log_perplexity): {perplexity:.6f}")

    save_metrics_csv(metrics, results_dir / "lda_metrics.csv")
    plot_metric(
        TOPIC_COUNTS,
        coherence_values,
        f"{label} LDA Coherence (c_v) vs K",
        "Coherence (c_v)",
        results_dir / "coherence_plot.png",
    )
    plot_metric(
        TOPIC_COUNTS,
        perplexities,
        f"{label} LDA Perplexity (log_perplexity) vs K",
        "Log Perplexity",
        results_dir / "perplexity_plot.png",
    )

    print(f"\n{label} 结果文件已输出到 {results_dir} 目录。")
    return metrics


def resolve_cli_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def infer_label(input_file: Path, output_dir: Path) -> str:
    combined = f"{input_file} {output_dir}".lower()
    if "jieba" in combined:
        return "jieba"
    if "hanlp" in combined:
        return "HanLP"
    return "LDA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gensim LDA experiments for K=3, K=5, and K=10.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE, help="LDA input text file, one document per line.")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR, help="Directory for metrics, topics, plots, and LDAvis HTML.")
    parser.add_argument("--label", default=None, help="Optional label used in console logs and plot titles.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_file = resolve_cli_path(args.input)
    output_dir = resolve_cli_path(args.output)
    label = args.label or infer_label(input_file, output_dir)
    run_experiment(input_file, output_dir, label)


if __name__ == "__main__":
    main()
