from __future__ import annotations

from pathlib import Path
from statistics import mean

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name == "src" else THIS_FILE.parent
PROCESSED_DIR = ROOT / "data" / "processed"

STOPWORDS_FILE = PROCESSED_DIR / "wyw_stopwords.txt"
HANLP_FILE = PROCESSED_DIR / "hanlp_result1.txt"
JIEBA_FILE = PROCESSED_DIR / "jieba_result1.txt"
OUT_HANLP = PROCESSED_DIR / "lda_input_hanlp.txt"
OUT_JIEBA = PROCESSED_DIR / "lda_input_jieba.txt"

MERGE_EVERY = 30
MIN_TOKEN_LEN = 2
MIN_DOC_TOKENS = 10


def load_stopwords(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def clean_line(line: str, stopwords: set[str]) -> list[str]:
    tokens = [tok.strip() for tok in line.strip().split(" ") if tok.strip()]
    return [tok for tok in tokens if len(tok) >= MIN_TOKEN_LEN and tok not in stopwords]


def process_file(input_path: Path, output_path: Path, stopwords: set[str]) -> dict[str, float]:
    raw_lines = input_path.read_text(encoding="utf-8").splitlines()
    original_lines = len(raw_lines)

    non_empty_lines = [line.strip() for line in raw_lines if line.strip()]
    after_remove_empty = len(non_empty_lines)

    # 保留原顺序去重
    deduped_lines = list(dict.fromkeys(non_empty_lines))
    after_dedup = len(deduped_lines)

    cleaned_line_tokens: list[list[str]] = [clean_line(line, stopwords) for line in deduped_lines]

    documents: list[list[str]] = []
    for i in range(0, len(cleaned_line_tokens), MERGE_EVERY):
        chunk = cleaned_line_tokens[i : i + MERGE_EVERY]
        merged_tokens = [tok for line_tokens in chunk for tok in line_tokens]
        if len(merged_tokens) >= MIN_DOC_TOKENS:
            documents.append(merged_tokens)

    output_lines = [" ".join(doc) for doc in documents]
    output_path.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")

    token_counts = [len(doc) for doc in documents]
    avg_tokens = mean(token_counts) if token_counts else 0.0
    vocab = {tok for doc in documents for tok in doc}

    return {
        "original_lines": original_lines,
        "after_remove_empty": after_remove_empty,
        "after_dedup": after_dedup,
        "merged_docs": len(documents),
        "avg_doc_tokens": avg_tokens,
        "vocab_size": len(vocab),
    }


def print_stats(name: str, stats: dict[str, float]) -> None:
    print(f"\n[{name}]")
    print(f"原始行数: {int(stats['original_lines'])}")
    print(f"去空行后的行数: {int(stats['after_remove_empty'])}")
    print(f"去重后的行数: {int(stats['after_dedup'])}")
    print(f"合并后的文档数: {int(stats['merged_docs'])}")
    print(f"平均每个文档 token 数: {stats['avg_doc_tokens']:.2f}")
    print(f"词表大小: {int(stats['vocab_size'])}")


def main() -> None:
    stopwords = load_stopwords(STOPWORDS_FILE)

    hanlp_stats = process_file(HANLP_FILE, OUT_HANLP, stopwords)
    jieba_stats = process_file(JIEBA_FILE, OUT_JIEBA, stopwords)

    print_stats("hanlp_result1.txt -> lda_input_hanlp.txt", hanlp_stats)
    print_stats("jieba_result1.txt -> lda_input_jieba.txt", jieba_stats)

    print("\n输出文件:")
    print(f"- {OUT_HANLP}")
    print(f"- {OUT_JIEBA}")


if __name__ == "__main__":
    main()
