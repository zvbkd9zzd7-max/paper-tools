from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train_gensim_lda import JIEBA_INPUT_FILE, JIEBA_RESULTS_DIR, run_experiment


def main() -> None:
    run_experiment(JIEBA_INPUT_FILE, JIEBA_RESULTS_DIR, "jieba")


if __name__ == "__main__":
    main()
