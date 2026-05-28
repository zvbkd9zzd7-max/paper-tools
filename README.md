# Plain-text

## LDA 数据准备

在进行 gensim LDA 实验前，请先运行以下命令完成数据清洗与文档重组：

```bash
python src/prepare_lda_data.py
```

该脚本会基于 `data/processed/hanlp_result1.txt`、`data/processed/jieba_result1.txt` 与 `data/processed/wyw_stopwords.txt` 生成：

- `data/processed/lda_input_hanlp.txt`
- `data/processed/lda_input_jieba.txt`

## 训练 gensim LDA

请在仓库根目录（包含 `src/`、`data/` 的目录）运行：

```bash
python src/train_gensim_lda.py
```

如果在 Windows 出现以下报错：

```text
can't open file '...\\src\\train_gensim_lda.py': [Errno 2] No such file or directory
```

通常是因为当前路径不是仓库根目录，或路径写成了不存在的多层目录。可按下面方式排查：

1. 先进入项目根目录再执行：

```powershell
cd E:\大二下\计算政治学\paper-tools-main
python .\src\train_gensim_lda.py
```

2. 先确认脚本文件是否存在：

```powershell
dir .\src\train_gensim_lda.py
```

3. 如果你有双层目录（如 `paper-tools-main\\paper-tools-main`），请以实际存在 `src` 的那一层为根目录运行。

训练脚本会输出：

- `results/lda_metrics.csv`
- `results/topics_K3.txt`
- `results/topics_K5.txt`
- `results/topics_K10.txt`
- `results/coherence_plot.png`
- `results/perplexity_plot.png`
- `results/lda_vis_K3.html` / `results/lda_vis_K5.html` / `results/lda_vis_K10.html`（若 pyLDAvis 可用）


## 无 `src/` 目录时的运行方式

如果你本地目录里没有 `src/`，可直接运行仓库根目录下的同名脚本：

```bash
python prepare_lda_data.py
python train_gensim_lda.py
```

这两个根目录脚本与 `src/` 目录版本保持一致，用于兼容不同下载/解压后的目录结构。

## jieba LDA 对照实验

HanLP 实验仍使用原来的输出目录 `results/`：

```bash
python train_gensim_lda.py
```

如需运行 jieba 分词结果的对照实验，请运行：

```bash
python train_gensim_lda_jieba.py
```

也可以使用 `src/` 入口：

```bash
python src/train_gensim_lda_jieba.py
```

jieba 对照实验会读取 `data/processed/lda_input_jieba.txt`，并将结果单独写入 `results_jieba/`，不会覆盖 HanLP 的 `results/` 目录。输出文件包括：

- `results_jieba/lda_metrics.csv`
- `results_jieba/topics_K3.txt`
- `results_jieba/topics_K5.txt`
- `results_jieba/topics_K10.txt`
- `results_jieba/coherence_plot.png`
- `results_jieba/perplexity_plot.png`
- `results_jieba/lda_vis_K3.html`
- `results_jieba/lda_vis_K5.html`
- `results_jieba/lda_vis_K10.html`
