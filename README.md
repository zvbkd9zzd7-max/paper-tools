# gensim LDA 文言文主题建模实验说明

## 一、项目说明

本项目用于对中文文言文语料进行 gensim LDA 主题建模实验。实验语料为《论语》相关文本，前期已完成 HanLP 和 jieba 两种分词处理。本实验主要比较不同主题数量 K 对 LDA 建模效果的影响，并进一步比较 HanLP 与 jieba 两种分词结果下的主题建模表现。

本实验重点比较：

* K = 3
* K = 5
* K = 10

输出内容包括：

* topic keywords
* coherence score
* log perplexity
* coherence 折线图
* perplexity 折线图
* LDAvis 可视化网页

---

## 二、运行环境

建议使用：

```bash
Python 3.11
```

主要依赖包括：

```bash
gensim
scipy==1.12.0
matplotlib
pyLDAvis
pandas
jieba
```

如需安装依赖，可运行：

```bash
pip install gensim==4.3.2 scipy==1.12.0 numpy pandas matplotlib pyLDAvis jieba
```

如果网络不稳定，可使用国内镜像源：

```bash
pip install gensim==4.3.2 scipy==1.12.0 numpy pandas matplotlib pyLDAvis jieba -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

---

## 三、数据文件说明

主要输入文件位于：

```text
data/processed/
```

其中：

```text
hanlp_result1.txt        HanLP 分词结果
jieba_result1.txt        jieba 分词结果
wyw_stopwords.txt        文言文停用词表
lda_input_hanlp.txt      HanLP 版本的 LDA 输入文件
lda_input_jieba.txt      jieba 版本的 LDA 输入文件
```

说明：
`hanlp_result1.txt` 和 `jieba_result1.txt` 是前期分词结果；`wyw_stopwords.txt` 是停用词表，不作为 LDA 输入。LDA 实际训练使用的是整理后的 `lda_input_hanlp.txt` 和 `lda_input_jieba.txt`。

---

## 四、数据准备

如果需要重新生成 LDA 输入文件，可在项目根目录运行：

```bash
python src/prepare_lda_data.py
```

该脚本会基于 HanLP / jieba 分词结果和文言文停用词表，生成：

```text
data/processed/lda_input_hanlp.txt
data/processed/lda_input_jieba.txt
```

主要处理包括：

* 去除空行
* 去除重复行
* 去除停用词
* 过滤过短 token
* 将短句合并为更适合 LDA 建模的文档单位

---

## 五、运行 HanLP 版本 LDA 实验

在项目根目录运行：

```bash
python src/train_gensim_lda_hanlp.py
```

输出结果保存在：

```text
results_hanlp/
```

主要结果包括：

```text
lda_metrics.csv
topics_K3.txt
topics_K5.txt
topics_K10.txt
coherence_plot.png
perplexity_plot.png
lda_vis_K3.html
lda_vis_K5.html
lda_vis_K10.html
```

---

## 六、运行 jieba 版本 LDA 对照实验

在项目根目录运行：

```bash
python src/train_gensim_lda_jieba.py
```

输出结果保存在：

```text
results_jieba/
```

主要结果包括：

```text
lda_metrics.csv
topics_K3.txt
topics_K5.txt
topics_K10.txt
coherence_plot.png
perplexity_plot.png
lda_vis_K3.html
lda_vis_K5.html
lda_vis_K10.html
```

---

## 七、实验结果概述

HanLP 版本结果显示：

```text
K=3  coherence = 0.4418
K=5  coherence = 0.5370
K=10 coherence = 0.5252
```

因此，在 HanLP 分词结果下，K=5 的 coherence score 最高，主题一致性最好。

jieba 版本结果显示：

```text
K=3  coherence = 0.4590
K=5  coherence = 0.4264
K=10 coherence = 0.4728
```

因此，在 jieba 分词结果下，K=10 的 coherence score 最高。

综合比较来看，HanLP 版本的最高 coherence score 高于 jieba 版本，且 HanLP + K=5 的主题解释性更好。因此，本实验最终选择：

```text
HanLP + K=5
```

作为本次文言文 LDA 主题建模的较优方案。

---

## 八、结果查看方式

### 1. 查看指标结果

打开：

```text
results_hanlp/lda_metrics.csv
results_jieba/lda_metrics.csv
```

可查看不同 K 值下的 coherence score 和 log perplexity。

### 2. 查看主题关键词

打开：

```text
results_hanlp/topics_K3.txt
results_hanlp/topics_K5.txt
results_hanlp/topics_K10.txt
```

或：

```text
results_jieba/topics_K3.txt
results_jieba/topics_K5.txt
results_jieba/topics_K10.txt
```

可查看不同 K 值下的 topic keywords。

### 3. 查看 LDAvis 可视化

双击打开：

```text
results_hanlp/lda_vis_K5.html
```

或：

```text
results_jieba/lda_vis_K10.html
```

即可在浏览器中查看交互式 LDAvis 可视化结果。

---

## 九、主要结论

本实验使用 gensim LDA 对《论语》文言文语料进行主题建模，重点比较了 K=3、K=5 和 K=10 三种主题数量设置，并对 HanLP 与 jieba 两种分词结果进行了对照实验。

实验结果表明：

1. HanLP 分词结果下，K=5 的 coherence score 最高；
2. jieba 分词结果下，K=10 的 coherence score 最高；
3. HanLP 的最佳 coherence score 高于 jieba；
4. HanLP + K=5 的主题数量适中，主题解释性较好；
5. LDA 在处理文言短文本时仍存在一定局限，例如高频人物词反复出现、主题之间存在重叠。

因此，最终认为：

```text
HanLP + K=5 是本次实验中较优的 LDA 建模方案。
```
