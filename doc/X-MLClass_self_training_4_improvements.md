# X-MLClass self-training 两项改进的已实现说明

## 1. 修改范围

这次落地的代码改动覆盖两个入口：

- `OpenWordMLTC/self_training/self_training.py`
- `OpenWordMLTC/zero-shot/zero-shot-AAPD.py`

原因是：`self_training.py` 的第 0 轮输入 `zero_shot_text_train0.jsonl` 并不是它自己生成的，而是来自初始 zero-shot 分类脚本。  
如果只改 `self_training.py`，那么第 0 轮仍然会沿用旧的 top-8 候选逻辑，无法完整体现改进 1。

---

## 2. 改动一：从 top-8 粗召回升级为“宽召回 + reranker 压缩 + NLI 终判”

### 2.1 旧实现

原始实现的链路是：

\[
s \rightarrow \mathrm{Top8}_{c \in C}\ \mathrm{dot}(e(s), e(c)) \rightarrow \mathrm{NLI}
\]

对应代码行为：

1. 用 `all-MiniLM-L6-v2` 对文本和当前标签空间做向量编码
2. 用 `util.dot_score` 从整个标签空间中取 top-8 候选标签
3. 用 `pipeline("zero-shot-classification", multi_label=False)` 在这 8 个候选里做 top-1 判别

这一步的主要问题是：  
如果正确标签在 embedding 召回阶段就没有进入 top-8，后面的 NLI 没法补救。

### 2.2 新实现

现在已经改成：

\[
s \rightarrow \mathrm{Cand}_K(s) \rightarrow \widetilde{\mathrm{Cand}}_M(s) \rightarrow \mathrm{NLI}
\]

其中：

- \(K =\) `candidate_top_k`，默认 32
- \(M =\) `rerank_top_m`，默认 8

具体流程：

1. 先用 `all-MiniLM-L6-v2` 从全标签空间做宽召回，保留 top-32 候选
2. 再用 cross-encoder reranker 对这 32 个文本-标签对重排
3. 取 rerank 后的 top-8 送入 NLI 做最终单标签判别

默认 reranker：

- `cross-encoder/ms-marco-MiniLM-L-6-v2`

新输出的 jsonl 里除了原本的 `labels` 和 `scores`，还额外保存：

- `retrieval_labels`
- `retrieval_scores`
- `reranker_labels`
- `reranker_scores`
- `reranker_top1`

这样后续 self-training 阶段就能直接使用 reranker 的结果，不需要重新猜测候选来源。

### 2.3 对应代码位置

- 初始 zero-shot：`OpenWordMLTC/zero-shot/zero-shot-AAPD.py`
- 迭代中的 zero-shot：`OpenWordMLTC/self_training/self_training.py` 里的 `zero_shot_training()`

---

## 3. 改动二：从“只看 top-1 分数”改成“综合不确定性”

### 3.1 旧实现

原始 `self_training()` 只按 top-1 分数选低置信样本：

\[
u_{\text{old}}(s)=1-p_1(s)
\]

也就是直接按 `json_raw_list[i]["scores"][0]` 从低到高排序，取 `tail_set_size` 个样本。

### 3.2 新实现

现在已经改成：

\[
u(s)=\lambda_1(1-p_1(s))+\lambda_2(1-(p_1(s)-p_2(s)))+\lambda_3 \cdot \mathbf{1}[\hat c_r(s)\neq \hat c_n(s)]
\]

其中默认参数为：

\[
\lambda_1 = 0.5,\quad \lambda_2 = 0.3,\quad \lambda_3 = 0.2
\]

含义分别是：

- `1 - p1`：top-1 预测越低，越不确定
- `1 - (p1 - p2)`：top-1 和 top-2 越接近，越不确定
- `disagreement`：reranker top-1 和 NLI top-1 不一致时，额外增加不确定性

如果读取的是旧格式 jsonl，里面没有 `reranker_top1`，代码会自动回退为“无分歧惩罚”的兼容行为，不会直接报错。

### 3.3 结果

tail 样本不再只是“top-1 偏低”的样本，而更接近：

- 当前标签空间覆盖不足的样本
- 候选边界模糊的样本
- reranker 和 NLI 判断不一致的边界样本

---

## 4. 新标签准入逻辑：本次没有重写，但文档口径已修正

这次没有系统性重写“新增标签准入规则”，仍然保留原有启发式框架：

1. 先从入选 tail chunk 中抽取候选 keyphrases
2. 用全局出现频次减去组内局部重复补偿做频次过滤
3. 先在新增候选之间做 embedding 去重
4. 再与当前标签空间比较，过滤掉过近邻的词

需要特别澄清的一点：

- 候选关键词不是“每个 tail group 的首个 chunk 的前 3 个 keyphrases”
- 实际代码是“每个入选低置信 chunk 自己的前 3 个 keyphrases”
- 同文档相邻 chunk 组成的 group 只用于计算该关键词在局部组内的重复补偿项

本次顺手做了两个稳定性修正：

- 修复了 tail group 向前扩展时的边界判断，避免 `index=0` 时错误访问最后一行
- `update_labelspace{iter}.txt` 改为覆盖写入，避免重复运行时不断追加旧内容

---

## 5. 兼容性与文件命名

为了兼容已有结果文件，`self_training.py` 现在会优先读取：

- `zero_shot_text_train0.jsonl`

如果不存在，再回退读取旧命名：

- `zero_shot_text_train_0.jsonl`

同时，新的 `zero-shot-AAPD.py` 会直接生成和 `self_training.py` 对齐的文件名：

- `zero_shot_text_train0.jsonl`
- `zero_shot_keyword_train0.jsonl`

---

## 6. 新增参数

### 6.1 `OpenWordMLTC/zero-shot/zero-shot-AAPD.py`

新增参数：

- `--embedding_model`
- `--reranker_model`
- `--candidate_top_k`
- `--rerank_top_m`

默认值：

```bash
--embedding_model sentence-transformers/all-MiniLM-L6-v2
--reranker_model cross-encoder/ms-marco-MiniLM-L-6-v2
--candidate_top_k 32
--rerank_top_m 8
```

### 6.2 `OpenWordMLTC/self_training/self_training.py`

除了上面 4 个参数，又新增了 3 个不确定性权重：

- `--lambda_top1`
- `--lambda_margin`
- `--lambda_disagreement`

默认值：

```bash
--lambda_top1 0.5
--lambda_margin 0.3
--lambda_disagreement 0.2
```

---

## 7. 推荐运行方式

### 7.1 先生成第 0 轮 zero-shot 结果

```bash
cd OpenWordMLTC/zero-shot
CUDA_VISIBLE_DEVICES=0 python zero-shot-AAPD.py \
    --path ../../datasets \
    --data_dir train_texts_split_50.txt \
    --keyphrase_dir qwen3_label_50.txt \
    --task AAPD \
    --dynamic_iter 3000 \
    --model MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33 \
    --candidate_top_k 32 \
    --rerank_top_m 8
```

### 7.2 再运行 self-training

```bash
cd OpenWordMLTC/self_training
CUDA_VISIBLE_DEVICES=0 python self_training.py \
    --path ../../datasets \
    --data_dir train_texts_split_50.txt \
    --keyphrase_dir qwen3_label_50.txt \
    --task AAPD \
    --llama_model qwen3 \
    --tail_set_size 500 \
    --majority_num 350 \
    --max_majority_num 5 \
    --sim_threshold 0.55 \
    --max_add_label 10 \
    --model MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33 \
    --candidate_top_k 32 \
    --rerank_top_m 8 \
    --lambda_top1 0.5 \
    --lambda_margin 0.3 \
    --lambda_disagreement 0.2
```

---

## 8. 当前版本的结论

本次已经真正落地的两项改动是：

1. 宽召回 + reranker 压缩 + NLI 终判
2. 基于 top-1、margin、模型分歧的综合不确定性 tail 选择

而“新增标签准入规则”这部分仍然基本保持原框架，只做了文档校正和小幅稳定性修正。  
如果后续还要继续优化，下一步最值得单独实验的方向会是：

- 重新设计 keyphrase 候选聚合方式
- 把新增标签准入从固定阈值启发式改成更稳的评分函数
- 对 reranker/NLI 的中间分数做日志化分析，检查新增标签是否真的来自边界样本
