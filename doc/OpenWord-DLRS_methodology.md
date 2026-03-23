# OpenWord-DLRS 方法论

## 1 问题定义

设训练语料由一组未标注文本块构成：

$$
\mathcal{D}=\{x_i\}_{i=1}^{N},
$$

其中，$x_i$ 表示第 $i$ 个文本块或文档片段。与封闭标签集设定不同，开放世界多标签文本分类并不预先给定完整标签空间，真实标签集合记为 $\mathcal{Y}^{\ast}$，且通常未知。模型需要在缺乏人工标注训练集的条件下，从原始语料中诱导初始标签空间，并在迭代过程中持续发现潜在新标签、删除低质量标签，从而逐步逼近 $\mathcal{Y}^{\ast}$。

结合当前实现，本文进一步区分两个相关但不完全相同的对象：

1. 累计标签池 $\mathcal{P}^{(t)}$：用于记录当前轮之前已保留的全部候选标签，并在新标签引入时承担全局近邻过滤作用；
2. 工作标签空间 $\mathcal{L}^{(t)}$：用于第 $t$ 轮 zero-shot 伪标注的实际标签集合。

因此，本文的目标可形式化为：在无人工标注监督的前提下，构建初始标签空间 $\mathcal{L}^{(0)}$，基于当前工作标签空间为样本分配伪标签，并通过高不确定性样本持续扩展和净化标签空间。需要说明的是，当前代码在训练阶段采用 `multi_label=False` 的零样本判别模式，即以“文本块级单标签伪标注”近似支撑开放世界多标签学习过程；多标签能力主要体现在标签空间的持续扩展以及测试阶段的标签融合与语义匹配结果之中。

## 2 方法总体框架

OpenWord-DLRS（Open-World Dynamic Label-space Retrieval and Self-training）由五个串联模块构成：

1. 基于局部大语言模型的关键词生成；
2. 基于语义聚类与标签归纳的初始标签空间构建；
3. 基于“检索-重排-NLI”三级判别的伪标签分配；
4. 基于综合不确定性的尾部样本挖掘；
5. 面向标签空间演化的标签扩展、去冗余与头尾平衡更新。

从迭代优化视角看，该过程可写为

$$
(\mathcal{P}^{(t+1)}, \mathcal{L}^{(t+1)})
=
\Phi\big(\mathcal{P}^{(t)}, \mathcal{L}^{(t)}, \mathcal{D}, \hat{\mathcal{Y}}^{(t)}\big),
$$

其中，$\hat{\mathcal{Y}}^{(t)}$ 表示第 $t$ 轮伪标注结果，$\Phi(\cdot)$ 表示由新标签发现、低频标签删除与头部标签抑制共同构成的标签空间更新算子。

## 3 初始标签空间构建

### 3.1 关键词生成

对于输入文本块 $x_i$，首先利用局部因果语言模型生成其语义压缩表示：

$$
K_i = G_k(x_i)=\{k_{i1},k_{i2},\dots,k_{im}\},
$$

其中，$K_i$ 为与文本块 $x_i$ 对应的关键词集合。当前实现由 [`OpenWordMLTC/keyword_generator/llama_keyword.py`](/e:/lwt/xml/OpenWordMLTC/keyword_generator/llama_keyword.py) 完成，并通过数据集特定提示词约束模型输出格式，使每个文本块均获得若干主题性较强的候选关键词或短语。该步骤不直接生成最终标签，而是为后续语义聚类与标签归纳提供更紧凑的语义载体。

### 3.2 基于关键词语义表示的聚类归纳

为避免直接在原始长文本上进行高成本标签诱导，本文以关键词文本为聚类输入。设聚类编码器为 $E_c(\cdot)$，则第 $i$ 个文本块的聚类表示为

$$
\mathbf{z}_i = E_c(K_i).
$$

在实现层面，系统采用 `hkunlp/instructor-large` 生成关键词语义向量；当样本规模较大时，进一步通过 UMAP 将高维表示映射到低维流形空间：

$$
\tilde{\mathbf{z}}_i = U(\mathbf{z}_i).
$$

随后，在降维表示上执行聚类。对于 AAPD、DBPedia-298、Reuters-21578 与 RCV1，使用高斯混合模型。若记聚类分配结果为 $c_i \in \{1,\dots,C\}$，则每个簇对应一组语义相近的文本块。

与“直接从关键词集合生成标签”不同，当前实现采用“关键词聚类 + 原始文本回看”的二阶段策略：先依据关键词表示完成聚类，再从每个簇中选取最接近簇中心的 3 个原始文本块，并将其拼接为该簇的代表性上下文。若簇 $j$ 的代表性文本集合记为 $\mathcal{R}_j$，则其标签归纳过程为

$$
\tilde{y}_j = G_l(\mathcal{R}_j),
$$

其中，$G_l(\cdot)$ 表示局部大语言模型驱动的簇级标签归纳器。该过程由 [`OpenWordMLTC/keyword_generator/dynamic_GMM.py`](/e:/lwt/xml/OpenWordMLTC/keyword_generator/dynamic_GMM.py) 实现，且默认使用“最终单标签”提示模板，以提升初始标签空间的紧凑性与可解释性。

### 3.3 初始标签空间去冗余

由于不同簇可能生成语义相近甚至上下位关系重叠的标签，本文进一步对初始标签集合执行语义去冗余。设候选标签间的相似度定义为

$$
\mathrm{sim}(y_a,y_b)=E_s(y_a)^\top E_s(y_b),
$$

其中，$E_s(\cdot)$ 表示标签语义编码器。若满足

$$
\mathrm{sim}(y_a,y_b)>\tau_{\text{init}},
$$

则调用局部大语言模型判断两者是否应视为同义或近义标签，并删除语义更窄或层级更低者。当前实现中，$\tau_{\text{init}}$ 由 [`OpenWordMLTC/keyword_generator/get_init_labelspace.py`](/e:/lwt/xml/OpenWordMLTC/keyword_generator/get_init_labelspace.py) 中的 `lower_bound` 控制，默认取值为 $0.80$。经去冗余后得到初始工作标签空间

$$
\mathcal{L}^{(0)}=\mathcal{P}^{(0)}.
$$

## 4 基于检索-重排-NLI 的伪标签分配

### 4.1 稠密检索候选召回

给定第 $t$ 轮工作标签空间 $\mathcal{L}^{(t)}$，首先使用统一的句向量编码器对文本块与标签进行编码：

$$
\mathbf{q}_i = E_r(x_i), \qquad \mathbf{h}^{(t)}_j = E_r(y^{(t)}_j).
$$

基于点积相似度计算文本与标签之间的相关性：

$$
s^{\text{ret}}_{ij} = {\mathbf{q}_i}^{\top}\mathbf{h}^{(t)}_j.
$$

随后保留得分最高的前 $K$ 个标签，形成宽召回候选集合

$$
\mathcal{C}^{(t)}_i = \operatorname{TopK}\big(\mathcal{L}^{(t)}, s^{\text{ret}}_{ij}, K\big).
$$

在代码实现中，$K$ 由 `candidate_top_k` 控制，默认取值为 32，对应文件 [`OpenWordMLTC/zero-shot/zero-shot-AAPD.py`](/e:/lwt/xml/OpenWordMLTC/zero-shot/zero-shot-AAPD.py) 与 [`OpenWordMLTC/self_training/self_training.py`](/e:/lwt/xml/OpenWordMLTC/self_training/self_training.py)。该设计的动机在于提高正确标签进入候选集合的概率，缓解单次粗召回造成的早期语义遗漏。

### 4.2 交叉编码器重排

宽召回带来了更高的覆盖率，但同时也引入了额外噪声。因此，本文在候选集合上进一步引入交叉编码器进行细粒度重排。设重排模型为 $R(\cdot,\cdot)$，则

$$
s^{\text{rer}}(x_i,y)=R(x_i,y), \qquad y\in\mathcal{C}^{(t)}_i.
$$

按照重排得分保留前 $M$ 个候选标签，得到

$$
\widetilde{\mathcal{C}}^{(t)}_i
=
\operatorname{TopM}\big(\mathcal{C}^{(t)}_i, s^{\text{rer}}(x_i,y), M\big).
$$

当前实现默认采用 `cross-encoder/ms-marco-MiniLM-L-6-v2` 作为重排器，$M$ 由 `rerank_top_m` 控制，默认取值为 8。同时，系统会额外记录交叉编码器 top-1 标签 $\hat{y}^{\text{rer}}_i$，用于后续不确定性估计中的模型分歧项。

### 4.3 零样本 NLI 最终判别

在重排后的候选集合上，本文利用自然语言推理模型执行最终零样本判别。设 NLI 判别函数为 $F(\cdot)$，则标签 $y \in \widetilde{\mathcal{C}}^{(t)}_i$ 的归一化得分为

$$
p_i^{(t)}(y)
=
F\big(x_i,y;\widetilde{\mathcal{C}}^{(t)}_i\big).
$$

对应的伪标签定义为

$$
\hat{y}_i^{(t)}
=
\arg\max_{y \in \widetilde{\mathcal{C}}^{(t)}_i} p_i^{(t)}(y).
$$

实现上，系统调用 `pipeline("zero-shot-classification", multi_label=False)` 执行单标签判别，默认模型为 `MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33`，假设模板为 `"This example is {}"`。因此，训练过程中的核心监督信号表现为块级 top-1 伪标签，而非严格意义上的文档级多标签真值替代。这一近似虽简化了自训练流程，但在开放标签发现任务中具有较好的工程稳定性。

## 5 基于综合不确定性的尾部样本挖掘

### 5.1 综合不确定性建模

传统伪标注方法通常仅依据 top-1 置信度衡量样本可靠性。然而，在开放世界场景下，低置信度既可能意味着分类错误，也可能意味着当前标签空间尚未覆盖对应语义。为此，本文构建如下综合不确定性函数：

$$
u(x_i)=
\lambda_1(1-p_1)
+ \lambda_2\big(1-(p_1-p_2)\big)
+ \lambda_3\mathbb{I}\big[\hat{y}^{\text{rer}}_i \neq \hat{y}^{\text{nli}}_i\big],
$$

其中，

$$
p_1=\max_y p_i^{(t)}(y), \qquad
p_2=\text{第二大标签得分}.
$$

$\hat{y}^{\text{nli}}_i$ 表示 NLI 的 top-1 预测标签，$\mathbb{I}[\cdot]$ 为示性函数。当前实现默认参数为

$$
\lambda_1=0.5,\qquad \lambda_2=0.3,\qquad \lambda_3=0.2.
$$

该定义分别从“主标签置信度不足”“类别边界模糊”与“检索重排模型和推理模型发生分歧”三个维度刻画样本的不确定性。

### 5.2 高不确定性样本筛选

设第 $t$ 轮全部样本的不确定性得分集合为

$$
\mathcal{U}^{(t)}=\{u(x_1),u(x_2),\dots,u(x_N)\},
$$

则选取得分最高的前 $B$ 个样本构成尾部样本集合：

$$
\mathcal{T}^{(t)}=\operatorname{TopB}\big(\mathcal{D}, \mathcal{U}^{(t)}, B\big),
$$

其中，$B$ 由 `tail_set_size` 控制，默认取值为 500。该集合被视为当前标签空间最可能失配或覆盖不足的语义边界区域。

### 5.3 基于关键词频率补偿的新标签发现

与直接调用大语言模型生成新增标签不同，当前代码对尾部样本采用更具约束性的关键词筛选策略。对于高不确定性样本 $x_i$，系统首先定位与其属于同一原始文档编号的相邻文本块，以构造局部上下文组；随后仅提取被选中样本自身的前 3 个关键词作为种子候选，并统计其在局部组内的重复次数 $r_i(a)$。同时，计算关键词 $a$ 在全体关键词集合中的全局出现频次 $f(a)$。于是，候选新增标签满足

$$
f(a)-r_i(a) \ge \gamma,
$$

其中阈值 $\gamma$ 在当前实现中固定为 15。该机制的含义在于：若某关键词仅在局部相邻块中重复出现，则其更可能只是文档内部的冗余表述；相反，若其在全局语料中具有足够支持，则更适合作为新增标签候选。

## 6 标签空间扩展与更新机制

### 6.1 新标签内部去冗余与全局过滤

设通过尾部样本挖掘得到的原始候选集合为 $\mathcal{A}^{(t)}_{\text{raw}}$。首先，系统在候选集合内部执行语义去冗余；若两个候选标签的向量相似度超过动态阈值

$$
\tau_t = \tau_0 + \frac{t}{100},
$$

则删除其中一个语义近邻标签。当前实现中，$\tau_0$ 对应参数 `sim_threshold`，默认取值为 0.55。

随后，将通过内部筛选的候选标签与累计标签池 $\mathcal{P}^{(t)}$ 进行相似度比较，仅当

$$
\max_{y\in\mathcal{P}^{(t)}} \mathrm{sim}(a,y) < \tau_t
$$

时，标签 $a$ 才被接纳为新增标签。该设计明确以累计标签池而非当前工作标签空间为过滤对象，从而避免历史上已出现过的近义标签在后续轮次中被重复引入。最终，每轮新增标签数量还受到 `max_add_label` 的约束，默认上限为 10。

### 6.2 低频标签删除

设第 $t$ 轮基于工作标签空间的伪标注频次函数为 $n_t(y)$。对于当前轮中极少被预测到的标签，系统将其从累计标签池中移除：

$$
\mathcal{R}^{(t)}_{\text{low}}
=
\{y\in\mathcal{L}^{(t)} \mid n_t(y)\le 6\}.
$$

该策略有助于抑制噪声标签的持续积累，防止标签空间在迭代过程中无约束膨胀。

### 6.3 头部标签抑制与工作标签空间重构

为了避免极少数高频标签在后续迭代中持续主导伪标注结果，本文引入跨轮次的头部标签记忆机制。设当轮高频标签集合为

$$
\mathcal{R}^{(t)}_{\text{head}}
=
\left\{
y \in \operatorname{TopH}(\mathcal{L}^{(t)}, n_t)
\;\middle|\;
n_t(y) > \tau_h
\right\},
$$

其中，$H$ 对应 `max_majority_num`，默认取值为 5；$\tau_h$ 对应 `majority_num`，默认取值为 350。系统将其累积到历史头部标签记忆集中：

$$
\mathcal{M}^{(t+1)} = \mathcal{M}^{(t)} \cup \mathcal{R}^{(t)}_{\text{head}}.
$$

于是，累计标签池与下一轮工作标签空间的更新分别为

$$
\mathcal{P}^{(t+1)}
=
\big(\mathcal{P}^{(t)} \setminus \mathcal{R}^{(t)}_{\text{low}}\big)
\cup
\mathcal{A}^{(t)},
$$

$$
\mathcal{L}^{(t+1)}
=
\mathcal{P}^{(t+1)} \setminus \mathcal{M}^{(t+1)}.
$$

这一设计与代码中的两个输出文件完全对应：`update_labelspace.txt` 用于维护累计标签池，`update_labelspace{t+1}.txt` 用于维护下一轮实际参与 zero-shot 分类的工作标签空间。二者的区分有助于在“保留历史语义记忆”与“控制当前分类偏置”之间取得平衡。

## 7 算法流程总结

综合上述模块，OpenWord-DLRS 的训练流程可概括为：

1. 对训练语料生成关键词表示；
2. 基于关键词语义聚类与代表样本归纳构建初始标签空间；
3. 在当前工作标签空间上执行稠密检索、交叉编码器重排和零样本 NLI 判别，得到块级伪标签；
4. 基于综合不确定性筛选高风险尾部样本；
5. 从尾部样本关键词中发现新增标签候选，并通过内部去冗余与累计标签池过滤控制标签质量；
6. 删除低频标签，记录头部标签记忆，并构造下一轮工作标签空间；
7. 重复步骤 3 至步骤 6，直至达到预设迭代轮数。

当前实现默认执行 10 轮自训练迭代。整体而言，OpenWord-DLRS 并非将标签空间视为固定先验，而是将其建模为一个可随伪标注反馈持续演化的动态对象；这使得模型能够在弱监督条件下逐步吸收开放语义、抑制冗余标签，并提升标签空间对真实语料语义结构的覆盖能力。

## 8 关键实现参数

为便于论文写作时统一口径，当前代码中的关键默认参数可整理为表 1。

| 模块 | 参数 | 默认值 |
| --- | --- | --- |
| 检索编码器 | `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` |
| 重排模型 | `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| NLI 模型 | `model` | `MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33` |
| 候选召回规模 | `candidate_top_k` | 32 |
| 重排保留规模 | `rerank_top_m` | 8 |
| 尾部样本规模 | `tail_set_size` | 500 |
| 主导标签阈值 | `majority_num` | 350 |
| 最大主导标签数 | `max_majority_num` | 5 |
| 新标签初始相似度阈值 | `sim_threshold` | 0.55 |
| 初始标签去冗余阈值 | `lower_bound` | 0.80 |
| 单轮新增标签上限 | `max_add_label` | 10 |
| 不确定性权重 | `lambda_top1, lambda_margin, lambda_disagreement` | 0.5, 0.3, 0.2 |

若论文仅展示某一具体数据集，还可在本节补充对应的 `dynamic_iter`、`cluster_size` 与局部大语言模型配置，以保证实验设置与方法叙述保持一致。
