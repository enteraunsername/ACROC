# ACROC 第4章：分解/合成反应改进（文献检索）

## 目标
- 不改动代码：仅从“开源/可公开访问论文”出发，提出对论文当前分解/合成反应痛点的更优替代与优化思路。

## 论文当前痛点（我建议你重点盯住的 3 个）
1) 分解反应依赖谱分解（λ2/特征向量）：对每个“分子/簇”重复做特征分解，代价高、对噪声/小图不稳。
2) 分解天然是 2-way（KMeans=2）：真实数据常需要多路切分或局部剥离，2-way 反复迭代会带来随机性与误切。
3) 合成反应候选对若按全局两两 KE 计算：计算量 O(m^2) 且偏向“大簇”，再叠加随机轮盘赌，导致收敛慢、可重复性差。

## 分解反应：不做谱分解的替代方法（更贴近“图切分/社区发现”）

### 1) 模块度优化的社区发现（Louvain / Leiden）
**思路**：把“分子内部图”看作社区发现问题，直接产出多个社区；再把它当作一次分解（可二分也可多分）。
- **为什么更好**：不需要求特征向量；Leiden 还保证社区连通性，减少“分解出断裂簇”的坏情况。
- **适配你论文框架**：分解触发后，对被选中的分子运行 Leiden/Louvain；用“能量/PE下降”或“模块度增益”做接受判据。
参考：
- Louvain（Fast unfolding / modularity）https://arxiv.org/abs/0803.0476
- Leiden（guaranteed connected communities）https://arxiv.org/abs/1810.08473

### 2) 基于随机游走信息压缩的 Infomap / Map Equation
**思路**：用随机游走的“编码长度”来定义社区结构（更像在拟合图上的扩散动力学）。
- **为什么更好**：对“瓶颈边/弱连接”更敏感，常比纯 cut 更符合流形/扩散结构；还能给层次结构（多尺度分解）。
- **适配框架**：把一次分解看作一次 Infomap 切分；把切分后的子社区当新分子。
参考：
- Map equation & Infomap 教程综述（开源预印本）https://arxiv.org/abs/2311.04036

### 3) 局部图划分（Personalized PageRank / PageRank Nibble / 非线性 PageRank）
**思路**：分解时往往只需要“从一个中心原子/边界点”剥离一个低导通率（low conductance）的子集，而不必对整个簇做全局谱分解。
- **为什么更好**：局部算法可近线性；对大分子特别划算；且天然输出“扫掠 cut（sweep cut）”一系列候选，能更稳健。
- **适配框架**：用你 syn 里已经有的“中心点/最近点对”作为 seed，跑 PPR/Nibble 得到候选子集，再用论文的能量判据选最优 cut。
参考（都可公开访问）：
- 非线性 PageRank 用于局部划分 https://arxiv.org/abs/2409.01834
- PageRank Nibble 在随机块模型上的分析 https://arxiv.org/abs/2303.06699
- 子线性时间 PageRank 计算（可作为加速思想来源）https://arxiv.org/abs/1202.2771
- 局部划分理论/改进 Cheeger 不等式（帮助理解“何时谱切靠谱”）https://arxiv.org/abs/1504.00686

### 4) 多层（Multilevel）图划分（coarsen → partition → refine）
**思路**：把分解看作经典 balanced partitioning / min-cut 类问题，用多层 coarsening+refinement 做近似最优切分。
- **为什么更好**：工程上非常快（尤其大图），且能显式控制平衡性/最小簇规模，减少“切出很碎的小簇”。
- **适配框架**：分解时对分子子图做 multilevel bipartition；用能量判据过滤。
参考：
- Deep multilevel graph partitioning https://arxiv.org/abs/2105.02022
- Jet: multilevel partitioning on GPUs（偏工程实现）https://arxiv.org/abs/2304.13194

### 5) 近线性启发式：标签传播及其稳定化（LabelRank / roLPA）
**思路**：标签传播类方法近线性时间给出社区；再用你的能量判据挑“值得分解的切”。
- **为什么更好**：极快，可作为“分解候选生成器”，再由能量/约束做二次筛选。
参考：
- LabelRank（稳定化的 LPA）https://arxiv.org/abs/1303.0868
- roLPA（role-based LPA，缓解 monster community）https://arxiv.org/abs/1601.06307

### 6) 先做 cut-sparsifier / sketch 再切分（降维保 cut）
**思路**：在不显著改变 cut 结构的前提下，把图稀疏化/压缩，再跑任一切分算法。
- **为什么更好**：把“贵”的分解反应前置成更便宜的图表示；对你这种反复分解特别有用。
参考：
- Graph cut sketching / sparsification 复杂度 https://arxiv.org/abs/1403.7058

## 合成反应：不改代码前提下，你可以优先考虑的优化方向（思想层面）

### A) 候选对生成：从全局 O(m^2) 变成“只看相邻分子”
- 痛点：如果每轮都对所有分子对算 KE，m 大时是瓶颈。
- 优化想法：只把 **G_nomutual 上确实存在跨簇边** 的分子对当作候选（用“分子图/簇间图”来维护邻接），KE 只在邻接边上更新。

### B) KE 归一化：减少“大簇天然更容易被选中”的偏置
- 痛点：纯求和的 KE 倾向选择边数多/点数大的簇对。
- 优化想法：把 KE 做规模归一化（例如除以 boundary 点数、或除以 vol 的函数），让“边界紧密的小簇”也有机会被合并。

### C) 选择策略：用 Top-K/优先队列替代轮盘赌（减少随机性）
- 痛点：轮盘赌会反复尝试明显不可能成功的合并，浪费迭代次数。
- 优化想法：每轮优先尝试 KE 最大的 Top-K 候选；随机性只用于探索（例如 epsilon-greedy）。

### D) 接受判据：把“图切分质量”纳入合成/分解的能量模型
- 痛点：如果能量项与图结构不一致，会出现“合成成功但结构更差”。
- 优化想法：在不改代码的前提下，你至少可以在实验分析里对比：
  - 合并前后 conductance / modularity / map equation 是否同步改善
  - 失败合并是否集中在某类边界（噪声点、kNN桥接边）

## 参考文献（本次检索引用，均可公开访问）
1. https://arxiv.org/abs/0803.0476
2. https://arxiv.org/abs/1810.08473
3. https://arxiv.org/abs/2311.04036
4. https://arxiv.org/abs/2409.01834
5. https://arxiv.org/abs/2303.06699
6. https://arxiv.org/abs/1202.2771
7. https://arxiv.org/abs/1504.00686
8. https://arxiv.org/abs/2105.02022
9. https://arxiv.org/abs/2304.13194
10. https://arxiv.org/abs/1303.0868
11. https://arxiv.org/abs/1601.06307
12. https://arxiv.org/abs/1403.7058

