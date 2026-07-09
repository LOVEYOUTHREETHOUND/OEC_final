# RQ1-精简版：遥感任务协同计算的问题建模

---

## 1. 任务模型

### 1.1 基本设定

预处理后的遥感影像**必须**经分块（tiling）后，每个 tile 为一个独立推理任务。tile 尺寸是可控的决策变量。时隙 $t$ 到达的任务记为：

$$i \triangleq (\mathbf{x}_i, \Phi_i^{(s)}, \delta_i, \text{src}_i, \text{type}_i)$$

| 字段 | 含义 |
|------|------|
| $\mathbf{x}_i$ | 预处理后的 tile，$\mathbb{R}^{H \times W \times C}$；tile 尺寸 $\tau_i$ 决定 $\mathbf{x}_i$ 的 $H \times W$ |
| $\Phi_i^{(s)}$ | 推理模型（从模型族中选择），决定精度、内存、计算量 |
| $\delta_i$ | 截止时间（deadline） |
| $\text{src}_i$ | 数据源卫星，标记数据"出生地" |
| $\text{type}_i$ | 任务类型：语义分割 / 目标检测 / 实例分割 |

### 1.2 模型族

所有模型共享 DeepLabV3+ 的 ASPP + Decoder 头部，仅变换 Backbone 网络。单星内存 100–200MB 是硬约束。引入三档 Backbone：

| 模型 $\Phi^{(s)}$ | Backbone | 总参数量 | 总内存需求 (含激活) | 相对 mIoU | 单星可行 |
|:---:|---|:---:|:---:|:---:|:--:|
| $\Phi^{\text{large}}$ | ResNet-101 | ~60M | ~740 MB | ★★★ | **否** |
| $\Phi^{\text{medium}}$ | ResNet-50 | ~43M | ~450 MB | ★★☆ | 临界 |
| $\Phi^{\text{small}}$ | MobileNetV2 | ~5.8M | ~80 MB | ★☆☆ | 是 |

> **Note**：需要调研一下当前低轨星座中的算力情况

### 1.3 空间与流程可行性

推理过程涉及两类独立的切分决策：

| 维度 | 切分什么 | 决策变量 | 是否强制 |
|:---:|---------|---------|:---:|
| **空间** | 影像 → tile | tile 尺寸 $\tau_i$ | **强制**（遥感图 GB 级，必须切） |
| **流程** | 模型计算管线 | 切分点 $\text{cut\_pt}_i$ | 可选（不切 = 单星完整推理） |

- （$\tau_i$）：小 tile → 更多 tile，更细粒度负载分配，更多边界需共享，但单 tile 内存更小；大 tile → 更少 tile，边界少，但单 tile 内存可能超限。
- （$\text{cut\_pt}_i$）：$\Phi^{\text{large}}$ 必须切分（740MB > 100MB 单星内存），$\Phi^{\text{small}}$ 切分无必要，$\Phi^{\text{medium}}$ 可选。
- 模型大小 + tile 大小 + 是否流程切分，三个维度共同决定单星内存是否可行、通信量多大。不存在一种组合在所有条件下最优——MARL 学习的是条件依赖的动态决策。

### 1.4 流程切分点（粗粒度管线切分）

当流程切分被启用（$\text{cut\_pt}_i \neq 0$），从以下三个预定义候选中选取：

| 切分方案 | 切分位置 | 星 A 执行 | 星 B 执行 | 中间通信量 |
|:---:|---|---------|---------|:---:|
| **不切 (0)** | — | 完整模型 | — | 0 |
| **特征提取后 (1)** | Backbone 之后 | Backbone (ResNet-101) | ASPP + Decoder | Backbone 输出 (32×32×2048, ~8 MB) |
| **Encoder 后 (2)** | Encoder (= Backbone + ASPP) 之后 | Backbone + ASPP | Decoder | ASPP 瓶颈 (32×32×256, ~1 MB) |

以 DeepLabV3+ (ResNet-101, tile 512×512) 为例，各切分方案的资源分布：

| 切分方案 | 星 A 参数内存 | 星 A 激活内存 | 星 B 参数内存 | 星 B 激活内存 | 通信量 (FP32) |
|:---:|---:|---:|---:|---:|---:|
| 不切 | 60M | — | — | — | 0 |
| 特征提取后切 | ~44M (ResNet) | ~400 MB | ~16M (ASPP+Decoder) | ~100 MB | ~8 MB |
| Encoder 后切 | ~59M | ~480 MB | ~1M (Decoder) | ~20 MB | ~1 MB |

> **管线切分动作编码**：$\text{cut\_pt}_i \in \{0, 1, 2\}$，分别对应不切(0)、Backbone 后切(1)、Encoder 后切(2)。

### 1.5 星座模型（简化假设）

星座由 $K$ 颗**同构**卫星组成，部署于**同一轨道面**，沿轨道方向编号为 $\text{sat}_1, \text{sat}_2, \dots, \text{sat}_K$。

**星间链路拓扑为静态环**，每颗星仅与前后两颗邻星直连：

$$G_{\text{topo}} = (\mathcal{S}, E_{\text{link}}), \quad \mathcal{S} = \{\text{sat}_1, \dots, \text{sat}_K\}$$

$$E_{\text{link}} = \{(\text{sat}_k, \text{sat}_{(k \bmod K)+1}) \mid k = 1, \dots, K\}$$

其中每条链路带宽为常数 $B$。

**同构卫星资源**（所有星相同）：

$$R = (M, C, E^{\text{budget}})$$

- $M$：单星可用内存
- $C$：单星计算能力（FLOPs/slot）
- $E^{\text{budget}}$：单星能耗预算（Wh/slot）

**多跳传输**：$\text{sat}_i$ 到 $\text{sat}_j$ 的最短路径跳数为 $h(i, j) = \min(|i-j|,\; K - |i-j|)$（取环上两个方向中的较短路径），通信延迟 = $h(i, j) \cdot \text{data\_size} / B$。

---

## 2. 多星分配约束

### 2.1 资源约束

对任务 $i$ 分配给 $\text{sat}_k$ 的计算负载（模型 $\Phi_i^{(s)}$ 的完整执行或切分后的部分）：

$$\text{mem}(\Phi_i^{(s)}, \text{cut\_pt}_i, \text{part}) \leq M \quad \text{(内存)}$$
$$\text{flops}(\Phi_i^{(s)}, \text{cut\_pt}_i, \text{part}) \leq C \cdot \delta_i \quad \text{(计算)}$$
$$\text{energy}(\Phi_i^{(s)}, \text{cut\_pt}_i, \text{part}) \leq E^{\text{budget}} \quad \text{(能耗)}$$

其中 $\text{part} \in \{\text{A}, \text{B}\}$ 表示切分后的前半段或后半段（不切时为完整模型，$\text{part} = \text{A}$）。资源参数由 §1.4 的切分表给出。

### 2.2 通信与截止时间约束

$$T_i^{\text{compute}}(\Phi_i^{(s)}, \text{cut\_pt}_i) + T_i^{\text{comm}}(\text{cut\_pt}_i, \text{src}_i) \leq \delta_i$$

- $T_i^{\text{compute}}$：切分方案下的推理延迟（不切 = 单星完整推理，切分 = 两星流水线并行）
- $T_i^{\text{comm}}$：跨星通信延迟。数据从 $\text{src}_i$ 到计算星的跳数 $h$ 由 §1.5 环拓扑决定：

$$T_i^{\text{comm}} = h(\text{src}_i, \text{sat}_A) \cdot \frac{|\mathbf{x}_i|}{B} + h(\text{sat}_A, \text{sat}_B) \cdot \frac{\text{comm\_size}(\text{cut\_pt}_i)}{B}$$

第一项为原始数据从源星到计算星 A 的传输延迟（不切时 $\text{sat}_A$ 即执行星）；第二项为切分中间激活从星 A 到星 B 的传输延迟（不切时为零）。

---

## 3. 跨星上下文共享（这是单独的问题，独立考虑）

### 3.1 问题定义

当不同 tile 分发到不同卫星独立推理时，全局上下文丢失导致边界误分割。跨星共享上下文信息可缓解此问题：

$$P(\hat{y}_i \mid \mathbf{x}_i) < P(\hat{y}_i \mid \mathbf{x}_i, \mathbf{z}_j)$$

其中 $\mathbf{z}_j$ 是来自其他卫星 tile $j$ 的上下文信息。

> **与标准做法的关系**：标准大图推理通过 overlap-tile 解决边界连续性（重叠像素取平均），不涉及中间特征交换。本方案将上下文共享建模为**有代价的外部决策**——融入邻居信息需消耗带宽/能量，MARL 学习的是"是否值得在当前约束下共享"，而非如何共享。

### 3.2 融合机制：轻量 Cross-Attention （或考虑概率推断模型、HMM之类的）

共享后的特征通过一个可插拔的**轻量 Transformer 融合模块**整合到当前 tile 的推理中：

$$\mathbf{z}_i' = \mathbf{z}_i + \text{CrossAttention}(W_Q \mathbf{z}_i,\; W_K \mathbf{z}_j,\; W_V \mathbf{z}_j)$$

其中 $\mathbf{z}_i$ 为当前 tile 的 Encoder 瓶颈特征，$\mathbf{z}_j$ 为邻星传来的特征，融合后的 $\mathbf{z}_i'$ 送入 Decoder。

**空间位置编码**：Encoder 输出的 2D 特征图通过全局坐标正弦编码标记每个特征点的空间位置，Cross-Attention 通过 query-key 距离自然学习空间邻近性优先。

### 3.3 两级共享模式

应对不同尺度的上下文丢失，将共享分为两个并行可选的层级：

| 级别 | 问题域 | 共享内容 | 注意力范围 | 通信量 | 融合方式 |
|:---:|------|---------|:---:|------|------|
| **L1** | 边界像素衔接 | 相邻 tile **边界区域的密集特征图** | 相邻 tile 仅 | MB 级 | Windowed Cross-Attention |
| **L2** | 跨多 tile 大要素 | 每个 tile 的**少量摘要 tokens** | 全星座任意 tile | KB 级 | Global Cross-Attention |

**L1 — 局部密集融合**：只传 tile 边界区域特征（如最右/最左 4 列），QK 注意力约束在相邻 tile 的边界窗口内。高效解决边界连贯性。

**L2 — 全局稀疏融合**：每个 tile 经 Token Aggregator（池化 + MLP）压缩为 $N_{\text{token}}$ 个摘要 token（如 3×256 维），仅传 token 不传原始特征图。轻量解决跨多 tile 的大尺度地物（河流、公路）的上下文补全。

> **L1 和 L2 可同时启用**，分别解决相邻局部和跨多 tile 全局两个正交的上下文丢失问题。

### 3.4 决策变量

上下文共享决策矩阵扩展为两级：

$$\mathbf{C}_i^{\text{local}} \in \{0, 1\}^{K \times K}, \quad \mathbf{C}_i^{\text{global}} \in \{0, 1\}^{K \times K}$$

- $C_{kl}^{\text{local}} = 1$：Star $k$ 和 $l$ 间启用 L1 局部融合
- $C_{kl}^{\text{global}} = 1$：Star $k$ 和 $l$ 间启用 L2 全局融合

实际有效组合：$\{(0,0), (1,0), (0,1), (1,1)\}$，对应无共享 / 仅局部 / 仅全局 / 全开。

> **与 GNN 的联系**：L1 局部融合 = GAT 的单跳邻居聚合，L2 全局融合 = virtual node 消息传递。在环拓扑上，MARL 智能体间的通信与特征融合共用同一图结构。

---

## 4. 目标函数（TBD）

### 4.1 mIoU 的分解与内生性

mIoU 不是外部常量，而是由决策变量**内生决定**的量：

$$\text{mIoU}(\Phi^{(s)}, \mathbf{C}) = \text{mIoU}_{\text{base}}(\Phi^{(s)}) \;+\; \Delta_{\text{L1}}(\mathbf{C}^{\text{local}}) \;+\; \Delta_{\text{L2}}(\mathbf{C}^{\text{global}})$$

| 分量 | 含义 | 决定因素 |
|------|------|---------|
| $\text{mIoU}_{\text{base}}(\Phi^{(s)})$ | 模型族固有精度 | $\Phi^{(s)}$（大 > 中 > 小） |
| $\Delta_{\text{L1}}(\mathbf{C}^{\text{local}})$ | 局部融合收益：消除 tile 边界伪影 | 与多少相邻星交换了边界特征 |
| $\Delta_{\text{L2}}(\mathbf{C}^{\text{global}})$ | 全局融合收益：补全大尺度跨 tile 上下文 | 与多少远距离星交换了摘要 tokens |

**精度是"买来"的**：$\Delta_{\text{L1}}$ 需传输 MB 级边界特征 → 消耗带宽和能量；$\Delta_{\text{L2}}$ 需传输 KB 级 tokens → 代价较小但收益也有上限。

### 4.2 主优化问题（多目标形式）

$$\min_{\mathbf{a}_i} \quad \Big( \; T(\mathbf{a}_i), \; E(\mathbf{a}_i), \; -\text{mIoU}(\mathbf{a}_i) \; \Big)$$

三个目标相互冲突：

- 选大模型 → mIoU 高，但 $T$ 高且可能迫使切分
- 开上下文共享 → mIoU 提升，但增加通信 $T$ 和 $E$
- 选小模型 + 不开共享 → $T$ 和 $E$ 最低，但 mIoU 最低

### 4.3 标量化形式（MARL 奖励用）

MARL 训练需要标量奖励，采用 $\ell_1$ 标量化（加权和）：

$$R_i = -\left( w_T \cdot \frac{T}{T_{\max}} + w_E \cdot \frac{E}{E_{\max}} \right) + w_{\text{mIoU}} \cdot \frac{\text{mIoU} - \text{mIoU}_{\min}}{\text{mIoU}_{\max} - \text{mIoU}_{\min}}$$

| 参数 | 约束 |
|------|------|
| $w_T, w_E, w_{\text{mIoU}}$ | 均 ≥ 0，$w_T + w_E + w_{\text{mIoU}} = 1$ |
| 归一化基准 | $T_{\max}, E_{\max}, \text{mIoU}_{\max}$ 为各指标在动作空间中的上下界 |

权重 $w_{\text{mIoU}}$ 决定了系统对精度的偏好强度，可看作任务场景参数（如农田监测 vs 军事侦察对精度容忍度不同）。

### 4.4 约束主目标形式

实际部署时可将某一维度作为硬约束，优化其余维度：

$$\min \; T(\mathbf{a}_i) \quad \text{s.t.} \quad E(\mathbf{a}_i) \leq E_{\text{budget}}, \quad \text{mIoU}(\mathbf{a}_i) \geq \text{mIoU}_{\text{task}}$$

### 4.5 Pareto 前沿

$$\mathcal{P}^* = \{ \mathbf{a}_i \mid \nexists \mathbf{a}_j \text{ 在 } (T, E, \text{mIoU}) \text{ 三项目标上均不劣于且至少一项严格优于 } \mathbf{a}_i \}$$

MARL 的目标是学习策略 $\pi(\mathbf{a}_i \mid s_i)$ 使得生成的动作落在 Pareto 前沿附近，并能根据任务权重偏好在前沿上滑动。

---

## 5. 决策变量与动作空间

$$\mathbf{a}_i = (\Phi_i^{(s)}, \tau_i, \text{cut\_pt}_i, \text{context\_share}_i)$$

| 维度 | 类型 | 含义 | 取值 |
|------|:---:|------|------|
| $\Phi_i^{(s)}$ | 模型 | 模型规模选择 | $\{\text{small}, \text{medium}, \text{large}\}$ |
| $\tau_i$ | **空间切分** | tile 尺寸（控制影像分块粒度） | $\{256, 512, 1024\}$ (px) |
| $\text{cut\_pt}_i$ | **流程切分** | 模型管线切分点 | $\{0\!:\!\text{不切},\, 1\!:\!\text{Backbone 后},\, 2\!:\!\text{Encoder 后}\}$ |
| $\text{context\_share}_i$ | 上下文 | 跨星上下文共享矩阵（两级独立） | $\mathbf{C}_i^{\text{local}} \in \{0,1\}^{K \times K}$，$\mathbf{C}_i^{\text{global}} \in \{0,1\}^{K \times K}$ |

### 空间切分 ($\tau_i$) vs 流程切分 ($\text{cut\_pt}_i$)

```
原始遥感影像 (GB级)
       │
       ├── 空间切分: 切成 N 个 tile，tile 大小 = τ × τ
       │        τ=256 → 更多tile，更细粒度分发
       │        τ=1024 → 更少tile，每tile上下文更丰富
       │
       └── 流程切分: 单个 tile 的计算管线是否跨星分段
                cut_pt=0 → 一星从头到尾（不切管线）
                cut_pt=1 → Backbone(A星) → 传特征 → ASPP+Decoder(B星)
                cut_pt=2 → Encoder(A星) → 传特征 → Decoder(B星)
```

**注意**：空间切分是强制的（影像必须切 tile 才能处理），tile 尺寸可控；流程切分是可选的（大模型因内存约束而必须启用），切分粒度可控。两维度正交：tile 512×512 且流程不切 = 每颗星独立完整推理一个 512×512 tile；tile 256×256 且 Encoder 后切 = 每颗星跑自己的 Encoder，发给另一颗星跑 Decoder。

### 合法性约束

| 模型 | 流程切分可用点 | 空间切分约束 |
|------|:---:|------|
| $\Phi^{\text{large}}$ | {1, 2}（不切不可行，内存不足） | tile + 激活内存 ≤ 100MB，大 tile 不可行 |
| $\Phi^{\text{medium}}$ | {0, 1, 2} | 不切管线且大 tile 时临界 |
| $\Phi^{\text{small}}$ | {0}（切分无必要） | 最大 tile 仍可行 |

---

## 6. 整体框架

```
                   遥感大图语义分割任务
                          │
              空间切分（强制的）: τ ∈ {256, 512, 1024}
                          │
                    N 个 tile 分配到 K 颗卫星
                          │
             ┌────────────┼────────────┐
             v            v            v
        选小/中模型    选大模型     选大模型
        流程不切       流程不切      流程切分
       (单星完整推理)  (不可行!)   (2星协同推理)
             │                        │
             │              ┌─────────┼─────────┐
             │              v         v         v
             │           Backbone后  Encoder后  需更大
             │             切(8MB)   切(1MB)    tile?
             │              │         │         │
     ┌───────┴───────┐      │         │         │
     v               v      └────┬────┘         │
 无上下文共享    有跨星上下文共享                    │
                 ├─ L1 边界融合                ────┘
                 └─ L2 全局摘要
                          │
                          v
    MARL 联合决策:
    (模型, τ, 切分点, 共享) → (T, E, mIoU)
    └── 动作决定精度 ────────→ mIoU = base(Φ) + Δ(L1) + Δ(L2)
```

---

## 7. 核心创新逻辑

本研究区别于传统卫星任务调度研究的两条主线：

### 创新 1：从"抽象任务参数"到"具象模型感知"

```
传统研究:
  任务 = {计算量, 数据量, 截止时间}     ← 三个抽象数，不区分 YOLO 还是 DeeplabV3+
  优化目标 = min(延迟, 能耗)             ← 精度不可知

本研究:
  任务 = 具体 DL 模型 (DeepLabV3+)      ← 模型的参数量、内存分布、切分点通信量精确已知
  优化目标 = min(延迟, 能耗, 1/mIoU)     ← 精度是核心决策依据
```

**由此引出三个子问题：**
- 大模型切分：740MB > 100MB → 必须切分才能部署
- 模型族选择：大模型高精度但需切分，小模型不切但精度低 → MARL 元决策
- 切分粒度选择：在哪切决定通信量与单星负载的平衡

### 创新 2：从"通信作为代价"到"通信换精度"

```
传统研究:
  星间通信 = 纯开销，尽量最小化           ← 通信 = cost

本研究:
  星间传输特征 = 精度投资                ← 通信 = investment
  → 相邻卫星传边界特征 (L1 局部融合)     → 消除 tile 边界伪影
  → 全局卫星传摘要 tokens (L2 全局融合)  → 补全跨多 tile 的大尺度地物上下文
  → MARL 决策: "当前带宽/能耗约束下，这批 tiles 之间是否值得做特征共享？"
```

**核心洞见**：通信换精度是分布式卫星场景的**独有优化自由度**——单星推理不存在"传还是不传"的决策（所有 tile 在同颗星上，上下文天然共享）；传统分布式调度不建模模型内部特征，无法评估"传了什么、换来了什么精度收益"。本研究的优化问题中，精度不是外部给定常量，而是被通信决策**内生决定**的变量。

### 两条创新的关系

```
创新 1（具象模型） ──→ 精度成为可量化的决策依据
                              │
                              ├── 精度需求驱动: 选大模型还是小模型？
                              ├── 内存约束驱动: 单星跑还是切分跑？
                              └── 切分点驱动: 通信负载多大？
                              │
创新 2（通信换精度） ──→ 精度是可投资提升的量
                              │
                              ├── 局部融合: 相邻 tile 的特征共享
                              └── 全局融合: 跨多 tile 的摘要共享
                              │
              ┌───────────────┘
              v
    MARL 四维决策空间:
    (模型选择, tile尺寸, 管线切分点, 上下文共享) → (延迟, 能耗, mIoU)
```

---

## 8. 核心贡献点

| # | 贡献点 | 对应创新 |
|:---:|------|:---:|
| 1 | **具象模型感知的任务调度**：以具体深度学习模型（DeepLabV3+）为任务表征单元，替代传统"计算量/数据量"抽象参数；模型的语义分割精度（mIoU）作为优化目标的核心维度，使得调度决策直接感知精度影响 | 创新 1 |
| 2 | **通信换精度的上下文共享**：利用星间链路在分布式推理间交换特征（局部边界特征 + 全局摘要 tokens），以通信开销换取分割精度提升，形成通信-精度-延迟的三维 Pareto 权衡，这是分布式卫星场景独有的优化自由度 | 创新 2 |
| 3 | **空间-流程双维切分**：将分布式推理拆解为两个正交维度——空间切分（tile 尺寸 $\tau$，强制的，控制负载粒度）和流程切分（管线切分点，可选的，应对大模型内存约束），避免将两者混为一谈导致的决策空间爆炸 | 创新 1 的工程支撑 |

---

## 9. 后续计划

```
Phase 1: 问题形式化
├── 模型假设集（星座规模、单星资源、链路模型、tile 尺寸、任务到达率）
├── 完整数学优化问题（目标函数 + 所有约束）
├── 复杂性分析
└── 关键设计维度提取

Phase 2: 场景实例化
├── 三档模型量化参数（计算量、内存、输出尺寸） + 各切分点的资源分布表
└── 卫星集群参数设定

Phase 3: MARL 衔接
├── 状态空间：卫星资源 + 链路拓扑 + 任务队列 + 可用模型族
├── 动作空间：模型选择 × tile 尺寸 × 管线切分点 × 上下文共享级别
├── 奖励函数：标准化加权组合
└── GNN 编码：计算图 → 图嵌入作为通信 backbone

Phase 4: 实验验证
├── 仿真平台搭建
├── 基线设计（集中式最优、贪心、随机、单模型固定、固定切分+固定tile）
└── 消融实验（模型选择消融、tile尺寸消融、上下文级别消融）
```
