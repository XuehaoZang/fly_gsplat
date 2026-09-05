# 3cam 稀疏视角超参数 Sweep 方案

状态：第1-3步（审查/假设/实验设计）已批判性修订，待执行第4步（实现自动化脚本）。

## 背景与关键发现

- **Pipeline 本质**：逐帧独立训练，每帧仅3张静态图像，2000 iters 实测 80~112s/帧，收敛后仅350~980个高斯点。96 GPU-hour 预算下真正稀缺的是**人工QC带宽**，不是算力。
- **"ratio3" 纠错**：`ratio3_sh0_dense` 里的 "ratio3" 是 `max-gauss-ratio 3.0`（高斯尺度比例正则化阈值），**不是**分辨率下采样。
- **核心异常**：对比 3cam（`ctrl_119_004`/`010`，480帧）vs 4cam baseline（`ctrl_009_002` `G2b_G9`，同套 `ratio3_sh0_dense`@2000iters）：`extent_overshoot` 均值 1.10(4cam) → 1.72~1.78(3cam)，>1.5占比达73%，>2.0占比32%，两视频高度一致。
- **新确认的混杂变量（本轮新增）**：3cam（Expr_119）拍摄帧率为 **8000fps**，4cam（ctrl_009等）为 **16000fps**（证据：`postprocessing/labeling/motion/density.py:204` 注释明确写"3相机数据集拍摄fps=8000，是这里16000fps的一半"；但 `select_frame_window.py`、`postprocessing/kinematics/diagnostics.py`、`wrap_mechanism_diag.py`、`real_data_validation.py`、`scene.py` 五处硬编码 `FPS=16000.0`，对3cam数据大概率是误用，这是超出本sweep范围的既有bug，仅记录不在本轮修复）。**影响**：4cam→3cam 的 `extent_overshoot` 对比同时混杂了"少一路相机"和"帧率减半→同等物理时间内每帧翅膀角速度扫过范围更大→sparse pixel重建出的轮廓边缘更容易模糊"两个变量，不能把全部劣化归因于纯粹的相机数减少。
- **附带信号**：`ctrl_119_010` 只有69%帧 T4 状态为 `ok`（`ctrl_119_004` 88.75%），失败集中在翅膀 RANSAC/leading-edge 判定，可能与点云质量相关。

## 对既有4cam sweep结果的批判性复盘（本轮新增，改变了Round 1设计）

项目里没有独立的"结论性分析文档"，结论都编码在 `outputs/ctrl_009_002_{8groups,densify_6groups}_100frames/summary.json` + `run/serial/batch_*.py` 的分组定义里。重新读了这两份的原始数据后，有一个**直接推翻我上一版方案默认前提**的发现：

1. **当前生产配置 `ratio3_sh0_dense` ≠ 未调优的基线**，它等于 `densify_6groups` 实验里表现最好的组合 `H6_grad_thresh_low_refine_fast`（`densify-grad-thresh 0.0004` + `refine-every 50`）叠加在 `G2b_G9` 基线之上——也就是说**3cam的问题是发生在"已经用了4cam验证过的最优密度化设置"之上的**，不是"还没调优"。这意味着 Round 1 不该是"从零试各种densify方向"，而应该是**"验证H6这个在4cam上验证有效的组合，在3cam弱约束下是否需要往回调（更保守）或者需要额外的漂移抑制手段来配合"**。
2. **`generate_hull.py` 的 `n_samples` 默认值是 `10_000`，生产pipeline（`gpu/schedule/schedule.py`）从未覆盖它**——README 里"1M points"的描述是过时文档，与代码不符（已用 memory 记录为待更正的文档偏差）。`densify_6groups` 实验里唯一测过的"加密hull"方案是 `n_samples=30_000`（3倍），对应 `H1/H4/H5` 三组，结果是：**单独加密hull（H1）floater_frac 从基线22.7%暴涨到48%**，n_gaussians只多了17%；加密hull叠加densify-grad-thresh（H4）floater 42%；只有**不动hull、只调densify-grad-thresh+refine-every的H6，floater反而比基线更低（13.6% vs 22.7%）、n_gaussians涨72%**。**结论：加密hull是被验证过的低效/有副作用方案，不该是本轮"让点云更稠密"的第一选择**——这直接否定了"想办法让点云更稠密就该增大hull采样点数"这个直觉，密度化的正确杠杆是densify调度参数，不是初始化点数。

## 候选超参数与假设（含本轮新增两项）

| # | 参数/改动 | 假设 | 调整方向 | 预期 |
|---|---|---|---|---|
| P1 | `densify-grad-thresh`/`refine-every` 往保守调 | 当前值已是4cam验证的"激进密度化"最优解(H6)，3cam弱约束下可能反而加剧漂移，需要验证往回调是否能压低overshoot而不过多损失点数 | 0.0004→0.0008(nerfstudio默认)/0.0012；refine-every 50→100 | 高优先级，直接验证核心矛盾 |
| P2 | `warmup-length`/`stop-split-at` 更早冻结 | 复用已验证的 `G6_densify_200_1200`(warmup200/stop-split-at1200) 配置，更早停止几何生长 | 200/1200 | 中，若P1显示漂移随训练时长单调恶化则此项优先级上升 |
| P3 | `camera-optimizer.mode` | 3cam冗余度更低，标定残差更容易转化为可观测漂移 | off → SO3xR3 | 中，需目视验证防退化解 |
| P4 | `cull-scale-thresh`/`cull-alpha-thresh` 收紧 | 复用已验证的 `G7_cull_strict`(alpha0.2/scale0.3/screen0.10) 起点，3cam overshoot远超4cam工况，可能需要比G7更激进 | 复用G7 + 一组更激进变体 | 中，直接压低overshoot数值 |
| P5 | `max_iters` | 验证漂移是否随训练时长单调恶化 | 1000/1500/3000 | 中，定方向用 |
| P6 | `sh-degree` | 输入是近乎二值剪影，SH>0预期无geometry收益 | 保持sh0，1次sh1确认 | 低 |
| P7（新增） | `binarize_mask` 阈值 | 8kfps下运动模糊更明显，当前threshold=1极低，模糊边缘的灰阶像素会被计入前景，引入轮廓噪声；提高阈值只保留高置信度前景 | 1→20/50 | 中，直接响应fps confound的可操作杠杆 |
| P8（新增，用户要求） | mask预处理去腿（形态学开运算/骨架剪枝） | 腿是细长附肢，当前 `dilate_mask` 只做膨胀不做腐蚀，腿部结构会被hull和训练一起保留，贡献细碎floater/误标注噪声；文献确认"形态学开运算/骨架剪枝去除细长附肢"是标准技术 | 新增 `erode_appendages()`：先开运算（腐蚀核尺寸大于腿宽小于身体/翅宽）再走原 dilate_mask 流程；**必须同步作用于 hull-voting mask 和实际训练用图像**，只改hull初始化而不改训练监督图像的话，photometric loss会在densify阶段把腿重新"长"回来 | 中，需要新代码，目视验证去腿不伤身体/翅膀 |
| P9（新增） | hull `n_samples` 修正 | 生产代码实际只用10k点（非README宣称的1M），且已知加密hull(30k)在4cam上floater代价高——本项不是孤立测试增大n_samples，而是与P1/P4的漂移抑制手段联合测试，单独测大概率重复H1的失败模式 | 30k(复用H1)/100k，仅与漂移抑制手段组合测 | 低-中，需配合P1/P4 |

**P0（原H0，hull投票阈值N选N→多数投票）**：用户已决定跳过，保留记录。

**fps/motion blur 混杂（原H7）**：不是可直接sweep的超参数，是数据特性差异，P7是响应它的唯一可操作杠杆；其余留作解释3cam vs 4cam差异时的重要限定条件，不单独起一轮实验。

## 文献佐证（简要）

- 2024-2025 稀疏视角3DGS文献（AD-GS、DNGaussian、CoR-GS、EAP-GS）普遍认为：视角少→初始点云质量差→朴素densify容易失控，需要更谨慎的densify策略（如AD-GS的"低/高密度化交替"）或额外正则化（深度先验、co-regularization）。与本项目实测（H1加密hull floater暴涨、H6纯densify调度反而更干净）方向一致：**杠杆应该在densify调度而非初始化暴力堆点**。深度先验类方法本项目无深度真值，不纳入本轮范围。
- 形态学开运算（腐蚀+膨胀）/骨架化后剪枝细枝，是去除细长附肢类结构的标准技术，支持P8的实现路线。

## 实验设计

### Dev 帧子集（不变）
`ctrl_119_004` f0730–f0830，`ctrl_119_010` f0373–f0473，各覆盖≥1完整拍打周期，共200帧。

### Round 0（免费）
目视QC坐实现有失败模式，baseline数值复用已有480帧全量结果。

### Round 1：核心超参数单变量消融（围绕"已是H6最优解"的生产基线）
P1(×2方向) + P2(复用G6) + P3 + P4(复用G7+激进变体) + P5(×3) + P6 = **9组** × 200帧 = 1800任务，≈6.4h。

### Round 1.5（新增）：预处理层面消融
P7(×2阈值) + P8(去腿) + P9(×2采样数，仅baseline密度化设置下先摸底) = **5组** × 200帧 = 1000任务，≈3.6h。P8需要新代码（`utils/image.py` 新增函数，通过参数开关控制，不改变其他sweep的默认路径）。

### Round 2：联合网格
从 Round1 + Round1.5 胜出项里选2-4个维度联合搜索，**显式兼顾"稠密"和"不漂移"两个目标**（例如 P1方向 × P4强度 × P8开关），不是单目标优化——H6的教训是单独追密度会牺牲floater，本轮网格必须同时看 `n_gaussians` 和 `extent_overshoot`/`dbscan_floater_frac` 的联合分布。预估14组 × 200帧 = 2800任务，≈8.1h。

### Round 3：全量验证（不变）
胜出配置+备选，两视频全480帧，对比已有baseline，含目视抽查+可选T4诊断。≈5h。

**合计 ≈ 23小时**，48h预算内留约25h余量。

### 评估指标（不变）
`n_gaussians`/`scale_ratio`/`opacity`/`extent_overshoot`（自动产出）+ `dbscan_floater_frac`（T2补跑）+ 目视QC + 可选leave-one-cam-out诊断 + Round3下游T4 diagnostics。

### 早停机制（不变）
轮次级早停，不做训练中途kill。

## 已确认决策
1. 跳过 P0（hull投票阈值代码改动）
2. Dev 子集：每视频100帧连续窗口（覆盖≥1完整拍打周期），共200帧
3. 新增 Round 1.5（预处理消融：mask阈值、去腿、hull采样数修正）
4. Round 1 的densify方向改为"围绕已验证的H6最优解双向探索"，不再假设当前配置是未调优基线

## 待办
- [ ] 第4步：写 Round 1（9组）+ Round 1.5（5组，含P8新函数）的 `gpu/schedule/` config + 结果聚合脚本
- [ ] 第5步：执行 Round 1/1.5，按结果决定 Round 2 网格
- [ ] 第6步：Round 3 全量验证 + 实验记录表 + 推荐配置 + 分析总结（含8k vs 16k fps 限定条件说明）
