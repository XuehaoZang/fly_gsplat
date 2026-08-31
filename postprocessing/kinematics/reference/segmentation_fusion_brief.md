# Segmentation fusion — 自主迭代任务书（motion + kmeans + 连续性，实质性改动）

**这是给一个在隔离 worktree 里自主运行 ~2-3 小时的 Claude Code agent 看的背景+任务文档，不是给人类协作者看的进展记录。** 你（执行者）没有本次规划对话的上下文，下面尽量把背景交代完整。写这份文档时的心态：宁可啰嗦也不要让你重新踩已经踩过的坑。

**这份任务书是 [`body_angle_roll_part1_brief.md`](./body_angle_roll_part1_brief.md) 的后续，不是平行任务。** 那一轮做的是"在红线内加一个新kmeans特征、验证、诚实报告负结果"，本轮的目标不一样：**真正把 motion-based 分割、kmeans 分割、跨帧连续性组合成一套新方法，如果验证通过，允许它成为新的生产默认行为**（这条红线本轮被放开了，见第6节）。开始前请完整读一遍 part1 的 brief 和 [`body_angle_roll_part1_progress.md`](./body_angle_roll_part1_progress.md)（同目录），那里的诊断结论、踩过的坑、被证伪的假设，本轮全部继承，不要重新验证一遍。

---

## 0. 一句话目标

在 `postprocessing/kinematics/simulate_gt/` 这套"造点云→跑T3分割→跑T4运动学→对比精确ground truth"的合成数据框架上，**设计并实现一个真正融合 motion-based 分割 + kmeans 分割 + 跨帧连续性的新分割方法**，用真数字（不是肉眼看图）验证它比现有 `segment_frame_kmeans_v2`（生产算法的忠实复刻）更接近 ground truth（`seg_accuracy` + `t3_roll_deg` mean/max 双指标，混淆矩阵佐证）。如果验证扎实（simulate_gt 上有效 + 真实数据 sanity check 不倒退），**把它实现进真正的生产模块**（`postprocessing/labeling/kmeans/kmeans_split.py` / `postprocessing/labeling/labeling.py` / `postprocessing/labeling/motion/`），成为新的默认行为，而不是留在 simulate_gt 的复刻代码里孤芳自赏。

---

## 1. 已确认的关键事实（继承自 part1，不用重新验证，但建议先复现一次确认没漂移）

### 1.1 误差来源定位（part1 stage 1-2 的结论）

- T4 的角度估计公式本身干净标注下误差很小（`t4only_roll_deg` mean≈2.4°）；真实误差大头来自 T3 分割质量（`t3_roll_deg` mean≈13.9°, max≈69.4°），`seg_accuracy` mean≈0.815 但方差很大（std≈0.244, min≈0.184）。
- 错分主导模式是 **body↔wing 混淆**（~85%），不是 wing_L↔wing_R 左右互换（~15%）。空间上软性集中在翅根附近（"halo"效应，不是硬边界）。
- 除了翅根 halo，还有**一批独立性质的灾难性失败帧**（100帧里约13帧：2,3,5,20,35,36,39,40,75,82,83,87,98,99，`seg_accuracy` 0.18-0.4），这些帧的失败模式是**整簇错配**（KMeans 把大片点分错，不是边界模糊），下游还伴随 `chord.py` 的 RANSAC 平面拟合失败。part1 没来得及深挖这批帧的具体机制，**本轮第一件事之一就是先诊断清楚这批帧**（第4.2节）。

### 1.2 现有 motion 方法的真实能力边界（part1 stage 1 的量化结论，这是本轮融合设计的关键输入）

`segment_frame_motion`（`postprocessing/labeling/motion/label.py` + `density.py` 生产算法的复刻，体素密度+`HALF_WINDOW=36`帧窗口，只能覆盖 `[36, 63]` 这类中段帧，边界帧完全不可用）在**跟 kmeans_v2 对齐同一帧子集**比较时（这一步的方法论见第4.1节，**不要跳过对齐这一步，part1 已经证明不对齐会得出相反结论**）：

- kmeans_v2 on [36,63]: seg_accuracy mean=0.8731
- motion on [36,63]: seg_accuracy mean=0.8095

**motion 单独跑整体上并不比 kmeans_v2 强**，但它的错误结构非常干净、方向单一：

```
motion 混淆矩阵（28帧累加）:
pred    body  wing_L  wing_R
gt
body    9156       0       0      <- 关键：body 召回 100%，从不把 body 点标成 wing
wing_L  1548    2707     253      <- 但会把 34% 的 wing_L 点误标成 body
wing_R  1367     219    2530      <- wing_R 同理，35%误标成body
```

**结论（本轮的设计起点，用户已确认按这个逻辑走）：motion 对"这个点是不是 body"这件事有极高精度（precision=1.0：只要 motion 说是 body，几乎肯定是 body）但召回率一般（会漏判一部分 wing 点，把它们也当成 body）。这意味着 motion 可以当一个"高置信度 body 先验"来用，而不是直接替代 kmeans 做三分类。** 具体融合方式不要在没看数据前就锁死实现细节——第4.3节要求你先做联合误差分析，再决定怎么融合。

### 1.3 生产代码结构（决定你在哪里写融合逻辑）

- `postprocessing/labeling/kmeans/kmeans_split.py`：kmeans 核心原语（`run_kmeans_v2`、`build_seed_mask`、`label_by_rule_a`、`standardize_v2` 等），`FEATURES_V2=[x,y,z,opacity,R]`，seed 规则 `opacity>=0.98 或 R<0.2`。
- `postprocessing/labeling/labeling.py::process_frame`：真正的生产入口，调用上面这些原语 + wing连通域merge + body-PCA左右锚定，产出最终标注。
- `postprocessing/labeling/motion/label.py` + `density.py`：motion 方法的生产实现（体素密度、`HALF_WINDOW=36`）。
- `postprocessing/kinematics/simulate_gt/segment.py`：**这不是独立重新实现，是直接 import `kmeans_split.py`/`motion/label.py` 的原语函数**（`run_kmeans_v2`、`label_by_rule_a` 等直接导入使用），只有"怎么组装这些原语"（对应 `labeling.py::process_frame` 的编排逻辑）是在 `segment.py` 里单独复刻的一份（`segment_frame_kmeans_v2`）。**这个结构对你很重要**：如果你的融合逻辑只改在 `segment.py` 里，`labeling.py::process_frame` 不会自动获得这个能力，两边会分叉。

**强烈建议**：把新的融合编排逻辑写成一个新的共享模块（例如 `postprocessing/labeling/fusion.py`），被 `simulate_gt/segment.py`（快速合成数据迭代用）和 `labeling.py::process_frame`（真实生产用，验证通过后再接入）两边同时 import 复用，而不是在两个文件里各写一份、迟早分叉。原语层（`run_kmeans_v2`/motion 的体素函数）继续复用 `kmeans_split.py`/`motion/label.py` 里已有的，不要重复实现。

---

## 2. 技术约束：点云没有跨帧 persistent ID

**这个约束决定"连续性"只能怎么实现，请先理解清楚再设计，不要假设点可以跨帧追踪。**

每一帧的 3D Gaussian 点云是**独立训练的 splatfacto checkpoint**（`f<NNNN>/splatfacto-checkpoint/<ts>/`），帧与帧之间的点**没有 ID 对应关系**，点数、点的空间分布每帧都不同（只有 world-space 坐标系是跨帧一致的，因为标定固定）。这就是为什么现有 `segment_frame_motion` 用的是**体素密度**（把 world-space 离散成体素格子，累加一个窗口内每个体素被多少帧命中）而不是逐点跨帧匹配。

任何你设计的"连续性条件"都要基于这个约束，大致两类可行方向（不排他，也可以是你想到的别的方式，只要不假设点级别 ID）：
- **体素/空间层面**：类似 motion 方法，在体素粒度上跨帧累积证据。
- **派生量层面**：不平滑点标签本身，而是平滑从标签算出来的下游量（比如 wing hinge 位置、roll 角），类似生产环境里 `correct_body_axis/sequence_axis.py` 对 x_body/yaw 做的"anchor + 连续性校正"（那是**已经上线、只管 x_body/yaw，不管 roll/y_body**，见第6节红线——你可以借鉴它的设计模式，但要做的是 roll/y_body 链路上一个新的机制，不是接入那个已有模块）。

---

## 3. 本轮放开的红线（跟 part1 的差异，务必看清楚）

part1 的红线是"新机制一律走新参数/新函数，生产默认值不能动"。**本轮用户已明确同意放开这一条**：如果你的融合方法在下面第7节的验证标准下证明有效，**允许把它写成 `kmeans_split.py`/`labeling.py`/`motion/` 里的新默认行为**。

但"允许"不等于"应该草率"——建议仍然按这个顺序推进，不要图省事直接改默认值再补验证：
1. 先在 `simulate_gt/`（理想是复用/新建的共享 `fusion.py` 模块）里实现融合方法，作为**新函数**，快速用合成数据迭代、调参、对比。
2. 在 simulate_gt 上验证扎实之后，**移植进生产模块**（`kmeans_split.py`/`labeling.py`/`motion/`），可以直接改默认行为（不需要再加开关），但改动要在报告里清楚写出"改了什么、为什么、验证依据是什么"。
3. 移植进生产后，**必须**跑一次真实数据 sanity check（第7.5节，这次不是可选项——见下方"仍然不变的红线"）。

---

## 4. 执行阶段与检查点（约2-3小时预算，可根据实际进度调整）

每个阶段结束都要留下一段可读的中间小结（数字/脚本路径），不要闷头做到最后才汇报——万一方向跑偏，中途能看出来。写进 `postprocessing/kinematics/reference/segmentation_fusion_progress.md`（新建，同目录下的 progress log，风格参考 `body_angle_roll_part1_progress.md`）。

### 4.1 ~15min 复现确认

重跑 part1 的对齐比较方法论（第1.2节的数字），确认现在仍然成立。同时重跑一遍完整的 kmeans_v2 baseline（`run_step2.py`，100帧），拿到本轮真正的起跑线数字（不要直接信这份文档里抄的数字，自己产出一份）。

### 4.2 ~20min 诊断灾难性失败帧的具体机制

part1 定位到 100 帧里约13帧（2,3,5,20,35,36,39,40,75,82,83,87,98,99）是"整簇错配"而非边界模糊，但没查清楚**为什么** KMeans 在这些帧上会整体错配。建议检查方向（不是必须全做，挑有效的）：
- 这些帧的 body 簇/wing 簇点数、种子点数量是否明显偏离正常帧？
- `labeling.py`/`segment.py` 里的 `_wing_merged`/强制中位数分裂逻辑是否在这些帧触发？
- 这些帧是否落在 motion 方法的有效窗口 `[36, 63]` 内？如果落在窗口外，motion 对这批帧天然无能为力，这是一个需要在报告里明确的**范围限制**，不是失败。
- 是否和拍打相位/翅膀速度有相关性（part1 提到"早期看了一眼没发现干净相关性，但没有严格确认"——如果你想严格确认一下也可以，不强制）。

这一步的产出应该是"这13帧的失败机制是 X"，用来判断融合方案（尤其是连续性那一环）对这批帧有没有针对性的帮助，还是需要专门处理。

### 4.3 ~20min 联合误差分析（决定怎么融合，不要跳过直接凭直觉设计）

对同一批帧（建议用 motion 的有效窗口 `[36,63]`，因为这是唯一能同时评估三种方法的子集），逐点对比 **motion 的判断 vs kmeans_v2 的判断 vs GT**，具体要回答：
- motion 和 kmeans_v2 都判对的点（不需要融合，占多数，跳过）。
- **motion 判对但 kmeans_v2 判错的点**：这些点长什么样（空间位置、特征）？是不是集中在第1.2节混淆矩阵之外的某类点？这是融合能带来净收益的点。
- **kmeans_v2 判对但 motion 判错的点**：对应第1.2节已知的"motion 把慢速翅根点误判成 body"——融合时如果无脑信任 motion 的 body 判断，这批点会被融合方法搞错，需要衡量这批点数量 vs 上一条的净收益点数量,谁更多。
- 这个净收益/净损失的比较结果，直接决定第4.4节该怎么设计融合规则（比如：只在 motion 和 kmeans 都同意时才提高置信度、只用 motion 覆盖 kmeans 判成 wing 但周围体素被 motion 长期占据的点、等等——具体规则从这一步的数据里推导，不要预设)。

### 4.4 ~45-60min 融合方法实现

基于 4.2/4.3 的诊断，实现融合方法。建议单一变量优先（比如先只做"motion 高置信度先验 + kmeans 兜底"这一层，跑通验证后再加连续性那一层），而不是三个机制一次性糊在一起——出问题时无法定位是哪个机制的锅。

允许你实现覆盖第4.2节找到的灾难性帧机制的针对性修复（如果诊断出来是可修的，比如种子点不足触发了某个 fallback），跟"motion+kmeans 主线融合"分开算一个独立变更，分别验证效果。

### 4.5 ~15min 合成数据上验证

用 4.1 同样的脚本/frame集合重新跑一遍，对比 `seg_accuracy`、混淆矩阵、`t3_roll_deg`（mean/max）相对 baseline 的变化，数字并排放。**如果要覆盖 motion 有效窗口外的帧（大多数帧），说清楚融合方法在窗口外退化成什么行为**（纯 kmeans？还是别的兜底?），并且窗口外的帧也要跑一遍全量对比，不能只看窗口内的漂亮数字。

### 4.6 ~20-30min 移植进生产模块 + 真实数据不倒退检查

如果 4.5 的验证过关（净改进，不是噪声），把融合逻辑移植进生产模块（第1.3节建议的共享 `fusion.py` 或直接改 `kmeans_split.py`/`labeling.py`/`motion/`），更新 `simulate_gt/segment.py` 让它继续复用生产实现而不是自己再写一份分叉的复刻。然后：
- `postprocessing/kinematics/tests/test_s2.py` 全部跑通（尤其 `test_clean_scenario_recovers_yaw_pitch_roll`，目前 roll 容差 6.0°，yaw/pitch 3.0°——如果需要放宽容差,必须在报告里明确说明并论证，不能悄悄改测试蒙混过关）。
- 用 `postprocessing/calc_kinematics.py` 在一个真实小数据集上跑一遍（不强求完整640帧，几十帧的子集也行，参考 `project_t4_s6a_real_data_smoke_test` 的经验：真实数据是 `_labeled.csv` 不是 `_marked.csv`，注意 `frame_glob`），看 `diagnostics.py` 报告里 roll 的 "concerning" 状态/跳变数是否**至少没有变差**——这一步这次**不是可选项**（跟 part1 不同，那时候没有真正的改动可验证；这次如果要动生产默认值，必须做这一步）。

---

## 5. 仍然不变的红线（继承自 part1，不要重新碰）

- **不要去动 `chord.py`/`wing_angles.py`（wing pitch/eta 那条线）**——那是另一个已经诊断过、独立的问题，跟本轮 body/roll 分割任务无关。
- **不要重新引入两个已经验证过、证伪的东西**：
  - wing hinge 算法换回 PCA-span-axis（已经被 far+CC 方法替代，真实数据验证过 roll 大跳变 25→13）。
  - "guide_axis 在多帧间切换来源导致符号错乱"这个假设（已诊断证伪，见 `project_robust_body_axis_guide_switch_disconfirmed.md`）。
- **不要碰 `correct_body_axis/sequence_axis.py`**（x_body/yaw 的连续性校正，已经上线且跟本轮 roll/y_body 链路是不同层面的问题）——如果你设计的连续性机制需要类似的模式，写一个新的、专门给 roll/y_body 用的，不要改这个已有模块的行为。
- **任何"变好了"的结论必须有可重跑脚本+固定frame集合支撑的数字**，不能是"肉眼看图感觉更平滑了"。第1.2节的"对齐帧集合再比较"教训贯彻到每一次 A/B 比较。
- `diag/` 目录里已有的 CSV 只是"曾经跑出来的产物"，不代表"已验证的结论"——可疑就重跑。

---

## 6. 用户已确认的设计方向（本轮跟 part1 不同的地方，来自澄清对话）

1. **红线放开**：验证通过后允许直接改 `kmeans_split.py`/`labeling.py` 的生产默认行为，不需要留在 opt-in 新参数后面。
2. **融合逻辑**：motion 当高置信度先验（它对"是不是body"精度高、召回一般），kmeans 补全其余判断（wing_L/wing_R 三分类 + 边界细化）+ 兜底覆盖 motion 无效窗口的帧；连续性机制作为最后一道平滑/纠错层。具体融合规则从第4.3节的联合误差分析里推导，不要预设死板公式。
3. **两种失败模式都要处理，分阶段**：先花小部分时间定位灾难性帧的具体机制（第4.2节，复用已有工具，成本低），再决定后续精力怎么分配到"翅根边界halo"（大部分点、已知方向）和"灾难性整簇错配"（少数帧、影响极端值）之间——不是提前二选一。
4. **会话结构**：单次更长的自主 worktree 会话，自己分阶段推进，中途留 progress log，结束时交一份结构化总结（第8节），不是 /loop 式的定期人工 check-in。

---

## 7. 交付物要求

结束时在 `segmentation_fusion_progress.md` 里留一份结构化总结（跟 part1 的 progress log 一样，边做边写，不要最后补），至少包含：

1. 复现结果：第1节的数字是否还成立，跟本文档写的是否有出入。
2. 灾难性帧诊断：13帧的具体失败机制是什么（第4.2节产出）。
3. 联合误差分析结果：motion vs kmeans 在哪些点上互补、哪些点上冲突，融合规则是怎么从这个分析推导出来的（第4.3节产出）。
4. 融合方法的设计和实现：做了什么、为什么选这个设计、连续性机制具体是什么（体素层面还是派生量层面）。
5. 改动前后对照数字（同一frame集合，`seg_accuracy` + `roll_deg` mean/max，混淆矩阵，窗口内和窗口外分开报告）。
6. 是否移植进了生产模块、`test_s2.py` 是否通过、真实数据 sanity check 的结果。
7. 还剩什么没做完，如果继续应该先做哪一步。

不要只留一堆代码diff和commit——回来的人应该能在10分钟内看懂这2-3小时发生了什么、结论是否站得住脚、生产代码是否真的被改了以及改动是否安全。
