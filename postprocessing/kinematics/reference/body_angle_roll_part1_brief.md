# T4 body-angle robustness, Part 1 — 自主迭代任务书（roll / T3 segmentation）

**这是给一个在隔离 worktree 里自主运行 ~30-60 分钟的 Claude Code agent 看的背景+任务文档，不是给人类协作者看的进展记录。** 你（执行者）没有本次规划对话的上下文，下面尽量把背景交代完整。写这份文档时的心态：宁可啰嗦也不要让你重新踩已经踩过的坑。

---

## 0. 一句话目标

在 `postprocessing/kinematics/simulate_gt/` 这套"造点云→跑T3分割→跑T4运动学→对比精确ground truth"的合成数据框架上，**把 body 角度（尤其是 roll）从"点云经过T3标注+T4估计"到"落地成 yaw/pitch/roll"这条链路的误差降下来**，用真数字（不是肉眼看图）衡量，且改动要经得起在真实数据上不倒退的检查。

不是让"看起来更平滑"，是让 `T3预测标注 → T4估计` 得到的 roll 更接近 `scene.py::FrameGroundTruth` 里那个精确构造出来的 `roll_deg`。

---

## 1. 已经确认的关键结论（不用你重新验证，但建议先复现一次确认没漂移）

`postprocessing/kinematics/simulate_gt/diag/step2_flapping_results.csv`（已存在于仓库里，`python -m postprocessing.kinematics.simulate_gt.run_step2` 的产物，100帧 `scene.scenario_step2_flapping`）里已经把问题定位清楚了，直接给出这组数字（`evaluate.py` 的三条件框架，见第4节）：

| 量 | `t4only_*`（喂**精确GT标注**给T4，隔离T4自身估计误差） | `t3_*`（喂**T3预测标注**给T4，真实端到端） |
|---|---|---|
| roll | mean=2.43°, max=6.53° | mean=13.90°, max=69.35° |
| x_body | mean=1.23°, max=2.89° | mean=16.50°, max=114.84° |
| y_body | mean=2.66°, max=6.34° | mean=23.13°, max=128.73° |
| z_body | mean=2.59°, max=6.37° | mean=18.36°, max=106.82° |
| n_sp | mean=2.05°, max=5.09° | mean=15.44°, max=72.27° |

`seg_accuracy`（T3预测标注 vs 精确GT标注的逐点准确率）mean=0.815, std=0.244, min=0.184 — 波动很大，不是稳定的高准确率。

**结论：T4的角度估计公式本身（`body_frame.py`/`geometry.py`）在拿到干净标注时误差很小（几度量级），真正的误差大头来自T3分割（`segment.segment_frame_kmeans_v2`，production算法的忠实复刻）质量不稳定，跟你的直觉一致。** 这不是猜测，是这份CSV里已经摆着的数字——你的第一件事应该是重跑一遍确认这个结论现在还成立（代码可能在这份CSV生成之后有过其它改动），而不是从零开始诊断。

**重要澄清，避免范围混淆**：上表里 x_body/y_body/z_body/n_sp 全部都被分割误差重创，不是只有y_body(roll的直接来源)。但 `evaluate.py`/`run_step2.py` 目前调用的是 `pl.PipelineConfig()` 默认配置——**没有**接入生产环境里已经上线的 `pipeline.run_dataset_with_sequence_correction`（sequence-level x_body连续性校正，见第2.2节）。也就是说上表里 `t3_x_body_deg` 的16.5°误差，一部分可能是"这个单帧PCA x_body本身的已知弱点"，这个弱点在真实数据上已经被sequence correction解决了，不是这次要重新解决的问题。**本轮聚焦 roll / y_body（wing-hinge链路），因为这条链路目前没有类似sequence correction的现有修复**——如果顺手验证x_body/yaw受益于同样的分割改进也可以记录，但不要把"给simulate_gt接入sequence correction"当成本轮的必答题，那是另一个独立的任务。

---

## 2. body角度计算链路背景（`body_frame.py`）

### 2.1 roll是怎么算出来的

```
x_body: body_xyz的PCA主轴（单帧兜底）或sequence-corrected表（真实数据生产环境用）
hinge_L/hinge_R: robust_body_axis.compute_wing_hinge_far_cc(wing_L_xyz, body_cm) / 同理wing_R
y_body: project_onto_plane(hinge_L - hinge_R, x_body)
z_body: cross(x_body, y_body)
roll: 由 y_body 相对"零roll参考系"(由yaw/pitch算出的e_y/e_z)的夹角决定 —— body_frame.py::_calculate_roll
```

关键点：**roll几乎完全由 `hinge_L`/`hinge_R` 决定**（yaw/pitch只提供参考系，不是误差主因）。而 `hinge_L`/`hinge_R` 是从 **T3标注给的 `wing_L`/`wing_R` 点集**里算出来的——如果T3把body点误标成wing、或者把wing点误标成body（尤其是翅根附近，body和wing在空间上最接近、外观特征也最容易混淆的区域），`compute_wing_hinge_far_cc` 拿到的就是被污染的点集，hinge位置偏移，y_body跟着偏移，roll跟着抖。这就是"segmentation不够鲁棒→roll不平滑"这条因果链的具体机制，不是含糊的直觉。

### 2.2 已经修过、不要重新碰的部分

- **wing hinge算法本身**（从PCA-span-axis极值点换成 far+CC 方法）：已经在生产环境验证过，真实640帧数据roll大跳变(>90°) 25→13（48%下降），见 `robust_body_axis.compute_wing_hinge_far_cc`。**这是给定一个wing点集之后怎么找hinge最准的问题，已经解决**。你要处理的是上游——给这个函数的wing点集本身对不对（T3分割质量），不是这个函数内部的算法。
- **x_body的符号/连续性问题**：真实数据上已经用sequence-level guide_axis链条+锚点校验解决了（yaw大跳变 40→2/630），见 `correct_body_axis/sequence_axis.py`。跟本轮无关，除非你在过程中发现simulate_gt也需要接入它才能公平评估——那也请明确写清楚这是不同层面的问题。
- **两个已经被验证排除的假设**（不要重新验证，浪费时间）：
  - guide_axis 在多帧之间切换来源（`prev_frame_guide`/`anchor_guide`/`pca_up_fallback`）导致符号错乱——诊断后**证伪**，真实数据上639/640帧全走`prev_frame_guide`，根本没有切换发生。
  - 见 `.claude/projects/-home-computer0-fly-project-fly-gsplat/memory/project_robust_body_axis_guide_switch_disconfirmed.md`（如果你能读到用户的memory目录的话；读不到就当作背景知识：**不要重新提出"guide_axis切换导致的符号错乱"这个假设**）。

---

## 3. 可用基础设施地图（`simulate_gt/`）

- **`mock.py`**（`postprocessing/kinematics/`下，不在`simulate_gt/`里）：最底层的正向几何构造（身体椭球+两片翅膀，给定yaw/pitch/roll/phi/theta/eta精确构造出点云和对应的精确GT向量），`scene.py`直接复用，不重新实现。
- **`scene.py`**：把`mock.py`的body+wing_L+wing_R拼成一整片未标注点云（`part_label`单独存放，不泄露给下游），并且：
  - `_resample_appearance_features`：把`mock.py`理想化的Gaussian形状/opacity/颜色，替换成从`real_appearance_reference.csv`（真实标注数据自举采样得到的body/wing_L/wing_R各自的`lam1-3, opacity, R,G,B`分布）里抽样的真实分布——这是为了让分割算法看到的特征分布尽量贴近真实数据，而不是`mock.py`自己那个"平面度0.89 vs 真实0.23"级别的失真理想化点云。**这个参考表来自单个真实数据集/单帧的采样**，改进方向如果高度依赖这个表的具体分布特征，要留意过拟合到这一份参考表的风险。
  - `scenario_step1_static`：静态pose，10帧，最简单的冒烟测试。
  - `scenario_step2_flapping`：100帧，body yaw/pitch/roll缓慢漂移+翅膀正弦拍打（`WINGBEAT_HZ=200`，跟`FPS=16000`一起给出`WINGBEAT_PERIOD_FRAMES=80`，100帧≈1.25个周期，跟真实f0000-f0099那份100帧数据规模匹配）——**这是本轮默认应该用的场景**，除非你发现它没有覆盖到需要覆盖的失败模式（见第5.1节的具体提醒，这条对wing pitch部分更相关，对body/roll这轮理论上问题不大，但仍建议留意）。
- **`segment.py`**：T3分割的**忠实in-memory复刻**（不是简化版），两个方法都已实现：
  - `segment_frame_kmeans_v2`：单帧kmeans（`labeling.py::process_frame`真实生产算法的复刻——`kmeans_split.py`的v2 seed-guided KMeans + rule-A语义映射 + wing连通域merge处理 + body-PCA左右锚定）。
  - `segment_frame_motion`：**已经存在**的跨帧方法（`labeling/motion/label.py`真实生产算法复刻，体素密度+`HALF_WINDOW=36`帧窗口，body因为始终占据同一批体素而被多帧命中、wing因为快速扫过每个体素只被短暂命中来区分）。**这就是"用上多帧连续性"这个方向本身在生产环境里已有的实现，不是要你从零设计**——第一件事应该是评估它在simulate_gt上的表现，而不是发明新的跨帧机制。
  - `segment_frame_binary_threshold`：更早、更弱的方法，仅供参考对比，`evaluate.py`已经不用它。
- **`evaluate.py`**：`evaluate_frame`一次跑三条件（T3预测标注→T4 / 精确GT标注→T4 / 精确GT角度本身），这正是第1节表格的来源，也是你区分"误差来自分割还是来自T4估计"的标准工具，不要重新发明。
- **`run_step1.py`/`run_step2.py`/`run_step2_motion.py`**：已有的跑批脚本+`diag/`产物，见下一节的重要提醒。
- **`postprocessing/kinematics/diagnostics.py`**：真实数据用的多帧平滑度诊断工具（`delta_report`/`run_diagnostics`，">5x median跳变数"+"wrap-artifact"这套指标）——**同一套函数可以直接喂合成序列的roll预测值vs GT roll值**，不用重新写平滑度指标，这是复用而不是重新发明的另一个例子。

---

## 4. 一个刚发现的真实陷阱案例（务必读完，这是本文档最重要的一课）

写这份文档时核对了仓库里已有的两份结果：

- `diag/step2_flapping_results.csv`（`segment_frame_kmeans_v2`，100帧全跑）：`seg_accuracy` mean = **0.815**。
- `diag/step2_motion_seg_results.csv`（`segment_frame_motion`，只能跑有效窗口内的帧，`[HALF_WINDOW, n_frames-1-HALF_WINDOW] = [36, 63]`，28帧）：`seg_accuracy` mean = **0.810**。

乍一看两个数字几乎一样，容易得出"多帧方法跟单帧方法差不多"的结论。**但把kmeans方法限制在同样的[36,63]这28帧子集上重新算一遍，均值是0.873——反而明显更高**。也就是说，如果不对齐帧范围直接比较两个mean，会得出跟事实相反的结论（"差不多" vs 真实情况"kmeans在这个子集上明显更好"）。

**教训，请贯彻到你做的每一次比较里**：
1. 任何A/B比较，先确认两边跑的是完全相同的帧集合，再看均值/分布。
2. 已经存在于`diag/`目录里的CSV，只是"曾经跑出来的产物"，不代表"已经验证过的正确结论"——用之前先看清楚它的frame range、生成它的代码版本是否还是当前版本，可疑就重跑，不要直接引用别人（包括你自己上一步）跑出来的数字。
3. 这也是为什么第9节要求你每个阶段都要留下"跑这个数字用的确切脚本/frame集合"，而不是只留一个数字。

---

## 5. 候选方向工具箱

三个方向，按 `segment.py`/`kmeans_split.py` 当前状态给出具体的可调节点，不是空泛的方向名。

### 5.1 调整分割参数（`kmeans_split.py`当前状态）

生产用的v2特征集是 `FEATURES_V2=[x,y,z,opacity,R]`（`standardize_v2`：xyz和[opacity,R]分别标准化，[opacity,R]整体乘`aux_weight`，生产锁定`AUX_WEIGHT_FINAL=1`）；seed规则是`opacity>=0.98 或 R<0.2`当高置信度body种子（`SEED_OPACITY_THRESH`/`SEED_R_THRESH`）；`K=3`。这些都是可调的旋钮，但注意`kmeans_split.py`自己的模块注释已经写了"planarity/scale_ratio/sphericity/linearity在真实阈值下不鲁棒，本次剔除"——如果你想把这几个特征加回来，至少先搞清楚当初为什么被剔除（真实数据上body~0.20 vs wing~0.23的planarity几乎不可分），不要盲目重新引入已经被验证无效的特征。

### 5.2 加其它高斯点参数的cue/聚类方法

`utils/gaussian_features.py`（T1产出）里还有哪些逐点特征没被现在的v2用上，可以去看一眼，评估是否对body/wing边界（尤其翅根附近的模糊点）有额外分离度。这条路径的验证标准应该是：在**翅根附近**这个具体容易错分的区域，新特征/新方法能不能把这批点分对——而不是整体`seg_accuracy`涨了多少（整体准确率可能被大量"容易分"的点稀释，看不出翅根附近这个真正影响roll的局部改进）。

### 5.3 多帧连续性

`segment_frame_motion`已经存在（见第3节），**第一步永远是先评估它在simulate_gt上、跟kmeans_v2在同一帧子集上公平对比的表现**（吸取第4节的教训），再决定：
- 如果它已经明显更好：那问题变成"怎么让它覆盖到边界帧"（`HALF_WINDOW=36`导致每段序列头尾各36帧没法用这个方法，真实数据集常常没有100帧那么长，边界问题可能很致命）——这是一个真实存在、已知的限制，`run_step2_motion.py`自己的docstring已经点出来了。
- 如果两者互有胜负：考虑能不能融合（比如用kmeans_v2兜底，motion方法只在有效窗口内介入/加权），而不是二选一。
- 不要重新设计一套全新的跨帧机制之前，先确认现有这一套的上限在哪——重新发明前先摸清楚已有实现的真实能力边界。

---

## 6. 硬性约束/红线

- **禁止修改默认生产路径的行为**。`kmeans_split.py`/`labeling.py`里任何生产用的函数，新机制一律走新参数（默认值=当前生产行为）或新函数，不能改一个已有函数的默认输出。这是本仓库一贯的约定（`chord.py`的`robust=False`/`use_gaussian_normals=False`默认关闭、`wing_angles.py`的`prev_tip=None`默认关闭，都是同一个模式）——continuity要靠"开关默认关"来保证，不是靠"我保证没改真实行为"这种口头保证。
- **任何"变好了"的结论必须有可重跑脚本+固定frame集合支撑的数字**，不能是"肉眼看图感觉更平滑了"。
- **不要去动`chord.py`/`wing_angles.py`(wing pitch/eta那条线)**——那是另一个已经诊断过、独立的问题（另有专门的诊断记录），跟这轮body/roll任务无关，不要因为顺手就去碰。
- **不要重新引入第2.2节列出的两个已经验证过的东西**（PCA-span-axis wing hinge、guide_axis切换假设）。
- 改动完成后，**跑一遍`postprocessing/kinematics/tests/test_s2.py`确认没有回归**（尤其`test_clean_scenario_recovers_yaw_pitch_roll`，目前roll容差是6.0°，yaw/pitch是3.0°——如果你的改动让这个测试需要放宽容差才能过，这是需要在报告里明确说明并论证的信号，不能悄悄改测试蒙混过关）。

---

## 7. 执行阶段与检查点（约1小时预算，可根据实际进度调整）

每个阶段结束都要留下一段可读的中间小结（数字/脚本路径），不要闷头做到最后才汇报——万一方向跑偏，中途能看出来。

1. **~10min 复现确认**：重跑`run_step2.py`（如需要也重跑`run_step2_motion.py`），确认第1节的数字（`t3_roll_deg` vs `t4only_roll_deg`的差距）现在依然成立，且按第4节的方法把`segment_frame_motion` vs `segment_frame_kmeans_v2`在同一帧子集上重新对比一次，作为本轮真正的起跑线（不要直接信第4节写的0.873这个数字，自己重新跑一遍产出自己的基线）。
2. **~10min 定位翅根区域的具体错分模式**：从上一步的混淆矩阵（`evaluate.py`已经产出`seg_confusion`）里，看错分主要发生在哪（body↔wing_L/R之间，还是wing_L↔wing_R之间），结合空间位置（是否集中在翅根附近）判断，这决定你该往5.1/5.2/5.3哪个方向投入。
3. **~30min 主体实现**：在第6节红线内，针对第2步定位到的具体错分模式做改动（可以是5.1/5.2/5.3之一，也可以组合，但不建议同时开三条战线——单一变量，方便判断哪个改动真正有效）。
4. **~10min 合成数据上验证**：用第1步同样的脚本/frame集合重新跑一遍，对比`seg_accuracy`、混淆矩阵、`t3_roll_deg`（mean/max）的变化，跟第1步的基线数字并排放。
5. **~5-10min 真实数据不倒退检查**：`test_s2.py`跑通；如果时间允许，用`postprocessing/calc_kinematics.py`在一个真实小数据集上跑一遍（不强求完整640帧，几十帧的子集也行），看`diagnostics.py`报告里roll的"concerning"状态/jump数是否至少没有变差——**这一步不是终审**（真实数据没有GT），只是"没有明显变差"的sanity check，跟其它诊断脚本的no-op guarantee一个风格。

如果时间不够，优先级是：**第1-2步（定位）> 第3步的最小可用实现 > 第4-5步（验证）**——宁可在时间不够时只交付"定位清楚+一个小的、验证过的改进"，也不要为了走完全部5步而在最后一步糊弄验证。

---

## 8. 交付物要求

结束时产出一份结构化小结（可以是一段文字，不需要单独建md文件），至少包含：

1. 复现结果：第1步的数字是否还成立，跟本文档写的是否有出入。
2. 错分定位：翅根附近的具体错分模式是什么。
3. 尝试了什么、为什么选这个方向。
4. 改动前后的对照数字（同一frame集合，`seg_accuracy` + `roll_deg` mean/max，最好还有混淆矩阵）。
5. `test_s2.py`是否通过；真实数据sanity check的结果（如果做了）。
6. 还剩什么没做完，如果继续应该先做哪一步。

不要只留一堆代码diff和commit——回来的人应该能在5分钟内看懂这一小时发生了什么、结论是否站得住脚。
