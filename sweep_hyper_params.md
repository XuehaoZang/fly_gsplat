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

## Round 1 / Round 1.5 结果复盘（本轮新增，`outputs/round1_summary.md`）

按 `extent_overshoot`（越低越好）排序后的关键结论：

1. **P5（max_iters）确认"漂移随训练时长单调恶化"**：1000/1500/2000/3000 iters 的 extent_overshoot 依次为 1.297/1.603/1.753/1.721（3000比2000略降是噪声，趋势仍是1000远优于其余）。**关键**：`n_gaussians` 在1000 iters（562.8）已经接近2000 iters收敛值（580.9），说明点数在前1000 iters基本长满，**多训的1000~2000 iters只贡献漂移、不贡献密度**——这是本轮最强的单变量信号，且训练更快（81s vs 115s/帧）。
2. **P1方向确认假设**：P1a（更保守，thresh 0.0008）overshoot降到1.460但n_gaussians跌到370.5（-36%，比baseline稀疏太多）；P1b（更激进，thresh 0.0002）overshoot反而恶化到2.002且n_gaussians暴涨到945.7——**证实"生产配置H6已是激进densify，3cam弱约束下更激进只会让漂移更差"**，P1b方向排除。
3. **P4（cull）单调收紧有效**：P4a（alpha0.2/scale0.3/screen0.10）overshoot 1.563，P4b（alpha0.3/scale0.15/screen0.08）overshoot 1.465、opacity_median从baseline 0.308冲到0.678——**P4b在更严格的同时overshoot和opacity都比P4a好**，只多付出约12%点数（376.7 vs 431.4），目前看到的cull强度里P4b全面占优P4a。
4. **P2（freeze_early）**overshoot 1.416（次优），点数604.4（比baseline还高）——但`low_opacity_frac`高达0.160（全表第二高，仅次于P1b的0.135... 实际上P2最高），暗示大量低置信度/半成型高斯没被及时cull掉，**很可能和P4的cull收紧强互补**。
5. **P3（camera-optimizer）**overshoot 1.496尚可，但wall_s 152.9s，比baseline的115.9s贵33%——收益中等、成本不低，需要联合验证是否值得。
6. **P6（sh1）确认性验证如预期方向但更差**：overshoot 2.709，全表最差，**证实sh0已是正确选择，不再需要测试**。
7. **P7（mask阈值）/P9（hull采样数）单独作用很弱**：四组overshoot都落在1.67~1.72，与baseline的1.753几乎没有区分度——**符合文档预判"这两项要配合漂移抑制手段联测才有意义，不是独立强杠杆"**，不再单独占用Round 2维度，只作为顺带验证的低成本搭配项。

### P8（去腿）异常：不是"去腿让漂移变差"，是 extent_overshoot 分母的度量口径问题

`p8_leg_erosion` 的 extent_overshoot=2.003，是全表第二差（仅次于P6），初看像是"去腿伤害了重建"，**但深挖后这是一个度量假象，不是真实劣化**：

- `extent_overshoot = splat_extent.max() / hull_extent.max()`（`gpu/schedule/common.py:147`），**分母 `hull_extent` 是该组自己用的 init hull 的extent**。P8为了去腿同时重新生成了images和hull（`remove_appendages=True` 同时作用于两者，`prepare_round1_5.py:61-63`），腿被腐蚀掉后hull的bbox天然缩小。
- 直接从 `_progress/*.jsonl` 里取绝对值核实：P8的`bbox_extent_max`均值 **0.006256**，baseline是 **0.006488** —— **P8的绝对重建extent反而比baseline小3.6%**，不是变大。是分母（hull_extent）缩得比分子还快，把比值"人为"推高到2.0，掩盖了P8实际上没有让重建膨胀这一事实。
- **结论**：P8对absolute drift没有负面证据，且它本来的设计目的是压低floater（细碎腿部误标注），不是压extent_overshoot——round1没跑T2/dbscan_floater_frac，尚未验证它真正的目标指标。**不能用round1的extent_overshoot数字否决P8**，也不能直接采信，需要在Round 2里用绝对指标（`bbox_extent_max`）+ 补跑floater_frac重新评估，本设计已按此处理（见下）。

## Round 2：联合网格（已按上述结果定稿，替换原方案里的示例性描述）

### 维度选择依据

Round 1确认了3种独立的"压制densify漂移"机制，彼此不冗余（时长/schedule/阈值），以及1个在其上独立起作用的cull强度杠杆：

- **D1 = iters1000**（BASE参数不变，只把max_iters从2000降到1000）—— 单变量最优，本轮的"骨架"配置
- **D2 = freeze_early**（warmup-length 200 / stop-split-at 1200，iters仍2000）
- **D3 = grad_conservative**（densify-grad-thresh 0.0008，iters仍2000）
- **C1 = cull_stricter**（P4b: cull-alpha-thresh 0.3 / cull-scale-thresh 0.15 / cull-screen-size 0.08）

P1b（更激进densify）、P6（sh1）已被round1证伪，不再进入Round 2。P7/P9单独效应太弱，只搭配D1做一次低成本验证，不单独占维度。

### 分组（10组新跑 + 4组复用Round1结果做参照，共14组，对齐原方案预估）

复用（免重跑，直接从`outputs/round1*/`取数）：
| 组 | 对应 |
|---|---|
| ref_D1 | round1 `P5_iters1000` |
| ref_D2 | round1 `P2_freeze_early` |
| ref_D3 | round1 `P1a_grad_conservative` |
| ref_C1 | round1 `P4b_cull_stricter` |

新跑（每组两视频×100帧=200任务）：
| # | 组名 | 参数 | 目的 |
|---|---|---|---|
| 1 | `D1_C1` | iters1000 + cull_stricter | 预期主力候选：叠加两个独立生效的杠杆 |
| 2 | `D2_C1` | freeze_early + cull_stricter | 验证cull是否能吃掉D2留下的高`low_opacity_frac`尾巴 |
| 3 | `D3_C1` | grad_conservative + cull_stricter | D3点数偏少(370)，cull再收紧是否让点数进一步不足 |
| 4 | `D1_D3` | iters1000 + grad_conservative | 两种"限制densify"机制叠加：验证是否协同还是冗余/过度稀疏 |
| 5 | `D1_camopt` | iters1000 + camera-optimizer SO3xR3 | P3独立收益能否在D1骨架上复现，且看叠加后wall_s涨幅是否仍值得 |
| 6 | `D1_C1_camopt` | iters1000 + cull_stricter + camopt | 三杠杆叠加的"旗舰候选A" |
| 7 | `D1_legerosion` | iters1000 + 去腿（复用`data/ctrl_119_3cam_p8_leg_erosion/{004,010}`已有数据，无需重新生成） | 用`bbox_extent_max`（非overshoot比值）+ T2 floater_frac重新评估P8真实效果 |
| 8 | `D1_C1_legerosion` | iters1000 + cull_stricter + 去腿 | 旗舰候选B |
| 9 | `iters750` | 仅把max_iters降到750（其余=BASE） | 摸底"点数-漂移"曲线拐点：确认750是否仍能保住~560点同时把overshoot压得比1000更低，还是点数已经开始不足 |
| 10 | `D1_thresh50` | iters1000 + mask threshold=50 | P7唯一还值得看一眼的搭配（低成本，复用p7_thresh50已生成的hull数据） |

**不预先纳入的"全家桶"组合**（D1+C1+camopt+legerosion四杠杆全叠）：留给Round 3——先看1/2/6/8结果是否显示收益可加性，再决定要不要在480帧全量验证前多花一组去确认叠加上限。

### 评估要求
- 除了round1沿用的4个免费指标，第7/8组（去腿）**必须**同时报告`bbox_extent_max`（绝对值对比）和`extent_overshoot`（并注明分母hull不同、仅供组内参考），避免重蹈本轮的度量口径误读。
- 对第1/7/8组（当前最可能的两个方向的交集）补跑T2 `dbscan_floater_frac`——这是P8设计初衷真正要看的指标，round1未跑。
- 联合视图必须同时画`n_gaussians` vs `extent_overshoot`（或第7/8组用`bbox_extent_max`）散点，不能只看排序表，避免H6教训重演（单追密度牺牲floater）。

### 预算估算
10组新跑 × 200任务 = 2000任务。按round1各组wall_s估算合计约51 GPU-hour计算量；按round1观测到的并发度（≈9路并行，207000s原始计算量压缩到6.4h墙钟），预计墙钟约**5.5~6h**，加上4组免费复用和T2补跑（CPU侧，成本可忽略），与原方案"Round2≈8.1h"的预算基本吻合、略有节省。

## Round 2 追加范围（用户2026-09-05追加需求，扩大到24-30h/2GPU）

在上面已定稿的Round 2基础上，用户追加了四类要求，本节记录最终定案(实现见
`gpu/schedule/generate_round2_configs.py` + `prepare_round2_leg_erosion.py`)：

1. **去腿erosion参数需要修正后重测**：round1.5的`p8_leg_erosion`(kernel_size=9,
   无安全网)被用户目视复查`outputs/round1/diag/reproj_f0730_p8_leg_erosion.png`发现
   CAM3几乎被腐蚀掉整只苍蝇。定量复核(对150帧×2视频×3相机=900个相机帧全扫描
   pre/post-erosion前景面积比)证实这不是个例：kernel_size=9在27%(241/900)的相机帧
   上erosion后面积比<0.7，最差到0.42(losing 58%前景)。`utils/image.py::erode_appendages`
   新增`min_area_ratio`安全网(单张图erosion后面积低于阈值就整张图退回不erosion的原mask)，
   扫描验证`min_area_ratio=0.5`太松(仍有个别帧卡在0.52左右不触发)，改用**0.65**后，
   900个相机帧里erosion后面积比的最小值被安全网正确兜到0.65，无一例外。Round 2新增
   4个erosion变体：`p8b_k5`/`p8b_k7`(更温和的核，不加安全网)、`p8b_k9_safe`/
   `p8b_k7_safe`(核9/7+0.65安全网)，加上round1.5原始的`p8_leg_erosion`(k9无安全网)
   做直接前后对比，共5个erosion数据变体。
2. **放开`max-gauss-ratio`**：新增5个值(2.0/2.5/4.0/6.0/10.0)叠加在D1(iters1000)骨架上，
   双向探测(比当前生产值3.0更严/更松)对`n_gaussians`和`extent_overshoot`的影响，此前
   round1从未变过这个参数。
3. **"更稠密"专门设计的新组合**：在D1骨架上新增`densify-grad-thresh=0.0003`(比生产值
   0.0004小幅更激进，但不到round1证伪的P1b那个0.0002)和`refine-every=25`(比生产值50
   更频繁的densify检查点)，赌"P1b失败是因为在2000 iters上激进densify让漂移滚雪球，
   iters1000的早停骨架本身已经拆掉了这个滚雪球机制，可以承受更温和的密度push"，并各自
   叠加cull_stricter做"稠密+不漂移"双目标对照组。
4. **每个pipeline都跑一遍完整kinematics(T1-T4)+两个可视化产出**：
   - 复用已有的`postprocessing.batch_calc_kinematics`链路(T1 gaussian_features -> T2
     mark_floaters -> T3 motion累加labeling -> T4 kinematics pipeline)，但必须显式传
     `half_window=18`(不能用默认的36——那是按16000fps标定的，3相机数据集是8000fps，
     用默认值会让累加窗口覆盖的物理时长变成2倍，见`density.py`docstring)。
   - **T3 motion累加窗口需要≥150帧连续数据才能稳定跑通**(半窗18帧，边界帧退化验证
     过)，round1的100帧dev窗口不够，Round 2起dev窗口拓宽到150帧/视频
     (004: 730-880, 010: 373-523)，所有组都在150帧上重新训练，不复用round1的100帧
     结果(即使参数相同的D1本身也要在150帧上重跑一份，为了配kinematics)。
   - `batch_calc_kinematics.run_one`对去腿/mask阈值/hull采样这类"数据变体"组会用错
     `raw_data_dir`(它固定读config顶层的base_name，不知道某个param_set单独覆盖了
     base_name)——只影响重投影QC图，不影响kinematics csv本身，但round2的kinematics
     runner(`gpu/schedule/analysis/run_round2_kinematics.py`)里已按每个param_set自己
     实际用的base_name显式传入，修掉这个不一致。
   - 角度-时间图复用现成的`plot_body_angles`+`ck.plot_wing_angles`，不用重写。
   - 新增标注点云视频(`gpu/schedule/analysis/render_labeled_video.py`)：仿照
     `postprocessing/kinematics/simulate_gt/animate.py`的固定bbox+`cv2.VideoWriter`
     思路，但读真实T3产出(`_labeled.csv`)而不是仿真数据，单一固定视角
     (`elev=45, azim=-60`，侧45度俯视)，颜色沿用`postprocessing.viz._colors.PART_COLORS`。

### Round 2 最终分组(实现于`generate_round2_configs.py`)
- `round2/ctrl_119_{mov}_d1`(max_iters=1000): 25个param_set，D1骨架 × {cull/ratio×5/
  densify阈值×2/refine频率/camera-optimizer/5个erosion变体/2个复用round1.5的hull100k+
  thresh50}，含4组cull_stricter叠加对照。
- `round2/ctrl_119_{mov}_iters750`(max_iters=750): 1个param_set，探测iters<1000时
  点数是否已经开始不足(elbow探底)。
- `round2/ctrl_119_{mov}_d2`(max_iters=2000): 2个param_set，D2(freeze_early)独立机制
  + cull_stricter对照。
- 共28个param_set × 150帧 × 2视频 = 8400个训练任务，估算约200计算小时，按round1观测
  的12-worker并发折算墙钟约17-22h GPU训练，加上训练完成后的CPU侧kinematics+视频渲染
  (无需GPU，跟训练阶段的12个worker共存但用剩余CPU核，预计几小时)，总墙钟落在用户要求
  的24-30h区间。

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
- [x] 第4步：写 Round 1（9组）+ Round 1.5（5组，含P8新函数）的 `gpu/schedule/` config + 结果聚合脚本
- [x] 第5步：执行 Round 1/1.5，按结果决定 Round 2 网格（见上方"Round 1/1.5结果复盘"与"Round 2：联合网格"两节，已定稿10组新跑+4组复用）
- [ ] 第5.5步：写 Round 2 的 `gpu/schedule/generate_round2_configs.py`（复用`generate_round1_configs.py`/`prepare_round1_5.py`模式，第7/8组直接指向已存在的`data/ctrl_119_3cam_p8_leg_erosion/`，不重新生成数据）+ 执行 + 聚合脚本（含`bbox_extent_max`绝对值对比 + 第1/7/8组T2 floater_frac补跑）
- [ ] 第6步：Round 3 全量验证 + 实验记录表 + 推荐配置 + 分析总结（含8k vs 16k fps 限定条件说明）
