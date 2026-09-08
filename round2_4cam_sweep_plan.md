# 4cam Round 2 超参数 Sweep 方案（跨branch洞察迁移）

状态：设计稿，待落地 `gpu/schedule/` config + 聚合脚本（本文档本身先在 test-3cam 分支的工作区写出，
计划随后迁移/cherry-pick 到 main；未生成任何 config，未跑任何训练）。

## 背景：为什么是"跨branch迁移"而不是直接照搬 round1 结论

`test-3cam` 分支跑的 3cam round1/1.5（见 `outputs/round1_summary.md`、`sweep_hyper_params.md`）在
`ctrl_119_004/010`（3相机、8000fps）上测了9+5组超参数。4cam（main，`ctrl_009_*`，**16000fps**）
自己也有更早的一轮历史sweep：`outputs/ctrl_009_002_8groups_100frames` + `_densify_6groups_100frames`
（单视频 `ctrl_009_002`，100帧），确立了当前生产配置 `ratio3_sh0_dense` = `G2b_scale_reg_ratio3`
（max-gauss-ratio 3.0）+ `H6_grad_thresh_low_refine_fast`（densify-grad-thresh 0.0004 + refine-every 50），
并已在 `valid480`（16视频×480帧）全量部署验证。

两轮sweep之间有4组参数用的是**完全相同的数值**，可以直接交叉验证方向是否一致——结果是**不一致**，
且方向相反的两组恰好是round1里表现"更好"的两组。如果只看round1的排行榜设计round2，会把两个
已经被4cam自己的历史数据证伪的方向重新引入。这是本文档存在的原因：不是"把round1搬过来"，而是
"用round1 + 4cam自己的历史结果做一次交叉过滤，只测两边都没覆盖、且机制上不依赖视角数/帧率这个
混杂变量的新方向"。

## 交叉验证表：round1 结论 vs 4cam 历史sweep（相同参数值）

| 参数 | 3cam round1 结果（vs baseline 1.753） | 4cam 历史sweep 结果（相同参数值，vs `G2b_G9` baseline 1.10） | 判定 |
|---|---|---|---|
| P4a cull-strict（alpha0.2/scale0.3/screen0.10） | overshoot 1.563，**变好** | `G7_cull_strict`：overshoot 1.172（变差），floater 0.42 vs 0.227（大幅变差） | **方向相反**。3cam下变好是因为视角少、约束弱，cull收紧在4cam上会误伤本就够用的密度。**已被4cam自己的数据证伪，round2不再测** |
| P9 hull 30k（3倍采样） | overshoot 1.706，略优于baseline | `H1_hull_dense`：floater 暴涨到0.48 | 两边都不看好加密hull（4cam证据更直接更强）。**round2跳过** |
| P8 leg_erosion（round1新增代码） | round1.5原始版(kernel_size=9,无安全网)：overshoot 2.003，比baseline还差 | 未测过（新功能，无4cam对照） | **诊断已完成，不再是待查异常**。`round1_diag_reproj.py`重投影核查发现round1.5那组在f0730 CAM3把整只苍蝇body都腐蚀掉了（不是翅膀被削——是单一kernel_size在透视缩短视角下把变窄的body本身当成"细长附肢"吃掉）。已在`utils/image.py::erode_appendages`加`min_area_ratio`安全网（单张图erosion后前景面积低于阈值就整张退回原mask）。test-3cam round2用修复后的多组kernel_size重测，结果见下方"P8去腿：修复后的结果与4cam建议"——**这是一个可迁移的机制性发现（"单kernel_size在多视角下不安全"和视角数/帧率无关），round2要带进4cam网格，但不能直接搬3cam选出的单一数值** |
| P1 densify-grad-thresh 方向 | 保守(0.0008)变好(1.460)、更激进(0.0002)大幅变差(2.002) | `H2_grad_thresh_low`(0.0004,即round1的"更激进"方向)：overshoot 1.145，比`G2b_G9`(1.10)略差但floater更低(0.188 vs 0.227) | 4cam从未测过比生产值(0.0004)更保守的方向，是真正空白；但4cam视角冗余度高，先验预期收益小于3cam。**低优先级，可作为单点确认性实验，不进主网格** |

## Round2 主网格：三个双边都没测过、且机制不依赖视角数/帧率的新轴

round1为3cam引入的P3/P5/P7三个轴，4cam的历史8groups/densify_6groups sweep从未覆盖，且它们的作用机制
（标定残差吸收、训练时长-漂移关系、mask边缘噪声）不依赖"视角少/帧率低"这个3cam特有的前提，属于
"值得在4cam上独立确认"的方向：

| 轴 | 候选值 | 3cam round1 参考结果 | 4cam上的预期与理由 |
|---|---|---|---|
| P3 camera-optimizer | off（现状）/ `SO3xR3` | 1.496，第5佳 | 4cam冗余度更高，标定残差更容易被多视角本身平均掉，预期收益小于3cam、但不应变差；风险是吸收出退化解（缩放/位姿漂移），需目视核对而非只看数值 |
| P5 max_iters | 1000 / 1500 / 2000（现状） | 1000最优(1.297)，1500转差(1.603)，3000打平baseline(1.721)——非单调，提示该指标在3cam下噪声较大 | 4cam本身几何约束强，1000 iters若能保持`H6`同等质量（n_gaussians/floater不明显下降），是直接把16视频×480帧规模sweep的GPU-hour砍半的效率发现，比"再降一点overshoot"更有实际价值 |
| P7 binarize_mask threshold | 1（现状）/ 20 | 1.675，第9佳，比baseline(1.753)略好 | threshold=1本身像是遗留疏忽（`utils/image.py::binarize_mask`），和帧率/视角数无关，是否改善4cam边缘噪声用同样成本可以直接测，不依赖3cam的"运动模糊"前提 |
| P8 leg_erosion（安全网修复版，多组） | 见下表 | 见下表 | 机制性修复（防止单kernel_size吃掉body），值得测，但**必须多组+人工核查**，不能只搬3cam选出的数值 |

三轴（P3/P5/P7）互相独立测试（不做联合网格——都是新轴，之前没有已知的交互假设，联合网格会在没有先验的情况下
浪费预算）。P8见下方单独说明，成本远低于round1的23小时。

## P8去腿：修复后的结果与4cam建议

test-3cam round2用`min_area_ratio=0.65`安全网重测了4组kernel_size，`outputs/round2_summary.json`
（150帧×2视频=300任务）：

| 组 | kernel_size | 安全网 | overshoot | 相对同条件baseline |
|---|---|---|---|---|
| D1_legerosion_orig | 9 | 无（round1.5原始灾难组） | 1.496 | 远差于D1_baseline(1.275) |
| D1_legerosion_k7 | 7 | 无 | 1.376 | 差 |
| D1_legerosion_k9safe | 9 | 有 | 1.355 | 差 |
| D1_legerosion_k5 | 5 | 无 | 1.269 | 与baseline持平 |
| D1_legerosion_k7safe | 7 | 有 | 1.32 | 差，但明显好于orig |
| D1_cullB（参照，无去腿） | — | — | 1.151 | — |
| **D1_cullB_legerosion_k7safe** | **7** | **有** | **1.141** | **唯一优于"不去腿"的组合** |
| D1_cullB_legerosion_k9safe | 9 | 有 | 1.168 | 略差于纯cullB |

单独看，去腿在3cam上大多数情况下仍然不如不去腿；只有 `kernel_size=7 + min_area_ratio=0.65` 叠加在
cullB之上时才略微超过"不去腿"的cullB基线。**这不代表4cam上kernel_size=7就是最优值**——4cam是不同批次的
相机架设/拍摄角度，body在各视角下的"最窄投影宽度"不同，同一个kernel_size在4cam某个相机视角下也可能
重演round1.5那次"吃掉body"的事故（安全网只兜底不失败，不保证效果最优）。

**4cam round2建议**：把安全网机制本身当作已验证、可直接采纳的通用改动（默认开启`min_area_ratio`，
不再有条件不带它跑去腿），但kernel_size留2~3档（如5/7/9，均带安全网）作为独立分组一起跑，产出后
用`round1_diag_reproj.py`同款重投影可视化对每组抽查若干帧，由你目视核查哪档在4cam的相机角度下
既去掉了腿又没伤到body/翅膀，而不是直接照搬3cam选出的k7_safe这个数值。

## 实验设计

### Dev 数据集
复用 `ctrl_009_002`（和历史 `8groups`/`densify_6groups` 同一视频，便于纵向对比生产配置的历史基线数值），
从已有 `valid480`/full sweep 结果里截取一段 ≥100 帧的连续窗口（覆盖≥1完整拍打周期，参考T3
motion-veto对连续帧数≥150的要求），复用已生成的 `transforms.json`/`init_points.ply`，Phase A可跳过。

### 网格
P3（×1新值）+ P5（×2新值，1500打平3cam无信息量可以只测1000）+ P7（×1新值）+ P8（×3
kernel_size档,均带`min_area_ratio`安全网,各自单独一组不叠加cullB——4cam是否需要cullB要等P4的
证伪结论最终确认，先看纯去腿效果） = **7组** × ~150帧 ≈ 1050任务，按round1 80~112s/帧估算 ≈ 2.5~3h。

P8的3个kernel_size变体需要先跑数据准备（复用`prepare_round2_leg_erosion.py`的模式，改
`VARIANTS`指向4cam的mask/hull重新生成，均固定`appendage_min_area_ratio=0.65`），训练完成后
必须过一遍`round1_diag_reproj.py`抽查再定档，不能只看`extent_overshoot`数值就下结论
（3cam的教训：数值差异不大的组之间，重投影可视化才是能看出"body有没有被吃掉"的地方）。

### P1保守方向（可选，独立单点）
单独测 `densify-grad-thresh=0.0008`，1组 × ~150帧，作为"4cam是否也存在更保守更优"的确认性实验，
不影响主网格结论，可视GPU余量决定是否跑。

### 评估指标
沿用 `n_gaussians`/`scale_ratio`/`opacity`/`extent_overshoot`（`run_task()`免费产出）+
`dbscan_floater_frac`（T2补跑，参考4cam历史sweep里floater对cull-strict/hull-dense的判别力明显
强于extent_overshoot）。

## 落地清单（未执行，等待确认）

1. 从 `test-3cam` 分支把 `gpu/schedule/analysis/{aggregate_round1,aggregate_round2,round1_diag_reproj}.py`、
   `generate_round1_configs.py`/`generate_round2_configs.py`、`prepare_round2_leg_erosion.py` 几个脚本的
   **结构**（不是内容）搬到 main，把 `VIDEOS`/`BASELINE_SWEEPS` 改成指向 `ctrl_009_002` 的dev窗口。
2. 把 `utils/image.py::erode_appendages` 的 `min_area_ratio` 安全网改动本身直接迁移到main（这是通用
   bugfix，不依赖3cam的任何东西，不需要等round2其余部分定稿）。
3. 用改造后的`prepare_round2_leg_erosion.py`同款脚本为4cam的3档kernel_size（均带安全网）生成mask/hull
   数据变体。
4. 生成7组（P3/P5×2/P7/P8×3）+ 可选P1保守方向1组的 `gpu/schedule/configs/round2/*.json`。
5. 网格训练 + 聚合（`aggregate_round2.py`同款）+ 对P8的3组过一遍`round1_diag_reproj.py`抽查定档
   + round1_summary.md同款汇总表。
