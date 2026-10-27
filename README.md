# fly_gsplat

用 4 相机高速录像，经逐帧 3D 高斯重建，得到果蝇的飞行运动学角度。

**交付物是角度**：身体 yaw / pitch / roll，左右翅 φ（stroke）/ θ（deviation）/ η（pitch）。
点云只是中间产物。所有角度都由本流程从高斯得到（不与旧 MATLAB 外壳流程混用）。当前的研究重点是 η。

## 流程

```
原始数据 ──► 1 dataset ──► 2 train ──► 3 gaussians ──► 4 segment ──► 5 kinematics ──► angles.csv
(sparse.mat,   图像+相机+hull   splat.ply    物理单位高斯表    body/wing 标签     角度
 easyWand.mat)                              (默认不去 floater)
```

| # | 模块 | 粒度 | 输入 | 输出 |
|---|---|---|---|---|
| 1 | `dataset/` | 逐帧 | `Camera{1-4}_sparse.mat`、`calibration_easyWandData.mat` | `f####/images/*.png`、`f####/transforms.json`、`f####/init_points.ply` |
| 2 | `train/` | 逐帧（2 个 GPU 并行） | `f####/` | `splat.ply`、`dataparser_transforms.json` |
| 3 | `gaussians/` | 逐帧 | `splat.ply` + `dataparser_transforms.json`（轮廓 clean 另读第 1 步的图像和 `transforms.json`） | `gaussian_features_f####.csv` 和 `_marked.csv`（多一列 `if_keep`，默认全保留） |
| 4 | `segment/` | **整段** | 一段连续帧的 `_marked.csv` | `gaussian_features_f####_labeled.csv` |
| 5 | `kinematics/` | **整段** | 一段连续帧的 `_labeled.csv`（平板法另读同目录 `splat.ply`） | `kinematics/angles.csv` |

横跨各模块的部分：`eval/viz/`（任意阶段的输出都能直接查看）、`eval/`（合成数据 GT 打分，**尚未实现**）、`utils/`（公共函数：`camera.py` 相机换算、`geometry.py` 几何原语、`pointcloud.py` kNN 连通分量）。


## 输入输出总览

一段视频（例：`ctrl_009` 第 003 段，帧 425–904）从原始数据到角度，盘上依次出现这些文件（`<run>` = 训练配置里的 `name`）：

```
/mnt/x/.../Expr_009_mov_003/Camera{1..4}_003_sparse.mat     原始稀疏录像（NAS，只读）
data/ctrl_009/003/calibration_easyWandData.mat              easyWand 标定（只读）
│
├─ 1 dataset ─────────────────────────────────────────────────────────────────────────────
│  data/ctrl_009/003/f####/images/P{f}CAM{1..4}.png        sparse_to_images   白底整幅灰度图
│  data/ctrl_009/003/f####/transforms.json                 make_transforms    4 台相机（nerfstudio 格式）
│  data/ctrl_009/003/f####/init_points.ply                 make_hull          可视外壳点云（训练初始化）
│
├─ 2 train ───────────────────────────────────────────────────────────────────────────────
│  outputs/ctrl_009/<run>/train_meta.json                  参数组、迭代数、data_dir（viz 靠它找回第 1 步的图像）
│  outputs/ctrl_009/<run>/_progress/*.jsonl, train_summary.json
│  outputs/ctrl_009/<run>/<param_set>/f####/splatfacto/<时间戳>/
│      splat.ply  dataparser_transforms.json  config.yml   nerfstudio 归一化空间里的高斯
│
├─ 3 gaussians（写在同一个 <时间戳>/ 目录里）─────────────────────────────────────────────
│      gaussian_features_f####.csv                         每个高斯一行，标定坐标系，米
│      gaussian_features_f####_marked.csv                  + if_keep（默认全 True）
│
├─ 4 segment（同上目录）──────────────────────────────────────────────────────────────────
│      gaussian_features_f####_labeled.csv                 + part_label ∈ {body, wing_L, wing_R}, confidence
│
└─ 5 kinematics ─────────────────────────────────────────────────────────────────────────
   outputs/ctrl_009/<run>/kinematics/angles.csv            每帧一行：yaw/pitch/roll、φ/θ/η（L/R）原始值和 *_smooth、诊断列、status

查看：eval.viz.export 把 <run> 打包到 outputs/viz/<run>/gsplat/，eval.viz.view 在浏览器里看点云、标签、相机叠图和角度曲线。
```

**每一步读什么、写什么**

| 步 | 读 | 写 | 必填参数 | 逐帧 / 整段 |
|---|---|---|---|---|
| 1a `dataset.sparse_to_images` | `Camera*_sparse.mat`、标定（取图像尺寸） | `f####/images/` | `--sparse --calib --out --frames a-b` | 逐帧 |
| 1b `dataset.make_transforms` | 标定 | `f####/transforms.json` | `--calib --out` | 逐帧（各帧相同） |
| 1c `dataset.make_hull` | 图像 + `transforms.json` | `f####/init_points.ply` | `--data` | 逐帧 |
| 2 `train.run` | 第 1 步的 `f####/` 全部三样 | `splat.ply` 等 | `--config`（配置里写 `frames`） | 逐帧，多 GPU |
| 3 `gaussians.run` | `splat.ply` + `dataparser_transforms.json`；`--silhouette-clean` 时另读第 1 步的 `f####/images/` 和 `transforms.json` | `_features.csv`、`_marked.csv` | `--run`（clean 时加 `--silhouette-clean --data`） | 逐帧 |
| 4 `segment.run` | 一段连续帧的 `_marked.csv`（≥20 帧，正常 ≥73 帧） | `_labeled.csv` | `--run --fps` | **整段** |
| 5 `kinematics.run` | 一段连续帧的 `_labeled.csv`；平板法另读同目录 `splat.ply` + `dataparser_transforms.json` | `kinematics/angles.csv` | `--run --up --fps` | **整段** |

**贯穿各步的约定**

- 帧范围一律写成 `a-b` / `{start, end}`，含 a 不含 b。帧号就是 sparse 录像里的帧号，文件名 `f####` 补零到 4 位。
- 坐标：第 1–4 步都在 easyWand 标定坐标系（米）。真实 4 相机装置里 **−z 朝上**，合成数据 +z 朝上；第 5 步用 `--up` 转到 +z 朝上的实验室坐标系再算角度。
- 帧率：真实 4 相机 16000 fps，合成数据 5000 fps，必须显式传（第 4 步按它换算累积窗口和阈值）。
- 后面的步骤只增加列、不删行：`_features` → `_marked`（+if_keep）→ `_labeled`（+part_label, confidence），行号和 `splat.ply` 里的高斯一一对应。
- 失败的处理：第 4 步某帧失败会删掉该帧旧的 `_labeled.csv`；第 5 步失败帧角度为 NaN、原因写在 `status`。
- 训练不是确定性的：同一帧重训，点云中心差约 0.06 mm，单帧角度可差到几度到几十度（roll、η 最明显）。比较两版第 3–5 步时要用同一次训练的结果。


---

## 1. dataset — 预处理 + 相机 + 外壳

三个脚本，按依赖关系依次运行（`--frames a-b` 含 a 不含 b，与 train 配置一致）：

| 脚本 | 输入 | 输出 |
|---|---|---|
| `sparse_to_images.py` | `--sparse <dir>`（`Camera{c}*_sparse.mat`，按文件名排序对应相机 1..N）、`--calib`（取图像尺寸） | `f####/images/P{f}CAM{c}.png`（白底 255、整幅、PNG 无损） |
| `make_transforms.py` | `--calib calibration_easyWandData.mat` | 每个含 `images/` 的帧目录写 `f####/transforms.json` |
| `make_hull.py` | 图像 + `transforms.json` | `f####/init_points.ply`（可视外壳，作为训练初始化） |

- 相机约定：EasyWand DLT → OpenCV（K, R_w2c, X0）→ OpenGL c2w（Y 朝上，Z 朝后），在 `utils/camera.py` 的 `CameraConfig`。世界坐标系就是 easyWand 标定坐标系（米；真实装置里 −z 朝上，见第 5 步）。
- `transforms.json`：`camera_model="OPENCV"`；`frames[]` 里每个相机一项，字段为 `file_path, fl_x, fl_y, cx, cy, w, h, transform_matrix`；`ply_file_path="init_points.ply"`。各帧相机相同，只有图片名不同。
- 外壳：前景 = 白底上灰度 < 255 的像素（膨胀两次）；四台相机掩码质心射线三角化出中心；在半径 2 mm 的球里撒 10000 个点（固定随机种子），保留四台相机都落在掩码内的点；统计去离群（20 近邻，2σ）。
- NAS 原始数据：`X:` 挂到 `/mnt/x`（`sudo mount -t drvfs X: /mnt/x`），`--sparse` 传 Linux 路径。
- TODO：裁剪（同时修改 cx/cy）、去腿。

## 2. train — 逐帧高斯训练

- 入口 `train/run.py`：`python -m train.run --config train/configs/<run>.json [--gpus 0,1] [--workers-per-gpu 6]`。GPU 长任务用 `nohup setsid` 放到后台。
- 配置（`train/configs/`，每段视频一个）：`name, data_dir, out_root, frames{start,end(不含)}, param_set, max_iters`。参数组只在 `train/param_sets.json` 定义一份，生产用 `ratio3_sh0_dense`。
- 输入：`<data_dir>/f####/{images, transforms.json, init_points.ply}`（第 1 步的输出；缺了就报错，train 不再顺手生成）。
- 每帧：`ns-train splatfacto`（`--vis tensorboard`、白背景、`--eval-mode all`）→ `ns-export gaussian-splat` → 记录高斯数、scale_ratio 中位数、耗时 → 删掉 checkpoint 和 tfevents。
- 输出：`<out_root>/<name>/<param_set>/f####/splatfacto/<时间戳>/{splat.ply, dataparser_transforms.json, config.yml}`；`<out_root>/<name>/_progress/*.jsonl`、`train_summary.json`、`train_meta.json`（参数组和迭代数，换参数必须换 name）。
- 多 GPU：每张卡起若干 worker，worker 通过原子重命名从 `_queue/pending/` 认领帧。已有 `splat.ply` 的帧会跳过，所以可以断点续跑。
- 训练不是确定性的：同一帧训两次，高斯中心的双向平均最近距离约 0.06 mm，高斯数相差可达 ~10%。
- `splat.ply` 的字段：`x,y,z`、`scale_0..2`（log）、`rot_0..3`（四元数 wxyz）、`opacity`（logit）、`f_dc_0..2`（颜色）。坐标在 nerfstudio 归一化空间里，**不能直接用**，要经过第 3 步换算。

## 3. gaussians — 预处理点云

- 入口 `gaussians/run.py`。输入：每帧最新时间戳目录下的 `splat.ply` + `dataparser_transforms.json`。输出：同目录下的 `gaussian_features_f####.csv` 和 `_marked.csv`（多一列 `if_keep`）。**默认不去 floater（if_keep 全为 True）**，因为去 floater 会删掉不少有用的点；加 `--remove-floaters` 才标记。
- `extract_features.py`：把高斯换算回标定坐标系（easyWand，真实数据里 −z 朝上）的物理单位（米），每个高斯一行，不删点：
  `x,y,z, dist_to_centroid, dist_to_principal_axis, R,G,B, color_oob, opacity, scale_phys_0..2, scale_ratio, linearity, planarity, sphericity, orientation_x/y/z, local_density`
  - 换算：nerfstudio 坐标 x_ns = s·(R x + t)。位置用 x = Rᵀ(x_ns/s − t)，方向向量用 v = Rᵀ v_ns，尺度用 exp(log_scale)/s。
  - `orientation_xyz` 是高斯最短轴的方向，已经转到实验室坐标系。**2026-10-20 之前盘上所有 CSV 的这一列都还在 nerfstudio 坐标系里，是错的**；其他列都正确。
- `remove_floaters.py`（可选）：在 k=10 近邻图上做连通分量（只保留邻距不超过 75 分位数的边），所在分量不超过 10 个点的高斯记为 floater，结果写入 `if_keep` 列。
- `silhouette_clean.py`（可选，`--silhouette-clean --data <第 1 步目录>`，推荐）：把每个高斯中心投影到 4 台相机的前景掩码（与 `make_hull` 同一掩码），去掉"白色（RGB 均值 > 0.8）且至少一台相机在掩码外"或"至少两台相机离掩码超过 2 px"的高斯，写入 `if_keep`（与 `--remove-floaters` 取与）。只要求多数相机一致：真实数据里常有一台相机看翅膀很淡（003 左翅约 24% 的高斯只落在 3 台相机的掩码里）。合成 LOW 上删 5.1% 的高斯，其中 99.5% 离 GT 表面 > 100 µm。
- 后续步骤用 `x,y,z`、`if_keep`；第 5 步的平板法另按行号从同目录 `splat.ply` 读完整协方差（第 3 步逐行写、不删点，读时用中心坐标核对）。

## 4. segment — 基于运动的 3D 点分割

- 入口 `segment/run.py`，算法在 `segment/motion.py`。输入：一段连续帧的 `_marked.csv`，帧率 `--fps`（必填）。输出：同目录下的 `_labeled.csv`，在输入表后面新增两列：`part_label ∈ {body, wing_L, wing_R}` 和 `confidence ∈ {high, low}`。
- body：取 ±round(36 × fps / 16000) 帧的窗口，统计每个体素（2.5e-4 m）被多少帧命中。身体几乎不动，命中帧数超过阈值的体素作为 body 候选（16000 fps、73 帧窗口时为 19 帧，其他帧率按窗口帧数等比换算，5000 fps 时为 6 帧），只保留其中最大的连通分量。
- 翅：其余点做 kNN 连通分量，最大的两个分量是两只翅，碎块按最近邻归入。如果两翅贴在一起，就按 x_body × up 方向的投影中位数强行切开，confidence 记为 low。左右由两翅在 x_body × up 上的投影决定。
- `if_keep=0` 的点不参与分割，最后按最近邻传播标签。
- 段首段尾的帧窗口较短。某帧分割失败时会删掉该帧旧的 `_labeled.csv`。

## 5. kinematics — 坐标系与角度

- 入口 `kinematics/run.py`。输入：`<run>/<param_set>/` 下一段连续帧的 `_labeled.csv`（平板法另读同目录的 `splat.ply` + `dataparser_transforms.json`），`--up`（输入点云坐标系里哪个方向朝上）和 `--fps`（真实 4 相机 16000，合成 5000），都必填。输出：`<run>/kinematics/angles.csv`（`--out-name` 可改名）。
- 坐标系：**真实 4 相机的 easyWand 坐标系里 −z 朝上**（三台侧向相机的图像上方都指向 −z；翅根中点相对体轴的偏移也指向 −z），所以真实数据用 `--up=-z`（要带等号，否则 `-z` 会被当成选项）；合成数据的 GT 是 +z 朝上，用 `--up=+z`。计算前先转到 +z 朝上的实验室坐标系（−z 时绕 x 轴转 180°），角度都在这个坐标系里定义；头默认取朝上的一端。TODO：按体点云的质心分布判断头尾。
- 步骤（默认 `--wing-method plate --roll keyframe`；旧方法 `diag`（对角线弦向）/`hinge`（逐帧翅根 roll）保留作对照）：
  1. 体长轴 x_body（`body_axis.py`）：先找锚点帧（两翅展开、体点云细长），再用上一帧的轴作引导逐帧取体点云的头/尾端点质心，最后用锚点校验并整段翻转。
  2. 翅平板（`wing_plate.py`）：每只翅用高斯的完整协方差，取 3σ 连通的最大块，按"中心离平面距离 + 沿法向 2σ 厚度"投票出平板（拒掉漂浮块和贴附的厚团）；法向 = 平板内点 2σ 椭球并集的最小主轴，翅展 = 连通块 3σ 并集的主轴（投影到平板平面）。
  3. roll（`run.py`）：每个拍动周期取一帧关键帧（两翅展夹角最大、两翅平板内点质量 ≥ 0.4），在关键帧上 y_body = span_L − span_R；对关键帧 roll 拟合 GCV 平滑样条（对应 MATLAB `calcBodyRoll.m`）。段首段尾 ±round(36 × fps / 16000) 帧（segment 累积窗口不完整、身体标签不可靠）不当关键帧。拍动周期由两翅展夹角序列的自相关估计。
  4. 身体平滑（`smooth.py`，移植 MATLAB `smoothBodyAngles.m`）：Hampel + 零相位 Butterworth（yaw/pitch 100 Hz，roll 200 Hz），段首段尾帧不参与、取最近值 → `yaw/pitch/roll_smooth`。拍动平面法向是 x_body 绕 y_body 转 −45°。
  5. 左右翻转纠正：平滑体坐标系里左翅展指向右侧且右翅展指向左侧时交换两翅（`wing_swapped`）。
  6. 弦向 = 平板法向 × 翅展；符号 = 前缘朝运动方向（翅展的帧间变化）+ 逐帧连续链（`wing.resolve_chord_signs`）。
  7. φ/θ/η 在**平滑后的体坐标系**里计算（逐帧体坐标系下的值保留为 `*_rawbody`）。φ、η 以整段圆均值为中心表示（不在 ±180° 处折回，切口远离数据，也不会像逐帧 unwrap 那样累积整圈漂移）。
  8. 翅角平滑（`smooth.py`，移植 MATLAB `smoothWingAngles.m`，窗口按 fps 换算）：`wing_masked` 帧（segment 低置信即两翅粘连，或平板内点质量 < 0.4）和段首段尾帧当作缺失，PCHIP 补；另加 φ 自身 Hampel；η 判反段（与迭代参考相差 > 90°）只在计算平滑曲线时翻转 180°，原始 η 和弦向量不改 → `phi/theta/eta_{L,R}_smooth`。
- `angles.csv` 每帧一行：`frame_id, yaw, pitch, roll, phi/theta/eta_{L,R}, sp_normal_*, chord_conf_L/R, span_L/R_*, chord_L/R_*, status`，plate 时另有 `plate_flat/mass/n_{L,R}, roll_keyframe, segment_low_conf, wing_swapped, wing_masked_{L,R}`，以及 `yaw/pitch/roll_smooth, *_rawbody, phi/theta/eta_{L,R}_smooth, eta_flipfix_{L,R}`。角度单位为度；失败帧为 NaN，原因写在 `status`。向量列写回输入点云的坐标系。
- 合成数据（LOW/MID/HIGH 各 1 段，排除段首段尾，扣除常数偏差后的 RMSE，2026-10-27）：身体 yaw 1.7–2.7°、pitch 0.8–1.0°、roll 3.8–8.4°；翅（平滑后）φ 6.5–10.4°、θ 10.5–13.5°、η 9.6–12.5°。改动前（不 clean、对角线法、原始值）分别约为 yaw 11–32°、φ 20–41°、θ 17–22°、η 63–111°。θ 是目前最大的缺口。
- 已知问题：真实数据里两翅粘连（segment 低置信）占 10–19% 的帧，换向处会出现 10–16 帧的屏蔽缺口，缺口旁若恰有离群帧会把平滑曲线拉出过冲（如 009 的 1100–1120 帧）。

---

## 数据（`data/`，gitignore，详见 `data/MANIFEST.md`）

- `data/ctrl_009/<mov>/f####/`：真实数据，第 1 步的输出。
- `outputs/ctrl_009/<run>/`：16 段真实视频（valid480），已经跑完第 2–5 步（2026-10-21 用当时主线重跑第 3–5 步，即旧的对角线法、不 clean）。
- `outputs/proto_wing_plate/{synth_LOW,synth_MID,synth_HIGH,real_003,real_009}/`：当前第 3–5 步（轮廓 clean + 平板法）的结果，`noclean/`、`clean/` 两套，`composite_v*_*.png` 是新旧对照图。
- `data/synth/{LOW,MID,HIGH}/`：合成数据，包括 GT（`gt.h5`）、渲染图、自重建结果、L2 标签，用于 `eval/` 打分。

## 运行

```bash
python -m dataset.sparse_to_images  --sparse /mnt/x/<...>/Expr_009_mov_003 --calib <easyWand.mat> --out data/<run> --frames 425-905
python -m dataset.make_transforms   --calib <easyWand.mat> --out data/<run>
python -m dataset.make_hull         --data data/<run> [--frames 425-905]
python -m train.run                 --config train/configs/<run>.json
python -m gaussians.run             --run <recon_dir> --silhouette-clean --data data/<run>
python -m segment.run               --run <recon_dir> --fps 16000
python -m kinematics.run            --run <recon_dir> --up=-z --fps 16000
python -m eval.viz.export           --run <recon_dir> [<recon_dir> ...] --out outputs/viz
python -m eval.viz.view             --data outputs/viz          # 浏览器打开 http://localhost:8765
# python -m eval.synth              --trial LOW                 # 尚未实现
```

## 环境
Python 3.10、PyTorch 2.1.2、CUDA 11.8、gsplat 1.4.0、nerfstudio 1.1.5（conda 环境 `fly_gsplat`，WSL2）。
NAS：`X:` 挂载在 `/mnt/x`（原始 sparse 数据）。
