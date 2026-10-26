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
| 3 | `gaussians/` | 逐帧 | `splat.ply` + `dataparser_transforms.json` | `gaussian_features_f####.csv` 和 `_marked.csv`（多一列 `if_keep`，默认全保留） |
| 4 | `segment/` | **整段** | 一段连续帧的 `_marked.csv` | `gaussian_features_f####_labeled.csv` |
| 5 | `kinematics/` | **整段** | 一段连续帧的 `_labeled.csv` | `kinematics/angles.csv` |

横跨各模块的部分：`viz/`（任意模块的输出都能直接查看）、`eval/`（合成数据 GT 打分）、`utils/`（公共函数）。

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
- 后续步骤目前只用 `x,y,z` 和 `if_keep`。

## 4. segment — 基于运动的 3D 点分割

- 入口 `segment/run.py`，算法在 `segment/motion.py`。输入：一段连续帧的 `_marked.csv`，帧率 `--fps`（必填）。输出：同目录下的 `_labeled.csv`，在输入表后面新增两列：`part_label ∈ {body, wing_L, wing_R}` 和 `confidence ∈ {high, low}`。
- body：取 ±round(36 × fps / 16000) 帧的窗口，统计每个体素（2.5e-4 m）被多少帧命中。身体几乎不动，命中帧数超过阈值的体素作为 body 候选（16000 fps、73 帧窗口时为 19 帧，其他帧率按窗口帧数等比换算，5000 fps 时为 6 帧），只保留其中最大的连通分量。
- 翅：其余点做 kNN 连通分量，最大的两个分量是两只翅，碎块按最近邻归入。如果两翅贴在一起，就按 x_body × up 方向的投影中位数强行切开，confidence 记为 low。左右由两翅在 x_body × up 上的投影决定。
- `if_keep=0` 的点不参与分割，最后按最近邻传播标签。
- 段首段尾的帧窗口较短。某帧分割失败时会删掉该帧旧的 `_labeled.csv`。

## 5. kinematics — 坐标系与角度

- 入口 `kinematics/run.py`。输入：`<run>/<param_set>/` 下一段连续帧的 `_labeled.csv`，以及 `--up`（输入点云坐标系里哪个方向朝上，必填）。输出：`<run>/kinematics/angles.csv`。
- 坐标系：**真实 4 相机的 easyWand 坐标系里 −z 朝上**（三台侧向相机的图像上方都指向 −z；翅根中点相对体轴的偏移也指向 −z），所以真实数据用 `--up=-z`（要带等号，否则 `-z` 会被当成选项）；合成数据的 GT 是 +z 朝上，用 `--up=+z`。计算前先转到 +z 朝上的实验室坐标系（−z 时绕 x 轴转 180°），角度都在这个坐标系里定义；头默认取朝上的一端。TODO：按体点云的质心分布判断头尾。
- 步骤：
  1. 体长轴 x_body（`body_axis.py`）：先找锚点帧（两翅展开、体点云细长），再用上一帧的轴作引导逐帧取体点云的头/尾端点质心，最后用锚点校验并整段翻转。
  2. 体坐标系（`body.py`）：y_body 由左右翅根确定（远端点 + 连通分量），拍动平面法向是 x_body 绕 y_body 转 -45°，由此得到 yaw / pitch / roll。
  3. 翅角（`wing.py`）：翅展为翅平面内点的 PCA 主轴，得到 φ、θ；弦向用 MATLAB `find_chords_quad.m` 的对角线法，符号按逐帧连续性确定，得到 η。
- `angles.csv` 每帧一行：`frame_id, yaw, pitch, roll, phi_L, theta_L, eta_L, phi_R, theta_R, eta_R, sp_normal_*, chord_conf_L/R, span_L_*, span_R_*, status`（角度单位为度，均为原始值）。失败帧的角度为 NaN，原因写在 `status`。
- 目前不做平滑、不做 η 解缠，所以 fps 暂时用不到。加入平滑时再增加 `--fps` 参数（真实 4 相机 16000，合成数据 5000）。
- 目前只用了高斯中心。用大小、颜色等其他属性，或者用模型拟合，是以后的研究方向，不改变这一步的输入和输出。

---

## 数据（`data/`，gitignore，详见 `data/MANIFEST.md`）

- `data/ctrl_009/<mov>/f####/`：真实数据，第 1 步的输出。
- `outputs/ctrl_009/<run>/`：16 段真实视频（valid480），已经跑完第 2–4 步，可以直接用于第 5 步和可视化。
- `data/synth/{LOW,MID,HIGH}/`：合成数据，包括 GT（`gt.h5`）、渲染图、自重建结果、L2 标签，用于 `eval/` 打分。

## 运行

```bash
python -m dataset.sparse_to_images  --sparse /mnt/x/<...>/Expr_009_mov_003 --calib <easyWand.mat> --out data/<run> --frames 425-905
python -m dataset.make_transforms   --calib <easyWand.mat> --out data/<run>
python -m dataset.make_hull         --data data/<run> [--frames 425-905]
python -m train.run                 --config train/configs/<run>.json
python -m gaussians.run             --run <recon_dir>
python -m segment.run               --run <recon_dir> --fps 16000
python -m kinematics.run            --run <recon_dir> --up=-z
python -m viz.view                  --run <recon_dir>
python -m eval.synth                --trial LOW
```

## 环境
Python 3.10、PyTorch 2.1.2、CUDA 11.8、gsplat 1.4.0、nerfstudio 1.1.5（conda 环境 `fly_gsplat`，WSL2）。
NAS：`X:` 挂载在 `/mnt/x`（原始 sparse 数据）。
