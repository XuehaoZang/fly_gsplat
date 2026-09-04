# seg2d 规格文档 — body/wing 2D 分割（v4：motion-based 主算法 + intensity 兜底）

## 背景与目标

将原有 MATLAB `seg_class`（果蝇 body/wing 运动分割算法）的核心逻辑
移植为 Python，用于给某一帧的 4 相机图像生成 body/wing 像素级伪标签，
进而重投影到 3D 高斯点云上。

**v1~v3 迭代记录（保留作背景）**：最初做了一版纯 intensity（灰度阈值）
的简化版，单帧独立、无时序信息。实测发现效果不够——body 和 wing 根部
物理相连、灰度是渐变过渡，纯靠一刀切的全局阈值，无论阈值怎么调
（双峰valley法 / Otsu），都会在某些相机/某些帧上出现 body 吞并 wing
根部的问题，且这个问题的严重程度和"视角相关的连接粗细"有关，单靠调
形态学参数（开运算半径）治标不治本。

**v4 修订**：加回原 MATLAB 里更鲁棒的核心方法——**motion-based**（利用
身体刚性平移、翅膀独立扑动这一物理先验，用 ±delta 帧的时序一致性来
分离 body/wing），intensity 方法降级为**兜底/补偿**，只在 motion-based
估计失败或退化时使用（这也是当初 MATLAB 代码里 `hybrid_seg` 引入
intensity 的原始动机：body/wing 有时会一起快速运动，纯 motion 的对齐
假设会失效，这时候需要 intensity 兜底）。

本文档只描述算法逻辑（伪代码级别），不含任何 MATLAB 原始代码。

**本轮仍明确排除**：wing 的 left/right 语义区分、wing 的 PCA 形状过滤
（长宽比/边界自交/曲率平滑度）。这两项留到 motion+intensity 混合版本
验证通过之后再评估是否需要。

---

## 1. 坐标约定（务必与 `utils/calib.py` 对齐）

| 概念 | 约定 |
|---|---|
| 原始稀疏数据 `(row, col, val)` | **1-based**；`row` = 图像纵轴(y)，`col` = 图像横轴(x)，`val` = 灰度强度 0~255 |
| 重建出的密集灰度图 `gray[H, W]` | 标准 numpy 0-based 数组，`gray[row-1, col-1] = val` |
| `utils/calib.py::proj()` 返回的 `(u, v)` | **0-based** OpenCV 像素坐标，`u` 对应 col 方向，`v` 对应 row 方向 |
| 换算关系 | `u = col - 1`，`v = row - 1`；反过来查 label map 时直接在 0-based 的 `label_map[H,W]` 上用 `label_map[round(v), round(u)]` |
| 图像尺寸 | `H = 800, W = 1280`（4相机固定） |

---

## 2. 输入数据

沿用现有 `generate_dataset.py` 里读取 `Camera{cam}_sparse.mat` 的方式
（h5py 读取 `-v7.3` 格式 `.mat`，字段路径 `/frames/indIm`，每帧一个
`ref`，指向的数组是某一帧的 `[row, col, val]` 列表，1-based）。建议抽成
公共函数放进 `utils/seg2d.py`（或 `utils/dataset.py`），供多处复用：

```
load_sparse_frame(sparse_path, frame_idx, frame_size=(800,1280)) -> np.ndarray
    # 返回 (H, W) uint8 密集灰度图，背景像素=0（intensity兜底用）
    #   1. h5py 打开 sparse_path，取 refs = f['/frames/indIm'][0]
    #   2. indIm = f[refs[frame_idx]][:]，若 shape[0]==3 则转置
    #   3. rows = indIm[:,0].astype(int)-1, cols = indIm[:,1].astype(int)-1, vals = indIm[:,2]
    #   4. 边界检查后写入 im[rows, cols] = vals

load_sparse_coords(sparse_path, frame_idx) -> np.ndarray  # (N,2) int, 0-based [row,col]
    # 只取坐标不取灰度值（motion对齐用，不需要为每帧都分配800x1280密集数组，
    # 窗口有上百帧时这样效率更好）
```

**v4 新增数据需求**：motion-based 算法需要目标帧 `frame_idx` 前后
`±(delta+fit_scope)` 范围的稀疏坐标数据（默认 `delta=36, fit_scope=100`，
也就是最多约 ±136 帧）。这比 v1~v3 的单帧方案 I/O 量大得多，实现时注意：

- 用 `load_sparse_coords` 而不是 `load_sparse_frame`（不需要为每帧生成
  完整 800×1280 密集图，只需要坐标列表，省内存省时间）
- 同一个相机、相邻目标帧之间的窗口有大量重叠，如果要处理连续多帧，
  考虑做增量式滑窗缓存，避免重复读取同一帧数据（这次先不用做这个优化，
  但代码结构上不要写死成"每次从头读全部"，方便以后优化）

---

## 3. 算法：body 分割

### 3.1 Motion-based 主算法

**原理**（沿用原 MATLAB `BodyCM_estimate` + `calc_dr` + `trans_frame` 的
核心思路）：body 是刚性躯干，在连续帧之间的运动近似于一个可以用低阶
多项式拟合的平滑轨迹；wing 是独立扑动的薄片，不跟随身体的整体平移。
因此：把邻近若干帧的前景像素，按照"身体的平滑运动轨迹"对齐平移到目标帧
的参考系下，再统计每个像素坐标在对齐后的窗口里出现的次数——**body 像素
应该在几乎所有对齐后的帧里都重复出现（次数高），wing 像素因为独立扑动，
对齐之后大概率对不齐，重复次数低**。

```
segment_body_motion(sparse_path, frame_idx, delta=36, fit_scope=100,
                     cm_poly_degree=2, body_th_ratio=0.5)
    -> (body_mask_motion: (H,W) bool, ok: bool, info: dict)

# --- Step A: 粗略 CM 轨迹估计（不对齐，纯粹靠"完全不动的像素"找近似body位置）---
for fr_i in range(frame_idx - fit_scope, frame_idx + fit_scope + 1):
    coords_list = [load_sparse_coords(sparse_path, f) for f in range(fr_i-delta, fr_i+delta+1)]
    all_coords = concat(coords_list)                      # (M, 2)
    uniq, counts = np.unique(all_coords, axis=0, return_counts=True)
    body_candidate = uniq[counts == 2*delta+1]             # 在整个窗口所有帧里都在同一像素出现
    CM_raw[fr_i] = mean(body_candidate) if body_candidate非空 else NaN

# --- Step B: 多项式平滑 + 计算每个邻近帧需要平移多少才能对齐到 frame_idx ---
valid = CM_raw 中非NaN的帧
if len(valid) < fit_scope // 2:                            # 有效帧太少，粗估质量不可靠
    return None, ok=False, info={'reason': 'too_few_valid_cm_frames', 'n_valid': len(valid)}

px = np.polyfit(valid_frames, CM_raw_x[valid], cm_poly_degree)
py = np.polyfit(valid_frames, CM_raw_y[valid], cm_poly_degree)
CM_smooth(fr) = (polyval(px,fr), polyval(py,fr))            # 对任意fr可评估的平滑轨迹

for fr_loop in range(frame_idx-delta, frame_idx+delta+1):
    dr[fr_loop] = round(CM_smooth(fr_loop) - CM_smooth(frame_idx))   # 对齐所需的整数像素平移量

# --- Step C: 对齐后重复像素投票，得到 body/wing 候选 ---
aligned_coords = []
for fr_loop in range(frame_idx-delta, frame_idx+delta+1):
    coords = load_sparse_coords(sparse_path, fr_loop)       # 该帧完整轮廓坐标（不是Step A的窗口计数）
    aligned_coords.append(coords - dr[fr_loop])
aligned_all = concat(aligned_coords)
uniq, counts = np.unique(aligned_all, axis=0, return_counts=True)

body_th = body_th_ratio * (2*delta + 1)                     # ⚠️ 原MATLAB默认 bodyTH=fit_scope=100，
                                                              # 但对齐窗口最多只有2*delta+1=73帧，
                                                              # 100这个默认值本身就不可能达到（可能是
                                                              # 原代码里delta实际配置得比默认36大，
                                                              # 或者bodyTH需要手动调）。这里改成按窗口
                                                              # 大小的比例给，不照抄绝对值100。
body_mask_motion = build_mask_from_coords(uniq[counts > body_th], frame_size)
wing_candidate_mask = build_mask_from_coords(uniq[(counts > 0) & (counts <= body_th)], frame_size)
                                                              # 留着，虽然v4还是用3.3之后统一的wing流程，
                                                              # 但这个中间量对调参/debug有用，建议也存下来

# --- 实现要点：Step A/B/C（尤其Step C的uniq/counts计算）是整个函数里最贵的部分，
#     而 body_th_ratio 需要经验调参、大概率要试好几个值对比效果。实现时把
#     "算counts"和"按body_th_ratio判定"拆成两个函数/两步，counts可以缓存/复用，
#     不要把body_th_ratio写死在算counts的函数内部，否则每次sweep参数都要重新
#     跑一遍Step A/B/C（很贵）。

# --- 退化检测 ---
body_px = body_mask_motion.sum()
if body_px < 20 or body_px > 0.9 * 该帧前景总像素数:
    return body_mask_motion, ok=False, info={'reason': 'degenerate_body_size', 'body_px': body_px}

return body_mask_motion, ok=True, info={'n_valid_cm_frames': len(valid), 'body_px': body_px}
```

### 3.2 Intensity 兜底（3.1 失败/退化时触发）

跟之前 v2/v3 版本相同的方法，改名为 `segment_body_intensity`，作为
fallback，不再是主路径：

```
segment_body_intensity(gray: (H,W) uint8) -> body_mask: (H,W) bool

1. silhouette = gray > 0
   if silhouette 像素数 < 50: 返回全 False

2. values = gray[silhouette]
   TH = otsu_threshold(values)             # skimage.filters.threshold_otsu，只用前景像素算

3. body_bin = (gray <= TH) & silhouette     # ⚠️ 极性需要用真实数据核实：
                                              # "更暗=body"这个假设是否成立，之前实测大体成立，
                                              # 但要留意如果某相机反常，可能是这个假设不适用
4. return body_bin
```

### 3.3 汇总 + 形态学清理（两条路径共用）

```
segment_body(sparse_path, gray, frame_idx, delta=36, fit_scope=100,
             cm_poly_degree=2, body_th_ratio=0.5, open_radius=5)
    -> (body_mask: (H,W) bool, source: str, info: dict)

body_mask_motion, ok, info = segment_body_motion(sparse_path, frame_idx,
                                                   delta, fit_scope, cm_poly_degree, body_th_ratio)
if ok:
    body_bin, source = body_mask_motion, "motion"
else:
    body_bin = segment_body_intensity(gray)
    source = "intensity_fallback"
    记录 warning，附上 info 里的退化原因（比如 too_few_valid_cm_frames /
    degenerate_body_size），方便统计"这一批数据里motion法的实际成功率"

# 形态学清理（沿用v3结论：body-wing根部物理相连、灰度/运动信号都可能连续过渡，
# 开运算是断开细长连接的关键步骤，不管body_bin来自哪条路径都要过这一步）
body_bin = remove_small_objects(body_bin, min_size=100)
body_bin = binary_opening(body_bin, disk(radius=open_radius))   # open_radius 默认5
body_bin = keep_largest_component(body_bin)
body_bin = binary_fill_holes(body_bin)

return body_bin, source, info
```

---

## 4. 算法：wing 分割

**目标**：在去掉 body 之后的前景区域里，找到面积最大的若干个连通域
作为 wing（本轮不区分 wing1/wing2 是左翅还是右翅，直接合并成一个
"wing" 标签）。这一步逻辑不变（不管 body_mask 是 motion 还是 intensity
兜底得到的，wing 分割方式一样）：

```
segment_wing(gray: (H,W) uint8, body_mask: (H,W) bool, leg_th=100) -> wing_mask: (H,W) bool

1. silhouette = gray > 0
2. body_dilated = binary_dilation(body_mask, disk(radius=7))
3. diff = silhouette & ~body_dilated
4. diff = binary_closing(diff, disk(radius=7))

5. labels, num = connected_components(diff)
   if num == 0: return 全False

6. areas = [各连通域像素数]
   取面积最大的最多2个连通域，且 area >= leg_th（默认100，过滤腿/触角碎片）
   若无连通域满足 area >= leg_th：保留面积最大的那一个，记录 warning

7. wing_mask = 选中连通域像素并集
8. return wing_mask
```

---

## 5. 汇总：生成 label map

```
segment_body_wing(sparse_path, cam, frame_idx, delta=36, fit_scope=100,
                   cm_poly_degree=2, body_th_ratio=0.5, open_radius=5, leg_th=100)
    -> label_map: (H,W) uint8, meta: dict

gray = load_sparse_frame(sparse_path, frame_idx)
body_mask, source, info = segment_body(sparse_path, gray, frame_idx, delta,
                                         fit_scope, cm_poly_degree, body_th_ratio, open_radius)
wing_mask = segment_wing(gray, body_mask, leg_th)

label_map = zeros((H,W), dtype=uint8)
label_map[gray > 0] = 3        # 前景未分类
label_map[body_mask] = 1       # body
label_map[wing_mask] = 2       # wing

meta = {'body_source': source, 'body_info': info}   # 每帧每相机都要记录body是motion还是fallback来的，
                                                       # 这是评估motion法实际鲁棒性的关键统计量

return label_map, meta

# label 约定：0=背景，1=body，2=wing（不分左右），3=前景未分类
```

---

## 6. 输出与冒烟测试建议

对单帧 4 相机跑完 `segment_body_wing` 后：

- 保存 4 张 overlay 图（灰度图为底，body绿/wing红/label=3黄，半透明叠加）
- 打印每相机统计：`{cam: {body_px, wing_px, unclassified_px, bg_px,
  body_source（"motion" 或 "intensity_fallback"）, fallback_reason（如果是fallback）}}`
- **新增重点**：如果这一批测试帧里 `intensity_fallback` 占比很高（比如
  超过一半），说明 motion-based 参数（delta/fit_scope/body_th_ratio）
  需要调，而不是正常现象——如实报告这个比例，不要不提。
- 人工确认：body 轮廓是否对应躯干位置、wing 是否覆盖翅膀区域、尤其
  关注之前 v1~v3 版本里出问题的场景（cam4曝光异常、cam1翅膀根部粘连）
  这次 motion-based 是否表现更好

---

## 7. 参数汇总表

| 参数 | 默认值 | 来源/含义 |
|---|---|---|
| `delta` | 36 | 对齐投票窗口半径（帧数），沿用原MATLAB默认值 |
| `fit_scope` | 100 | CM轨迹多项式拟合用的帧范围半径，沿用原MATLAB默认值 |
| `cm_poly_degree` | 2 | CM轨迹多项式拟合阶数，沿用原MATLAB默认值 |
| `body_th_ratio` | **0.85（已定案）** | **v4新定义**，替代原MATLAB绝对值`bodyTH`。实测：frame90-100共11帧×4相机验证，count分布在≈58~67处有浅谷、count=73处有孤立尖峰，0.85（阈值≈62）落在浅谷中点，比默认0.5（把长尾大量误判为body）稳健得多，也比0.95更安全（更靠近73尖峰，容易因帧间噪声误切真实body边缘）|
| 粗估CM最少有效帧数 | `fit_scope // 2` | 低于此认为motion粗估不可靠，触发intensity兜底 |
| body退化判定 | `body_px < 20` 或 `> 0.9*前景总像素` | 触发intensity兜底 |
| intensity兜底 `TH` 估计方法 | Otsu（只用前景像素） | v2结论，仍适用于兜底路径 |
| body 最小连通域面积 | 100 | `remove_small_objects` |
| body 形态学开运算半径 `open_radius` | 5 px | v3结论：断开body-wing根部细长连接的关键参数，不管body来自motion还是intensity都要过这一步 |
| body 开运算后 | 只保留最大连通域 | body生理上应为一整块 |
| body 膨胀半径（提取wing候选区用） | 7 px | 对应原 `strel('disk',7)` |
| wing 闭运算半径 | 7 px | 弥合缝隙 |
| `leg_th`（wing 最小连通域面积） | 100 | 过滤腿/触角/噪声碎片 |
| wing 最多保留连通域数 | 2 | 合并为同一个 wing 标签 |
| 前景最小像素数（跳过空帧判定） | 50 | 避免空/坏帧报错 |

---

## 8. 明确排除在本轮之外

- wing 的 left/right 语义区分
- wing 的 PCA 形状过滤（长宽比/边界自交检测/边界曲率平滑度），如果
  motion-based 主算法效果已经足够好，这项可能不再需要，留到评估后再定
- 各参数（`delta/fit_scope/body_th_ratio/open_radius`等）都是照搬或
  合理外推自旧 MATLAB 默认值，需要用真实数据跑一遍看效果，必要时调参

---

## 9. Task2：单帧重投影 + 单相机标签查表

**目标**：把 splat.ply 的点投影到4个相机图像上，查询每个点在每个相机
上落在 seg2d 的 body/wing/背景/未分类 里的哪一类，作为后续多相机投票
融合（Task3）的输入。这一步先不做投票融合，只做"投影+单相机查表"，
并且做一次可视化自检。

```
1. 加载 splat.ply（load_ply_with_attrs 或直接读xyz），得到点集 pts_rescale
   （dataparser 归一化坐标空间）

2. 加载 dataparser_transforms.json，取出 R_ns, t_ns, scale
   pts_phys = unrescale(pts_rescale, R_ns, t_ns, scale)   # 物理坐标

3. 构造4个相机的 CameraConfig：
   ⚠️ 直接用 CameraConfig.easywand_dlt(ew_data, i) 从 calibration_easyWandData.mat
   构造，不要用 CameraConfig.from_opengl(transforms.json的frame) ——
   如果当初生成 transforms.json 时开了 if_crop（裁剪），from_opengl 拿到的
   内参是裁剪后坐标系的，会和 seg2d 的 label_map（未裁剪的原始800x1280
   像素空间）对不上。用 easywand_dlt 直接从标定文件构造，保证和
   label_map 同一套坐标系。实现前先确认这批数据当初是否用了 if_crop。

4. 对每个相机 i、每个点 X（物理坐标）：
   u, v, depth = proj(K_i, R_w2c_i, X0_i, X)      # utils/calib.py::proj
   若 depth <= 0：该点在这个相机背后，label = -1（"不在此相机视野"）
   否则：
       row, col = round(v), round(u)               # label_map 是0-based数组，和gray一致
       若 0<=row<H 且 0<=col<W：
           label = label_map_cam_i[row, col]
       否则：
           label = -1

5. 输出：(N_points, 4) 的 label 数组 + (N_points, 4) 的 depth 数组

6. 冒烟测试：
   a. 打印每相机 label 分布直方图（-1/0/1/2/3各多少个点，占比）
   b. 挑1个相机，把该相机的 seg2d label_map 当底图画出来，再把所有点在
      这个相机上的投影位置(u,v)画成散点叠加上去（散点按查到的label上色）。
      散点应该和底图对应颜色区域高度重合；如果整体偏移/镜像/错位，
      说明坐标系对应关系搞反了，必须肉眼确认过再往下走。
```