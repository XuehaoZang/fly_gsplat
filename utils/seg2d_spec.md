# seg2d 规格文档 — 单帧独立 body/wing 2D 分割（简化版）

## 背景与目标

将原有 MATLAB `seg_class`（果蝇 body/wing 运动分割算法）的核心逻辑，
以**单帧独立、无时序修正、无 PCA 形状过滤**的简化形式移植为 Python，
用于给某一帧的 4 相机图像生成 body/wing 像素级伪标签，进而重投影到
3D 高斯点云上。

本文档只描述算法逻辑（伪代码级别），不含任何 MATLAB 原始代码。
本轮范围明确排除：left/right 翅膀区分、多帧时序一致性修正、PCA 翅膀
形状过滤——这些留到简化版验证通过之后再补。

---

## 1. 坐标约定（务必与 `utils/calib.py` 对齐）

| 概念 | 约定 |
|---|---|
| 原始稀疏数据 `(row, col, val)` | **1-based**；`row` = 图像纵轴(y)，`col` = 图像横轴(x)，`val` = 灰度强度 0~255 |
| 重建出的密集灰度图 `gray[H, W]` | 标准 numpy 0-based 数组，`gray[row-1, col-1] = val` |
| `utils/calib.py::proj()` 返回的 `(u, v)` | **0-based** OpenCV 像素坐标，`u` 对应 col 方向，`v` 对应 row 方向 |
| 换算关系 | `u = col - 1`，`v = row - 1`；反过来查 label map 时 `row = round(v) + 1`，`col = round(u) + 1`，或者直接在 0-based 的 `label_map[H,W]` 上用 `label_map[round(v), round(u)]` |
| 图像尺寸 | `H = 800, W = 1280`（4相机固定） |

---

## 2. 输入数据

沿用现有 `generate_dataset.py` 里读取 `Camera{cam}_sparse.mat` 的方式
（h5py 读取 `-v7.3` 格式 `.mat`，字段路径 `/frames/indIm`，每帧一个
`ref`，指向的数组是某一帧的 `[row, col, val]` 列表，1-based）。

建议把这部分逻辑从 `generate_dataset.py` 抽成一个公共函数，放进
`utils/seg2d.py`（或 `utils/dataset.py`），供两处复用：

```
load_sparse_frame(sparse_path: Path, frame_idx: int, frame_size=(800,1280)) -> np.ndarray
    # 返回 (H, W) uint8 密集灰度图，背景像素=0
    # 内部实现：
    #   1. 用 h5py 打开 sparse_path
    #   2. 取 refs = f['/frames/indIm'][0]
    #   3. indIm = f[refs[frame_idx]][:]，若 shape[0]==3 则转置
    #   4. rows = indIm[:,0].astype(int) - 1   # 转 0-based
    #      cols = indIm[:,1].astype(int) - 1
    #      vals = indIm[:,2].astype(uint8)
    #   5. 边界检查后写入 im[rows, cols] = vals
```

---

## 3. 算法：body 分割

**目标**：从密集灰度图里分出苍蝇身体（躯干，运动幅度小、通常灰度值
偏低/偏亮取决于成像，需按实际数据核实极性）对应的像素。

```
segment_body(gray: (H,W) uint8) -> body_mask: (H,W) bool

1. silhouette = gray > 0                          # 苍蝇整体轮廓（前景）
   if silhouette 像素数 < 某个最小值（如 50）：直接返回全 False，跳过（该帧几乎没有前景，可能是空/坏帧）

2. 阈值估计（单帧版，不做跨帧采样）：
   values = gray[silhouette]                      # 前景像素灰度值
   hist, edges = histogram(values, bins=100)
   smooth_hist = 滑动平均(hist, window≈25)
   peaks = find_peaks(smooth_hist, prominence>=10, distance>=40)   # scipy.signal.find_peaks

   if len(peaks) >= 2:
       取前两个峰 i1, i2（按峰位置排序，非按高度）
       valley_idx = argmin(smooth_hist[i1:i2]) + i1
       TH_candidate = bin_centers[valley_idx]
       if 30 <= TH_candidate <= 60:
           TH = TH_candidate
       else:
           TH = DEFAULT_TH   # 越界，回退默认值，并记录一条 warning 日志
   else:
       TH = DEFAULT_TH       # 找不到双峰，回退默认值，并记录 warning
   # DEFAULT_TH = 40（沿用原 MATLAB 默认值，可调）

3. body_bin = (gray <= TH) & silhouette           # 注意极性：原逻辑是"更暗"的部分是body
                                                    # ⚠️ 需要用真实数据核实这个极性方向是否正确，
                                                    #    如果反了就是 (gray > TH) & silhouette

4. 形态学清理（对应 MATLAB bwareaopen + imopen + imfill）：
   body_bin = remove_small_objects(body_bin, min_size=100)
   body_bin = binary_opening(body_bin, disk(radius=2))
   body_bin = binary_fill_holes(body_bin)

5. return body_bin
```

---

## 4. 算法：wing 分割

**目标**：在去掉 body 之后的前景区域里，找到面积最大的若干个连通域
作为 wing（本轮不区分 wing1/wing2 是左翅还是右翅，直接合并成一个
"wing" 标签；若识别到 >=1 个有效连通域就都算 wing）。

```
segment_wing(gray: (H,W) uint8, body_mask: (H,W) bool) -> wing_mask: (H,W) bool

1. silhouette = gray > 0
2. body_dilated = binary_dilation(body_mask, disk(radius=7))
3. diff = silhouette & ~body_dilated               # 减去膨胀后的body，剩下的候选是wing+腿+噪声
4. diff = binary_closing(diff, disk(radius=7))      # 对应原 imclose(SE=disk(7))，弥合小缝隙

5. labels, num = connected_components(diff)         # scipy.ndimage.label 或 skimage.measure.label
   if num == 0:
       return 全False

6. areas = [各连通域像素数]
   排序取面积最大的最多2个连通域，且要求 area >= LEG_TH（默认100，用于过滤腿/触角等小碎片）
   若没有任何连通域满足 area >= LEG_TH：
       退化处理——保留面积最大的那一个（即使小于阈值），并记录 warning
       （对应原 MATLAB "⚠️ No valid wing passed filters, keeping largest as fallback"）

7. wing_mask = 上一步选中的连通域像素的并集
8. return wing_mask
```

---

## 5. 汇总：生成 label map

```
segment_body_wing(gray: (H,W) uint8, default_th=40, leg_th=100) -> label_map: (H,W) uint8

label_map = zeros((H,W), dtype=uint8)
body_mask = segment_body(gray, default_th)
wing_mask = segment_wing(gray, body_mask, leg_th)

label_map[gray > 0] = 3        # 3 = 前景但未分类（腿/触角/噪声），先占位再被下面覆盖
label_map[body_mask] = 1       # 1 = body
label_map[wing_mask] = 2       # 2 = wing        （wing_mask 和 body_mask 理论上不重叠，若有重叠 wing 优先覆盖，需在实现里显式处理谁覆盖谁）

return label_map

# label 约定：
#   0 = 背景（不在苍蝇轮廓内）
#   1 = body
#   2 = wing（不分左右）
#   3 = 前景但未分类（腿/触角/形态学噪声/被legTH过滤掉的小碎片）
```

> 关于 label=3：这是与旧 MATLAB 逻辑的一个有意区别——旧代码里没被分到
> body/wing1/wing2 的前景像素直接被丢弃、不导出。这里显式保留为
> label=3，是为了在后续 3D 重投影投票时能区分"确定是背景"和"是苍蝇
> 但分类算法没处理到"，避免二者被静默合并成同一种误判。

---

## 6. 输出与冒烟测试建议

对单帧 4 相机跑完 `segment_body_wing` 后：

- 保存 4 张 overlay 图（原始灰度图为底，body 涂绿、wing 涂红、
  label=3 涂黄，透明度叠加），文件名如
  `seg2d_debug_cam{cam}_frame{fr}.png`
- 打印每相机的统计：`{cam: {body_px, wing_px, unclassified_px,
  bg_px, threshold_used}}`
- 人工确认：body 轮廓是否大致对应躯干位置、wing 是否覆盖翅膀区域、
  是否有明显把整只苍蝇都判成 body（阈值极性反了）或者全判成背景
  （轮廓提取失败）等异常情况

---

## 7. 参数汇总表

| 参数 | 默认值 | 来源/含义 |
|---|---|---|
| `DEFAULT_TH` | 40 | body/wing 灰度阈值回退默认值 |
| `TH` 有效范围 | [30, 60] | 超出则视为估计失败，回退 `DEFAULT_TH` |
| body 最小连通域面积 | 100 | `remove_small_objects` |
| body 形态学开运算半径 | 2 px | 去噪 |
| body 膨胀半径（用于扣除得到 wing 候选区） | 7 px | 对应原 `strel('disk',7)` |
| wing 闭运算半径 | 7 px | 弥合缝隙 |
| `LEG_TH`（wing 最小连通域面积） | 100 | 过滤腿/触角/噪声碎片 |
| wing 最多保留连通域数 | 2 | 合并为同一个 wing 标签 |
| 前景最小像素数（跳过空帧判定） | 50（建议值，需按实际数据调） | 避免空/坏帧报错 |

---

## 8. 明确排除在本轮之外（记录以防遗忘）

- wing 的 left/right 语义区分
- ±delta 帧的时序一致性修正（原 `hybrid_seg` 里对 body 的运动一致性
  校正、原 `BodyCM_estimate`/`trans_frame` 的运动对齐方法）
- wing 的 PCA 形状过滤（长宽比、边界自交检测、边界曲率平滑度）
- body 阈值极性、有效范围 [30,60]、各形态学半径参数，都是照搬旧
  MATLAB 默认值，未必适配新数据集，实现后需要用真实图像跑一遍看效果，
  必要时调参
