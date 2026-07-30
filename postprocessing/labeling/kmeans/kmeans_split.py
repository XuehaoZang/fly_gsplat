"""
T3 body/wing 二分类，第二版：无监督聚类(KMeans)方法，作为规则阈值法
(binary_split.py，现归档在 postprocessing/labeling/binary/ 供对照)的替代方案。

背景(阈值法失败的根本原因): body -> wing 是连续渐变(尤其翼根附近)，不是双峰分布，
硬阈值(dist_to_principal_axis / planarity 任一超阈值)只能抓住翼尖附近的极值点，
漏掉根部/中段的wing点。改用无监督聚类，让多维特征的联合分布决定分界，而不是
单特征硬切。

方法:
1. 特征: [x, y, z, planarity, scale_ratio, opacity] 六维(先用这六维；R/G/B/
   color_oob不用，历史上这批数据颜色信号不稳定)。每维单独standardize
   (zero mean unit variance)后再做kmeans，避免xyz的米级尺度和0~1的形状特征
   混在一起时被某一类特征暗中主导。
2. k=3(body/wing_L/wing_R一次性聚类，不再分两步做body-vs-all)，sklearn KMeans
   (n_init>=10，固定random_state)。
3. cluster_id -> 语义标签，两条规则都跑、都打印，不自动选一个:
   - 规则A: 点数最多的簇 = body；其余两簇按质心投影到"次主轴"(第二大方差方向，
     猜测对应展翅时的左右展向)的正负暂称wing_A/wing_B(不代表真实L/R语义，
     那是ST3的事，这里不做)。
   - 规则B: mean planarity最低的簇 = body(翅膀是膜状结构，planarity该更高)。
   两规则如果对body的判定一致最好；不一致时都打印，供人工判断哪个更可信，
   不在代码里自动拍板选一个。
4. 出图: 前视+俯视两视角散点图(排版同postprocessing/labeling/binary/diag/check_binary_split.py，
   方便直接对比)，三簇三色(按raw cluster_id着色，不是body/wing语义)，存到
   eda_outputs/，文件名带"kmeans"以区分threshold版的图。
5. 稳定性检查: kmeans对初始化敏感，不能假设它天然稳定——同一帧用5个不同
   random_state重跑，两两算adjusted_rand_score(ARI，对簇编号permutation不敏感，
   能正确处理"同一划分但簇编号换了"的情况)比较cluster分配一致性。

不做的事(留给之后的步骤): L/R真实语义判断(ST3)、颜色特征加入判据、跨帧对齐簇号。

---
v2 更新(body种子引导KMeans初始化 + 精简特征集):

背景: 已验证 opacity>=0.98 和 R<0.2(灰度暗，R/G/B基本相同只留R这一维)两个信号能
干净地圈出body高置信度核心点；planarity/scale_ratio/sphericity/linearity在真实
阈值下不鲁棒，本次剔除，不再进v2特征集。原6维随机初始化流程(FEATURES/standardize/
run_kmeans)保留不动，供v2跟"旧版本"对比用，也是 eda_body_wing_features.py 的
ABLATION_SETS["full6"]依赖的接口，不能改语义。

v2 方法:
1. 特征集精简为 FEATURES_V2 = [x, y, z, opacity, R] 五维。xyz一组、[opacity, R]
   一组分别standardize，再给[opacity, R]这组乘一个可调权重(AUX_WEIGHTS)后拼接，
   避免颜色/不透明度信号被xyz的三个维度稀释或反过来主导，具体见standardize_v2()。
2. 种子引导初始化(build_seed_init): body高置信度种子 = opacity>=0.98 或 R<0.2 的点。
   取这批种子点在标准化特征空间里的质心，作为KMeans三个初始质心之一；其余两个初始
   质心用kmeans++(sklearn.cluster.kmeans_plusplus)在"剩余(非种子)点"里选，理由:
   种子点已经被专门分配给第一个质心，剩余两个质心不该再被种子点主导，让它们在剩下
   的点(以wing点为主)里找分散的起点。具体拼法见build_seed_init()的实现注释。
   因为init是显式给定的(3, n_features)数组，KMeans只需要跑一次Lloyd迭代
   (n_init=1)，不是sklearn默认的kmeans++随机重启。
3. cluster_id->语义标签: 沿用规则A(点数最多的簇=body，label_by_rule_a，历史验证过
   的规则)，同时新增规则C(label_by_seed_rule): 包含种子点数量最多的簇=body。两条
   规则都打印，对比是否一致。
4. 权重扫描: AUX_WEIGHTS=[1,3,5]，不做自动网格搜索，固定跑这几档，每档单独出一张图
   (文件名带kmeans_v2字样)，标题打印簇点数占比、权重、种子点数。
5. 对比: 每一档v2结果 vs 旧6维随机初始化(FEATURES/MAIN_RANDOM_STATE)的ARI；以及
   簇间/簇内kNN距离表(复用 eda_body_wing_features.py 里
   typical_intra_knn_dist/cross_cluster_nn_dist 的同款计算逻辑，这里原样复制一份
   避免循环import——eda_body_wing_features.py反过来import这个模块的FEATURES等)，
   观察"疑似同一结构被切两段"(cross/intra比值接近1)的情况是否比旧版减少。

---
v3 更新(双翼种子引导初始化，三个簇都有显式init):

背景: v2只给body一个种子，剩下两个wing簇仍用kmeans++随机挑起点，f0061/f0069等帧
不同random_state之间ARI低(不稳定)，且body簇有时把翼根也吞进去。复用T1已验证的
发现(axis_diag诊断，见postprocessing/labeling/binary/diag/eda_outputs/axis_diag_*.png)：全局第一
PCA主轴(最大方差方向)大致沿"翼尖到翼尖"方向，body在轴中段附近，两翼分别向轴的
正负两端延伸——用这条轴给两翼各找一个种子点。

v3 方法:
1. 全局主轴: 对该帧全部if_keep点做weighted_pca，取最大特征值对应的特征向量
   (primary_axis，eigvecs[:,-1])和质心。注意这是"第一主轴"，跟secondary_axis()
   用的"第二主轴"(v1/v2里给wing_A/wing_B定序用的次主轴)不是同一条轴。
2. 带符号投影: t = (xyz - 质心) · 主轴方向(signed_axis_projection)。
   dist_to_principal_axis(既有列)是到轴的无符号距离，不能直接当wing种子判据，
   这里另算带符号的t值区分两端。
3. 两翼候选种子(wing_seed_mask): 分别取t最大和t最小的一段极值点(WING_SEED_FRAC=
   7.5%，5~10%区间内的固定值，不做网格搜索)。极值点里可能混入孤立噪点，对每组
   候选点用utils/ply.py里现成的connected_component_sizes(候选点自己的局部kNN图)
   做一次紧凑性检查，只保留候选点里最大连通分量，丢掉游离的离群点，剩下的点才
   算种子。跟v2的body种子(seed_mask)有重叠的点直接从wing种子里剔除(避免同一个点
   被算进两个不同簇的种子质心里)，重叠数量如实打印，不静默忽略。
4. 三质心init(build_seed_init_v3): [body种子质心, wingA种子质心, wingB种子质心]
   都在标准化特征空间(FEATURES_V2=[x,y,z,opacity,R]，aux_weight固定用1x，不再像
   v2一样扫[1,3,5]三档——三个质心都已经显式给定，权重扫描不是这次要验证的变量)
   里取均值，直接传给KMeans(init=...)，n_init=1(同v2)。
5. 簇id->语义: 不假设KMeans迭代后簇的索引顺序还对应init时的[body,wingA,wingB]
   顺序(Lloyd迭代理论上不会互换质心身份，但不应该不做验证就假设)。
   label_by_seed_rule_v3: 算"每个簇 x 每组种子"的重叠点数矩阵(3x3)，用
   scipy.optimize.linear_sum_assignment找总重叠最大的一一对应，比硬编码索引顺序
   更稳妥；如果得到的对应和"索引顺序不变"的朴素假设不一致会打印提示。
6. 诊断复用v1/v2同款两套: 5个random_state的两两ARI稳定性(用
   stability_check_with_init，一个通用版本，同时给v2的body-only种子init和v3的
   三种子init复用，因为v2下kmeans++选的另两个质心仍受random_state影响、v3下三个
   质心都是种子给定、理论上应该不受random_state影响——但"应该"不代表一定，照样跑
   一遍如实报告，不预设结果)；簇间/簇内kNN距离表(cross_vs_intra_table，判断是否
   疑似硬切)。
7. 汇总: build_comparison_table在v1(无种子)/v2(仅body种子,aux_weight=1x，跟v3
   同一权重才可比)/v3(三种子)三版本、DEV_FRAMES全部帧上取
   [ARI均值, ARI最小值, 疑似硬切簇对数]，存成一张长表(每帧x每版本一行)，方便直接
   按frame筛选对比f0061/f0069这两帧改善没有，而不是只看全局均值掩盖单帧的问题。
8. 出图: 同款前视+俯视双视角3D散点图，文件名带"kmeans_v3"。v3的图和汇总表存到
   kmeans/k_means_results/v3/(不是eda_outputs/，跟v1/v2的输出目录分开，方便单独归档)。

用法:
    python -m postprocessing.labeling.kmeans.kmeans_split
"""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from postprocessing.cleaning.viz_floater_check import find_features_csv  # noqa: E402
from postprocessing.kinematics.geometry import weighted_pca  # noqa: E402
from postprocessing.labeling.kmeans.diag.select_dev_frames import DEV_FRAMES, DATASET_DIR  # noqa: E402
from utils.ply import connected_component_sizes  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "eda_outputs"

FEATURES = ["x", "y", "z", "planarity", "scale_ratio", "opacity"]
K = 3
N_INIT = 10
MAIN_RANDOM_STATE = 0              # 用来出图/定语义标签的那一次kmeans
STABILITY_SEEDS = [0, 1, 2, 3, 4]  # 稳定性检查用的5个种子(含MAIN_RANDOM_STATE)

CLUSTER_COLORS = [to_rgb("#1f77b4"), to_rgb("#ff7f0e"), to_rgb("#2ca02c")]

# ---- v2: 精简特征集 + body种子引导初始化，见模块docstring "v2 更新" ----
FEATURES_V2 = ["x", "y", "z", "opacity", "R"]
XYZ_COLS = ["x", "y", "z"]
AUX_COLS_V2 = ["opacity", "R"]           # 单独standardize+加权的那组
SEED_OPACITY_THRESH = 0.98
SEED_R_THRESH = 0.2
AUX_WEIGHTS = [1, 3, 5]                  # [opacity, R]标准化后乘的权重，几档固定值，不做网格搜索
K_NN_INTRA = 5                           # 簇内"典型"kNN距离用的近邻数，同eda_body_wing_features.py

# ---- v3: 双翼种子引导初始化，见模块docstring "v3 更新" ----
V3_OUT_DIR = Path(__file__).resolve().parent / "k_means_results" / "v3"
V3_AUX_WEIGHT = 1                        # 跟v2同权重才可比，三质心都已显式给定，不再扫权重
WING_SEED_FRAC = 0.075                   # 主轴t值最大/最小的一段极值点比例(5~10%区间内的固定值)
WING_SEED_CC_K = 10                      # 候选点自己的局部kNN图用的k，同utils.ply.connected_component_sizes默认
WING_SEED_CC_PERCENTILE = 75.0           # 同上，同该函数默认dist_percentile


def load_kept(frame: str) -> pd.DataFrame:
    """加载该帧的_marked表，只取if_keep=True的点(跟binary/同样的口径)。"""
    features_csv = find_features_csv(frame, DATASET_DIR)
    marked_csv = features_csv.with_name(features_csv.stem + "_marked.csv")
    df = pd.read_csv(marked_csv)
    return df[df["if_keep"]].reset_index(drop=True)


def standardize(df: pd.DataFrame) -> np.ndarray:
    """FEATURES 每维单独zero-mean/unit-variance后再喂给kmeans，见模块docstring。"""
    return StandardScaler().fit_transform(df[FEATURES].to_numpy(dtype=float))


def run_kmeans(X: np.ndarray, random_state: int) -> np.ndarray:
    km = KMeans(n_clusters=K, n_init=N_INIT, random_state=random_state)
    return km.fit_predict(X)


def seed_mask(df: pd.DataFrame) -> np.ndarray:
    """body高置信度种子: opacity>=SEED_OPACITY_THRESH 或 R<SEED_R_THRESH(灰度暗)，
    见模块docstring "v2 更新"。"""
    return ((df["opacity"] >= SEED_OPACITY_THRESH) | (df["R"] < SEED_R_THRESH)).to_numpy()


def standardize_v2(df: pd.DataFrame, aux_weight: float) -> np.ndarray:
    """FEATURES_V2=[x,y,z,opacity,R]。xyz和[opacity,R]分别standardize(避免xyz的米级
    尺度和0~1的opacity/R混在一起时被暗中主导)，再给标准化后的[opacity,R]整体乘
    aux_weight，最后按FEATURES_V2的列顺序拼接(x,y,z先，opacity,R后)。"""
    xyz_scaled = StandardScaler().fit_transform(df[XYZ_COLS].to_numpy(dtype=float))
    aux_scaled = StandardScaler().fit_transform(df[AUX_COLS_V2].to_numpy(dtype=float)) * aux_weight
    return np.hstack([xyz_scaled, aux_scaled])


def build_seed_init(X: np.ndarray, seeds: np.ndarray, random_state: int) -> np.ndarray:
    """拼出KMeans要用的显式(3, n_features)初始质心数组:
      - 第0行 = 种子点(seeds=True的行)在标准化特征空间X里的质心，即
        X[seeds].mean(axis=0)。这一行代表"body高置信度种子"这个初始猜测。
      - 第1、2行 = 在"剩余(非种子)点" X[~seeds] 上跑
        sklearn.cluster.kmeans_plusplus(n_clusters=2) 选出的两个初始点。种子点已经
        被专门分配给第0行，这里不想让种子点再主导另外两个质心的选取，所以kmeans++
        只在剩余点(以wing点为主)里找两个分散的起点。
    三行按顺序 np.vstack 成 (3, n_features)，直接传给 KMeans(init=...)。"""
    seed_centroid = X[seeds].mean(axis=0, keepdims=True)
    remaining = X[~seeds]
    other_centers, _ = kmeans_plusplus(remaining, n_clusters=2, random_state=random_state)
    return np.vstack([seed_centroid, other_centers])


def run_kmeans_v2(X: np.ndarray, seeds: np.ndarray, random_state: int) -> np.ndarray:
    """用build_seed_init给出的显式初始质心跑KMeans。init是显式数组时只需要跑一次
    Lloyd迭代，所以n_init=1(不是sklearn默认的kmeans++多次随机重启)。"""
    init = build_seed_init(X, seeds, random_state)
    km = KMeans(n_clusters=K, init=init, n_init=1, random_state=random_state)
    return km.fit_predict(X)


def primary_axis(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """全局第一PCA主轴(最大方差方向)+质心，T1已验证大致沿"翼尖到翼尖"方向，
    见模块docstring "v3 更新"。跟secondary_axis()用的次主轴不是同一条轴。"""
    _, eigvecs, centroid = weighted_pca(xyz)
    return eigvecs[:, -1], centroid


def signed_axis_projection(xyz: np.ndarray, axis: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """点在主轴上的带符号投影t=(xyz-质心)·axis。既有的dist_to_principal_axis列是到
    轴的无符号距离，分不出两翼在轴的哪一端，这里另算带符号值。"""
    return (xyz - centroid) @ axis


def wing_seed_mask(xyz: np.ndarray, t: np.ndarray, take_top: bool,
                    frac: float = WING_SEED_FRAC, k: int = WING_SEED_CC_K,
                    percentile: float = WING_SEED_CC_PERCENTILE) -> np.ndarray:
    """t最大(take_top=True)或最小(take_top=False)的一段极值点里，用
    connected_component_sizes(候选点自己的局部kNN图)剔除游离离群点，只保留候选点
    里最大连通分量，返回布尔mask(长度=len(xyz))，见模块docstring "v3 更新" 第3点。"""
    n = len(t)
    n_take = max(int(np.ceil(n * frac)), k + 2)
    order = np.argsort(t)
    idx = order[-n_take:] if take_top else order[:n_take]

    mask = np.zeros(n, dtype=bool)
    cand_xyz = xyz[idx]
    k_use = min(k, len(cand_xyz) - 1)
    if k_use < 1:
        mask[idx] = True
        return mask

    comp_sizes = connected_component_sizes(cand_xyz, k=k_use, dist_percentile=percentile)
    keep_local = comp_sizes == comp_sizes.max()
    mask[idx[keep_local]] = True
    return mask


def build_seed_init_v3(X: np.ndarray, mask_body: np.ndarray, mask_wing_a: np.ndarray,
                        mask_wing_b: np.ndarray) -> np.ndarray:
    """三个初始质心都是种子给定(不再靠kmeans++猜)：[body种子质心, wingA种子质心,
    wingB种子质心]在标准化特征空间X里的均值，按顺序vstack成(3, n_features)。"""
    return np.vstack([
        X[mask_body].mean(axis=0),
        X[mask_wing_a].mean(axis=0),
        X[mask_wing_b].mean(axis=0),
    ])


def run_kmeans_v3(X: np.ndarray, init: np.ndarray, random_state: int) -> np.ndarray:
    """用build_seed_init_v3给出的三质心init跑KMeans，同run_kmeans_v2一样n_init=1。"""
    km = KMeans(n_clusters=K, init=init, n_init=1, random_state=random_state)
    return km.fit_predict(X)


def label_by_seed_rule_v3(labels: np.ndarray, mask_body: np.ndarray, mask_wing_a: np.ndarray,
                          mask_wing_b: np.ndarray) -> dict[int, str]:
    """三个簇的语义由"簇 x 种子组"重叠点数(3x3矩阵)决定，用linear_sum_assignment找
    总重叠最大的一一对应——不假设KMeans迭代后簇索引顺序还等于init时的
    [body,wingA,wingB]顺序，见模块docstring "v3 更新" 第5点。"""
    groups = ["body", "wing_A", "wing_B"]
    masks = [mask_body, mask_wing_a, mask_wing_b]
    overlap = np.array([[int(np.sum(m & (labels == c))) for m in masks] for c in range(K)])
    row_ind, col_ind = linear_sum_assignment(-overlap)  # 最大化重叠 = 最小化负重叠
    return {int(c): groups[g] for c, g in zip(row_ind, col_ind)}


def stability_check_with_init(X: np.ndarray, init_fn, seeds: list[int] = STABILITY_SEEDS) -> dict:
    """通用稳定性检查：init_fn(random_state)->(3,n_features)显式初始质心数组，
    KMeans只跑一次Lloyd迭代(n_init=1)。v2下init_fn靠kmeans_plusplus选两翼质心(受
    random_state影响)，v3下三个质心都是种子给定(理论上不受random_state影响)，两边
    复用同一个稳定性检查逻辑才可比，见模块docstring "v3 更新" 第6点。"""
    runs = {}
    for s in seeds:
        km = KMeans(n_clusters=K, init=init_fn(s), n_init=1, random_state=s)
        runs[s] = km.fit_predict(X)
    pairwise = [(s1, s2, adjusted_rand_score(runs[s1], runs[s2]))
                for s1, s2 in combinations(seeds, 2)]
    aris = [ari for _, _, ari in pairwise]
    return {"pairwise": pairwise, "mean_ari": float(np.mean(aris)), "min_ari": float(np.min(aris))}


def cluster_sizes(labels: np.ndarray) -> np.ndarray:
    return np.array([int((labels == c).sum()) for c in range(K)])


def secondary_axis(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """次主轴(第二大方差方向)+质心，猜测对应展翅时的左右展向，仅用来给wing_A/wing_B
    一个暂时的、带点物理意义的排序方向；不代表真实L/R语义(ST3的事，这里不做)。"""
    _, eigvecs, centroid = weighted_pca(xyz)
    return eigvecs[:, -2], centroid


def _split_others_by_axis(df: pd.DataFrame, labels: np.ndarray, others: list[int],
                           axis: np.ndarray, centroid: np.ndarray) -> tuple[int, int]:
    """把body之外的两个簇按质心在axis上的投影排序，返回(wing_A簇id, wing_B簇id)
    (投影较大的一侧记作wing_A)。规则A/B共用同一条排序规则，方便直接对比。"""
    xyz = df[["x", "y", "z"]].to_numpy()
    projs = {c: float(np.dot(xyz[labels == c].mean(axis=0) - centroid, axis)) for c in others}
    ordered = sorted(others, key=lambda c: projs[c], reverse=True)
    return ordered[0], ordered[1]


def label_by_rule_a(df: pd.DataFrame, labels: np.ndarray, axis: np.ndarray, centroid: np.ndarray) -> dict[int, str]:
    """规则A: 点数最多的簇 = body；其余两簇按质心在次主轴上的投影正负分wing_A/wing_B。"""
    body_id = int(np.argmax(cluster_sizes(labels)))
    others = [c for c in range(K) if c != body_id]
    wing_a, wing_b = _split_others_by_axis(df, labels, others, axis, centroid)
    return {body_id: "body", wing_a: "wing_A", wing_b: "wing_B"}


def label_by_rule_b(df: pd.DataFrame, labels: np.ndarray, axis: np.ndarray, centroid: np.ndarray) -> dict[int, str]:
    """规则B: mean planarity最低的簇 = body；其余两簇的wing_A/wing_B排序方式同规则A，
    方便直接对比两条规则是否选出同一个body簇。"""
    planarity_means = {c: float(df.loc[labels == c, "planarity"].mean()) for c in range(K)}
    body_id = min(planarity_means, key=planarity_means.get)
    others = [c for c in range(K) if c != body_id]
    wing_a, wing_b = _split_others_by_axis(df, labels, others, axis, centroid)
    return {body_id: "body", wing_a: "wing_A", wing_b: "wing_B"}


def label_by_seed_rule(df: pd.DataFrame, labels: np.ndarray, seeds: np.ndarray,
                       axis: np.ndarray, centroid: np.ndarray) -> dict[int, str]:
    """规则C(v2新增): 包含body高置信度种子点数量最多的簇 = body；其余两簇的
    wing_A/wing_B排序方式同规则A，方便和规则A(点数最多=body)直接对比。"""
    seed_counts = {c: int(np.sum(seeds & (labels == c))) for c in range(K)}
    body_id = max(seed_counts, key=seed_counts.get)
    others = [c for c in range(K) if c != body_id]
    wing_a, wing_b = _split_others_by_axis(df, labels, others, axis, centroid)
    return {body_id: "body", wing_a: "wing_A", wing_b: "wing_B"}


def typical_intra_knn_dist(xyz_cluster: np.ndarray, k: int = K_NN_INTRA) -> float:
    """簇内部"典型"局部近邻距离，同eda_body_wing_features.py的同名函数(这里复制一份
    避免循环import，因为那边反过来import本模块的FEATURES/run_kmeans等)：
    tree.query(xyz, k=k+1)再丢弃自身列，取全簇kNN距离的中位数。"""
    n = len(xyz_cluster)
    kk = min(k, n - 1)
    if kk < 1:
        return float("nan")
    tree = cKDTree(xyz_cluster)
    dists, _ = tree.query(xyz_cluster, k=kk + 1)
    dists = dists[:, 1:]  # 去掉自身(距离0)
    return float(np.median(dists))


def cross_cluster_nn_dist(xyz_a: np.ndarray, xyz_b: np.ndarray) -> float:
    """两簇之间的最近点对距离(单链接)，同eda_body_wing_features.py的同名函数。"""
    tree = cKDTree(xyz_b)
    dists, _ = tree.query(xyz_a, k=1)
    return float(dists.min())


def cross_vs_intra_table(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """簇间/簇内kNN距离对比表，同eda_body_wing_features.py的同名函数：ratio接近1
    提示"疑似同一结构被硬切"，明显更大提示"真正分开的结构"。"""
    xyz = df[["x", "y", "z"]].to_numpy()
    intra = {c: typical_intra_knn_dist(xyz[labels == c]) for c in range(K)}
    rows = []
    for i, j in combinations(range(K), 2):
        cross = cross_cluster_nn_dist(xyz[labels == i], xyz[labels == j])
        ref = min(intra[i], intra[j])
        ratio = cross / ref if ref > 0 else float("inf")
        note = "接近(疑似同一结构被硬切)" if ratio < 1.5 else "明显更大(像是真正分开的结构)"
        rows.append({
            "cluster_pair": f"{i}-{j}", "cross_nn_dist": cross,
            f"intra_typical_k{K_NN_INTRA}_i": intra[i], f"intra_typical_k{K_NN_INTRA}_j": intra[j],
            "ratio_cross_over_min_intra": ratio, "note": note,
        })
    return pd.DataFrame(rows)


def stability_check(X: np.ndarray, seeds: list[int] = STABILITY_SEEDS) -> dict:
    """kmeans对初始化敏感，不能假设天然稳定：同一帧用多个不同random_state重跑，
    两两算adjusted_rand_score(ARI，对簇编号permutation不敏感)比较cluster分配的
    一致性。"""
    runs = {s: run_kmeans(X, s) for s in seeds}
    pairwise = [(s1, s2, adjusted_rand_score(runs[s1], runs[s2]))
                for s1, s2 in combinations(seeds, 2)]
    aris = [ari for _, _, ari in pairwise]
    return {"pairwise": pairwise, "mean_ari": float(np.mean(aris)), "min_ari": float(np.min(aris))}


def plot_kmeans_clusters(df: pd.DataFrame, labels: np.ndarray, frame: str, out_path: Path,
                          features_label: list[str] | None = None, extra_info: str = "") -> None:
    xyz = df[["x", "y", "z"]].to_numpy()
    colors = np.array([CLUSTER_COLORS[c] for c in labels])
    sizes = cluster_sizes(labels)
    n_total = len(df)
    features_label = features_label if features_label is not None else FEATURES

    views = [
        ("front (elev=0, azim=-90)", dict(elev=0, azim=-90)),
        ("top (elev=90, azim=-90)", dict(elev=90, azim=-90)),
    ]

    fig = plt.figure(figsize=(12, 5.5))
    for i, (title, view_kw) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=14, alpha=0.9, depthshade=False)
        ax.view_init(**view_kw)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(title, fontsize=10)

    size_str = "  ".join(
        f"cluster{c}={sizes[c]}({100 * sizes[c] / n_total:.1f}%)" for c in range(K))
    extra_line = f"\n{extra_info}" if extra_info else ""
    fig.suptitle(
        f"{frame}: KMeans k={K} on standardized {features_label}  |  n_total={n_total}  |  {size_str}\n"
        f"colors = raw cluster_id (blue=0 orange=1 green=2, NOT body/wing semantics "
        f"-- see printed rule mapping){extra_line}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(frame: str) -> dict:
    df = load_kept(frame)
    X = standardize(df)

    labels = run_kmeans(X, MAIN_RANDOM_STATE)
    sizes = cluster_sizes(labels)
    n_total = len(df)

    xyz = df[["x", "y", "z"]].to_numpy()
    axis, centroid = secondary_axis(xyz)
    mapping_a = label_by_rule_a(df, labels, axis, centroid)
    mapping_b = label_by_rule_b(df, labels, axis, centroid)
    body_a = next(c for c, lab in mapping_a.items() if lab == "body")
    body_b = next(c for c, lab in mapping_b.items() if lab == "body")
    agree = body_a == body_b

    stability = stability_check(X)

    out_path = OUT_DIR / f"kmeans_split_{frame}.png"
    plot_kmeans_clusters(df, labels, frame, out_path)

    print(f"\n[{frame}] n_total={n_total}")
    print("  簇点数占比: " + "  ".join(
        f"cluster{c}={sizes[c]}({100 * sizes[c] / n_total:.1f}%)" for c in range(K)))
    print("  规则A(点数最多=body): " + "  ".join(
        f"cluster{c}={lab}" for c, lab in sorted(mapping_a.items())))
    print("  规则B(mean planarity最低=body): " + "  ".join(
        f"cluster{c}={lab}" for c, lab in sorted(mapping_b.items())))
    print(f"  规则A/B对body簇的判定{'一致' if agree else '不一致，需要人工判断'}: "
          f"A选cluster{body_a}, B选cluster{body_b}")
    if not agree:
        planarity_means = {c: float(df.loc[labels == c, "planarity"].mean()) for c in range(K)}
        print("    [人工判断参考] 各簇mean planarity: " +
              "  ".join(f"cluster{c}={planarity_means[c]:.3f}" for c in range(K)) +
              f"  (点数最多的是cluster{int(np.argmax(sizes))})")
    print(f"  稳定性检查(5个random_state两两ARI): mean={stability['mean_ari']:.3f} "
          f"min={stability['min_ari']:.3f}  "
          f"{'稳定' if stability['min_ari'] > 0.9 else '[警告]不同初始化的聚类结果有明显差异，需留意'}")
    print(f"  plot -> {out_path}")

    return {
        "frame": frame, "sizes": sizes, "mapping_a": mapping_a, "mapping_b": mapping_b,
        "agree": agree, "stability": stability,
    }


def run_v2(frame: str, old_labels: np.ndarray) -> dict:
    """v2: 精简特征集[x,y,z,opacity,R] + body种子引导初始化，在AUX_WEIGHTS几档权重上
    各跑一次，见模块docstring "v2 更新"。old_labels是同一帧旧6维随机初始化
    (FEATURES/MAIN_RANDOM_STATE)的结果，用来算新旧ARI对比。"""
    df = load_kept(frame)
    n_total = len(df)
    seeds = seed_mask(df)
    n_seed = int(seeds.sum())

    xyz = df[["x", "y", "z"]].to_numpy()
    axis, centroid = secondary_axis(xyz)

    print(f"\n[{frame}] (v2) n_total={n_total}  种子点(opacity>={SEED_OPACITY_THRESH} 或 "
          f"R<{SEED_R_THRESH})数量={n_seed}({100 * n_seed / n_total:.1f}%)")

    weight_results = []
    for aux_weight in AUX_WEIGHTS:
        X = standardize_v2(df, aux_weight)
        labels = run_kmeans_v2(X, seeds, MAIN_RANDOM_STATE)
        sizes = cluster_sizes(labels)

        mapping_a = label_by_rule_a(df, labels, axis, centroid)
        mapping_seed = label_by_seed_rule(df, labels, seeds, axis, centroid)
        body_a = next(c for c, lab in mapping_a.items() if lab == "body")
        body_seed = next(c for c, lab in mapping_seed.items() if lab == "body")
        agree = body_a == body_seed

        ari_old_vs_new = float(adjusted_rand_score(old_labels, labels))
        dist_df = cross_vs_intra_table(df, labels)

        out_path = OUT_DIR / f"kmeans_v2_{frame}_w{aux_weight}.png"
        plot_kmeans_clusters(
            df, labels, frame, out_path, features_label=FEATURES_V2,
            extra_info=(f"v2: seed-centroid init, aux_weight={aux_weight}x on [opacity,R], "
                        f"n_seed={n_seed}({100 * n_seed / n_total:.1f}%)"))

        print(f"  [aux_weight={aux_weight}x] 簇点数占比: " + "  ".join(
            f"cluster{c}={sizes[c]}({100 * sizes[c] / n_total:.1f}%)" for c in range(K)))
        print(f"    规则A(点数最多=body): " + "  ".join(
            f"cluster{c}={lab}" for c, lab in sorted(mapping_a.items())))
        print(f"    规则C(种子点数量最多=body): " + "  ".join(
            f"cluster{c}={lab}" for c, lab in sorted(mapping_seed.items())))
        print(f"    规则A/C对body簇的判定{'一致' if agree else '不一致，需要人工判断'}: "
              f"A选cluster{body_a}, C选cluster{body_seed}")
        print(f"    ARI(旧6维随机初始化 vs 新5维种子初始化) = {ari_old_vs_new:.3f}")
        print("    簇间/簇内kNN距离:")
        for _, row in dist_df.iterrows():
            print(f"      cluster{row['cluster_pair']}: cross={row['cross_nn_dist']:.5f}  "
                  f"intra_i={row[f'intra_typical_k{K_NN_INTRA}_i']:.5f}  "
                  f"intra_j={row[f'intra_typical_k{K_NN_INTRA}_j']:.5f}  "
                  f"ratio={row['ratio_cross_over_min_intra']:.2f}  {row['note']}")
        print(f"    plot -> {out_path}")

        weight_results.append({
            "aux_weight": aux_weight, "sizes": sizes, "mapping_a": mapping_a,
            "mapping_seed": mapping_seed, "agree": agree,
            "ari_old_vs_new": ari_old_vs_new, "dist_df": dist_df,
        })

    return {"frame": frame, "n_seed": n_seed, "weight_results": weight_results}


def n_hardcut_pairs(dist_df: pd.DataFrame) -> int:
    """疑似硬切簇对数：cross_vs_intra_table里ratio<1.5(note标"疑似同一结构被硬切")的行数。"""
    return int((dist_df["ratio_cross_over_min_intra"] < 1.5).sum())


def run_v3(frame: str) -> dict:
    """v3: body种子(同v2) + 两翼种子(全局主轴带符号投影极值段 + 连通分量紧凑性过滤)
    三质心引导初始化，aux_weight固定1x，见模块docstring "v3 更新"。"""
    df = load_kept(frame)
    n_total = len(df)
    xyz = df[["x", "y", "z"]].to_numpy()

    axis, centroid = primary_axis(xyz)
    t = signed_axis_projection(xyz, axis, centroid)

    mask_body = seed_mask(df)
    mask_wing_a = wing_seed_mask(xyz, t, take_top=True)
    mask_wing_b = wing_seed_mask(xyz, t, take_top=False)

    n_overlap_a = int(np.sum(mask_body & mask_wing_a))
    n_overlap_b = int(np.sum(mask_body & mask_wing_b))
    mask_wing_a = mask_wing_a & ~mask_body
    mask_wing_b = mask_wing_b & ~mask_body

    n_seed_body, n_seed_a, n_seed_b = int(mask_body.sum()), int(mask_wing_a.sum()), int(mask_wing_b.sum())
    print(f"\n[{frame}] (v3) n_total={n_total}  种子点数: body={n_seed_body} "
          f"wing_A={n_seed_a} wing_B={n_seed_b}"
          + (f"  [与body种子重叠已剔除: wing_A={n_overlap_a} wing_B={n_overlap_b}]"
             if n_overlap_a or n_overlap_b else ""))

    X = standardize_v2(df, V3_AUX_WEIGHT)
    init = build_seed_init_v3(X, mask_body, mask_wing_a, mask_wing_b)
    labels = run_kmeans_v3(X, init, MAIN_RANDOM_STATE)
    sizes = cluster_sizes(labels)

    mapping = label_by_seed_rule_v3(labels, mask_body, mask_wing_a, mask_wing_b)
    naive_mapping = {0: "body", 1: "wing_A", 2: "wing_B"}
    if mapping != naive_mapping:
        print(f"  [提示] 种子重叠对应关系({mapping})和init索引顺序假设({naive_mapping})不一致，"
              f"以重叠关系为准")

    stability = stability_check_with_init(X, lambda s: build_seed_init_v3(X, mask_body, mask_wing_a, mask_wing_b))
    dist_df = cross_vs_intra_table(df, labels)
    n_hardcut = n_hardcut_pairs(dist_df)

    out_path = V3_OUT_DIR / f"kmeans_v3_{frame}.png"
    plot_kmeans_clusters(
        df, labels, frame, out_path, features_label=FEATURES_V2,
        extra_info=(f"v3: body+wingA+wingB seed-centroid init, aux_weight={V3_AUX_WEIGHT}x, "
                    f"n_seed body={n_seed_body}/wingA={n_seed_a}/wingB={n_seed_b}"))

    print(f"  簇点数占比: " + "  ".join(
        f"cluster{c}={sizes[c]}({100 * sizes[c] / n_total:.1f}%)" for c in range(K)))
    print(f"  簇语义(种子重叠一一对应): " + "  ".join(
        f"cluster{c}={lab}" for c, lab in sorted(mapping.items())))
    print(f"  稳定性检查(5个random_state两两ARI): mean={stability['mean_ari']:.3f} "
          f"min={stability['min_ari']:.3f}  "
          f"{'稳定' if stability['min_ari'] > 0.9 else '[警告]不同初始化的聚类结果有明显差异，需留意'}")
    print("  簇间/簇内kNN距离:")
    for _, row in dist_df.iterrows():
        print(f"    cluster{row['cluster_pair']}: cross={row['cross_nn_dist']:.5f}  "
              f"intra_i={row[f'intra_typical_k{K_NN_INTRA}_i']:.5f}  "
              f"intra_j={row[f'intra_typical_k{K_NN_INTRA}_j']:.5f}  "
              f"ratio={row['ratio_cross_over_min_intra']:.2f}  {row['note']}")
    print(f"  plot -> {out_path}")

    return {
        "frame": frame, "n_seed_body": n_seed_body, "n_seed_a": n_seed_a, "n_seed_b": n_seed_b,
        "mapping": mapping, "stability": stability, "dist_df": dist_df, "n_hardcut": n_hardcut,
    }


def build_comparison_table(v1_rows: list[dict], v2_rows: list[dict], v3_rows: list[dict]) -> pd.DataFrame:
    """v1(无种子)/v2(仅body种子,aux_weight=1x)/v3(三种子)在DEV_FRAMES每一帧上的
    [ARI均值, ARI最小值, 疑似硬切簇对数]长表，一行=一帧x一版本，见模块docstring
    "v3 更新" 第7点：按frame筛选就能直接对比f0061/f0069改善没有，不被全局均值掩盖。"""
    rows = []
    for version, version_rows in [("v1_no_seed", v1_rows), ("v2_body_seed_w1", v2_rows), ("v3_dual_wing_seed", v3_rows)]:
        for r in version_rows:
            rows.append({
                "frame": r["frame"], "version": version,
                "ari_mean": r["stability"]["mean_ari"], "ari_min": r["stability"]["min_ari"],
                "n_hardcut_pairs": r["n_hardcut"],
            })
    return pd.DataFrame(rows)


def main_v3() -> None:
    """v3流水线：跑body+双翼种子引导的KMeans，并跟v1(无种子)/v2(仅body种子,w1x)做
    三版本对比，见模块docstring "v3 更新"。独立于main()，会重新计算v1/v2(w=1)的诊断
    (不复用main()里的结果)，图和汇总表存到k_means_results/v3/。"""
    V3_OUT_DIR.mkdir(parents=True, exist_ok=True)

    v1_rows = []
    for frame in DEV_FRAMES:
        df = load_kept(frame)
        X = standardize(df)
        labels = run_kmeans(X, MAIN_RANDOM_STATE)
        stability = stability_check(X)
        dist_df = cross_vs_intra_table(df, labels)
        v1_rows.append({"frame": frame, "stability": stability, "n_hardcut": n_hardcut_pairs(dist_df)})

    v2_rows = []
    for frame in DEV_FRAMES:
        df = load_kept(frame)
        seeds = seed_mask(df)
        X = standardize_v2(df, V3_AUX_WEIGHT)
        labels = run_kmeans_v2(X, seeds, MAIN_RANDOM_STATE)
        stability = stability_check_with_init(X, lambda s: build_seed_init(X, seeds, s))
        dist_df = cross_vs_intra_table(df, labels)
        v2_rows.append({"frame": frame, "stability": stability, "n_hardcut": n_hardcut_pairs(dist_df)})

    print(f"\n{'=' * 70}\nv3: body+双翼种子引导初始化 + 精简特征集[x,y,z,opacity,R], aux_weight={V3_AUX_WEIGHT}x\n{'=' * 70}")
    v3_rows = [run_v3(f) for f in DEV_FRAMES]

    table = build_comparison_table(v1_rows, v2_rows, v3_rows)
    csv_path = V3_OUT_DIR / "version_comparison_v1_v2_v3.csv"
    table.to_csv(csv_path, index=False)

    print(f"\n{'=' * 70}\nv1/v2/v3 三版本 x {len(DEV_FRAMES)}帧 汇总对比\n{'=' * 70}")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(table.to_string(index=False))
    print(f"  汇总表 -> {csv_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [run(f) for f in DEV_FRAMES]

    n_agree = sum(r["agree"] for r in results)
    mean_min_ari = float(np.mean([r["stability"]["min_ari"] for r in results]))
    print(f"\n{'=' * 70}\n{len(results)}帧汇总(旧版 6维随机初始化)\n{'=' * 70}")
    print(f"  规则A/B对body簇判定一致的帧数: {n_agree}/{len(results)}")
    print(f"  各帧稳定性(min ARI)的均值: {mean_min_ari:.3f}")

    print(f"\n{'=' * 70}\nv2: body种子引导初始化 + 精简特征集[x,y,z,opacity,R]\n{'=' * 70}")
    old_labels_by_frame = {f: run_kmeans(standardize(load_kept(f)), MAIN_RANDOM_STATE) for f in DEV_FRAMES}
    v2_results = [run_v2(f, old_labels_by_frame[f]) for f in DEV_FRAMES]

    print(f"\n{'=' * 70}\nv2 {len(v2_results)}帧汇总\n{'=' * 70}")
    for aux_weight in AUX_WEIGHTS:
        n_agree_v2 = sum(
            1 for r in v2_results
            for wr in r["weight_results"] if wr["aux_weight"] == aux_weight and wr["agree"])
        mean_ari = float(np.mean([
            wr["ari_old_vs_new"] for r in v2_results for wr in r["weight_results"]
            if wr["aux_weight"] == aux_weight]))
        print(f"  [aux_weight={aux_weight}x] 规则A/C对body簇判定一致的帧数: "
              f"{n_agree_v2}/{len(v2_results)}  |  旧vs新ARI均值: {mean_ari:.3f}")


if __name__ == "__main__":
    main()
    main_v3()
