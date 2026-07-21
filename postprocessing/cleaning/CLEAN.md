# postprocessing/cleaning (T2)

## 1. 做什么

输入 T1 (`utils/gaussian_features.py`) 输出的逐点特征表，输出同一张表加一列
`if_keep` (bool)。不删行、不删列、不改变列顺序，只标记孤立的floater点团块。
**不处理**贴着翼缘的尖刺——那部分在单帧特征空间和真实薄片结构连续分布，分不开。

## 2. 判据

连通分量法 (`utils.ply.connected_component_sizes`)：构建 k-近邻图，只保留邻距
不超过全局 `dist_percentile` 分位数的边，做连通分量分析；分量大小
`patch_size <= min_patch_size` 判为floater。

参数锁定为 `k=10, dist_percentile=75, min_patch_size=10`，不开放成CLI可调参数。
选连通分量而不是逐点密度/形状特征，是因为真实解剖结构(躯干/翅膀/腿)对应的分量
和孤立噪点之间有清晰的**物理尺度gap**(真实结构>=17点，孤立噪点<=6点，中间无值)，
跨帧、跨大幅度翅膀运动帧压测验证过这个gap不会被填上，对k/dist_percentile的选取
不敏感；真实尖刺形状上同样细长，但空间上仍连着主体，用连通性可以正确保留它们。

## 3. 已验证范围

- `G2b_scale_reg_ratio3`：基线 f0090/f0091/f0092(floater占比8.6%~10.1%)，
  外加3帧大幅度翅膀运动压测(展开顶点/折叠瞬间/最紧凑状态)，floater占比
  5.2%~15.3%，判据依据的size gap全部保留。
- `G2b_G9`：全量100帧(f0000~f0099)批处理，floater占比5.8%~15.0%
  (mean 9.5%, median 9.5%)。

## 4. 已知问题 / TODO

- **翼缘尖刺不处理**：特征空间连续谱和真实薄片结构分不开，已验证多种方法失败，
  本阶段搁置。
- **G2b_G9约21%帧(21/100)存在11~16点的边界分量**，紧贴阈值但身份未验证
  (可能是触角或腿尖)，当前判据下均正确保留(size>10)非漏判；已测试收紧阈值，
  代价是误伤真实结构，故维持现状不改。
- 判据参数是在 `G2b_scale_reg_ratio3` 上校准锁定的，迁移到 `G2b_G9` 未重新
  独立调优。
- 已知异常帧：`G2b_scale_reg_ratio3` 的 **f0062** floater占比跳到15.35%，
  明显高于基线区间(8.6%~10.1%)；排查确认是该帧翅膀折叠到最紧凑状态时上游
  3D重建噪点本身更多(高opacity离群点)，不是判据失效。

## 5. 如何跑

```bash
# 单帧
python -m postprocessing.cleaning.mark_floaters --csv path/to/gaussian_features_f0090.csv

# 批处理(整段数据集，默认输出汇总到 eda_outputs/)
python -m postprocessing.cleaning.mark_floaters --data-root outputs/.../G2b_G9 --start 0 --end 99
```
