# 磁盘诊断报告 (gpu/disk/audit.py)

## 磁盘空间
- WSL2 (outputs/、data/ 所在文件系统): `/dev/sdd       1007G  137G  819G  15% /`
- 文件系统类型: **是原生ext4 (不是DrvFS/9p挂进来的Windows盘)。** (outputs/→/dev/sdd(ext4); data/→/dev/sdd(ext4))
- Windows C盘 (经 /mnt/c): `C:\             476G  361G  116G  76% /mnt/c`

## 每run分类别平均大小 (抽样均值)
| 分类 | 平均大小 | 备注 |
|---|---|---|
| config.yml | 8.0K |  |
| dataparser_transforms.json | 4.0K |  |
| debug_checkpoints | 20.0K | **不可删/不计入可省空间** |
| nerfstudio_models | 9.6M |  |
| other | 148.0K |  |
| splat.ply | 48.0K |  |
| tensorboard_events | 567.2K |  |

## 空间预估
- 当前总run数: **2307** ，当前总占用: 约 **23.0G**
- 删除所有run的 `nerfstudio_models/`(训练ckpt) 能省: 约 **21.6G** (2307 × 9.6M)
- 不再产生tensorboard事件文件能省: 约 **1.2G** (2307 × 567.2K)
- `debug_checkpoints/` 不计入以上可省空间也不能删 (唯一指标数据源，见脚本docstring)，平均 20.0K/run

### 跑到更多帧数预计还需要的总空间 (假设每帧1个run；多个param_set并行跑同一帧要再乘以param_set数)
| 目标run数 | 预计总占用 | 相对当前(2307个run)的增量 |
|---|---|---|
| 1000 | 10.0G | -13.0G |
| 2000 | 19.9G | -3.1G |
| 5000 | 49.9G | 26.9G |
