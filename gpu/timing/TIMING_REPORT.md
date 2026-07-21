# G2: 单次训练全流程耗时分解报告

**日期**: 2026-07-21
**方法**: `gpu/timing/time_pipeline.py`，固定 (frame=0 of ctrl_009_002, MAX_ITERS=2000,
warmup-length=50, stop-split-at=1800, background=white) 组合，重复 5 次，全程不修改
`generate_dataset.py` / `generate_hull.py` / `models/` 任何代码。
scratch 数据目录 `data/timing_g2_scratch/f0000`，不触碰正式实验帧
(`data/ctrl_009_002/f0000~f0099`)；calibration mat 只读复用 `data/ctrl_009_002/`。

复现: `python3 gpu/timing/time_pipeline.py 5` 生成 `results/timing_raw.json` ->
`python3 gpu/timing/analyze.py` 生成表格+图。

---

## 1. 分阶段耗时表 (n=5, 均值 ± 标准差)

| 阶段 | 均值 (s) | 标准差 (s) | 占端到端 % |
|---|---|---|---|
| 1. generate_dataset | 0.436 | 0.009 | 0.5% |
| 2. generate_hull | 0.071 | 0.007 | 0.1% |
| 3. ns-train 冷启动 (进程起→第一个iteration真正开始) | 12.747 | 0.268 | 15.8% |
| 4. ns-train 训练循环 (2000 iterations) | 55.756 | 4.528 | 69.2% |
| 5. 训练后处理 (export_splat + load_ply + DBSCAN clean) | 11.614 | 0.245 | 14.4% |
| **端到端总计** | **80.623** | **4.719** | **100%** |

阶段5内部再拆分 (n=5)：`ns-export` 子进程 11.60±0.24s，`load_ply_with_attrs` 0.005±0.0002s，
DBSCAN `clean_ply` 0.007±0.002s ——阶段5几乎全部耗时都在 `ns-export` 这一次独立子进程启动上，
真正的数值计算(读ply+聚类)是毫秒级、可忽略的。

阶段1+2 (数据准备) 合计约 0.5s，方差极小(σ<0.01s)，**在整条流水线里是噪声量级**，即使
X:盘(drvfs 9p)比原生 ext4 慢5倍这个 G1 结论仍然成立，但因为单帧要读的数据量本身很小
(~1MB，2.29±0.05 MB/s 的读吞吐)，绝对耗时可忽略不计。

---

## 2. 资源利用率

### CPU (数据准备阶段，阶段1+2，进程内采样)
- `generate_dataset`: cpu_busy_frac 明显 >1（多线程 h5py/numpy/cv2 解码），wall=0.436s，
  实际CPU-busy时间(user+sys求和跨线程)约0.6s → **判定为CPU-bound的小任务**，不是IO-bound：
  哪怕读盘慢(drvfs)，读取量太小(~1MB)以至于IO等待时间被CPU侧的解码/写PNG开销盖过。
- `generate_hull`: cpu_busy_frac ≈ 5.1±0.25（即用满了约5个核心的等效时间），wall仅0.07s ——
  三角化+1万点采样+可见性投票+outlier removal 全部是CPU(numpy/open3d)计算，**GPU完全不参与**，
  且因为点数少(万级)+相机数少(4个)，多线程BLAS/numpy把它压到70ms量级，同样噪声量级。

### GPU (阶段3+4，ns-train子进程全程流式采样，200ms间隔)
详见 `results/gpu_cpu_util.png`（上图GPU%曲线，5次重复叠加；下图CPU%曲线）。

- **冷启动阶段 (0~12.7s)**：GPU利用率几乎全程为0%，仅在 t≈6-8s 出现一个短暂小峰
  (~10-13%，推测是 gsplat/CUDA 惰性初始化时的一次探测性kernel调用或图像预处理里的GPU判断)。
  同一时间窗口内 CPU 侧有两个明显特征：t=0~3s 出现 **1100~1200% 的CPU占用尖峰**
  (11-12个核心同时忙碌，是 torch/nerfstudio/gsplat import + 多线程BLAS/JIT 的典型特征)，
  随后 t≈5-11s 出现几次 250~300% 的次级峰值(对应"Caching / undistorting train images"
  这一步用多worker并行做图像缓存)。**结论：冷启动阶段是纯CPU/内存密集型阶段，GPU全程空闲**，
  这12.7s里显卡完全没被用上。
- **训练循环阶段 (12.7s之后)**：GPU利用率跳升后稳定在 **均值26.3%、稳态区间大致30~40%**
  的窄带内抖动，穿插少量到 72~89%(均值83.2%)的短促尖峰(推测对应densify/split或
  checkpoint写盘时的批量操作，不是持续性满载)。CPU侧同期稳定在 **~100~110%(约1个核心)**，
  说明训练循环期间：**GPU没有被跑满，且CPU也只用了1个核心** —— 这是典型的
  "小工作负载 + Python/kernel-launch开销主导" 特征：本帧初始高斯数很少(hull点~1700，
  训练后~350~380个高斯)，单次iteration的实际计算量太小，不足以填满GPU，
  也不需要多核CPU参与，大部分墙钟时间花在Python侧单线程的kernel调度/CUDA同步等待上。

---

## 3. 一句话结论

**瓶颈不在单一"资源耗尽"，而在"资源没用满"——训练循环占了69%的墙钟时间，但GPU均值利用率只有26%
（稳态30~40%），CPU也只吃了1个核心；此外冷启动(15.8%)+后处理里的`ns-export`子进程(14.4%)
合计约30%的时间纯粹花在torch/nerfstudio/gsplat的进程级import和CUDA context初始化上
（且这个开销每帧要付两次：一次ns-train一次ns-export），跟数据规模无关；而数据准备阶段
(generate_dataset+generate_hull)只占0.6%，可以忽略。**

对 G3 单卡并发扫描设计的直接含义：
1. **不需要给数据准备单独设并发档位** —— 它占比<1%，即使给它无限并发也节省不了多少绝对时间，
   数据生成和训练可以合并到同一个并发档位里一起扫，没有必要拆成两条独立的并发曲线。
2. **训练阶段本身值得重点扫并发数** —— 因为单进程训练时GPU远未跑满(26%均值)、CPU只占1核，
   直觉上一张显卡上跑2~3个并发`ns-train`进程仍有较大空间，G3应把"每卡并发训练进程数"作为
   主要扫描维度，实测多进程叠加后总吞吐(frames/hour)而非只看单进程耗时。
3. **冷启动+ns-export的进程级开销(~30%)是并发数越高、边际收益越有限的部分** —— 这部分是
   CPU密集+一次性开销，如果G3发现高并发下CPU(28线程)先被冷启动阶段的多线程import挤满，
   那么真正的瓶颈会从"GPU利用率"转移到"CPU核数"，需要在G3报告里同时给出GPU和CPU两个维度的
   并发上限，不能只看显存/GPU利用率一个指标。
4. （超出G2范围，供后续参考）由于冷启动+export的开销与"每帧都是一个全新subprocess"直接相关，
   如果未来要跑到几千帧规模，用长驻进程(一次import、循环处理多帧)替代"每帧一个ns-train子进程
   +一个ns-export子进程"，理论上能省掉这~30%目前跟计算量无关的固定开销——这是架构层面的优化点，
   不在本次G2的"不改训练代码"范围内，仅作为后续阶段(G3/G4之后)的参考方向记录在此。

---

## 4. 产出文件

- `gpu/timing/samplers.py` — GPU(nvidia-smi流式采样) + CPU(psutil递归子进程) 采样工具
- `gpu/timing/time_pipeline.py` — 5阶段计时主脚本（`python3 time_pipeline.py [N_REPEATS]`）
- `gpu/timing/analyze.py` — 汇总表格+绘图（不重跑训练，纯读取 `results/timing_raw.json`）
- `gpu/timing/results/timing_raw.json` — 5次重复的完整原始记录(含GPU/CPU采样点序列)
- `gpu/timing/results/timing_summary.csv` / `.json` — 汇总表
- `gpu/timing/results/gpu_cpu_util.png` — GPU/CPU利用率曲线图(5次重复叠加)
