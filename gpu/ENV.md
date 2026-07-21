# G1 环境与硬件基线审计报告

**日期**: 2026-07-21
**范围**: 纯环境探测，未执行任何训练/数据生成代码
**目的**: 为 G2（耗时分解）、G3（单卡并发扫描）、G4（双卡调度器）提供统一、实测的硬件/软件基线

---

## 1. WSL2 资源限制（重要发现：内存被 `.wslconfig` 限制）

`.wslconfig` **存在**，位于 `/mnt/c/Users/computer0/.wslconfig`：

```ini
[wsl2]
memory=28GB
swap=16GB
```

`/etc/wsl.conf`（WSL 内部配置）：
```ini
[boot]
systemd=true

[user]
default=computer0
```

**实测（`free -h`）**：
```
               total        used        free      shared  buff/cache   available
Mem:            27Gi       8.2Gi        14Gi       3.5Mi       5.4Gi        19Gi
Swap:           16Gi          0B        16Gi
```

**结论**：宿主机总物理内存约 32,477 MB (~31.7GB)，但 WSL2 通过 `.wslconfig` 显式限制为 `memory=28GB`，`free -h` 实测可见总量为 27Gi（约合 28GB，误差来自 GiB/GB 换算），当前已用 8.2GB，可用 19GB。**这不是 WSL2 默认行为的资源泄漏，而是人为配置的上限**——WSL2 默认行为其实是宿主机总内存的 50%，这里被显式抬高到 28GB。跑批规模扩大到几千次训练前，若单次训练进程数 × 单进程内存峰值逼近 19GB 可用量，需要考虑调高 `.wslconfig` 里的 `memory` 值（需要重启 WSL2 生效：`wsl --shutdown`）。

**CPU**：`.wslconfig` 未设置 `processors` 限制项，`nproc` 实测为 **28**，与宿主机 i9-10940X（14 核 / 28 线程）完全一致，WSL2 对 CPU 线程无限制。

```
=== lscpu ===
Model name:        Intel(R) Core(TM) i9-10940X CPU @ 3.30GHz
CPU(s):            28
Thread(s) per core: 2
Core(s) per socket: 14
Socket(s):         1
NUMA node(s):      1
```

---

## 2. GPU 可见性（WSL2 vs Windows 对比）

### WSL2 内 `nvidia-smi`：
```
NVIDIA-SMI 595.45.03    Driver Version: 595.71    CUDA Version: 13.2

GPU 0: NVIDIA RTX A5000   17MiB / 24564MiB    0% util
GPU 1: NVIDIA RTX A5000  888MiB / 24564MiB    1% util
No running processes found (WSL 侧看不到 Windows 进程占用的显存来源)
```

### Windows 侧（PowerShell）`nvidia-smi`：
```
NVIDIA-SMI 595.71    Driver Version: 595.71    CUDA Version: 13.2

GPU 0 (WDDM):  17MiB / 24564MiB   0% util
GPU 1 (WDDM): 934MiB / 24564MiB   0% util
Processes on GPU 1: ~19 个 Windows 桌面进程 (explorer.exe, chrome.exe, VS Code, WindowsTerminal 等，全部 C+G 类型)
```

**结论与建议**：
- 两张 GPU 型号、总显存（24564MiB）、驱动版本（595.71）在 WSL2 与 Windows 侧**完全一致**，显卡对 WSL2 完整可见，驱动模式确认为 **WDDM**（非 TCC，与已知信息一致）。
- WSL2 侧 `nvidia-smi` 工具自身版本号（595.45.03）与 Driver Version（595.71）不同——这是 WSL2 CUDA 兼容层已知的正常现象（`nvidia-smi` 二进制随 WSL 驱动栈打包，版本号可能滞后于实际驱动），**不是异常**，无需处理。
- **重要（影响 G3/G4 并发规划）**：GPU 1 是当前 Windows 桌面显示主显卡（Disp.A = On），常驻占用约 900MB 显存用于桌面合成（Windows Terminal、Chrome、VS Code 等 ~19 个进程）。GPU 0 几乎空闲（17MiB）。这意味着：
  - GPU 0 可用显存 ≈ 24547MiB，GPU 1 可用显存 ≈ 23630MiB（且会随 Windows 桌面活动波动）。
  - 后续 G3 单卡并发扫描时，两张卡的可用显存基线**不对称**，GPU 1 上能塞下的并发训练进程数可能略少于 GPU 0，调度器（G4）分配任务时不应假设两卡显存基线完全相同。
- `torch.cuda.get_device_properties()` 确认两卡 `total_memory=24563MB`，`multi_processor_count=64`（两卡型号、算力完全对称）。

---

## 3. Conda 环境 `fly_gsplat` 实际安装版本

| 库 | README.md 声称 | CLAUDE.md 声称 | **实测版本** |
|---|---|---|---|
| PyTorch | 2.1.2 | 2.0.1 | **2.1.2+cu118** |
| gsplat | 1.4.0 | — | **1.4.0+pt20cu118** |
| nerfstudio | 1.1.5 | — | **1.1.5** |

**结论**：`README.md` 的记录准确；**`CLAUDE.md` 里 PyTorch 2.0.1 是过期/错误信息**，实际装的是 2.1.2+cu118，建议后续更新 CLAUDE.md 予以修正（本次任务范围不含修改代码/文档，仅记录此发现）。

`gsplat` 的 wheel 标签 `+pt20cu118` 表明该 wheel 是针对 PyTorch 2.0 + CUDA 11.8 编译的，但实际运行环境是 PyTorch 2.1.2——`torch.cuda.is_available()` 返回 `True` 且能正确枚举 2 张 GPU，说明这个版本不匹配在当前环境下**未表现出运行时故障**，但如果 G2/G3 阶段出现难以解释的 CUDA kernel 层面报错，这是一个值得回头检查的线索。

`ns-train --help` 确认 `splatfacto-checkpoint` 方法已正确安装并可见（自定义方法，非 nerfstudio 内置）。

---

## 4. CUDA Toolkit 版本与驱动兼容性

```
=== nvcc --version（conda 环境内） ===
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
路径: /home/computer0/anaconda3/envs/fly_gsplat/bin/nvcc（conda 环境自带的 cudatoolkit，非系统级）

=== torch.version.cuda ===
11.8

=== 系统级 CUDA（/usr/local/，与 conda 环境无关）===
cuda -> cuda-12.6（symlink）
cuda-12
cuda-12.6
```

**结论**：
- conda 环境 `fly_gsplat` 内部自带独立的 CUDA 11.8 toolkit（`nvcc` 取自环境内 `bin/`），与系统级 `/usr/local/cuda-12.6` **互不干扰**（`CUDA_HOME` 环境变量未设置，说明当前 shell 不依赖系统级 CUDA）。
- 驱动版本 595.71 支持的 CUDA 上限是 13.2（向前兼容），而 conda 环境用的是 CUDA 11.8 运行时——**驱动新、环境用旧 CUDA runtime 属于标准的向后兼容场景**，`torch.cuda.is_available()==True` 且能枚举双卡，实测未见任何 warning 或报错，兼容性正常。
- 系统级存在 CUDA 12.6，但只要不设置 `CUDA_HOME` 指向它、且 conda 环境的 PATH 优先级正确（实测 `which nvcc` 命中的是 conda 环境内的 11.8），就不会被误用。这是一个环境卫生的潜在风险点（如果未来有人手动 `export CUDA_HOME=/usr/local/cuda`，会切换到 12.6 并可能与 gsplat 编译的 cu118 扩展不兼容），值得记录但当前无需处理。

---

## 5. 数据盘（X:）挂载情况

**结论：X: 盘目前未挂载到 WSL2。**

```
=== mount | grep drvfs ===
C:\ on /mnt/c type 9p (rw,noatime,aname=drvfs,...)
（无 X: 相关条目）

=== /mnt/x 目录状态 ===
存在但为空目录（drwxr-xr-x, 无内容）

=== /etc/fstab ===
# UNCONFIGURED FSTAB FOR BASE SYSTEM
（无任何自动挂载配置）

=== Windows 侧盘符确认（wmic logicaldisk） ===
C:  X:  Y:  Z:   （X: 在 Windows 下确实存在）

=== 尝试手动挂载 ===
sudo mount -t drvfs X: /mnt/x
→ 失败：当前会话下 sudo 需要交互式密码，无法在本次审计中静默完成挂载
```

**建议**：X: 盘在 Windows 侧存在且此前显然被使用过（`/mnt/x` 挂载点已预先创建），但当前 WSL2 会话里既没有开机自动挂载（`/etc/fstab` 为空），也没有手动挂载。如果后续 G2+ 阶段的数据集在 X: 盘上，需要用户手动执行一次（需要 sudo 密码）：
```bash
sudo mount -t drvfs X: /mnt/x
```
或在 `/etc/wsl.conf` 里加 `[automount]` 配置实现开机自动挂载。**本次未能测试 X: 盘的读取速度**（因为访问不了），这是本报告唯一未完成的量化项，需要用户提供 sudo 权限后补测。

### 已完成的替代读速量级测试（暖缓存，非严谨 benchmark，仅供量级参考）

用 50MB 随机数据文件测试 `dd write` + `cp` 拷贝耗时：

| 挂载点 | 类型 | 50MB 写入速度 | 50MB 拷贝(读)耗时 | 量级 |
|---|---|---|---|---|
| `/mnt/c`（drvfs, Windows C 盘） | 9p/drvfs | 67.1 MB/s | 0.504s | ~百 MB/s 量级 |
| `/`（原生 ext4，repo 所在盘 `/dev/sdd`） | ext4 | 341 MB/s | 0.071s | ~几百 MB/s 量级 |
| `/dev/shm` | tmpfs（内存） | 428 MB/s | 0.036s | 最快，内存速度 |

**注意**：这些数字都是刚写入后立即读取，页缓存（page cache）很可能是热的，不代表冷读性能，仅用于确认三类挂载点之间**数量级差异**——`drvfs`（Windows 文件系统跨 9p 协议访问）明显比原生 ext4 慢（约 5 倍），`/dev/shm` 最快。若数据集存放在 `/mnt/c` 或 `/mnt/x`（同为 drvfs），跑几千次训练时的 I/O 很可能成为瓶颈，建议 G2 耗时分解阶段专门测量数据加载耗时占比。

---

## 6. 项目仓库与磁盘布局

```
=== df -h . （repo 所在位置）===
Filesystem      Size  Used Avail Use% Mounted on
/dev/sdd       1007G  130G  826G  14% /
```

`fly_gsplat` 仓库位于 WSL2 的原生虚拟磁盘 `/dev/sdd`（ext4，挂载在 `/`），总容量 1007G，已用 130G，可用 826G。**这是原生文件系统，不是 drvfs**，I/O 性能应接近上表中 `/` 那一行的量级，是几类存储里较快的选项。

```
=== df -h 全量输出（关键行）===
none      14G  ...  /dev/shm       ← tmpfs，可用空间见下
/dev/sdd  1007G 130G 826G 14%  /
C:\       476G  347G 130G 73%  /mnt/c
```

---

## 7. `/dev/shm` 可用空间

```
Filesystem      Size  Used Avail Use% Mounted on
none             14G     0   14G   0% /dev/shm
```

**`/dev/shm` 当前完全空闲，可用 14G**（这是内存映射的 tmpfs，占用会计入 WSL2 的内存预算，即前述被 `.wslconfig` 限制到 28GB 的那部分内存，不是独立于内存的额外空间）。后续磁盘优化阶段若想用 `/dev/shm` 做临时数据/checkpoint 缓存，需注意它和训练进程共享同一块 28GB 内存上限，大量占用 `/dev/shm` 会挤压可用于训练进程的内存。

---

## 8. 汇总结论（给 G2/G3/G4 的基线要点）

1. **内存上限是 28GB（人为配置），不是 32GB**，当前可用约 19GB。G3 单卡并发扫描时，并发进程数的内存维度天花板要按 28GB 算，不是宿主机标称的 31.7GB。
2. **CPU 28 线程完整可见**，无 WSL2 限制，多进程并发在 CPU 侧没有额外约束。
3. **双 GPU 型号/驱动/显存在 WSL2 内完整可见**，与 Windows 侧一致；但 **GPU 1 因承担 Windows 桌面显示，常驻占用 ~900MB 显存**，两卡可用显存基线不完全对称，G4 调度器分配任务时应分别探测两卡实时可用显存，不要假设对称。
4. **torch/gsplat/nerfstudio 实际版本已核实**：2.1.2+cu118 / 1.4.0+pt20cu118 / 1.1.5，与 README.md 一致，**CLAUDE.md 里的 PyTorch 2.0.1 是过期信息**。
5. **驱动 595.71（支持 CUDA 13.2）与 conda 环境内 CUDA 11.8 runtime 之间未发现任何运行时 warning 或报错**，向后兼容正常。
6. **X: 盘当前未挂载**，需要用户提供 sudo 权限手动挂载或配置 `/etc/wsl.conf` automount 后才能测试其真实读取速度——**这是本报告唯一遗留的待测项**。
7. **存储速度量级**：`/dev/shm`（tmpfs）> 原生 ext4（仓库所在盘）> `/mnt/c` drvfs，约有 5-10 倍量级差异。若数据集在 drvfs 上，I/O 很可能是几千次训练跑批时的瓶颈来源之一，值得在 G2 耗时分解里专门测量。
