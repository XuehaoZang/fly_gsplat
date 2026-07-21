"""
samplers_multi.py
G3并发扫描专用的多进程CPU%聚合采样器：G2的 gpu/timing/samplers.py::ProcResourceSampler
只支持单个根pid，这里扩展成支持一组根pid(每个并发worker一个)，各自递归追踪子进程
(worker本身 + 它fork出的ns-train/ns-export子进程)，把所有并发worker的CPU%加总，
用来判断"冷启动尖峰叠加后是否逼近/超过28线程上限"。

GPU侧不需要类似扩展：nvidia-smi查询的是整卡利用率，天然是所有并发进程的聚合值，
直接复用 gpu/timing/samplers.py::GPUSampler 即可。
"""
import threading
import time
import psutil


class MultiProcResourceSampler:
    """按多个根pid(各自含递归子进程)聚合采样CPU% (对单核百分比,可超100；
    超过2800%代表28个逻辑核全部跑满)。"""

    def __init__(self, root_pids: list, interval_s: float = 0.2):
        self.root_pids = list(root_pids)
        self.interval_s = interval_s
        self.samples = []  # list of (t_rel_s, cpu_pct_sum, n_tracked_procs)
        self._thread = None
        self._t0 = None
        self._stop = False
        self._tracked = {}

    def start(self):
        self._t0 = time.perf_counter()
        self._stop = False
        self._tracked = {}
        for pid in self.root_pids:
            try:
                proc = psutil.Process(pid)
                proc.cpu_percent(interval=None)  # prime baseline
                self._tracked[pid] = proc
            except psutil.NoSuchProcess:
                continue
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            try:
                newly_primed = set()
                for root_pid in self.root_pids:
                    root = self._tracked.get(root_pid)
                    if root is None:
                        try:
                            root = psutil.Process(root_pid)
                            root.cpu_percent(interval=None)
                            self._tracked[root_pid] = root
                            newly_primed.add(root_pid)
                        except psutil.NoSuchProcess:
                            continue
                    try:
                        for child in root.children(recursive=True):
                            if child.pid not in self._tracked:
                                try:
                                    child.cpu_percent(interval=None)
                                    self._tracked[child.pid] = child
                                    newly_primed.add(child.pid)
                                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                                    continue
                    except psutil.NoSuchProcess:
                        continue

                total = 0.0
                n_alive = 0
                dead = []
                for pid, p in self._tracked.items():
                    if pid in newly_primed:
                        # 刚在本轮discovery阶段第一次prime的进程：cpu_percent()的baseline
                        # 就是几微秒前设的，这一轮如果立刻拿来算sum，wall-time分母趋近于0，
                        # 而该进程可能已经用多线程跑了一段时间的CPU-time，会算出物理上不可能
                        # 的畸高瞬时值(实测见过>10000%，28线程机器物理上限是2800%)。
                        # 正确做法(遵循psutil文档): prime调用的返回值本身就该丢弃，
                        # 真正有意义的读数从下一轮迭代开始，所以这一轮先跳过，只统计计数。
                        n_alive += 1
                        continue
                    try:
                        total += p.cpu_percent(interval=None)
                        n_alive += 1
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        dead.append(pid)
                for pid in dead:
                    self._tracked.pop(pid, None)

                t_rel = time.perf_counter() - self._t0
                self.samples.append((t_rel, total, n_alive))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def stop(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2)
        return list(self.samples)
