"""
samplers.py
G2耗时分解用的资源采样工具：GPU利用率流式采样(nvidia-smi -lms 常驻子进程)、
进程级CPU%/IO字节采样(psutil, 支持递归子进程)。纯外部观测，不侵入
generate_dataset/generate_hull/ns-train 的代码。
"""

import subprocess
import threading
import time
import psutil


class GPUSampler:
    """常驻 `nvidia-smi -lms` 子进程，逐行读取(不受被测子进程stdout缓冲影响)。"""

    def __init__(self, gpu_index: int = 0, interval_ms: int = 200):
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self.samples = []  # list of (t_rel_s, util_pct, mem_used_mib)
        self._proc = None
        self._thread = None
        self._t0 = None
        self._stop = False

    def start(self):
        self._t0 = time.perf_counter()
        self._stop = False
        self._proc = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                f"-lms", str(self.interval_ms),
                "-i", str(self.gpu_index),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        for line in self._proc.stdout:
            if self._stop:
                break
            t_rel = time.perf_counter() - self._t0
            try:
                util_s, mem_s = line.strip().split(",")
                self.samples.append((t_rel, float(util_s), float(mem_s)))
            except ValueError:
                continue

    def stop(self):
        self._stop = True
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return list(self.samples)


class ProcResourceSampler:
    """按pid(含递归子进程)采样CPU% (对单核百分比,可超100)。"""

    def __init__(self, pid: int, interval_s: float = 0.2):
        self.pid = pid
        self.interval_s = interval_s
        self.samples = []  # list of (t_rel_s, cpu_pct_sum)
        self._thread = None
        self._t0 = None
        self._stop = False

    def start(self):
        self._t0 = time.perf_counter()
        self._stop = False
        self._tracked = {}  # pid -> psutil.Process, persistent across polls so cpu_percent() has a baseline
        try:
            root = psutil.Process(self.pid)
            root.cpu_percent(interval=None)  # prime baseline, first read is always 0
            self._tracked[self.pid] = root
        except psutil.NoSuchProcess:
            pass
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            try:
                root = self._tracked.get(self.pid) or psutil.Process(self.pid)
                for child in root.children(recursive=True):
                    if child.pid not in self._tracked:
                        try:
                            child.cpu_percent(interval=None)  # prime new child's baseline
                            self._tracked[child.pid] = child
                        except (psutil.NoSuchProcess, psutil.ZombieProcess):
                            continue

                total = 0.0
                dead = []
                for pid, p in self._tracked.items():
                    try:
                        total += p.cpu_percent(interval=None)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        dead.append(pid)
                for pid in dead:
                    self._tracked.pop(pid, None)

                t_rel = time.perf_counter() - self._t0
                self.samples.append((t_rel, total))
            except psutil.NoSuchProcess:
                pass
            time.sleep(self.interval_s)

    def stop(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2)
        return list(self.samples)


def read_proc_io(pid: int) -> dict:
    """返回{read_chars, write_chars}，rchar/wchar在9p/drvfs挂载上也能统计到真实syscall字节数，
    不同于block-device级别的io_counters().read_bytes（对drvfs/9p恒为0）。"""
    try:
        p = psutil.Process(pid)
        io = p.io_counters()
        return {"read_chars": io.read_chars, "write_chars": io.write_chars}
    except (psutil.NoSuchProcess, AttributeError):
        return {"read_chars": None, "write_chars": None}


def gpu_busy_start_time(samples, util_threshold: float = 15.0, sustain: int = 2) -> float:
    """从GPU利用率曲线里找第一次"持续sustain个采样点都超过阈值"的时刻，
    近似认为这是真正的CUDA计算(第一个iteration)开始的时间点，
    用来把ns-train子进程耗时切成"冷启动(import/初始化)"和"训练循环"两段。
    找不到就返回None(调用方应回退为不切分)。"""
    n = len(samples)
    for i in range(n - sustain + 1):
        window = samples[i:i + sustain]
        if all(s[1] >= util_threshold for s in window):
            return window[0][0]
    return None
