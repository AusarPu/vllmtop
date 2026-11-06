#!/usr/bin/env python3
"""vllmtop — 本机所有 vLLM 实例的终端看板（TUI / 一次性快照 / CSV 存档）。

只读来源，不改任何 vLLM 配置：
- 发现：扫 /proc/net/tcp{,6} 的 LISTEN 端口，逐个探 /metrics，响应里有 ``vllm:`` 指标的才算实例；
  主进程再从 /proc/*/cmdline 里按 ``--port`` 对上，用于显示 PID / 启动参数，以及关停。
- 吞吐：间隔内累计 counter 差分；cache hit rate = Δprefix_cache_hits / Δprefix_cache_queries
  （同时给出启动以来的 lifetime 命中率，两者差别大说明近期流量结构变了）。
- 延迟：TTFT / ITL 用直方图的 ``_sum / _count`` 差分，得到区间平均值。
- GPU：nvidia-smi --query-gpu；每个实例的卡号优先取主进程环境里的 CUDA_VISIBLE_DEVICES。
  容器外其它任务占的显存没有可见进程，不归属到任何实例。
- CSV：每个实例每个采样一行，按天分文件 ``logs/vllmtop/vllm-<日期>.csv``；
  默认保留 7 天（--csv-days），总额超过上限时删最旧的文件（--csv-max-mb），不会无限增长。

用法::

    vllmtop              # 默认 TUI：q 退出 / r 重扫 / k 关停实例
    vllmtop --once       # 打一屏彩色快照就退出
    vllmtop --watch 2    # 原地刷新，不占终端全屏
    vllmtop --kill 8888  # 命令行关停（--yes 跳过确认）
    vllmtop --report     # 读 CSV 存档，输出分钟级汇总 + 吞吐曲线（跑完回看用）
    vllmtop --info       # 看实例运行配置（max_model_len / max_num_seqs / KV / 特性 / 启动命令）

只展示到“实例”这一层，不铺进程树明细。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------- 常量

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_DIR = SCRIPT_DIR / "logs" / "vllmtop"

# vLLM 的 counter / gauge / histogram 名（不同版本有别名，聚合时都认）
COUNTERS = {
    "prompt_tokens_total",
    "generation_tokens_total",
    "prefix_cache_hits_total",
    "prefix_cache_queries_total",
    "prompt_tokens_cached_total",
    "request_success_total",
    "external_prefix_cache_hits_total",
    "external_prefix_cache_queries_total",
}
HIST_SUM = {
    "time_to_first_token_seconds_sum",
    "time_to_first_token_seconds_count",
    "inter_token_latency_seconds_sum",
    "inter_token_latency_seconds_count",
}
GAUGE_SUM = {
    "num_requests_running": "running",
    "num_requests_waiting": "waiting",
}
KV_GAUGES = {"kv_cache_usage_perc", "gpu_cache_usage_perc"}

SPARK_BLOCKS = "▁▂▃▄▅▆▇█"
METRIC_LINE_RE = re.compile(
    r'^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>\S+)\s*$'
)
LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


# ---------------------------------------------------------------- 数据结构


@dataclass
class RawMetrics:
    """一次 /metrics 抓取聚合后的值（跨 engine 求和 / 求平均）。"""

    ts: float
    counters: dict[str, float] = field(default_factory=dict)
    running: float = 0.0
    waiting: float = 0.0
    kv_pcts: list[float] = field(default_factory=list)
    engines: set[str] = field(default_factory=set)
    models: set[str] = field(default_factory=set)
    prefix_caching: bool | None = None

    @property
    def kv_avg(self) -> float | None:
        return sum(self.kv_pcts) / len(self.kv_pcts) if self.kv_pcts else None


@dataclass
class ProcInfo:
    """一个 vllm serve 主进程。"""

    pid: int
    argv: list[str]
    port: int | None
    model: str | None
    dp: int | None
    tp: int | None
    gpus: list[int]
    started: float  # epoch 秒
    opts: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class GpuInfo:
    index: int
    util: float | None
    mem_used: float | None  # MiB
    mem_total: float | None  # MiB
    temp: float | None
    power: float | None


@dataclass
class View:
    """渲染 / 落 CSV 用的一行（一个实例）。"""

    port: int
    model: str
    dp: int | None
    tp: int | None
    gpus: list[int]
    gpu_util: float | None
    gpu_mem_used: float | None
    gpu_mem_total: float | None
    pid: int | None
    nproc: int
    uptime: float | None
    in_tps: float | None
    out_tps: float | None
    running: float
    waiting: float
    kv_pct: float | None
    cache_hit: float | None       # 本采样区间的命中率 %
    cache_life: float | None      # 启动以来的命中率 %
    ttft_ms: float | None
    itl_ms: float | None
    req_tps: float | None
    done_total: float
    max_model_len: int | None
    max_num_seqs: int | None
    ts: float
    prefix_caching: bool | None = None
    series_in: deque[float] = field(default_factory=lambda: deque(maxlen=600))
    series_out: deque[float] = field(default_factory=lambda: deque(maxlen=600))
    series_hit: deque[float] = field(default_factory=lambda: deque(maxlen=600))
    series_run: deque[float] = field(default_factory=lambda: deque(maxlen=600))


# ---------------------------------------------------------------- 发现


def _read_text(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def listen_ports() -> set[int]:
    """本网络命名空间里处于 LISTEN 的 TCP 端口（不依赖 ss/netstat）。"""
    ports: set[int] = set()
    for file in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in _read_text(file).splitlines()[1:]:
            parts = line.split()
            if len(parts) < 4 or parts[3] != "0A":  # 0A = LISTEN
                continue
            try:
                ports.add(int(parts[1].rsplit(":", 1)[1], 16))
            except (IndexError, ValueError):
                continue
    return ports


_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch_metrics(port: int, timeout: float = 1.0) -> str | None:
    """抓一个端口的 /metrics；不是 vLLM 就返回 None。"""
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/metrics",
            headers={"Accept-Encoding": "identity", "Connection": "close"},
        )
        with _opener.open(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        return None
    return body if "vllm:" in body else None


def parse_metrics(text: str, ts: float) -> RawMetrics | None:
    raw = RawMetrics(ts=ts)
    engines: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        m = METRIC_LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")[len("vllm:"):]
        labels = dict(LABEL_RE.findall(m.group("labels") or ""))
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        if value != value:  # NaN
            continue
        if labels.get("model_name"):
            raw.models.add(labels["model_name"])
        if labels.get("engine"):
            engines.add(labels["engine"])
        if name in COUNTERS or name in HIST_SUM:
            raw.counters[name] = raw.counters.get(name, 0.0) + value
        elif name in GAUGE_SUM:
            setattr(raw, GAUGE_SUM[name], getattr(raw, GAUGE_SUM[name]) + value)
        elif name in KV_GAUGES:
            raw.kv_pcts.append(value)
        elif name == "cache_config_info" and raw.prefix_caching is None:
            raw.prefix_caching = labels.get("enable_prefix_caching") == "True"
    if not engines and not raw.counters and not raw.kv_pcts:
        return None
    raw.engines = engines
    return raw


def _parse_serve_argv(argv: list[str]) -> dict[str, str]:
    opts: dict[str, str] = {}
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--"):
            body = arg[2:]
            if "=" in body:
                key, val = body.split("=", 1)
                opts[key] = val
                i += 1
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                opts[body] = argv[i + 1]
                i += 2
            else:
                opts[body] = "true"
                i += 1
        else:
            i += 1
    return opts


def _is_vllm_main(argv: list[str]) -> bool:
    joined = " ".join(argv)
    if "vllm" not in joined.lower() or "VLLM::" in joined:
        return False
    # 排除 bash -c 'vllm serve ...' 这种包装进程：它只该被当成 shell，不该当成实例主进程
    if os.path.basename(argv[0]) in {"bash", "sh", "zsh", "dash", "fish"}:
        return False
    return "serve" in joined.split() or "api_server" in joined or "vllm.entrypoints" in joined


def _boot_epoch() -> float:
    """开机时刻（epoch 秒），把 /proc/<pid>/stat 的 starttime 换成启动时间。"""
    parts = _read_text("/proc/uptime").split()
    try:
        return time.time() - float(parts[0])
    except (IndexError, ValueError):
        return time.time()


def _read_argv(pid: int) -> list[str]:
    return [p for p in _read_text(f"/proc/{pid}/cmdline").split("\0") if p]


def _read_environ(pid: int) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in _read_text(f"/proc/{pid}/environ").split("\0"):
        if "=" in item:
            key, val = item.split("=", 1)
            env[key] = val
    return env


def _pid_start_epoch(pid: int, boot_epoch: float) -> float:
    stat = _read_text(f"/proc/{pid}/stat")
    try:
        fields = stat.rsplit(")", 1)[1].split()
        start_ticks = float(fields[19])  # starttime（右括号后第 20 个字段）
        return boot_epoch + start_ticks / os.sysconf("SC_CLK_TCK")
    except (IndexError, ValueError, OSError):
        return time.time()


def ppid_map() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat = _read_text(f"/proc/{pid}/stat")
            parent = int(stat.rsplit(")", 1)[1].split()[1])
        except (IndexError, ValueError):
            continue
        children.setdefault(parent, []).append(pid)
    return children


def descendants(pid: int, children: dict[int, list[int]]) -> list[int]:
    out: list[int] = []
    stack = list(children.get(pid, []))
    while stack:
        cur = stack.pop()
        out.append(cur)
        stack.extend(children.get(cur, []))
    return out


def scan_main_procs() -> dict[int, ProcInfo]:
    """扫出所有 vllm serve 主进程，按它自己的 --port 建索引（默认 8000）。"""
    boot = _boot_epoch()
    result: dict[int, ProcInfo] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        argv = _read_argv(pid)
        if not argv or not _is_vllm_main(argv):
            continue
        opts = _parse_serve_argv(argv)
        env = _read_environ(pid)
        try:
            port = int(opts["port"])
        except (KeyError, ValueError):
            port = 8000
        gpus: list[int] = []
        for token in re.split(r"[,:]", env.get("CUDA_VISIBLE_DEVICES", "")):
            if token.strip().isdigit():
                gpus.append(int(token.strip()))
        model = opts.get("served-model-name") or opts.get("model") or None
        if model and "," in model:
            model = model.split(",")[0]
        try:
            dp = int(opts.get("data-parallel-size", "1"))
        except ValueError:
            dp = None
        try:
            tp = int(opts.get("tensor-parallel-size", "1"))
        except ValueError:
            tp = None
        result[port] = ProcInfo(
            pid=pid,
            argv=argv,
            port=port,
            model=model,
            dp=dp,
            tp=tp,
            gpus=gpus,
            started=_pid_start_epoch(pid, boot),
            opts=opts,
            env=env,
        )
    return result


# ---------------------------------------------------------------- GPU


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _smi(query: str) -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout


def query_gpus() -> dict[int, GpuInfo]:
    gpus: dict[int, GpuInfo] = {}
    for line in _smi("gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw").splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != 6:
            continue
        try:
            idx = int(cells[0])
        except ValueError:
            continue
        gpus[idx] = GpuInfo(idx, _to_float(cells[1]), _to_float(cells[2]),
                            _to_float(cells[3]), _to_float(cells[4]), _to_float(cells[5]))
    return gpus


def query_compute_apps() -> dict[int, str]:
    """pid -> gpu uuid；容器里只能看到本容器的进程。"""
    apps: dict[int, str] = {}
    for line in _smi("compute-apps=gpu_uuid,pid").splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) == 2 and cells[1].isdigit():
            apps[int(cells[1])] = cells[0]
    return apps


def query_gpu_uuids() -> dict[str, int]:
    mapping: dict[str, int] = {}
    for line in _smi("gpu=index,uuid").splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) == 2 and cells[0].isdigit():
            mapping[cells[1]] = int(cells[0])
    return mapping


def resolve_gpus(proc: ProcInfo | None, children: dict[int, list[int]]) -> list[int]:
    """实例占的卡号：优先主进程环境变量，退回到 worker 进程的显存占用。"""
    if proc is None:
        return []
    if proc.gpus:
        return proc.gpus
    apps = query_compute_apps()
    uuid_to_index = query_gpu_uuids()
    found: set[int] = set()
    for child in descendants(proc.pid, children):
        uuid = apps.get(child)
        if uuid and uuid in uuid_to_index:
            found.add(uuid_to_index[uuid])
    return sorted(found)


# ---------------------------------------------------------------- 实例配置


def _fetch_json(port: int, path: str, timeout: float = 1.5) -> dict | None:
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            headers={"Accept-Encoding": "identity", "Connection": "close"},
        )
        with _opener.open(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return data if isinstance(data, dict) else None
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        return None


def _fetch_cache_config(port: int) -> dict[str, str]:
    text = fetch_metrics(port)
    if not text:
        return {}
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info"):
            m = METRIC_LINE_RE.match(line)
            if m:
                return dict(LABEL_RE.findall(m.group("labels") or ""))
    return {}


def _parse_gen_config(text: str | None) -> dict:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def collect_instance_details(port: int) -> dict:
    """实例的运行配置：启动参数（/proc cmdline）+ 环境变量 + /metrics 的 cache_info + /v1/models + /version。

    配置是启动期固定的，主界面不展示；TUI 按 i 或 `--info` 时现拉。
    """
    proc = scan_main_procs().get(port)
    opts = proc.opts if proc else {}
    env = {k: v for k, v in (proc.env if proc else {}).items()
           if k == "CUDA_VISIBLE_DEVICES" or k.startswith("VLLM_")}
    cache = _fetch_cache_config(port)
    models = _fetch_json(port, "/v1/models") or {}
    model_list = models.get("data") or []
    model_info = model_list[0] if model_list and isinstance(model_list[0], dict) else {}
    version = (_fetch_json(port, "/version") or {}).get("version")
    gen_cfg = _parse_gen_config(opts.get("override-generation-config"))
    max_model_len = model_info.get("max_model_len")
    if max_model_len is None:
        try:
            max_model_len = int(opts["max-model-len"])
        except (KeyError, ValueError):
            max_model_len = None
    return {
        "port": port,
        "model": model_info.get("id") or (proc.model if proc else None) or "?",
        "root": model_info.get("root"),
        "pid": proc.pid if proc else None,
        "uptime": (time.time() - proc.started) if proc else None,
        "dp": proc.dp if proc else None,
        "tp": proc.tp if proc else None,
        "gpus": proc.gpus if proc else [],
        "host": opts.get("host"),
        "version": version,
        "max_model_len": max_model_len,
        "max_num_seqs": opts.get("max-num-seqs"),
        "api_server_count": opts.get("api-server-count"),
        "gpu_mem_util": opts.get("gpu-memory-utilization") or cache.get("gpu_memory_utilization"),
        "kv_dtype": opts.get("kv-cache-dtype") or cache.get("cache_dtype"),
        "block_size": cache.get("block_size"),
        "kv_size_tokens": cache.get("kv_cache_size_tokens"),
        "kv_max_concurrency": cache.get("kv_cache_max_concurrency"),
        "num_gpu_blocks": cache.get("num_gpu_blocks"),
        "quantization": opts.get("quantization"),
        "prefix_caching": opts.get("enable-prefix-caching") or cache.get("enable_prefix_caching"),
        "chunked_prefill": opts.get("enable-chunked-prefill"),
        "tool_call_parser": opts.get("tool-call-parser"),
        "reasoning_parser": opts.get("reasoning-parser"),
        "auto_tool_choice": opts.get("enable-auto-tool-choice"),
        "trust_remote_code": opts.get("trust-remote-code"),
        "language_model_only": opts.get("language-model-only"),
        "max_new_tokens": gen_cfg.get("max_new_tokens"),
        "env": env,
        "argv": proc.argv if proc else [],
    }


def _fmt_int(value: object) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "—"


def _fmt_float2(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


def _flag(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        low = value.lower()
        if low in {"true", "1", "on", "yes"}:
            return "on"
        if low in {"false", "0", "off", "no"}:
            return "off"
        return value
    return "on" if value else "off"


def render_instance_details(details_list: list[dict]) -> "Group":
    """把实例配置渲染成面板；TUI 按 i 的浮层和 `--info` 共用。"""
    from rich import box
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    parts: list[object] = []
    for d in details_list:
        title = Text()
        title.append(f":{d['port']} ", style="bold cyan")
        title.append(str(d["model"]), style="bold")
        title.append("   运行配置", style="dim")

        grid = Table.grid(padding=(0, 2), expand=True)
        for _ in range(2):
            grid.add_column(style="dim", no_wrap=True)
            grid.add_column(style="bold")

        def blank() -> None:
            grid.add_row(Text(""), Text(""), Text(""), Text(""))

        def section(name: str) -> None:
            blank()
            grid.add_row(Text(name, style="bold cyan"), Text(""), Text(""), Text(""))

        def row(*pairs: tuple[str, object]) -> None:
            cells: list[Text] = []
            for key, value in pairs:
                cells.append(Text(key, style="dim"))
                if key == "" and value in (None, ""):
                    cells.append(Text(""))  # 占位空格，不画横线
                elif isinstance(value, Text):
                    cells.append(value)
                else:
                    text = "—" if value in (None, "") else str(value)
                    cells.append(Text(text, style="bold"))
            while len(cells) < 4:
                cells.append(Text(""))
            grid.add_row(*cells[:4])

        section("运行")
        row(("PID", d["pid"]), ("up", _fmt_duration(d["uptime"])))
        row(("DP×TP", f"{(d['dp'] or '?')}×{(d['tp'] or '?')}"),
            ("卡", ",".join(map(str, d["gpus"])) or "—"))
        row(("host", d["host"]), ("vLLM", d["version"]))

        section("模型 / 限制")
        row(("max_model_len", _fmt_int(d["max_model_len"])), ("max_num_seqs", d["max_num_seqs"]))
        row(("api_server_count", d["api_server_count"]), ("gpu_mem_util", d["gpu_mem_util"]))

        section("KV / 显存")
        row(("kv_cache_dtype", d["kv_dtype"]), ("block_size", _fmt_int(d["block_size"])))
        row(("kv_cache_size", f"{_fmt_int(d['kv_size_tokens'])} tok"),
            ("kv_max_concurrency", _fmt_float2(d["kv_max_concurrency"])))
        row(("num_gpu_blocks", _fmt_int(d["num_gpu_blocks"])), ("", ""))

        section("特性")
        row(("quantization", d["quantization"]), ("prefix_caching", _flag(d["prefix_caching"])))
        row(("chunked_prefill", _flag(d["chunked_prefill"])), ("auto_tool_choice", _flag(d["auto_tool_choice"])))
        row(("tool_parser", d["tool_call_parser"]), ("reasoning_parser", d["reasoning_parser"]))
        row(("trust_remote_code", _flag(d["trust_remote_code"])),
            ("language_model_only", _flag(d["language_model_only"])))
        row(("max_new_tokens", _fmt_int(d["max_new_tokens"])), ("", ""))

        lines: list[object] = [grid]
        if d.get("root"):
            lines.append(Text(""))
            lines.append(Text.assemble(("权重目录  ", "dim"), (str(d["root"]), "")))
        env = d.get("env") or {}
        if env:
            lines.append(Text(""))
            lines.append(Text.assemble(("环境      ", "dim"),
                                       ("   ".join(f"{k}={v}" for k, v in env.items()), "")))
        argv = d.get("argv") or []
        if argv:
            lines.append(Text(""))
            lines.append(Text.assemble(("启动命令  ", "dim"), (" ".join(argv), "dim")))
        parts.append(Panel(Group(*lines), title=title, title_align="left",
                           border_style="grey37", box=box.ROUNDED, expand=True))
    return Group(*parts)


# ---------------------------------------------------------------- 采集


class Collector:
    """维护“端口 → 上一次指标快照”，每个采样周期输出一组 View。"""

    def __init__(
        self,
        interval: float = 2.0,
        rescan_every: float = 30.0,
        csv_logger: "CsvLogger | None" = None,
    ) -> None:
        self.interval = interval
        self.rescan_every = rescan_every
        self.csv = csv_logger
        self.known_ports: set[int] = set()
        self.prev: dict[int, RawMetrics] = {}
        self.procs: dict[int, ProcInfo] = {}
        self.gpus: dict[int, GpuInfo] = {}
        self.series: dict[int, dict[str, deque[float]]] = {}
        self.config_cache: dict[int, tuple[int | None, int | None]] = {}
        self.last_scan = 0.0
        self.proc_scanned_at = 0.0
        self.last_views: list[View] = []
        self.scanned_ports = 0
        self.errors: list[str] = []
        self._rescan_requested = True

    def request_rescan(self) -> None:
        self._rescan_requested = True

    def sample(self, log_csv: bool = True) -> list[View]:
        now = time.time()
        if self._rescan_requested or now - self.last_scan >= self.rescan_every:
            self._discover()
        self.errors = []
        raw_by_port: dict[int, RawMetrics] = {}
        for port in sorted(self.known_ports):
            text = fetch_metrics(port)
            if text is None:
                self.known_ports.discard(port)  # 实例停了 / 换端口：等下一轮重扫
                continue
            parsed = parse_metrics(text, now)
            if parsed is not None:
                raw_by_port[port] = parsed
        self.gpus = query_gpus()
        if not self.gpus:
            self.errors.append("nvidia-smi 不可用")
        children = ppid_map()
        views = [self._build_view(port, raw, children) for port, raw in raw_by_port.items()]
        views.sort(key=lambda v: v.port)
        self.last_views = views
        if self.csv is not None and log_csv:
            try:
                self.csv.log(views)
            except OSError as exc:
                self.errors.append(f"CSV 写入失败: {exc}")
        return views

    # -- 内部
    def _discover(self) -> None:
        self.last_scan = time.time()
        self._rescan_requested = False
        ports = listen_ports()
        self.scanned_ports = len(ports)
        found: set[int] = set()
        if ports:
            with ThreadPoolExecutor(max_workers=min(16, len(ports))) as pool:
                for port, text in zip(sorted(ports), pool.map(fetch_metrics, sorted(ports))):
                    if text is not None:
                        found.add(port)
        self.known_ports = found
        self.config_cache.clear()
        if time.time() - self.proc_scanned_at > 10 or not self.procs:
            self.procs = scan_main_procs()
            self.proc_scanned_at = time.time()

    def _config_for(self, port: int) -> tuple[int | None, int | None]:
        """启动后固定的两项配置：max_model_len（优先 /v1/models，退回启动参数）与 max_num_seqs。"""
        if port not in self.config_cache:
            proc = self.procs.get(port)
            opts = proc.opts if proc else {}
            models = _fetch_json(port, "/v1/models") or {}
            model_list = models.get("data") or []
            info = model_list[0] if model_list and isinstance(model_list[0], dict) else {}
            max_model_len = info.get("max_model_len")
            if max_model_len is None:
                try:
                    max_model_len = int(opts["max-model-len"])
                except (KeyError, ValueError):
                    max_model_len = None
            try:
                max_num_seqs = int(opts["max-num-seqs"])
            except (KeyError, ValueError):
                max_num_seqs = None
            self.config_cache[port] = (max_model_len, max_num_seqs)
        return self.config_cache[port]

    def _build_view(self, port: int, raw: RawMetrics, children: dict[int, list[int]]) -> View:
        proc = self.procs.get(port)
        prev = self.prev.get(port)
        counters = raw.counters
        max_model_len, max_num_seqs = self._config_for(port)

        def delta(name: str) -> float | None:
            if prev is None:
                return None
            change = counters.get(name, 0.0) - prev.counters.get(name, 0.0)
            return change if change > 0 else 0.0  # counter 回退（实例重启）当 0

        def rate(name: str) -> float | None:
            change = delta(name)
            if change is None or prev is None:
                return None
            dt = raw.ts - prev.ts
            return change / dt if dt > 0 else None

        def interval_avg_ms(sum_name: str, count_name: str) -> float | None:
            total = delta(sum_name)
            count = delta(count_name)
            return total / count * 1000 if total is not None and count and count > 0 else None

        cache_hit = None
        hit_delta = delta("prefix_cache_hits_total")
        query_delta = delta("prefix_cache_queries_total")
        if hit_delta is not None and query_delta is not None and query_delta > 0:
            cache_hit = hit_delta / query_delta * 100
        hits_total = counters.get("prefix_cache_hits_total", 0.0)
        queries_total = counters.get("prefix_cache_queries_total", 0.0)
        cache_life = hits_total / queries_total * 100 if queries_total > 0 else None

        gpus = resolve_gpus(proc, children)
        infos = [self.gpus[g] for g in gpus if g in self.gpus]
        utils = [i.util for i in infos if i.util is not None]
        mems = [i.mem_used for i in infos if i.mem_used is not None]
        totals = [i.mem_total for i in infos if i.mem_total is not None]

        view = View(
            port=port,
            model=(proc.model if proc and proc.model else next(iter(sorted(raw.models)), "?")),
            dp=(proc.dp if proc and proc.dp else (len(raw.engines) or None)),
            tp=(proc.tp if proc else None),
            gpus=gpus,
            gpu_util=(sum(utils) / len(utils)) if utils else None,
            gpu_mem_used=sum(mems) if mems else None,
            gpu_mem_total=sum(totals) if totals else None,
            pid=proc.pid if proc else None,
            nproc=(len(descendants(proc.pid, children)) + 1) if proc else 1,
            uptime=(time.time() - proc.started) if proc else None,
            in_tps=rate("prompt_tokens_total"),
            out_tps=rate("generation_tokens_total"),
            running=raw.running,
            waiting=raw.waiting,
            kv_pct=(raw.kv_avg * 100) if raw.kv_avg is not None else None,
            cache_hit=cache_hit,
            cache_life=cache_life,
            ttft_ms=interval_avg_ms("time_to_first_token_seconds_sum", "time_to_first_token_seconds_count"),
            itl_ms=interval_avg_ms("inter_token_latency_seconds_sum", "inter_token_latency_seconds_count"),
            req_tps=rate("request_success_total"),
            done_total=counters.get("request_success_total", 0.0),
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            ts=raw.ts,
            prefix_caching=raw.prefix_caching,
        )
        # 曲线历史跨采样保留（600 点 ≈ 2s 间隔下 20 分钟）
        hist = self.series.setdefault(port, {
            "in": deque(maxlen=600), "out": deque(maxlen=600),
            "hit": deque(maxlen=600), "run": deque(maxlen=600),
        })
        if view.in_tps is not None:
            hist["in"].append(view.in_tps)
        if view.out_tps is not None:
            hist["out"].append(view.out_tps)
        if view.cache_hit is not None:
            hist["hit"].append(view.cache_hit)
        hist["run"].append(view.running)
        view.series_in, view.series_out = hist["in"], hist["out"]
        view.series_hit, view.series_run = hist["hit"], hist["run"]
        self.prev[port] = raw
        return view


# ---------------------------------------------------------------- CSV 存档


class CsvLogger:
    """按天分文件的采样存档；保留 N 天 + 总容量上限，不会无限长。"""

    FIELDS = [
        "ts", "port", "model", "dp", "tp", "gpus",
        "in_tps", "out_tps", "running", "waiting", "kv_pct",
        "cache_hit_pct", "cache_life_pct", "ttft_ms", "itl_ms",
        "pid", "uptime_s", "req_tps", "done_total", "max_model_len", "max_num_seqs",
    ]

    def __init__(self, directory: Path, days: int = 7, max_mb: float = 200) -> None:
        self.directory = Path(directory)
        self.days = days
        self.max_bytes = int(max_mb * 1024 * 1024)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._pruned_for_day: str | None = None
        self._make_current(self._path())
        self._prune()

    def _path(self, ts: float | None = None) -> Path:
        day = datetime.fromtimestamp(ts if ts is not None else time.time()).strftime("%Y-%m-%d")
        return self.directory / f"vllm-{day}.csv"

    def _read_header(self, path: Path) -> list[str] | None:
        try:
            with path.open(newline="", encoding="utf-8") as f:
                return next(csv.reader(f), None)
        except OSError:
            return None

    def _make_current(self, path: Path) -> None:
        """当天文件必须是当前列格式；旧格式先轮转成 -vN.csv 保留，已采到的数据不丢。"""
        header = self._read_header(path) if path.exists() else None
        if header is not None and header != self.FIELDS:
            n = 1
            while True:
                target = path.with_name(f"{path.stem}-v{n}.csv")
                if not target.exists():
                    break
                n += 1
            path.rename(target)
            header = None
        if header != self.FIELDS:
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.FIELDS)

    def log(self, views: Iterable[View]) -> None:
        views = list(views)
        if not views:
            return
        now = time.time()
        path = self._path(now)
        if self._pruned_for_day != path.stem:
            self._make_current(path)
            self._prune()
            self._pruned_for_day = path.stem
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for v in views:
                writer.writerow([
                    datetime.fromtimestamp(v.ts).strftime("%Y-%m-%d %H:%M:%S"),
                    v.port, v.model, v.dp, v.tp, ",".join(map(str, v.gpus)),
                    _round(v.in_tps), _round(v.out_tps), v.running, v.waiting, _round(v.kv_pct),
                    _round(v.cache_hit), _round(v.cache_life), _round(v.ttft_ms), _round(v.itl_ms),
                    v.pid, _round(v.uptime, 0), _round(v.req_tps), _round(v.done_total, 0),
                    _csv_int(v.max_model_len), _csv_int(v.max_num_seqs),
                ])
            f.flush()

    def _prune(self) -> None:
        """按保留天数 + 总容量上限清旧文件；当前写的这份永远不删。"""
        files = sorted(self.directory.glob("vllm-*.csv"))
        cutoff = time.time() - self.days * 86400
        for path in files:
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue
        files = [p for p in sorted(self.directory.glob("vllm-*.csv")) if p.exists()]
        current = self._path()
        total = sum(p.stat().st_size for p in files)
        remaining = len(files)
        for path in files:  # 从最旧开始删，直到总量回到上限内
            if total <= self.max_bytes or remaining <= 1:
                break
            if path == current:  # 当前这天的不删，等它自然变旧
                continue
            try:
                size = path.stat().st_size
                path.unlink()
                total -= size
                remaining -= 1
            except OSError:
                continue

    def describe(self) -> str:
        return f"{self.directory}/{self._path().name} · 保留 {self.days} 天"


def _round(value: float | None, digits: int = 1) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _csv_int(value: int | None) -> str:
    return "" if value is None else str(int(value))


# ---------------------------------------------------------------- 渲染


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def _fmt_count(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(max(seconds, 0))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d{hours}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def sparkline(values: Iterable[float], width: int = 30) -> str:
    vals = [float(v) for v in values][-width:]
    if not vals:
        return "—"
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return SPARK_BLOCKS[3] * len(vals)
    return "".join(SPARK_BLOCKS[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


PARTIAL_BLOCKS = "▁▂▃▄▅▆▇"


def area_graph(values: Iterable[float], width: int, height: int, style: str = "green") -> list["Text"]:
    """把一串数值画成 height 行的面积图（min-max 缩放、拉伸铺满宽度），给单实例时的剩余高度填内容。"""
    from rich.text import Text

    if height <= 0:
        return []
    vals = [float(v) for v in values if v is not None][-width:]
    if not vals:
        return [Text("") for _ in range(height)]
    if len(vals) != width:
        # 历史点不足宽度时拉伸画满（时间轴按当前历史自适应），避免单实例时左边一大片空白
        idx = [int(round(i * (len(vals) - 1) / max(1, width - 1))) for i in range(width)]
        vals = [vals[i] for i in idx]
    lo, hi = min(vals), max(vals)
    units_total = height * 8
    if hi <= 0:
        units = [0] * len(vals)
    elif hi - lo < 1e-9:
        units = [units_total // 2] * len(vals)  # 平线画中间
    else:
        units = [1 + int(round((v - lo) / (hi - lo) * (units_total - 1))) for v in vals]
    pad = max(0, width - len(units))
    units = [None] * pad + units
    # 填充用暗色、轮廓用亮色（像 nvitop 的面积图那样，整块纯色会太重）
    edge_style = "bold bright_green" if style == "green" else style
    fill_style = "dim green" if style == "green" else f"dim {style}"
    lines: list[Text] = []
    for row in range(height - 1, -1, -1):
        text = Text()
        base = row * 8
        for unit in units:
            if unit is None or unit <= base:
                text.append(" ")
            elif unit >= base + 8:
                text.append("█", style=edge_style if unit <= base + 8 else fill_style)
            else:
                text.append(PARTIAL_BLOCKS[unit - base - 1], style=edge_style)
        lines.append(text)
    return lines


def render_header(views: list[View], meta: dict) -> "Text":
    from rich.text import Text

    header = Text()
    header.append(" vllmtop ", style="bold reverse")
    header.append(f" {meta.get('host', '')}", style="bold")
    header.append(f"  {meta.get('clock', '')}", style="dim")
    header.append(f"  {len(views)} 实例", style="bold cyan" if views else "bold yellow")
    header.append(f"  采样 {meta.get('interval', 2.0):.1f}s", style="dim")
    return header


def _cache_text(v: View, text: "Text") -> None:
    """cache 命中率：区间值 + lifetime 值（未启用 prefix caching 的实例直接说明）。"""
    text.append("cache ", style="dim")
    if v.prefix_caching is False:
        text.append("未启用", style="dim")
        return
    if v.cache_hit is None:
        text.append("—", style="dim")
        return
    color = "green" if v.cache_hit >= 30 else "yellow" if v.cache_hit >= 10 else "red"
    text.append(f"{v.cache_hit:.1f}%", style=color)
    if v.cache_life is not None:
        text.append(f" (life {v.cache_life:.1f}%)", style="dim")


def render_instance(v: View, graph_height: int = 0, graph_width: int = 80) -> "Panel":
    """一个实例的面板；graph_height > 0 时把 out tok/s 历史面积图画满剩余高度。"""
    from rich import box
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    title = Text()
    title.append(f":{v.port} ", style="bold cyan")
    title.append(v.model, style="bold")
    if v.dp:
        title.append(f"  DP{v.dp}", style="magenta")
        if v.tp:
            title.append(f"×TP{v.tp}", style="magenta")
    if v.pid is None:
        title.append("  未对上主进程", style="yellow")
    if graph_height > 0:
        vals = list(v.series_out)[-graph_width:]
        if vals:
            lo, hi = min(vals), max(vals)
            title.append(f"   ▸ out tok/s {_fmt_rate(lo)}–{_fmt_rate(hi)}", style="dim green")
        else:
            title.append("   ▸ out tok/s", style="dim green")

    # 第 1 行：吞吐 / 队列 / KV
    line1 = Text()
    line1.append("in ", style="dim")
    line1.append(f"{_fmt_rate(v.in_tps)} tok/s", style="bold cyan")
    line1.append("   out ", style="dim")
    line1.append(f"{_fmt_rate(v.out_tps)} tok/s", style="bold green")
    line1.append("   run ", style="dim")
    line1.append(f"{v.running:.0f}", style="bold")
    line1.append("  wait ", style="dim")
    line1.append(f"{v.waiting:.0f}", style="bold yellow" if v.waiting else "dim")
    line1.append("   KV ", style="dim")
    if v.kv_pct is None:
        line1.append("—", style="dim")
    else:
        pct = max(0.0, min(100.0, v.kv_pct))
        color = "green" if pct < 60 else "yellow" if pct < 85 else "red"
        filled = int(round(pct / 100 * 10))
        line1.append("█" * filled + "░" * (10 - filled), style=color)
        line1.append(f" {pct:.0f}%", style=color)

    # 第 2 行：cache 命中率 / TTFT / ITL
    line2 = Text()
    _cache_text(v, line2)
    line2.append("   TTFT ", style="dim")
    line2.append(f"{v.ttft_ms:.0f}ms" if v.ttft_ms is not None else "—", style="dim")
    line2.append("   ITL ", style="dim")
    line2.append(f"{v.itl_ms:.1f}ms" if v.itl_ms is not None else "—", style="dim")
    line2.append("   done ", style="dim")
    line2.append(_fmt_count(v.done_total), style="bold")
    line2.append(f"  {v.req_tps:.1f}/s" if v.req_tps is not None else "  —", style="cyan")

    parts: list[object] = [line1, line2]
    if graph_height > 0:  # 面积图占满剩余高度
        parts.extend(area_graph(v.series_out, max(10, graph_width), graph_height, style="green"))
        parts.append(Text(""))

    # 底部一行：in / cache / run 的小曲线
    slim = Text()
    slim.append("in ", style="dim cyan")
    slim.append(sparkline(v.series_in, 14), style="cyan")
    slim.append("   cache ", style="dim yellow")
    slim.append(sparkline(v.series_hit, 12), style="yellow")
    slim.append("   run ", style="dim")
    slim.append(sparkline(v.series_run, 8), style="dim")
    parts.append(slim)

    # 静态元信息
    meta = Text(style="dim")
    meta.append(f"GPU {','.join(map(str, v.gpus)) or '—'}")
    if v.gpu_util is not None:
        meta.append(f"  {v.gpu_util:.0f}%")
    if v.gpu_mem_used is not None and v.gpu_mem_total:
        meta.append(f"  {v.gpu_mem_used / 1024:.0f}/{v.gpu_mem_total / 1024:.0f}G")
    meta.append(f"  {v.nproc} 进程")
    meta.append(f"  {_fmt_duration(v.uptime)}")
    if v.pid:
        meta.append(f"  PID {v.pid}")
    parts.append(meta)

    panel_height = None
    if graph_height > 0:
        # 内容 4 行 + 图 graph_height 行 + 空行 1 行，再加边框 2 行
        panel_height = graph_height + 7
    return Panel(Group(*parts), title=title, title_align="left", border_style="grey37",
                 box=box.ROUNDED, expand=True, height=panel_height)


def render_body(views: list[View], meta: dict) -> "Group":
    from rich.console import Group
    from rich.text import Text

    if not views:
        hint = Text()
        hint.append("没发现 vLLM 实例", style="yellow")
        hint.append(f"（扫了 {meta.get('scanned_ports', 0)} 个监听端口，均无 vllm: 指标）", style="dim")
        return Group(hint)
    n = len(views)
    body_height = int(meta.get("body_height") or 0)
    body_width = int(meta.get("body_width") or 0)
    if body_height > 0:
        # 实例少时每块分更多高度：单实例把整屏剩余高度画成面积图，实例多了按数量均分（余数给前几块）
        base, extra = divmod(body_height, n)
        budgets = [max(6, base + (1 if i < extra else 0)) for i in range(n)]
        graph_heights = [max(0, budget - 7) for budget in budgets]
    else:
        graph_heights = [1] * n  # --once 这种非全屏模式保持紧凑
    graph_width = max(10, (body_width or 120) - 6)
    total_in = sum(v.in_tps or 0 for v in views)
    total_out = sum(v.out_tps or 0 for v in views)
    totals = Text(style="dim")
    totals.append(f"合计 in {_fmt_rate(total_in)} tok/s · out {_fmt_rate(total_out)} tok/s", style="dim")
    return Group(totals, *(render_instance(v, gh, graph_width) for v, gh in zip(views, graph_heights)))


def render_status(meta: dict) -> "Text":
    from rich.text import Text

    status = Text(style="dim")
    if meta.get("tui"):
        status.append("q 退出  r 重扫  k 关停实例  i 实例配置", style="bold")
    else:
        status.append("TUI 下：q 退出  r 重扫  k 关停实例  i 实例配置", style="dim")
    if meta.get("csv"):
        status.append(f"   CSV {meta['csv']}", style="dim")
    if meta.get("errors"):
        status.append(f"   ⚠ {'；'.join(meta['errors'])}", style="yellow")
    return status


def render_dashboard(views: list[View], meta: dict) -> "Group":
    from rich.console import Group

    return Group(render_header(views, meta), render_body(views, meta), render_status(meta))


# ---------------------------------------------------------------- TUI


def build_tui(collector: Collector, csv_logger: CsvLogger | None) -> "App":
    from rich.console import Group
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.containers import Vertical, VerticalScroll
    from textual.screen import ModalScreen
    from textual.widgets import Button, Static

    class ConfirmKill(ModalScreen[bool]):
        BINDINGS = [("escape", "dismiss_false", "取消"), ("n", "dismiss_false", "取消")]

        def __init__(self, port: int, model: str, pid: int | None) -> None:
            super().__init__()
            self._port = port
            self._model = model
            self._pid = pid

        def compose(self) -> ComposeResult:
            with Vertical(classes="dialog"):
                yield Static(
                    f"关停 :{self._port} {self._model}（PID {self._pid}）？\n"
                    "会向主进程发 SIGTERM，显存与端口随后释放。",
                    markup=False,
                )
                yield Button("确认关停", variant="error", id="yes")
                yield Button("取消", variant="default", id="no")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            self.dismiss(event.button.id == "yes")

        def action_dismiss_false(self) -> None:
            self.dismiss(False)

    class KillSelect(ModalScreen[int | None]):
        BINDINGS = [("escape", "dismiss_none", "取消")]

        def __init__(self, views: list[View]) -> None:
            super().__init__()
            self._views = views

        def compose(self) -> ComposeResult:
            with Vertical(classes="dialog"):
                yield Static("选择要关停的实例（Esc 取消）", markup=False)
                for v in self._views:
                    yield Button(f":{v.port}  {v.model}  PID {v.pid}", id=f"kill-{v.port}",
                                 variant="error")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            button_id = event.button.id or ""
            if button_id.startswith("kill-"):
                self.dismiss(int(button_id[len("kill-"):]))
            else:
                self.dismiss(None)

        def action_dismiss_none(self) -> None:
            self.dismiss(None)

    class InfoScreen(ModalScreen[None]):
        BINDINGS = [("escape", "close", "关闭"), ("i", "close", "关闭"), ("q", "close", "关闭")]

        def __init__(self, details_list: list[dict]) -> None:
            super().__init__()
            self._details = details_list

        def compose(self) -> ComposeResult:
            with VerticalScroll(classes="info-dialog"):
                yield Static(render_instance_details(self._details))

        def action_close(self) -> None:
            self.dismiss(None)

    class VllmtopApp(App):
        TITLE = "vllmtop"
        CSS = """
        Screen { layout: vertical; }
        #title { height: 1; padding: 0 1; background: $boost; }
        #scroll { height: 1fr; }
        #body { padding: 0 1; }
        #status { height: 1; padding: 0 1; color: $text-muted; }
        ModalScreen { align: center middle; }
        .dialog {
            width: auto; max-width: 80;
            height: auto;
            border: thick $error;
            padding: 1 2;
            background: $surface;
        }
        .dialog Button { margin-top: 1; width: 100%; }
        .info-dialog {
            width: 92%;
            height: 85%;
            border: thick $accent;
            padding: 1 2;
            background: $surface;
        }
        """
        BINDINGS = [("q", "quit", "退出"), ("r", "rescan", "重扫"), ("k", "kill", "关停实例"),
                    ("i", "info", "实例配置")]

        def __init__(self) -> None:
            super().__init__()
            self._collector = collector
            self._csv = csv_logger
            self._task: asyncio.Task | None = None
            self._views: list[View] = []
            self._kill_port: int | None = None
            self._error: str = ""

        def compose(self) -> ComposeResult:
            yield Static("", id="title")
            with VerticalScroll(id="scroll"):
                yield Static("", id="body")
            yield Static("", id="status")

        def on_mount(self) -> None:
            self._task = asyncio.create_task(self._sampler())

        async def on_unmount(self) -> None:
            if self._task is not None:
                self._task.cancel()

        async def _sampler(self) -> None:
            try:
                await asyncio.to_thread(self._collector.sample, False)  # 基线，首个间隔后才有速率
                while True:
                    await asyncio.sleep(self._collector.interval)
                    self._views = await asyncio.to_thread(self._collector.sample)
                    self._paint()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 采样线程异常不该让 TUI 整个挂掉
                self._error = f"采样异常: {exc}"
                self._paint()

        def _meta(self) -> dict:
            errors = ([self._error] if self._error else []) + self._collector.errors
            return {
                "host": os.uname().nodename,
                "clock": datetime.now().strftime("%H:%M:%S"),
                "interval": self._collector.interval,
                "scanned_ports": self._collector.scanned_ports,
                "csv": self._csv.describe() if self._csv else "",
                "errors": errors,
                "tui": True,
                "body_height": max(4, self.size.height - 3),
                "body_width": max(20, self.size.width),
            }

        def on_resize(self, event) -> None:  # noqa: ANN001 - Textual 事件
            if self._views:
                self._paint()

        def _paint(self) -> None:
            meta = self._meta()
            self.query_one("#title", Static).update(render_header(self._views, meta))
            self.query_one("#body", Static).update(render_body(self._views, meta))
            self.query_one("#status", Static).update(render_status(meta))

        def action_rescan(self) -> None:
            self._collector.request_rescan()
            self.notify("已请求重扫监听端口", timeout=1.5)

        def action_kill(self) -> None:
            if not self._views:
                self.notify("没有可关停的实例", severity="warning", timeout=2)
                return
            self.push_screen(KillSelect(self._views), self._after_select)

        def action_info(self) -> None:
            if not self._views:
                self.notify("没有可查看的实例", severity="warning", timeout=2)
                return
            details = [collect_instance_details(v.port) for v in self._views]
            self.push_screen(InfoScreen(details))

        def _after_select(self, port: int | None) -> None:
            if port is None:
                return
            view = next((v for v in self._views if v.port == port), None)
            if view is None:
                return
            self._kill_port = port
            self.push_screen(ConfirmKill(view.port, view.model, view.pid), self._after_confirm)

        def _after_confirm(self, confirmed: bool | None) -> None:
            port = self._kill_port
            self._kill_port = None
            if not confirmed or port is None:
                return
            ok, message = kill_instance(port)
            if ok:
                self.notify(message, timeout=3)
                self._collector.request_rescan()
            else:
                self.notify(message, severity="error", timeout=5)

    return VllmtopApp()


# ---------------------------------------------------------------- 关停


def kill_instance(port: int) -> tuple[bool, str]:
    """只关停本工具认得出的 vllm serve 主进程，只发 SIGTERM。"""
    info = scan_main_procs().get(port)
    if info is None:
        return False, f"找不到 :{port} 对应的 vllm serve 主进程，未做任何操作"
    if not Path(f"/proc/{info.pid}").exists():
        return False, f"PID {info.pid} 已不存在"
    try:
        os.kill(info.pid, signal.SIGTERM)
    except OSError as exc:
        return False, f"SIGTERM 失败: {exc}"
    return True, f"已向 :{port}（PID {info.pid}）发 SIGTERM"


def cmd_kill(port: int, assume_yes: bool, wait_seconds: float = 15.0) -> int:
    info = scan_main_procs().get(port)
    if info is None:
        print(f"找不到 :{port} 的 vllm serve 主进程，未做任何操作", file=sys.stderr)
        return 1
    print(f"目标: :{port} {info.model or ''}  PID {info.pid}  GPU {','.join(map(str, info.gpus)) or '—'}")
    if not assume_yes:
        answer = input("确认发送 SIGTERM？[y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("已取消")
            return 1
    ok, message = kill_instance(port)
    print(message)
    if not ok:
        return 1
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if not Path(f"/proc/{info.pid}").exists():
            print(f"PID {info.pid} 已退出")
            return 0
        time.sleep(0.5)
    print(f"PID {info.pid} 在 {wait_seconds:.0f}s 内未退出（SIGTERM 可能被忽略）")
    return 1


# ---------------------------------------------------------------- CSV 回看


def _collect_report_paths(spec: list[str]) -> list[Path]:
    targets = [Path(s).expanduser() for s in spec] if spec else [DEFAULT_CSV_DIR]
    paths: list[Path] = []
    for target in targets:
        if target.is_dir():
            paths.extend(sorted(target.glob("vllm-*.csv")))
        elif target.is_file():
            paths.append(target)
        else:
            print(f"找不到 {target}", file=sys.stderr)
    return paths


def _read_csv_rows(paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        try:
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    ts = _parse_ts(row.get("ts", ""))
                    if ts is None:
                        continue
                    row["_ts"] = ts
                    rows.append(row)
        except (OSError, csv.Error) as exc:
            print(f"读 {path} 失败: {exc}", file=sys.stderr)
    return rows


def _parse_ts(text: str) -> "datetime | None":
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _num(row: dict, key: str) -> float | None:
    try:
        return float(row.get(key) or "")
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _bucket_rows(rows: list[dict], limit: int = 60) -> list[tuple["datetime", list[dict]]]:
    """先按分钟分桶；分钟数超过 limit 就改成 5 分钟一桶，报告不至于太长。"""
    minutes: dict[datetime, list[dict]] = {}
    for row in rows:
        key = row["_ts"].replace(second=0, microsecond=0)
        minutes.setdefault(key, []).append(row)
    if len(minutes) <= limit:
        return sorted(minutes.items())
    coarse: dict[datetime, list[dict]] = {}
    for minute, bucket in minutes.items():
        key = minute.replace(minute=minute.minute - minute.minute % 5)
        coarse.setdefault(key, []).extend(bucket)
    return sorted(coarse.items())


def _report_instance(console, port: str, model: str, rows: list[dict]) -> None:
    from rich import box
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

    rows = sorted(rows, key=lambda r: r["_ts"])
    first, last = rows[0]["_ts"], rows[-1]["_ts"]
    span = (last - first).total_seconds()
    outs = [v for r in rows if (v := _num(r, "out_tps")) is not None]
    ins = [v for r in rows if (v := _num(r, "in_tps")) is not None]
    runs = [v for r in rows if (v := _num(r, "running")) is not None]
    waits = [v for r in rows if (v := _num(r, "waiting")) is not None]
    caches = [v for r in rows if (v := _num(r, "cache_hit_pct")) is not None]
    dones = [v for r in rows if (v := _num(r, "done_total")) is not None]
    completed = (dones[-1] - dones[0]) if len(dones) >= 2 and dones[-1] >= dones[0] else None
    req_rate = (completed / span) if completed is not None and span > 0 else _mean(
        [v for r in rows if (v := _num(r, "req_tps")) is not None]
    )

    title = Text()
    title.append(f":{port} ", style="bold cyan")
    title.append(model, style="bold")
    title.append(f"  {len(rows)} 行 · {first:%m-%d %H:%M}–{last:%H:%M}（{_fmt_duration(span)}）",
                 style="dim")
    console.print(title)

    cfg_bits: list[str] = []
    mmls = sorted({r.get("max_model_len") for r in rows if r.get("max_model_len")})
    mnss = sorted({r.get("max_num_seqs") for r in rows if r.get("max_num_seqs")})
    dpts = {(r.get("dp"), r.get("tp")) for r in rows if r.get("dp")}
    if mmls:
        cfg_bits.append("max_model_len=" + "/".join(_fmt_int(x) for x in mmls))
    if mnss:
        cfg_bits.append("max_num_seqs=" + "/".join(_fmt_int(x) for x in mnss))
    if len(dpts) == 1:
        dp, tp = next(iter(dpts))
        cfg_bits.append(f"DP{dp}×TP{tp}" if tp else f"DP{dp}")
    if cfg_bits:
        console.print(Text("配置  " + " · ".join(cfg_bits), style="dim"))

    summary = Text()
    summary.append("平均 out ", style="dim")
    summary.append(f"{_fmt_rate(_mean(outs))} tok/s", style="bold green")
    if outs:
        summary.append(f"（峰 {_fmt_rate(max(outs))}）", style="dim")
    summary.append("   平均 in ", style="dim")
    summary.append(f"{_fmt_rate(_mean(ins))} tok/s", style="cyan")
    summary.append("   平均 run ", style="dim")
    summary.append(f"{_fmt_rate(_mean(runs))}", style="bold")
    if waits:
        summary.append(f"（wait 峰 {max(waits):.0f}）", style="dim")
    if caches:
        summary.append(f"   cache 均值 {_mean(caches):.1f}%", style="yellow")
    if completed is not None:
        summary.append(f"   期间完成 {_fmt_count(completed)} 请求", style="bold")
        if req_rate is not None:
            summary.append(f"（{req_rate:.1f}/s）", style="dim")
    elif req_rate is not None:
        summary.append(f"   请求完成 {req_rate:.1f}/s", style="dim")
    console.print(summary)

    # 采样中断（长时间没写行）单独提示，别把断档当成塌速
    gaps = []
    for prev, cur in zip(rows, rows[1:]):
        delta = (cur["_ts"] - prev["_ts"]).total_seconds()
        if delta > 60:
            gaps.append(delta)
    if gaps:
        console.print(Text(f"采样中断 {len(gaps)} 处，最长 {_fmt_duration(max(gaps))}", style="yellow"))

    buckets = _bucket_rows(rows)
    if len(buckets) >= 2:
        graph_w = max(10, min(100, console.width - 4))
        graph_h = 6
        values = []
        for _, bucket in buckets:
            vals = [v for r in bucket if (v := _num(r, "out_tps")) is not None]
            values.append(_mean(vals) or 0.0)
        console.print(Group(*area_graph(values, graph_w, graph_h, style="green")))
        console.print(Text(f"{buckets[0][0]:%H:%M} → {buckets[-1][0]:%H:%M}", style="dim"))

    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, expand=False)
    for column in ("时间", "采样", "in tok/s", "out tok/s", "run 均/峰", "wait 峰", "cache %", "req/s"):
        table.add_column(column, justify="right" if column != "时间" else "left")
    for start, bucket in buckets:
        b_ins = [v for r in bucket if (v := _num(r, "in_tps")) is not None]
        b_outs = [v for r in bucket if (v := _num(r, "out_tps")) is not None]
        b_runs = [v for r in bucket if (v := _num(r, "running")) is not None]
        b_waits = [v for r in bucket if (v := _num(r, "waiting")) is not None]
        b_caches = [v for r in bucket if (v := _num(r, "cache_hit_pct")) is not None]
        b_reqs = [v for r in bucket if (v := _num(r, "req_tps")) is not None]
        table.add_row(
            f"{start:%H:%M}",
            str(len(bucket)),
            _fmt_rate(_mean(b_ins)),
            _fmt_rate(_mean(b_outs)),
            f"{_fmt_rate(_mean(b_runs))}/{_fmt_rate(max(b_runs) if b_runs else None)}",
            f"{max(b_waits):.0f}" if b_waits else "—",
            f"{_mean(b_caches):.1f}" if b_caches else "—",
            f"{_mean(b_reqs):.1f}" if b_reqs else "—",
        )
    console.print(table)
    console.print()


# ---------------------------------------------------------------- 入口


def _build(args: argparse.Namespace) -> tuple[Collector, CsvLogger | None]:
    csv_logger = None if args.no_csv else CsvLogger(
        Path(args.csv_dir), days=args.csv_days, max_mb=args.csv_max_mb
    )
    return Collector(interval=args.interval, rescan_every=args.rescan, csv_logger=csv_logger), csv_logger


def cmd_once(args: argparse.Namespace) -> int:
    from rich.console import Console

    collector, csv_logger = _build(args)
    collector.sample(log_csv=False)  # 基线
    time.sleep(args.interval)
    views = collector.sample()
    console = Console()
    meta = {
        "host": os.uname().nodename,
        "clock": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "interval": args.interval,
        "scanned_ports": collector.scanned_ports,
        "csv": csv_logger.describe() if csv_logger else "",
        "errors": collector.errors,
        "body_width": console.width,
    }
    console.print(render_dashboard(views, meta))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.text import Text

    paths = _collect_report_paths(args.report)
    if not paths:
        spec = " ".join(args.report) if args.report else str(DEFAULT_CSV_DIR)
        print(f"没找到 CSV 存档：{spec}", file=sys.stderr)
        return 1
    rows = _read_csv_rows(paths)
    if not rows:
        print("CSV 里没有可解析的采样行", file=sys.stderr)
        return 1
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row.get("port", "?"), row.get("model", "?")), []).append(row)
    console = Console()
    console.print(Text(f"vllmtop --report · {len(paths)} 个文件 · {len(rows)} 行", style="bold reverse"))
    for (port, model), group_rows in sorted(groups.items(), key=lambda kv: str(kv[0][0])):
        _report_instance(console, port, model, group_rows)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from rich.console import Console

    if args.info >= 0:
        if fetch_metrics(args.info) is None:
            print(f":{args.info} 上没有 vLLM /metrics，实例没在跑或端口不对", file=sys.stderr)
            return 1
        ports = [args.info]
    else:
        collector = Collector(interval=1.0, rescan_every=60, csv_logger=None)
        collector.sample(log_csv=False)
        ports = sorted(collector.known_ports)
        if not ports:
            print("没发现 vLLM 实例", file=sys.stderr)
            return 1
    details = [collect_instance_details(port) for port in ports]
    if args.json:
        print(json.dumps(details, ensure_ascii=False, indent=2))
        return 0
    Console().print(render_instance_details(details))
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.live import Live

    collector, csv_logger = _build(args)
    collector.sample(log_csv=False)  # 基线
    try:
        with Live(console=Console(), refresh_per_second=8, screen=False) as live:
            while True:
                time.sleep(args.watch)
                views = collector.sample()
                size = live.console.size
                meta = {
                    "host": os.uname().nodename,
                    "clock": datetime.now().strftime("%H:%M:%S"),
                    "interval": args.watch,
                    "scanned_ports": collector.scanned_ports,
                    "csv": csv_logger.describe() if csv_logger else "",
                    "errors": collector.errors,
                    "body_height": max(4, size.height - 3),
                    "body_width": size.width,
                }
                live.update(render_dashboard(views, meta))
    except KeyboardInterrupt:
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllmtop",
        description="本机所有 vLLM 实例的终端看板（吞吐 / cache hit / KV / GPU / CSV 存档）",
    )
    parser.add_argument("--once", action="store_true", help="打一屏快照就退出")
    parser.add_argument("--watch", type=float, metavar="SECONDS", help="原地刷新模式，指定刷新间隔")
    parser.add_argument("--report", nargs="*", metavar="CSV",
                        help="读 CSV 存档输出分钟级汇总与吞吐曲线（可给文件/目录，省略用默认目录）")
    parser.add_argument("--info", nargs="?", const=-1, type=int, metavar="PORT",
                        help="查看实例运行配置（省略 PORT 则列出全部实例）")
    parser.add_argument("--json", action="store_true",
                        help="配合 --info 输出 JSON（单独用等于列出全部实例的配置）")
    parser.add_argument("--kill", type=int, metavar="PORT", help="关停指定端口上的 vLLM 实例")
    parser.add_argument("--yes", action="store_true", help="--kill 时跳过确认")
    parser.add_argument("--interval", type=float, default=2.0, help="采样间隔秒（默认 2.0）")
    parser.add_argument("--rescan", type=float, default=30.0, help="重新发现端口的间隔秒（默认 30）")
    parser.add_argument("--no-csv", action="store_true", help="不写 CSV 存档")
    parser.add_argument("--csv-dir", default=str(DEFAULT_CSV_DIR), help="CSV 目录（默认 logs/vllmtop）")
    parser.add_argument("--csv-days", type=int, default=7, help="CSV 保留天数（默认 7）")
    parser.add_argument("--csv-max-mb", type=float, default=200.0, help="CSV 总容量上限 MB（默认 200）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.kill is not None:
        return cmd_kill(args.kill, args.yes)
    if args.report is not None:
        return cmd_report(args)
    if args.json and args.info is None:
        args.info = -1
    if args.info is not None:
        return cmd_info(args)
    if args.once:
        return cmd_once(args)
    if args.watch:
        return cmd_watch(args)
    collector, csv_logger = _build(args)
    build_tui(collector, csv_logger).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
