#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""
Score Bridge Daemon for Attention-Aware Eviction
(注意力感知驱逐策略的评分桥接守护进程)

【架构概览】
这是一个纯用户态的 Python 脚本。它通过 Linux 底层的 bpf() 系统调用，
直接修改挂载在文件系统中的 eBPF Map，从而指挥内核里的驱逐策略。
"""

import argparse
import ctypes
import ctypes.util
import os
import signal
import struct
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Constants (必须与 attention_aware_eviction.bpf.c 中的定义严格对齐)
# ---------------------------------------------------------------------------

# 语义分级
TIER_TRASH = 0
TIER_COOL = 1
TIER_HOT = 2

# 显存 2MB 大页对齐偏移量（用于将虚拟地址转换为紧凑的 page_id）
VA_SHIFT = 21  

# eBPF Map 在 Linux 虚拟文件系统中的持久化“钉扎”路径
SCORE_MAP_PIN = "/sys/fs/bpf/attention_score_map"
STATS_MAP_PIN = "/sys/fs/bpf/attention_stats_map"

# 统计指标名称（与内核里的统计数组索引一一对应）
STAT_NAMES = [
    "activate_total",
    "score_hit",
    "move_head_trash",
    "move_tail_hot",
    "tier_cool",
    "t1_protect",
    "score_miss",
]

# StreamingLLM 启发式算法的默认参数
DEFAULT_SINK_TOKENS = 4       # 开头永远保住的 System Prompt 词数
DEFAULT_RECENT_WINDOW = 256   # 结尾永远保住的最近上下文词数

# Linux 内核 bpf() 系统调用的标准命令字 (见 linux/bpf.h)
BPF_MAP_LOOKUP_ELEM = 1  # 查 Map
BPF_MAP_UPDATE_ELEM = 2  # 写 Map
BPF_MAP_DELETE_ELEM = 3  # 删 Map
BPF_MAP_GET_NEXT_KEY = 4 # 遍历 Map
BPF_OBJ_GET = 7          # 根据路径获取已钉扎的 Map/Prog 的文件描述符 (fd)
BPF_ANY = 0              # 更新 Map 时的 flag：键不存在则创建，存在则覆盖

# ---------------------------------------------------------------------------
# Low-level BPF map operations via bpf() syscall 
# (通过底层的 bpf 系统调用实现 Map 操作)
# ---------------------------------------------------------------------------

_NR_bpf = 321  # x86_64 架构下，bpf() 系统调用的系统调用号是 321

_libc: Optional[ctypes.CDLL] = None

def _get_libc() -> ctypes.CDLL:
    """加载标准的 C 运行库 (libc)，我们需要借用它里面的 syscall() 函数"""
    global _libc
    if _libc is None:
        lib_name = ctypes.util.find_library("c")
        if not lib_name:
            lib_name = "libc.so.6"
        # use_errno=True 非常重要：当系统调用失败时，Python 可以通过 ctypes.get_errno() 拿到具体的 C 错误码
        _libc = ctypes.CDLL(lib_name, use_errno=True)
    return _libc

# 【黑魔法警告】以下三个类是在 Python 中精确复刻 C 语言里的 `union bpf_attr`。
# BPF 的系统调用极其复杂，它接收一个庞大的 C 联合体作为参数。
# 我们必须使用 ctypes 严格对齐内存布局，哪怕差一个字节，内核都会拒绝访问。

class _BpfObjAttr(ctypes.Structure):
    """对应 bpf_attr 中用于 BPF_OBJ_GET 的匿名结构体"""
    _fields_ = [
        ("pathname", ctypes.c_uint64), # 指向文件路径字符串的指针
        ("bpf_fd", ctypes.c_uint32),
        ("file_flags", ctypes.c_uint32),
    ]

class _BpfMapAttr(ctypes.Structure):
    """对应 bpf_attr 中用于 Map 增删改查的匿名结构体"""
    _fields_ = [
        ("map_fd", ctypes.c_uint32),       # Map 的文件描述符
        ("_pad0", ctypes.c_uint32),        # C 语言的 64 位内存对齐填充（极度关键，丢了就会内存错位）
        ("key", ctypes.c_uint64),          # 指向 Key 内存缓冲区的指针
        ("value_or_next_key", ctypes.c_uint64), # 指向 Value 或 Next Key 缓冲区的指针
        ("flags", ctypes.c_uint64),        # 操作标志 (如 BPF_ANY)
    ]

class _BpfAttr(ctypes.Union):
    """C 语言里的 union bpf_attr，将上面的结构体叠在同一块内存上"""
    _fields_ = [
        ("obj", _BpfObjAttr),
        ("map", _BpfMapAttr),
        ("_raw", ctypes.c_char * 128), # 强制预留 128 字节，适配未来的内核扩展
    ]

def _bpf_syscall(cmd: int, attr: _BpfAttr) -> int:
    """执行终极系统调用"""
    libc = _get_libc()
    # 相当于 C 里的：syscall(321, cmd, &attr, sizeof(attr))
    ret = libc.syscall(
        ctypes.c_long(_NR_bpf),
        ctypes.c_int(cmd),
        ctypes.byref(attr),
        ctypes.c_uint(ctypes.sizeof(attr)),
    )
    if ret < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno)) # 抛出易读的 Python 异常
    return ret

# ----------------- 下面是对业务暴露的 BPF CRUD 接口 -----------------

def bpf_obj_get(pathname: str) -> int:
    """
    通过挂载点路径（如 /sys/fs/bpf/attention_score_map）拿到 Map 的文件描述符 (fd)。
    相当于在 C 程序加载 BPF 后，Python 进程拿到了操控这块内存的“钥匙”。
    """
    path_bytes = pathname.encode("utf-8") + b"\0" # C 语言的字符串必须以 \0 结尾
    path_buf = ctypes.create_string_buffer(path_bytes)

    attr = _BpfAttr()
    ctypes.memset(ctypes.byref(attr), 0, ctypes.sizeof(attr)) # 初始化清零，防止脏内存
    attr.obj.pathname = ctypes.addressof(path_buf) # 传递内存地址

    return _bpf_syscall(BPF_OBJ_GET, attr)

def bpf_map_update(map_fd: int, key: bytes, value: bytes, flags: int = BPF_ANY) -> None:
    """向 BPF Map 写入数据（控制面向数据面下发分数的入口）"""
    key_buf = (ctypes.c_char * len(key)).from_buffer_copy(key)
    val_buf = (ctypes.c_char * len(value)).from_buffer_copy(value)

    attr = _BpfAttr()
    ctypes.memset(ctypes.byref(attr), 0, ctypes.sizeof(attr))
    attr.map.map_fd = map_fd
    attr.map.key = ctypes.addressof(key_buf)
    attr.map.value_or_next_key = ctypes.addressof(val_buf)
    attr.map.flags = flags

    _bpf_syscall(BPF_MAP_UPDATE_ELEM, attr)

def bpf_map_lookup(map_fd: int, key: bytes, value_size: int) -> bytes:
    """读取 BPF Map 的数据（例如读取监控统计的大盘数据）"""
    key_buf = (ctypes.c_char * len(key)).from_buffer_copy(key)
    val_buf = (ctypes.c_char * value_size)() # 创建一个空的 Buffer 等待内核写入

    attr = _BpfAttr()
    ctypes.memset(ctypes.byref(attr), 0, ctypes.sizeof(attr))
    attr.map.map_fd = map_fd
    attr.map.key = ctypes.addressof(key_buf)
    attr.map.value_or_next_key = ctypes.addressof(val_buf)

    _bpf_syscall(BPF_MAP_LOOKUP_ELEM, attr)
    return bytes(val_buf) # 将 C Buffer 转换回 Python 的 bytes

def bpf_map_delete(map_fd: int, key: bytes) -> None:
    """删除 Map 中的某个条目"""
    key_buf = (ctypes.c_char * len(key)).from_buffer_copy(key)

    attr = _BpfAttr()
    ctypes.memset(ctypes.byref(attr), 0, ctypes.sizeof(attr))
    attr.map.map_fd = map_fd
    attr.map.key = ctypes.addressof(key_buf)

    _bpf_syscall(BPF_MAP_DELETE_ELEM, attr)

def bpf_map_get_next_key(map_fd: int, key: Optional[bytes], key_size: int) -> Optional[bytes]:
    """
    遍历 BPF Map 的迭代器底层接口。
    BPF 没有 `keys()` 方法，只能传当前的 key 进去，让内核返回下一个 key。
    如果传 None，内核返回第一个 key。
    """
    next_key_buf = (ctypes.c_char * key_size)()

    attr = _BpfAttr()
    ctypes.memset(ctypes.byref(attr), 0, ctypes.sizeof(attr))
    attr.map.map_fd = map_fd
    attr.map.value_or_next_key = ctypes.addressof(next_key_buf)

    if key is not None:
        key_buf = (ctypes.c_char * len(key)).from_buffer_copy(key)
        attr.map.key = ctypes.addressof(key_buf)
    else:
        attr.map.key = 0 # 传空指针，获取第一个 Key

    try:
        _bpf_syscall(BPF_MAP_GET_NEXT_KEY, attr)
    except OSError:
        return None # 遍历到尽头了，内核会抛出 ENOENT 错误
    return bytes(next_key_buf)


# ---------------------------------------------------------------------------
# Score Map helpers (分数映射表辅助函数：负责 Python 与 C 的序列化/反序列化)
# ---------------------------------------------------------------------------

def pack_page_id(page_id: int) -> bytes:
    """
    将 Python 整数打包为 C 语言的 `u32` (无符号 32 位整型)。
    用途：作为 bpf_map_update_elem 写入 score_map 时所用的 Key。
    
    细节解析：
    - `<` 代表 Little-Endian (小端序)。x86_64 架构的 Linux 内核默认是小端序。
    - `I` 代表 unsigned int (4 个字节)。
    """
    return struct.pack("<I", page_id)


def pack_block_score(score: int, tier: int, flags: int = 0) -> bytes:
    """
    将 Python 数据打包为内核中定义的 `struct block_score` 结构体。
    用途：作为写入 score_map 时的 Value。
    
    在 attention_aware_eviction.bpf.c 中，该结构体定义为：
    struct block_score {
        u16 attention_score; // 2 字节
        u8  tier;            // 1 字节
        u8  flags;           // 1 字节
    }; // 总计 4 字节
    
    细节解析：
    - `<`：小端序。
    - `H`：unsigned short (2 字节)，对应 `u16`。
    - `B`：unsigned char (1 字节)，对应 `u8`。
    - `& 0xFFFF` 和 `& 0xFF`：这是极其严谨的防溢出保护。防止 Python 传进来的
      整数超过了 C 语言类型的最大上限，导致 struct.pack 抛出异常或内存错位。
    """
    return struct.pack("<HBB", score & 0xFFFF, tier & 0xFF, flags & 0xFF)


def unpack_block_score(data: bytes) -> dict:
    """
    将内核传回来的底层字节流，反序列化还原为 Python 字典。
    用途：当我们需要用 BPF_MAP_LOOKUP_ELEM 查表调试时使用。
    
    细节解析：
    - `data[:4]`：防御性切片，只取前 4 个字节进行解包，防止内核返回的数据
      因为某些原因带了多余的 Padding（内存对齐填充）导致解包失败。
    """
    score, tier, flags = struct.unpack("<HBB", data[:4])
    return {"score": score, "tier": tier, "flags": flags}


def va_to_page_id(va: int) -> int:
    """
    将 64 位的虚拟内存地址 (Virtual Address) 转换为紧凑的 2MB 大页 ID。
    用途：与 BPF 内核代码中的 `chunk_va >> VA_SHIFT` 逻辑保持绝对一致。
    
    细节解析：
    - `VA_SHIFT` 通常是 21（因为 2^21 = 2MB）。
    - `& 0xFFFFFFFF`：Python 的位移操作不会自动截断位数。这里强制做一个掩码操作，
      确保生成的 page_id 严格落在一个 32 位无符号整数 (`u32`) 的范围内，
      完美匹配内核侧定义好的 Map Key 类型。
    """
    return (va >> VA_SHIFT) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# StreamingLLM scoring heuristic
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# StreamingLLM scoring heuristic (StreamingLLM 启发式评分算法)
# ---------------------------------------------------------------------------

class StreamingLLMScorer:
    """
    使用 StreamingLLM 的启发式规则为 KV Cache 块打分。

    【核心洞察】StreamingLLM 论文指出，大模型的注意力（Attention）分布极度不均：
    1. Attention Sink (注意力池)：最开头的几个 Token（如 System Prompt）吸收了极大的注意力，绝对不能丢。
    2. Recent Window (局部窗口)：刚刚生成的最近几个 Token，对于保持上下文连贯性至关重要。
    3. Middle Tokens (中间废话)：处于中间位置的长尾 Token，几乎不被后续生成关注，可以安全扔掉。

    评分公式：
      - Sink tokens (开头):   直接拉满，score = 65535, tier = HOT
      - Recent window (结尾): 极高分，score = 50000 + 新鲜度加成, tier = HOT
      - Old tokens (中间):    按位置比例打低分, tier = TRASH (极度靠前) 或 COOL (偏中后)
    """

    def __init__(self, sink_tokens: int = DEFAULT_SINK_TOKENS,
                 recent_window: int = DEFAULT_RECENT_WINDOW,
                 trash_percentile: float = 0.2,
                 hot_percentile: float = 0.5):
        self.sink_tokens = sink_tokens
        self.recent_window = recent_window
        self.trash_percentile = trash_percentile  # 处于前 20% 位置的中间词直接判定为垃圾
        self.hot_percentile = hot_percentile      # 处于后 50% 位置的中间词暂定为普通 (COOL)

    def compute_scores(self, num_blocks: int, tokens_per_block: int,
                       total_tokens: int) -> list[dict]:
        """
        计算每个逻辑 Block 的得分。
        注意：这里的 Block 是 vLLM 等推理引擎的逻辑概念（比如 1 个 Block 存 16 个 Token），
        还不是操作系统的物理页。
        """
        if num_blocks == 0 or tokens_per_block == 0:
            return []

        results = []
        for block_idx in range(num_blocks):
            # 计算当前 block 包含的 token 索引范围
            token_start = block_idx * tokens_per_block
            token_end = min(token_start + tokens_per_block, total_tokens)

            # 如果这个 block 还没被分配任何 token，直接标记为垃圾，方便内核随时回收
            if token_start >= total_tokens:
                results.append({
                    "block_idx": block_idx,
                    "score": 0,
                    "tier": TIER_TRASH,
                    "flags": 0,
                })
                continue

            # 判定当前 block 是否属于 Sink (开头) 或 Recent (结尾)
            is_sink = token_start < self.sink_tokens
            recent_start = max(0, total_tokens - self.recent_window)
            is_recent = token_end > recent_start

            if is_sink:
                score = 65535 # 16位无符号整数的上限
                tier = TIER_HOT
                flags = 0x01  # bit0: is_sink
            elif is_recent:
                # 越靠近当前的 Token，新鲜度 (recency) 越高，加成越高
                recency = 1.0 - (total_tokens - token_end) / max(
                    self.recent_window, 1
                )
                score = int(50000 + recency * 15000)
                tier = TIER_HOT
                flags = 0x02  # bit1: is_recent_window
            else:
                # 中间被遗忘的 Token
                position_ratio = token_end / max(total_tokens, 1)
                score = int(position_ratio * 40000)
                
                # 越靠前（越老）的 Token 越没用，打入 TRASH 层优先斩首
                if position_ratio < self.trash_percentile:
                    tier = TIER_TRASH
                elif position_ratio > self.hot_percentile:
                    tier = TIER_COOL
                else:
                    tier = TIER_COOL
                flags = 0

            results.append({
                "block_idx": block_idx,
                "score": min(score, 65535), # 绝对防御溢出
                "tier": tier,
                "flags": flags,
            })

        return results


# ---------------------------------------------------------------------------
# Score Bridge (连接大脑与内核的执行机构)
# ---------------------------------------------------------------------------

class ScoreBridge:
    """
    连接计分逻辑与底层 eBPF Map 的桥梁。
    """

    def __init__(self, score_map_path: str = SCORE_MAP_PIN,
                 stats_map_path: str = STATS_MAP_PIN):
        self.score_map_fd = -1
        self.stats_map_fd = -1
        self._score_map_path = score_map_path
        self._stats_map_path = stats_map_path

    def connect(self) -> None:
        """通过挂载路径，拿到内核 Map 的文件描述符 (fd)"""
        try:
            self.score_map_fd = bpf_obj_get(self._score_map_path)
            print(f"Opened score_map (fd={self.score_map_fd})")
        except OSError as e:
            raise RuntimeError(
                f"Cannot open {self._score_map_path}: {e}\n"
                "Is the attention_aware_eviction loader running?"
            ) from e

        try:
            self.stats_map_fd = bpf_obj_get(self._stats_map_path)
            print(f"Opened stats_map (fd={self.stats_map_fd})")
        except OSError as e:
            print(f"Warning: cannot open stats_map: {e}", file=sys.stderr)

    def close(self) -> None:
        """关闭 fd，释放资源。这不会销毁内核里的 Map，因为它们已经被钉扎(pinned)了"""
        if self.score_map_fd >= 0:
            os.close(self.score_map_fd)
            self.score_map_fd = -1
        if self.stats_map_fd >= 0:
            os.close(self.stats_map_fd)
            self.stats_map_fd = -1

    def update_score(self, page_id: int, score: int, tier: int,
                     flags: int = 0) -> None:
        """调用前面的底层 C 封装，写入单条数据"""
        key = pack_page_id(page_id)
        value = pack_block_score(score, tier, flags)
        bpf_map_update(self.score_map_fd, key, value)

    def update_scores_for_va_range(
        self,
        kv_base_va: int,
        block_scores: list[dict],
        block_size_bytes: int,
    ) -> int:
        """
        【系统工程的精髓：逻辑空间到物理空间的降维打击】
        
        大模型（vLLM）只知道 "Block"，不知道内存地址。
        操作系统（eBPF）只知道 "2MB 物理大页 (page_id)"，不知道什么叫 Block。
        这个函数负责做翻译：把连续的虚拟内存地址切成 2MB 的页，并把对应 Block 的分数赋给这些页。
        """
        page_size = 1 << VA_SHIFT  # 2MB (2097152 字节)
        entries = 0

        for bs in block_scores:
            # 计算这个逻辑 Block 在虚拟内存空间中的具体地址范围
            block_va = kv_base_va + bs["block_idx"] * block_size_bytes
            block_end = block_va + block_size_bytes

            va = block_va
            # 一个逻辑 Block 可能跨越多个 2MB 的物理大页，也可能比 2MB 小。
            # 这里以 2MB 为步长滑动，把这个 Block 涉及到的所有大页都打上相同的分数。
            while va < block_end:
                page_id = va_to_page_id(va)
                self.update_score(page_id, bs["score"], bs["tier"],
                                  bs["flags"])
                va += page_size
                entries += 1

        return entries

    def clear_all(self) -> int:
        """
        清空 Map。
        eBPF Map 没有一次性 clear 的接口，只能像链表一样，不断请求下一个 Key 并执行删除。
        """
        count = 0
        key = None
        while True:
            next_key = bpf_map_get_next_key(self.score_map_fd, key,
                                            key_size=4)
            if next_key is None:
                break
            try:
                bpf_map_delete(self.score_map_fd, next_key)
                count += 1
            except OSError:
                pass
            key = None  # 删除当前项后，重置 key，内核会自动给下一个（如果继续用旧 key 会报错）

        return count

    def read_stats(self) -> dict[str, int]:
        """
        读取并汇总内核中各个 CPU 核的遥测数据。
        """
        if self.stats_map_fd < 0:
            return {}

        # 【PERCPU_ARRAY 的读取机制】
        # 当你从 PERCPU_ARRAY 读取一个 Key 时，内核会一次性把所有 CPU 核对应的 Value 
        # 作为一个连续的数组吐给你。
        num_cpus = os.cpu_count() or 1
        # 分配一个足够大的缓冲区。每个 CPU 的计数器是 u64 (8 字节)。
        buf_size = max(num_cpus, 256) * 8 

        stats = {}
        for i, name in enumerate(STAT_NAMES):
            key = struct.pack("<I", i) # 指标的索引 (0, 1, 2...)
            try:
                raw = bpf_map_lookup(self.stats_map_fd, key, buf_size)
                total = 0
                # 遍历所有 CPU 核心，把它们各自记的账加起来
                for c in range(min(num_cpus, 256)):
                    # "<Q" 表示解析一个 8 字节的无符号长整型 (u64)
                    val = struct.unpack_from("<Q", raw, c * 8)[0]
                    total += val
                stats[name] = total
            except OSError:
                stats[name] = 0

        return stats
# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CLI commands (命令行指令具体实现)
# ---------------------------------------------------------------------------

_running = True

def _signal_handler(sig, frame):
    """
    优雅退出的信号处理函数。
    当用户在终端按下 Ctrl+C (SIGINT) 或系统发送 kill 信号 (SIGTERM) 时，
    将 _running 设为 False，让主循环安全退出，确保后续能正常关闭文件描述符。
    """
    global _running
    _running = False


def cmd_standalone(args: argparse.Namespace) -> None:
    """
    【核心测试模式：独立模拟运行 (Standalone Mode)】
    用途：在你还没有修改真实的 vLLM 代码前，用这个模式来模拟 LLM 引擎的显存布局，
    并周期性地向内核注入 StreamingLLM 启发式分数。
    """
    # 解析 KV Cache 的虚拟内存基地址 (支持 16 进制字符串输入)
    kv_base_va = int(args.kv_base_va, 16) if isinstance(
        args.kv_base_va, str
    ) else args.kv_base_va
    # 将 KB 转换为字节
    block_size_bytes = args.block_size_kb * 1024

    # 1. 实例化“大脑”：加载 StreamingLLM 算分器
    scorer = StreamingLLMScorer(
        sink_tokens=args.sink_tokens,
        recent_window=args.recent_window,
    )

    # 2. 实例化“桥梁”：连接到内核的 eBPF Map
    bridge = ScoreBridge()
    bridge.connect()

    # 注册信号，准备随时优雅退出
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # 打印当前的模拟参数
    print(f"\nKV cache layout:")
    print(f"  Base VA       : 0x{kv_base_va:x}")
    print(f"  Num blocks    : {args.num_blocks}")
    print(f"  Block size    : {args.block_size_kb} KB")
    print(f"  Tokens/block  : {args.tokens_per_block}")
    print(f"  Total tokens  : {args.num_tokens}")
    print(f"  Sink tokens   : {args.sink_tokens}")
    print(f"  Recent window : {args.recent_window}")
    print()

    iteration = 0
    while _running:
        # Step 1: 在用户态计算所有逻辑 Block 的分数与层级 (HOT/COOL/TRASH)
        scores = scorer.compute_scores(
            num_blocks=args.num_blocks,
            tokens_per_block=args.tokens_per_block,
            total_tokens=args.num_tokens,
        )

        # Step 2: 将逻辑 Block 的分数，映射为物理大页 ID，并写入内核
        entries = bridge.update_scores_for_va_range(
            kv_base_va, scores, block_size_bytes
        )

        # 统计本轮生成的 Tier 分布
        tier_counts = {TIER_TRASH: 0, TIER_COOL: 0, TIER_HOT: 0}
        for s in scores:
            tier_counts[s["tier"]] += 1

        iteration += 1
        print(
            f"[iter {iteration}] Updated {entries} page entries  "
            f"(TRASH={tier_counts[TIER_TRASH]}, "
            f"COOL={tier_counts[TIER_COOL]}, "
            f"HOT={tier_counts[TIER_HOT]})"
        )

        # 如果开启了 --stats，则读取内核遥测大盘并打印
        if args.stats:
            stats = bridge.read_stats()
            if stats:
                for name, val in stats.items():
                    print(f"  {name:22s} {val}")

        # 如果开启了 --once，执行一次就退出
        if args.once:
            break

        # 挂起进程，等待下一个周期（模拟 LLM 随时间推移生成新的 Token）
        time.sleep(args.interval)

    # 释放系统资源
    bridge.close()
    print("Score bridge stopped.")


def cmd_watch(args: argparse.Namespace) -> None:
    """
    【纯监控模式：Watch Mode】
    用途：不写入任何分数，仅仅像 top 命令一样，周期性地打印内核 eBPF 的驱逐统计数据。
    这在真实的生产环境调试中极其有用。
    """
    bridge = ScoreBridge()
    bridge.connect()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print("Watching attention-aware eviction stats (Ctrl-C to stop)\n")

    while _running:
        stats = bridge.read_stats()
        if stats:
            print(f"--- {time.strftime('%H:%M:%S')} ---")
            for name, val in stats.items():
                print(f"  {name:22s} {val}")
            print()
        time.sleep(args.interval)

    bridge.close()


def cmd_clear(args: argparse.Namespace) -> None:
    """
    【清理模式：Clear Mode】
    用途：排障/重置。强制清空内核 score_map 里的所有数据。
    """
    bridge = ScoreBridge()
    bridge.connect()

    count = bridge.clear_all()
    print(f"Cleared {count} entries from score_map")

    bridge.close()


# ---------------------------------------------------------------------------
# Argument parser (命令行参数解析器)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score Bridge Daemon for Attention-Aware Eviction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 使用 subparsers 实现类似 git 的多命令设计 (git clone, git commit...)
    subparsers = parser.add_subparsers(dest="command", help="sub-command")

    # --- 注册 standalone 子命令及参数 ---
    p_standalone = subparsers.add_parser(
        "standalone",
        help="Populate scores using StreamingLLM heuristic",
    )
    # 模拟显存布局的关键参数
    p_standalone.add_argument("--kv-base-va", required=True, help="Base VA of KV cache tensor (hex, e.g. 0x7f0000000000)")
    p_standalone.add_argument("--num-blocks", type=int, required=True, help="Number of KV cache blocks")
    p_standalone.add_argument("--block-size-kb", type=int, default=256, help="Block size in KB (default: 256)")
    p_standalone.add_argument("--num-tokens", type=int, required=True, help="Total number of tokens in the sequence")
    p_standalone.add_argument("--tokens-per-block", type=int, default=16, help="Tokens stored per KV block (default: 16)")
    # 算法调优参数
    p_standalone.add_argument("--sink-tokens", type=int, default=DEFAULT_SINK_TOKENS, help=f"Number of attention sink tokens (default: {DEFAULT_SINK_TOKENS})")
    p_standalone.add_argument("--recent-window", type=int, default=DEFAULT_RECENT_WINDOW, help=f"Recent token window size (default: {DEFAULT_RECENT_WINDOW})")
    p_standalone.add_argument("--interval", type=float, default=1.0, help="Update interval in seconds (default: 1.0)")
    p_standalone.add_argument("--once", action="store_true", help="Update once and exit")
    p_standalone.add_argument("--stats", action="store_true", help="Print BPF stats after each update")
    # 绑定回调函数
    p_standalone.set_defaults(func=cmd_standalone)

    # --- 注册 watch 子命令及参数 ---
    p_watch = subparsers.add_parser("watch", help="Monitor BPF eviction stats")
    p_watch.add_argument("--interval", type=float, default=2.0, help="Poll interval in seconds (default: 2.0)")
    p_watch.set_defaults(func=cmd_watch)

    # --- 注册 clear 子命令 ---
    p_clear = subparsers.add_parser("clear", help="Clear all score_map entries")
    p_clear.set_defaults(func=cmd_clear)

    # 解析用户输入
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 魔法路由：根据用户输入的子命令，直接调用对应的函数 (如 cmd_standalone)
    args.func(args)


if __name__ == "__main__":
    main()