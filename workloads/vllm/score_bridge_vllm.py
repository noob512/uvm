#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""vLLM-integrated score bridge (position/category based).

This module no longer computes KV importance via StreamingLLM heuristics.
Instead, vLLM assigns fixed importance score/tier by memory category:
- KV cache pages: fixed KV score/tier
- Model weights pages: fixed weight score/tier

The eBPF side then distinguishes eviction behavior using these assigned tiers.
"""

import argparse
import os
import signal
import sys
import threading
import time
from typing import Any, Iterator, Optional

try:
    from .score_bridge import (
        SCORE_MAP_PIN,
        STATS_MAP_PIN,
        TIER_COOL,
        TIER_HOT,
        VA_SHIFT,
        ScoreBridge,
        va_to_page_id,
    )
except ImportError:
    from score_bridge import (
        SCORE_MAP_PIN,
        STATS_MAP_PIN,
        TIER_COOL,
        TIER_HOT,
        VA_SHIFT,
        ScoreBridge,
        va_to_page_id,
    )


DEFAULT_KV_SCORE = 20000
DEFAULT_KV_TIER = TIER_COOL
DEFAULT_WEIGHT_SCORE = 65535
DEFAULT_WEIGHT_TIER = TIER_HOT

KV_FLAGS = 0x20
WEIGHT_FLAGS = 0x10


def _iter_tensors(obj: Any) -> Iterator[Any]:
    """Yield torch.Tensor objects from nested container structures."""
    try:
        import torch
    except Exception:
        return

    if isinstance(obj, torch.Tensor):
        yield obj
        return

    if isinstance(obj, dict):
        for value in obj.values():
            for tensor in _iter_tensors(value):
                yield tensor
        return

    if isinstance(obj, (list, tuple)):
        for value in obj:
            for tensor in _iter_tensors(value):
                yield tensor


# import sys
# import threading
# from typing import Any, Iterator, Optional

# 假设以下常量和类在外部定义
# VA_SHIFT, SCORE_MAP_PIN, STATS_MAP_PIN, DEFAULT_KV_SCORE, DEFAULT_KV_TIER,
# DEFAULT_WEIGHT_SCORE, DEFAULT_WEIGHT_TIER, KV_FLAGS, WEIGHT_FLAGS,
# ScoreBridge, va_to_page_id, _iter_tensors

class VLLMScoreBridge:
    """
    连接 vLLM 内存布局与底层 eBPF 注意力分数 Map 的桥梁。
    
    主要作用：将用户态程序（vLLM）中关键数据结构（如 KV Cache 和 模型权重）所在的
    虚拟内存地址映射为操作系统的物理/虚拟内存页 ID，并赋予不同的“分数(Score)”和“层级(Tier)”。
    操作系统内核（通过 eBPF）可以根据这些分数来进行智能的内存调度（例如：冷热内存分层、CXL 内存换入换出等）。
    """

    def __init__(
        self,
        sink_tokens: int | None = None,
        recent_window: int | None = None,
        score_map_path: str = SCORE_MAP_PIN,     # BPF 评分 Map 的文件系统挂载路径
        stats_map_path: str = STATS_MAP_PIN,     # BPF 统计 Map 的挂载路径
        verbose: bool = False,
        kv_score: int = DEFAULT_KV_SCORE,        # KV Cache 默认分数
        kv_tier: int = DEFAULT_KV_TIER,          # KV Cache 默认内存层级 (如高速/低速内存)
        weight_score: int = DEFAULT_WEIGHT_SCORE,# 模型权重默认分数
        weight_tier: int = DEFAULT_WEIGHT_TIER,  # 模型权重默认内存层级
    ):
        # 兼容旧版本基于启发式的调用者，这两个参数已被废弃但予以保留
        _ = sink_tokens
        _ = recent_window

        # 实例化底层的 C/C++ 或 BPF 桥接对象，负责真正的内核通信
        self.bridge = ScoreBridge(score_map_path, stats_map_path)
        self.verbose = verbose

        # 保存 KV Cache 和模型权重的策略配置
        self.kv_score = kv_score
        self.kv_tier = kv_tier
        self.weight_score = weight_score
        self.weight_tier = weight_tier

        # 状态控制变量
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def connect(self) -> None:
        """建立与底层 BPF Map 的连接。"""
        if self._connected:
            return
        self.bridge.connect()
        self._connected = True
        if self.verbose:
            print("[VLLMScoreBridge] Connected to BPF maps")

    def close(self) -> None:
        """关闭与底层 BPF Map 的连接。"""
        if not self._connected:
            return
        self.bridge.close()
        self._connected = True
        if self.verbose:
            print("[VLLMScoreBridge] Closed BPF maps")

    def get_stats(self) -> dict[str, int]:
        """从 BPF Map 读取当前的统计信息（例如命中率、更新次数等）。"""
        if not self._connected:
            return {}
        return self.bridge.read_stats()

    def clear_scores(self) -> int:
        """清空底层 BPF Map 中的所有评分记录。"""
        if not self._connected:
            raise RuntimeError("Not connected to BPF maps")
        return self.bridge.clear_all()

    def _iter_page_ids_for_range(self, addr: int, size_bytes: int) -> Iterator[int]:
        """
        [核心底层方法] 将一段内存地址范围 [addr, addr + size_bytes) 转换为操作系统内存页 ID 的迭代器。
        
        为什么需要这个？因为底层操作系统管理内存的最小单位是“页 (Page)”，而不是字节。
        """
        if size_bytes <= 0:
            return

        # 计算页大小 (通常 VA_SHIFT 为 12，即 2^12 = 4096 bytes = 4KB)
        page_size = 1 << VA_SHIFT 
        # 创建页掩码，例如 4KB 页的掩码是 0xFFFFFFFFFFFFF000，用于将地址向下对齐到页边界
        page_mask = ~(page_size - 1)

        # 获取这段内存跨越的起始页和结束页的对齐起始地址
        start_page_va = addr & page_mask
        end_page_va = (addr + size_bytes - 1) & page_mask

        # 遍历涉及的每一个内存页
        current = start_page_va
        while current <= end_page_va:
            # va_to_page_id 是外部函数，将虚拟地址(Virtual Address)转换为页 ID (Page Framework ID)
            yield va_to_page_id(current)
            current += page_size

    def _mark_block_span(
        self,
        base_va: int,
        num_blocks: int,
        block_size_bytes: int,
        score: int,
        tier: int,
        flags: int,
    ) -> int:
        """
        标记指定数量和大小的连续内存块序列。
        将这些内存块所在的物理/虚拟页写入 BPF Map。
        """
        if not self._connected:
            raise RuntimeError("Not connected to BPF maps. Call connect() first.")

        if num_blocks <= 0 or block_size_bytes <= 0:
            return 0

        # 使用集合记录已经标记过的页，防止跨越同一页的多个块造成重复系统调用，提升效率
        page_seen: set[int] = set()
        entries = 0

        for block_idx in range(num_blocks):
            # 计算当前块的起始地址
            block_start = base_va + block_idx * block_size_bytes
            # 找到该块涉及的所有内存页
            for page_id in self._iter_page_ids_for_range(block_start, block_size_bytes):
                if page_id in page_seen:
                    continue
                page_seen.add(page_id)
                # 调用底层 C/C++ 接口更新 BPF Map
                self.bridge.update_score(page_id, score, tier, flags)
                entries += 1

        return entries

    def hint_kv_cache_layout(
        self,
        kv_cache_base_va: int,
        num_blocks: int,
        block_size_bytes: int,
    ) -> int:
        """
        将整段 KV Cache 内存区域打上固定的 KV Cache 标签（分数和层级）。
        这是通知内核：“这片内存属于 KV Cache，请按照 KV Cache 的策略进行管理”。
        """
        entries = self._mark_block_span(
            base_va=kv_cache_base_va,
            num_blocks=num_blocks,
            block_size_bytes=block_size_bytes,
            score=self.kv_score,
            tier=self.kv_tier,
            flags=KV_FLAGS,
        )

        if self.verbose:
            print(
                "[VLLMScoreBridge] Marked "
                f"{entries} KV pages (score={self.kv_score}, tier={self.kv_tier})"
            )

        return entries

    def update_from_kv_cache(
        self,
        kv_cache_base_va: int,
        num_blocks: int,
        block_size_bytes: int,
        tokens_per_block: int | None = None,
        total_tokens: int | None = None,
    ) -> int:
        """兼容性包装器：旧版本可能有 token 级别的细粒度控制，现在统一采用纯位置/类别基准。"""
        _ = tokens_per_block
        _ = total_tokens
        return self.hint_kv_cache_layout(
            kv_cache_base_va=kv_cache_base_va,
            num_blocks=num_blocks,
            block_size_bytes=block_size_bytes,
        )

    def _get_worker_kv_layout(self, worker: Any) -> Optional[tuple[int, int, int]]:
        """
        [非常核心的反射方法 - 内存嗅探器] 
        接收一个黑盒的 vLLM Worker 对象，通过 Python 的 getattr 机制深入其内部，
        提取出 KV Cache 所在的真实虚拟内存起始地址、总块数以及单块字节数。
        
        返回值: (base_va起始虚拟地址, num_blocks块总数量, block_size_bytes单块大小)
        如果 Worker 尚未初始化完毕，则安全地返回 None。
        """
        
        # 1. 探针深入第一层：获取 model_runner
        # Worker 是 vLLM 的顶层执行单元，它内部包装了真正的模型运行器 model_runner。
        # 使用 getattr(..., None) 是一种防御性编程，防止在初始化早期抛出 AttributeError。
        model_runner = getattr(worker, "model_runner", None)
        if model_runner is None:
            return None

        # 2. 探针深入第二层：获取配置信息
        # kv_cache_config 中包含了 vLLM 启动时计算好的显存规划（比如能分出多少个 Block）。
        kv_cache_config = getattr(model_runner, "kv_cache_config", None)
        if kv_cache_config is None:
            return None

        # 3. 探针深入第三层：获取实际的内存张量列表
        # kv_caches 是一个嵌套结构，通常包含了模型每一层（Layer）对应的 K 和 V 缓存。
        kv_caches = getattr(model_runner, "kv_caches", None)
        
        # 4. 定位内存锚点 (Anchor Tensor)
        # 通过外部传入的辅助迭代器 `_iter_tensors`，拍平复杂的嵌套结构，拿到第一个 KV Tensor。
        # 为什么只拿第一个？因为 vLLM 在分配 KV Cache 时，通常是在底层申请一整块极其巨大的连续内存池，
        # 然后再把它切分给各个层。拿到第一个 Tensor 的指针，就等于拿到了这整片连续显存区域的起始基址。
        first_tensor = next(_iter_tensors(kv_caches), None)
        if first_tensor is None:
            return None

        # 5. 指针解构与物理参数计算
        try:
            # 提取 Tensor 在显存/内存空间中的真实 C/C++ 级别指针（Base Virtual Address）。
            # 这个地址就是操作系统/eBPF 能够听懂的语言。
            base_va = int(first_tensor.data_ptr())
            
            # 从配置中提取当前 GPU 节点上分配的物理块 (Block) 总数。
            # PagedAttention 的核心就是将显存划分为固定大小的 Block。
            num_blocks = int(getattr(kv_cache_config, "num_gpu_blocks", 0))
            
            # 计算该 Tensor 占据的总字节数：
            # numel() 返回元素总个数，element_size() 返回单个元素大小（例如 float16 是 2 字节）。
            total_bytes = int(first_tensor.numel() * first_tensor.element_size())
        except Exception:
            # 如果在提取指针或配置时发生任何类型转换错误或属性缺失，安全退出。
            return None

        # 6. 数据合法性校验
        # 确保显存块数量和总字节数合法，防止后续出现除以零或死循环等致命错误。
        if num_blocks <= 0 or total_bytes <= 0:
            return None

        # 7. 计算关键参数：单块大小
        # 将连续内存的总字节数平均分给所有的 Block。
        # 这个参数非常关键，后续 eBPF 桥接器需要根据这个大小和 block_idx 来精准计算
        # 每一个 Block 所在的内存页范围。
        block_size_bytes = total_bytes // num_blocks
        
        # 再次校验，防止除法截断后出现 0 或负数的情况。
        if block_size_bytes <= 0:
            return None

        # 8. 返回给上层调用者
        # 到此为止，我们成功把一个复杂的 Python 对象，降维成操作系统可以消费的 3 个纯粹的数字。
        return (base_va, num_blocks, block_size_bytes)

    def update_from_vllm_worker(self, worker: Any) -> int:
        """
        从 vLLM worker 实例中自动读取 KV Cache 的内存布局，并将其标记到内核的 BPF Map 中。
        
        这是一个高度封装的便捷方法（High-level API）。外部调用者（如 vLLM 的初始化流程）
        不需要关心底层 KV Cache 的物理连续性、虚拟地址或块大小，只需要把当前的 worker 对象
        传进来，这个方法就会自动“扫描”并完成一切注册工作。
        """
        
        # 1. 探针反射获取布局信息
        # 调用内部隐藏的 `_get_worker_kv_layout` 方法。
        # 该方法会像“X光机”一样，通过 Python 的 getattr 反射机制，层层穿透 worker 的内部属性
        # (worker -> model_runner -> kv_cache_config/kv_caches)，
        # 最终挖掘出底层 PyTorch Tensor 的真实内存指针、分块数量和单块字节数。
        layout = self._get_worker_kv_layout(worker)
        
        # 2. 状态校验与防御性拦截
        # 如果获取到的 layout 为 None，说明当前的 vLLM worker 还处于非常早期的初始化阶段，
        # 底层的 GPU 显存（KV Cache）可能还没有真正被 PyTorch 分配出来。
        if layout is None:
            if self.verbose:
                # 打印提示信息，说明时机未到，直接返回 0（表示处理了 0 个内存页），不引发异常。
                print("[VLLMScoreBridge] KV cache layout not ready")
            return 0

        # 3. 解包布局参数
        # 成功获取到了 KV Cache 在当前进程虚拟内存空间中的精准“地图”：
        # - kv_cache_base_va: KV Cache 连续内存块的起始虚拟地址 (Base Virtual Address)
        # - num_blocks: vLLM PagedAttention 机制分配的显存块总数
        # - block_size_bytes: 每个显存块占用的字节数
        kv_cache_base_va, num_blocks, block_size_bytes = layout
        
        # 4. 执行底层的 BPF 注册
        # 将解析出来的物理内存布局参数，转发给专门负责 BPF 交互的 `hint_kv_cache_layout` 方法。
        # 该方法会将这段连续的内存地址切分为操作系统的标准页（如 4KB/页），并为这些页打上 
        # KV Cache 专属的分数（Score）和层级（Tier），最终返回成功标记的页面总数。
        return self.hint_kv_cache_layout(
            kv_cache_base_va=kv_cache_base_va,
            num_blocks=num_blocks,
            block_size_bytes=block_size_bytes,
        )

    def hint_model_weights(self, model: Any) -> int:
        """
        遍历给定的 PyTorch 模型，将其所有的权重（Parameters）和缓冲区（Buffers）所在的内存页，
        打上模型权重的专用标签（分数和层级）。
        这通常是为了告诉内核：“这些是极其重要的只读数据，无论如何尽量不要把它们 Swap 出去”。
        """
        # 1. 前置检查：确保与底层 eBPF Map 的通信通道已打开
        if not self._connected:
            raise RuntimeError("Not connected to BPF maps. Call connect() first.")

        # 2. 延迟加载 PyTorch
        # 为什么在这里 import？为了避免在类模块加载时就强制依赖 PyTorch，
        # 从而加快模块导入速度，并减少不必要的显存上下文初始化。
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch is required for hint_model_weights") from exc

        # 3. 初始化去重集合
        # page_seen: 记录已经向 BPF Map 汇报过的内存页 ID。
        # 多个小 Tensor 极有可能挤在同一个 4KB 的内存页中，去重可以大幅减少与内核交互的系统调用(syscall)开销。
        page_seen: set[int] = set()
        
        # storage_seen: 极其关键的设计！用于跟踪 PyTorch 的底层 Storage。
        # 在 PyTorch 中，多个 Tensor 可以是同一个底层连续内存（Storage）的不同视图（View）。
        # 比如 tensor.view() 或分片操作。如果不针对 Storage 去重，同一块物理内存会被重复处理。
        storage_seen: set[int] = set() 
        entries = 0

        # 4. 定义处理单个 Tensor 的内部辅助函数
        def process_tensor(tensor: Any) -> None:
            nonlocal entries # 允许内部函数修改外部的计数器变量
            
            # 过滤非 Tensor 对象（防御性编程）
            if not isinstance(tensor, torch.Tensor):
                return
            
            # 仅处理驻留在 CUDA (GPU) 上且包含实际数据的 Tensor
            # 如果是 CPU Tensor 或空 Tensor 则直接跳过
            if not tensor.is_cuda or tensor.numel() == 0:
                return

            # 5. 提取底层物理存储层级的指针 (Storage Pointer)
            # 获取 Tensor 的底层存储指针，避免处理指向同一块物理内存的多个视图 Tensor。
            try:
                # 现代 PyTorch API：获取底层的 UntypedStorage 的真实内存地址
                storage_ptr = int(tensor.untyped_storage().data_ptr())
            except Exception:
                # 兼容老版本 PyTorch 的回退方案
                storage_ptr = int(tensor.data_ptr())

            # 检查这块底层存储是否已经被处理过
            if storage_ptr in storage_seen:
                return
            storage_seen.add(storage_ptr)

            # 6. 计算当前 Tensor 在虚拟内存中的具体范围
            # start: Tensor 数据区域的起始虚拟地址
            start = int(tensor.data_ptr())
            # size_bytes: 占用总字节数 = 元素个数 (numel) * 单个元素的字节数 (如 FP16 是 2 bytes)
            size_bytes = int(tensor.numel() * tensor.element_size())
            
            # 7. 将内存地址范围转换为页 ID，并更新至内核
            # 计算该 Tensor 跨越的所有内存页，并更新到 BPF
            for page_id in self._iter_page_ids_for_range(start, size_bytes):
                # 页面级去重，防止多个不同 Storage 的数据正好首尾挤在同一个页内
                if page_id in page_seen:
                    continue
                page_seen.add(page_id)
                
                # 调用桥接接口，将该页标记为"模型权重"专属的分数(weight_score)和层级(weight_tier)
                self.bridge.update_score(page_id, self.weight_score, self.weight_tier, WEIGHT_FLAGS)
                entries += 1

        # 8. 遍历模型的计算图状态
        # 遍历模型所有的可训练参数 (如 Linear 层的 weight 和 bias)
        # recurse=True 表示递归遍历所有子模块
        for _, param in model.named_parameters(recurse=True):
            process_tensor(param)

        # 遍历模型所有的非训练缓冲区（Buffers）
        # 例如 BatchNorm 层的 running_mean 和 running_var，或者固定生成的绝对位置编码 (Positional Embeddings)
        for _, buf in model.named_buffers(recurse=True):
            process_tensor(buf)

        # 9. 打印统计结果
        if self.verbose:
            print(
                "[VLLMScoreBridge] Marked "
                f"{entries} weight pages (score={self.weight_score}, tier={self.weight_tier})"
            )

        # 返回成功标记的内存页总数
        return entries

    def start_background_thread(
        self,
        kv_cache_base_va: int,
        num_blocks: int,
        block_size_bytes: int,
        tokens_per_block: int | None = None,
        total_tokens: int | None = None,
        interval: float = 1.0,
    ) -> None:
        """
        启动一个守护线程（后台线程），定期（默认每 1.0 秒）向 BPF Map 刷新 KV Cache 的布局和评分。
        
        为什么要定期刷新？
        1. 内核/eBPF 程序内部可能有自己的老化、驱逐或垃圾回收机制，定期刷新可以保持“心跳”。
        2. 防止因页面迁移（Page Migration）等内核行为导致的映射失效。
        """
        _ = tokens_per_block
        _ = total_tokens

        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Background thread already running")

        self.connect()

        def _loop() -> None:
            # 只要没有收到停止信号，就持续循环
            while not self._stop_event.is_set():
                try:
                    self.hint_kv_cache_layout(
                        kv_cache_base_va=kv_cache_base_va,
                        num_blocks=num_blocks,
                        block_size_bytes=block_size_bytes,
                    )
                except Exception as exc:
                    if self.verbose:
                        print(f"[VLLMScoreBridge] Update error: {exc}", file=sys.stderr)
                # 等待指定间隔，如果期间收到了 stop_event，wait 会立即中断
                self._stop_event.wait(interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

        if self.verbose:
            print(f"[VLLMScoreBridge] Started background thread (interval={interval}s)")

    def stop_background_thread(self, timeout: float = 5.0) -> None:
        """停止定期刷新的后台线程，等待其安全退出。"""
        if self._thread is None or not self._thread.is_alive():
            return

        # 发送停止信号
        self._stop_event.set()
        # 等待线程结束
        self._thread.join(timeout=timeout)

        if self._thread.is_alive():
            print("[VLLMScoreBridge] Warning: thread did not stop cleanly", file=sys.stderr)
        elif self.verbose:
            print("[VLLMScoreBridge] Background thread stopped")

        # 清理状态
        self._thread = None
        self._stop_event.clear()


_running = True


def _signal_handler(_sig: int, _frame: Any) -> None:
    global _running
    _running = False


def cmd_daemon(args: argparse.Namespace) -> None:
    kv_base_va = int(args.kv_cache_ptr, 16) if isinstance(args.kv_cache_ptr, str) else int(args.kv_cache_ptr)
    block_size_bytes = args.block_size_kb * 1024

    print("[VLLMScoreBridge] Starting daemon mode")
    print(f"  KV cache base: 0x{kv_base_va:x}")
    print(f"  Num blocks: {args.num_blocks}")
    print(f"  Block size: {args.block_size_kb} KB")
    print(f"  Update interval: {args.interval}s")

    bridge = VLLMScoreBridge(
        verbose=True,
        kv_score=args.kv_score,
        kv_tier=args.kv_tier,
        weight_score=args.weight_score,
        weight_tier=args.weight_tier,
    )

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        bridge.connect()

        iteration = 0
        while _running:
            iteration += 1
            start = time.time()

            entries = bridge.hint_kv_cache_layout(
                kv_cache_base_va=kv_base_va,
                num_blocks=args.num_blocks,
                block_size_bytes=block_size_bytes,
            )

            elapsed_ms = (time.time() - start) * 1000.0
            print(f"[Iteration {iteration}] Updated {entries} entries in {elapsed_ms:.2f}ms")

            if args.stats:
                stats = bridge.get_stats()
                if stats:
                    print(f"  BPF stats: {stats}")

            if args.once:
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[VLLMScoreBridge] Interrupted by user")
    finally:
        bridge.close()
        print("[VLLMScoreBridge] Daemon stopped")


def cmd_test_integration(_args: argparse.Namespace) -> None:
    print("[VLLMScoreBridge] Testing vLLM integration...")
    try:
        from vllm.v1.worker.gpu_worker import Worker  # noqa: F401
    except ImportError as exc:
        print(f"  vLLM import failed: {exc}")
        print("  Run with: uv run --directory workloads/vllm")
        return

    print("  vLLM imports available")
    print("  Embedded use:")
    print("    bridge = VLLMScoreBridge(verbose=True)")
    print("    bridge.connect()")
    print("    bridge.update_from_vllm_worker(worker)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="vLLM-integrated score bridge daemon (position/category based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    p_daemon = subparsers.add_parser("daemon", help="Run as standalone daemon")
    p_daemon.add_argument("--kv-cache-ptr", required=True, help="KV cache base address in hex")
    p_daemon.add_argument("--num-blocks", type=int, required=True, help="Number of KV cache blocks")
    p_daemon.add_argument("--block-size-kb", type=int, default=256, help="Block size in KB")
    p_daemon.add_argument("--tokens-per-block", type=int, default=16, help="Deprecated, ignored")
    p_daemon.add_argument("--num-tokens", type=int, default=0, help="Deprecated, ignored")
    p_daemon.add_argument("--kv-score", type=int, default=DEFAULT_KV_SCORE, help="Fixed score for KV pages")
    p_daemon.add_argument("--kv-tier", type=int, default=DEFAULT_KV_TIER, help="Fixed tier for KV pages")
    p_daemon.add_argument("--weight-score", type=int, default=DEFAULT_WEIGHT_SCORE, help="Fixed score for weight pages")
    p_daemon.add_argument("--weight-tier", type=int, default=DEFAULT_WEIGHT_TIER, help="Fixed tier for weight pages")
    p_daemon.add_argument("--interval", type=float, default=1.0, help="Update interval in seconds")
    p_daemon.add_argument("--once", action="store_true", help="Update once and exit")
    p_daemon.add_argument("--stats", action="store_true", help="Print BPF stats after each update")
    p_daemon.set_defaults(func=cmd_daemon)

    p_test = subparsers.add_parser("test-integration", help="Test vLLM integration imports")
    p_test.set_defaults(func=cmd_test_integration)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
