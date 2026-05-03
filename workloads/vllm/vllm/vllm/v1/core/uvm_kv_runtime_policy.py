# SPDX-License-Identifier: Apache-2.0
"""Stage J UVM KV runtime pressure policy.

This module intentionally stays at the vLLM block-manager boundary.  It traces
and optionally denies new KV block admission; it does not migrate raw allocator
pointers or alter active request block tables behind the scheduler's back.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from vllm.logger import init_logger
from vllm.device_allocator.uvm_pool_coordinator import (
    record_uvm_pool_pressure,
    request_uvm_pool_action,
)
from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


def _read_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _read_int_env(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return max(int(raw), 0)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; expected non-negative integer.", name, raw)
        return default


def _read_mode_env() -> str:
    mode = os.environ.get("VLLM_UVM_KV_RUNTIME_MODE", "trace_only")
    mode = mode.strip().lower()
    if mode not in ("trace_only", "enforce"):
        logger.warning(
            "Ignoring invalid VLLM_UVM_KV_RUNTIME_MODE=%r; using trace_only.",
            mode,
        )
        return "trace_only"
    return mode


def _block_hash_to_text(block_hash: Any) -> str | None:
    if block_hash is None:
        return None
    if isinstance(block_hash, bytes):
        return block_hash.hex()
    return str(block_hash)


class UvmKvRuntimePolicy:
    """Trace-only or soft-admission Stage J policy for KV block pressure."""

    def __init__(self, kv_cache_config: KVCacheConfig) -> None:
        self.enabled = _read_bool_env("VLLM_UVM_KV_RUNTIME_ENABLE", False)
        self.mode = _read_mode_env()
        self.trace_file = os.environ.get("VLLM_UVM_KV_RUNTIME_TRACE_FILE", "")
        self.policy = os.environ.get(
            "VLLM_UVM_KV_RUNTIME_EVICTION_POLICY", "lru_prefix_cache"
        ).strip().lower()
        self.candidate_limit = _read_int_env(
            "VLLM_UVM_KV_RUNTIME_CANDIDATE_LIMIT", 16
        )
        self.prefix_evict_enable = _read_bool_env(
            "VLLM_UVM_KV_RUNTIME_PREFIX_EVICT_ENABLE", False
        )
        self.prefix_evict_max_blocks = _read_int_env(
            "VLLM_UVM_KV_RUNTIME_PREFIX_EVICT_MAX_BLOCKS", 0
        )
        self.budget_bytes = _read_int_env("VLLM_UVM_KV_RUNTIME_BUDGET_BYTES", 0)
        budget_blocks_override = _read_int_env(
            "VLLM_UVM_KV_RUNTIME_BUDGET_BLOCKS", 0
        )
        self.total_bytes = sum(tensor.size for tensor in kv_cache_config.kv_cache_tensors)
        self.total_blocks = max(kv_cache_config.num_blocks - 1, 0)
        self.bytes_per_block = (
            self.total_bytes // kv_cache_config.num_blocks
            if kv_cache_config.num_blocks > 0 and self.total_bytes > 0
            else 0
        )
        if budget_blocks_override > 0:
            self.budget_blocks = budget_blocks_override
        elif self.budget_bytes > 0 and self.bytes_per_block > 0:
            self.budget_blocks = max(self.budget_bytes // self.bytes_per_block, 1)
        else:
            self.budget_blocks = 0
        if self.policy not in ("lru_prefix_cache", "scheduler_aware"):
            logger.warning(
                "Ignoring invalid VLLM_UVM_KV_RUNTIME_EVICTION_POLICY=%r; "
                "using lru_prefix_cache.",
                self.policy,
            )
            self.policy = "lru_prefix_cache"
        self._lock = threading.Lock()
        self._config_recorded = False
        self._summary = {
            "allocation_pressure_records": 0,
            "over_budget_records": 0,
            "would_deny_records": 0,
            "denied_records": 0,
            "candidate_records": 0,
            "prefix_evict_attempt_records": 0,
            "prefix_evict_success_blocks": 0,
            "prefix_evict_noop_records": 0,
            "prefix_evict_failed_records": 0,
            "prefix_evict_coordinator_reject_records": 0,
            "prefix_cache_eviction_records": 0,
            "unsafe_prefix_cache_eviction_records": 0,
            "reuse_failure_records": 0,
        }

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.trace_file)

    def record_config(self) -> None:
        if not self.active or self._config_recorded:
            return
        self._config_recorded = True
        self._write(
            {
                "action": "runtime_config",
                "enabled": self.enabled,
                "mode": self.mode,
                "policy": self.policy,
                "budget_bytes": self.budget_bytes,
                "budget_blocks": self.budget_blocks,
                "bytes_per_block": self.bytes_per_block,
                "total_bytes": self.total_bytes,
                "total_blocks": self.total_blocks,
                "candidate_limit": self.candidate_limit,
                "prefix_evict_enable": self.prefix_evict_enable,
                "prefix_evict_max_blocks": self.prefix_evict_max_blocks,
            }
        )

    def should_allow_allocation(
        self,
        *,
        request_id: str,
        num_blocks_to_allocate: int,
        block_pool: Any,
    ) -> bool:
        """Trace pressure and decide whether new KV block admission is allowed."""
        if not self.active or num_blocks_to_allocate <= 0:
            return True
        self.record_config()
        pressure = self._pressure_snapshot(block_pool, num_blocks_to_allocate)
        over_budget = bool(pressure["over_budget"])
        record_uvm_pool_pressure(
            pool="kv",
            pressure_bytes=int(pressure.get("pressure_bytes") or 0),
            pressure_ratio=(
                (
                    float(pressure.get("projected_used_blocks") or 0)
                    / float(pressure.get("budget_blocks") or 1)
                )
                if int(pressure.get("budget_blocks") or 0) > 0
                else None
            ),
            action_queue_depth=num_blocks_to_allocate,
            metadata={
                "stage": "stage_j",
                "request_id": request_id,
                "over_budget": over_budget,
            },
        )
        self._write(
            {
                "action": "allocation_pressure",
                "request_id": request_id,
                "num_blocks_to_allocate": num_blocks_to_allocate,
                **pressure,
            }
        )
        self._summary["allocation_pressure_records"] += 1
        if over_budget:
            self._summary["over_budget_records"] += 1
            self._trace_candidates(
                request_id=request_id,
                block_pool=block_pool,
                reason="runtime_budget_pressure",
            )
            self._maybe_execute_prefix_eviction(
                request_id=request_id,
                block_pool=block_pool,
                pressure=pressure,
                reason="runtime_budget_pressure",
            )
            self._write(
                {
                    "action": "would_deny_allocation",
                    "request_id": request_id,
                    "mode": self.mode,
                    "policy": self.policy,
                    "num_blocks_to_allocate": num_blocks_to_allocate,
                    **pressure,
                }
            )
            self._summary["would_deny_records"] += 1
            allowed = self.mode != "enforce"
            if not allowed:
                self._summary["denied_records"] += 1
                self._write(
                    {
                        "action": "deny_allocation",
                        "request_id": request_id,
                        "mode": self.mode,
                        "policy": self.policy,
                        "num_blocks_to_allocate": num_blocks_to_allocate,
                        **pressure,
                    }
                )
            self._record_summary()
            return allowed
        self._record_summary()
        return True

    def record_reuse_failure(
        self,
        *,
        request_id: str,
        num_blocks_to_allocate: int,
        block_pool: Any,
    ) -> None:
        if not self.active:
            return
        self.record_config()
        self._write(
            {
                "action": "allocation_no_free_blocks",
                "request_id": request_id,
                "num_blocks_to_allocate": num_blocks_to_allocate,
                **self._pressure_snapshot(block_pool, num_blocks_to_allocate),
            }
        )
        self._summary["reuse_failure_records"] += 1
        self._record_summary()

    def record_prefix_cache_eviction(self, *, block: Any, block_hash: Any) -> None:
        """Record a real vLLM-safe prefix-cache eviction/reuse event."""
        if not self.active:
            return
        self.record_config()
        safe_ref_cnt_zero = getattr(block, "ref_cnt", None) == 0
        self._write(
            {
                "action": "evict_prefix_cache_block",
                "block_id": getattr(block, "block_id", None),
                "ref_cnt": getattr(block, "ref_cnt", None),
                "is_null": getattr(block, "is_null", None),
                "block_hash": _block_hash_to_text(block_hash),
                "safe_ref_cnt_zero": safe_ref_cnt_zero,
            }
        )
        self._summary["prefix_cache_eviction_records"] += 1
        if not safe_ref_cnt_zero:
            self._summary["unsafe_prefix_cache_eviction_records"] += 1
        self._record_summary()

    def _pressure_snapshot(
        self, block_pool: Any, num_blocks_to_allocate: int
    ) -> dict[str, Any]:
        total_blocks = max(getattr(block_pool, "num_gpu_blocks", 0) - 1, 0)
        free_blocks = max(block_pool.get_num_free_blocks(), 0)
        used_blocks = max(total_blocks - free_blocks, 0)
        projected_used_blocks = used_blocks + max(num_blocks_to_allocate, 0)
        budget_blocks = self.budget_blocks
        pressure_blocks = (
            max(projected_used_blocks - budget_blocks, 0)
            if budget_blocks > 0
            else 0
        )
        return {
            "budget_blocks": budget_blocks,
            "budget_bytes": self.budget_bytes,
            "bytes_per_block": self.bytes_per_block,
            "total_blocks": total_blocks,
            "free_blocks": free_blocks,
            "used_blocks": used_blocks,
            "projected_used_blocks": projected_used_blocks,
            "pressure_blocks": pressure_blocks,
            "pressure_bytes": pressure_blocks * self.bytes_per_block,
            "over_budget": budget_blocks > 0 and pressure_blocks > 0,
        }

    def _trace_candidates(
        self,
        *,
        request_id: str,
        block_pool: Any,
        reason: str,
    ) -> None:
        if self.candidate_limit <= 0:
            return
        try:
            free_blocks = block_pool.free_block_queue.get_all_free_blocks()
        except Exception as exc:  # pragma: no cover - defensive trace path.
            self._write(
                {
                    "action": "candidate_snapshot_failed",
                    "request_id": request_id,
                    "reason": reason,
                    "error": repr(exc),
                }
            )
            return
        for index, block in enumerate(free_blocks[: self.candidate_limit]):
            block_hash = getattr(block, "block_hash", None)
            self._write(
                {
                    "action": "would_evict_candidate"
                    if block_hash is not None
                    else "would_reuse_free_block",
                    "request_id": request_id,
                    "reason": reason,
                    "candidate_index": index,
                    "block_id": getattr(block, "block_id", None),
                    "ref_cnt": getattr(block, "ref_cnt", None),
                    "is_null": getattr(block, "is_null", None),
                    "has_block_hash": block_hash is not None,
                    "block_hash": _block_hash_to_text(block_hash),
                    "safe_ref_cnt_zero": getattr(block, "ref_cnt", None) == 0,
                }
            )
            self._summary["candidate_records"] += 1

    def _maybe_execute_prefix_eviction(
        self,
        *,
        request_id: str,
        block_pool: Any,
        pressure: dict[str, Any],
        reason: str,
    ) -> None:
        if not self.prefix_evict_enable:
            return
        max_blocks = self.prefix_evict_max_blocks
        if max_blocks <= 0:
            max_blocks = int(pressure.get("pressure_blocks") or 0)
        if max_blocks <= 0:
            return

        self._summary["prefix_evict_attempt_records"] += 1
        requested_bytes = max_blocks * int(pressure.get("bytes_per_block") or 0)
        coordinator_decision = request_uvm_pool_action(
            pool="kv",
            action="prefix_cache_evict",
            requested_bytes=requested_bytes,
            scope_key=f"kv_request:{request_id}",
            metadata={
                "stage": "stage_j",
                "request_id": request_id,
                "reason": reason,
                "max_blocks": max_blocks,
                "policy": self.policy,
            },
        )
        if not coordinator_decision.allowed:
            self._summary["prefix_evict_coordinator_reject_records"] += 1
            self._write(
                {
                    "action": "prefix_evict_coordinator_reject",
                    "request_id": request_id,
                    "reason": reason,
                    "max_blocks": max_blocks,
                    "requested_bytes": requested_bytes,
                    "coordinator_mode": coordinator_decision.mode,
                    "coordinator_reason": coordinator_decision.reason,
                    **pressure,
                }
            )
            return
        self._write(
            {
                "action": "prefix_evict_attempt",
                "request_id": request_id,
                "reason": reason,
                "max_blocks": max_blocks,
                **pressure,
            }
        )
        try:
            evicted_blocks = int(block_pool.evict_cached_free_blocks(max_blocks))
        except Exception as exc:  # pragma: no cover - defensive trace path.
            self._summary["prefix_evict_failed_records"] += 1
            self._write(
                {
                    "action": "prefix_evict_failed",
                    "request_id": request_id,
                    "reason": reason,
                    "max_blocks": max_blocks,
                    "error": repr(exc),
                }
            )
            return

        if evicted_blocks > 0:
            self._summary["prefix_evict_success_blocks"] += evicted_blocks
            action = "prefix_evict_success"
        else:
            self._summary["prefix_evict_noop_records"] += 1
            action = "prefix_evict_noop"
        self._write(
            {
                "action": action,
                "request_id": request_id,
                "reason": reason,
                "requested_blocks": max_blocks,
                "evicted_blocks": evicted_blocks,
                **pressure,
            }
        )

    def _record_summary(self) -> None:
        if not self.active:
            return
        self._write(
            {
                "action": "runtime_summary",
                "mode": self.mode,
                "policy": self.policy,
                "budget_blocks": self.budget_blocks,
                "budget_bytes": self.budget_bytes,
                "prefix_evict_enable": self.prefix_evict_enable,
                "prefix_evict_max_blocks": self.prefix_evict_max_blocks,
                **self._summary,
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        if not self.trace_file:
            return
        record.setdefault("ts", time.time())
        record.setdefault("stage", "stage_j")
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._lock:
            os.makedirs(os.path.dirname(self.trace_file) or ".", exist_ok=True)
            with open(self.trace_file, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
