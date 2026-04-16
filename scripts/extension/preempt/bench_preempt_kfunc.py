#!/usr/bin/env python3
"""
Benchmark: kfunc preempt latency vs ioctl preempt latency

Strategy:
  1. Load BPF tool → attach kprobes + struct_ops
  2. Create CUDA context A → captures handles via kprobes
  3. Arm trigger with repeat=N
  4. Create N more CUDA contexts → each fires struct_ops → bpf_wq → kfunc preempt
  5. Read latency samples from BPF map
"""
import ctypes
import subprocess
import time
import os
import signal
import sys
import struct

N_PREEMPTS = int(sys.argv[1]) if len(sys.argv) > 1 else 10

def run_cmd(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"CMD FAILED: {cmd}\n{r.stderr}")
    return r

def bpftool_map_id(name):
    """Find BPF map ID by name"""
    r = run_cmd(f"sudo bpftool map list -j 2>/dev/null", check=False)
    if r.returncode != 0:
        return None
    import json
    maps = json.loads(r.stdout)
    for m in maps:
        if m.get("name") == name:
            return m["id"]
    return None

def bpftool_map_lookup(map_id, key_bytes):
    """Read a value from BPF map"""
    key_hex = " ".join(f"0x{b:02x}" for b in key_bytes)
    r = run_cmd(f"sudo bpftool map lookup id {map_id} key {key_hex} -j 2>/dev/null", check=False)
    if r.returncode != 0:
        return None
    import json
    data = json.loads(r.stdout)
    val = data.get("value", [])
    # bpftool -j may return hex strings ("0x01") or ints (1)
    if isinstance(val, list) and len(val) > 0:
        if isinstance(val[0], str):
            return bytes(int(v, 16) for v in val)
        return bytes(val)
    return None

def bpftool_map_update(map_id, key_bytes, val_bytes):
    """Write a value to BPF map"""
    key_hex = " ".join(f"0x{b:02x}" for b in key_bytes)
    val_hex = " ".join(f"0x{b:02x}" for b in val_bytes)
    run_cmd(f"sudo bpftool map update id {map_id} key {key_hex} value {val_hex}")

def read_u64(map_id, key_u32):
    raw = bpftool_map_lookup(map_id, struct.pack("<I", key_u32))
    if raw and len(raw) >= 8:
        return struct.unpack("<Q", raw[:8])[0]
    return 0

def read_u32(map_id, key_u32):
    raw = bpftool_map_lookup(map_id, struct.pack("<I", key_u32))
    if raw and len(raw) >= 4:
        return struct.unpack("<I", raw[:4])[0]
    return 0

print(f"=== kfunc Preempt Latency Benchmark (N={N_PREEMPTS}) ===\n")

# Clear trace
os.system("sudo sh -c 'echo > /sys/kernel/tracing/trace'")

# Step 1: Start BPF tool (piped, non-interactive)
print("[1] Loading BPF tool...")
# Build command sequence with enough time
cmds = "sleep 999\n"  # keep stdin open
tool_proc = subprocess.Popen(
    ["sudo", "./test_preempt_kfunc"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../extension")
)
time.sleep(2)

# Step 2: Create CUDA target
print("[2] Creating target CUDA context...")
libcuda = ctypes.CDLL("libcuda.so")
libcuda.cuInit(0)
dev = ctypes.c_int()
libcuda.cuDeviceGet(ctypes.byref(dev), 0)
target_ctx = ctypes.c_void_p()
libcuda.cuCtxCreate_v2(ctypes.byref(target_ctx), 0, dev)
print(f"  Target context created (PID={os.getpid()})")
time.sleep(2)

# Find BPF maps
stats_id = bpftool_map_id("stats")
trigger_id = bpftool_map_id("trigger")
latency_id = bpftool_map_id("latency_samples")
tsg_count_id = bpftool_map_id("tsg_count")

if not all([stats_id, trigger_id, latency_id, tsg_count_id]):
    print(f"ERROR: Missing BPF maps: stats={stats_id} trigger={trigger_id} latency={latency_id} tsg_count={tsg_count_id}")
    tool_proc.terminate()
    sys.exit(1)

tsg_count = read_u32(tsg_count_id, 0)
print(f"  Captured TSGs: {tsg_count}")
if tsg_count == 0:
    print("ERROR: No TSGs captured! Is nvidia module loaded?")
    tool_proc.terminate()
    sys.exit(1)

# Step 3: Arm trigger with repeat
print(f"\n[3] Arming trigger: target=idx0, repeat={N_PREEMPTS}...")
# trigger[0] = 1 (idx 0 + 1)
bpftool_map_update(trigger_id, struct.pack("<I", 0), struct.pack("<I", 1))
# trigger[1] = N_PREEMPTS (repeat count)
bpftool_map_update(trigger_id, struct.pack("<I", 1), struct.pack("<I", N_PREEMPTS))

# Step 4: Create N CUDA contexts to trigger struct_ops hooks
print(f"\n[4] Creating {N_PREEMPTS} CUDA contexts (each fires struct_ops → kfunc)...")
t0 = time.monotonic()

for i in range(N_PREEMPTS):
    ctx = ctypes.c_void_p()
    libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    time.sleep(0.3)  # let bpf_wq complete
    libcuda.cuCtxDestroy_v2(ctx)
    time.sleep(0.2)

elapsed = time.monotonic() - t0
print(f"  Done in {elapsed:.1f}s")

time.sleep(2)

# Step 5: Read results
print(f"\n[5] Results:")
wq_fired = read_u64(stats_id, 3)
preempt_ok = read_u64(stats_id, 1)
preempt_err = read_u64(stats_id, 2)
last_kfunc_ns = read_u64(stats_id, 5)
last_wq_ns = read_u64(stats_id, 6)

print(f"  preempt_ok:  {preempt_ok}")
print(f"  preempt_err: {preempt_err}")
print(f"  wq_fired:    {wq_fired}")
print(f"  last_kfunc:  {last_kfunc_ns} ns = {last_kfunc_ns/1000:.0f} us")
print(f"  last_wq:     {last_wq_ns} ns = {last_wq_ns/1000:.0f} us")

# Read latency samples
print(f"\n  kfunc latency samples:")
latencies = []
for i in range(min(int(wq_fired), 128)):
    ns = read_u64(latency_id, i)
    if ns > 0:
        latencies.append(ns)
        print(f"    [{i}] {ns} ns = {ns/1000:.0f} us")

if latencies:
    avg = sum(latencies) / len(latencies)
    mn = min(latencies)
    mx = max(latencies)
    print(f"\n  Summary ({len(latencies)} samples):")
    print(f"    avg = {avg:.0f} ns = {avg/1000:.0f} us")
    print(f"    min = {mn} ns = {mn/1000:.0f} us")
    print(f"    max = {mx} ns = {mx/1000:.0f} us")

    print(f"\n=== Comparison with ioctl path ===")
    print(f"  ioctl (test_preempt_demo):  avg=369 us  (min=320, max=401)")
    print(f"  kfunc (bpf_wq → kfunc):    avg={avg/1000:.0f} us  (min={mn/1000:.0f}, max={mx/1000:.0f})")
    print(f"  kfunc = pure RM+GSP latency (no userspace round-trip)")

# Print trace
print(f"\n=== trace_pipe ===")
r = run_cmd("sudo timeout 2 cat /sys/kernel/tracing/trace_pipe 2>/dev/null || true", check=False)
print(r.stdout[:3000] if r.stdout else "(empty)")

# Cleanup
tool_proc.terminate()
libcuda.cuCtxDestroy_v2(target_ctx)
print("\n=== Benchmark Complete ===")
