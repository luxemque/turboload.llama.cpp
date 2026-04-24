# Parallel RPC loading

## TL;DR

On a 4-node cluster (1 head + 3 RPC workers with CUDA unified-memory GPUs, local NVMe, RDMA interconnect, model replicated at the same path on each node), cold-loading Kimi-K2.6-Q2 (318 GB GGUF, 8 shards) went from **~701 s on unmodified llama.cpp to ~66 s** at a typical production launch config — about **~10.7×** end-to-end. Same hardware, same model, same `llama-server` command line, same `-c 262144 -np 4` with Q4 KV cache; only the binary changed.

Inference throughput is unchanged — these refactors only touch model loading.

| config | cold load (s) | vs stock |
|---|---:|---:|
| Stock llama.cpp, prodconf (`-c 262144 -np 4`), local-cache model | **701** | **1.00×** |
| Stock llama.cpp, `-c 8192 -np 1`, local-cache model | ~702 | 1.00× |
| Stock llama.cpp, NFS-shared model (historical) | ~835 | 0.84× |
| + pinned staging on rpc-server | ~114 | 6.2× |
| + worker-side QD=8 pread pool | ~87 | 8.1× |
| + head-side QD=8 pread pool | ~60 | 11.7× |
| **+ worker tset serialisation (final), `-c 8192 -np 1`** | **~58–63** | **~11–12×** |
| **+ final, prodconf (`-c 262144 -np 4`)** | **66** | **10.7×** |

(The smaller-config and prodconf numbers differ only by ~5 s — the extra KV allocation at 262 K×4 slots adds ~5 s on the improved path but is invisible on stock where 700 s is dominated by tensor transfer.)

The branch keeps full backward compatibility: every new path is gated on env vars and falls back to the stock code path if anything it needs is missing (no worker-read capability, no pinned host buffer type, etc.).

## The underlying bottleneck

Stock llama.cpp's RPC load path is **head-push**:

1. Head reads every tensor from disk.
2. Head sends each tensor over the socket to the relevant worker.
3. Worker receives, copies to pageable host memory, calls `cudaMemcpyAsync` + sync to upload to the GPU.
4. Worker acks, head moves to next tensor.

On a 3-worker cluster with a 318 GB model, this serialises everything through one head process and one pageable HtoD. Profiling showed the head was spending most of its wall in `ggml_backend_tensor_set` sync on the worker side (~340 ms per 294 MiB tensor, i.e. ~0.8 GiB/s pageable HtoD) — not in disk reads as we first assumed. Once that was fixed, pread at QD=1 became the bottleneck on both head and worker. Once pread parallelism was added, CUDA context contention from N pool threads inflated per-tensor HtoD from ~5 ms to ~130 ms. This document describes the fixes in the order they landed.

## The fixes

### 1. Pinned staging on the rpc-server

`ggml/src/ggml-rpc/ggml-rpc.cpp`: each rpc-server maintains a pinned host buffer (via the backend's host buffer type — `cudaHostAlloc` on CUDA) and reads tensor bytes into that before `ggml_backend_tensor_set`. This turns a ~340 ms pageable HtoD into a ~5 ms pinned HtoD per tensor.

- Always on, no flag.
- Transparent fallback to a `std::vector` if the backend has no host buffer type or pinned alloc fails.
- **Do NOT try the mmap fast-path** — tried it, got a 3× regression because `cudaMemcpyAsync` on unpinned mmap'd memory demand-faults pages one at a time.

### 2. Worker-side file reads (`SET_TENSOR_FROM_FILE`)

New RPC command. Instead of head pushing bytes, head sends `{path, file_offset, tensor_offset, size}` and the worker pread's the bytes from its own filesystem copy of the same model file.

- Gated on env vars: `LLAMA_RPC_PARALLEL_LOAD=1` (probes the capability) and `LLAMA_RPC_PARALLEL_LOAD_ASYNC=1` (enables fire-and-forget dispatch with a per-endpoint receiver thread on the head side).
- Requires the model file to be reachable at the same absolute path on head and each worker. The cleanest arrangement is to replicate the file onto each worker's local disk; a shared filesystem at the same mount point works too but inherits its latency.
- If the capability probe fails (old worker), the head falls back to the stock head-push path silently.

Head-side plumbing (`src/llama-model-loader.cpp`, `src/llama-model.cpp`) includes a per-endpoint async dispatcher thread, a receiver thread draining acks, and `flush_pending_rpc_reads()` at end of load to join them.

### 3. Worker-side QD=8 pread pool (`stff_pool`) + dedicated ack-sender thread

Each rpc-server now runs a thread pool (`stff_pool`) that issues `pread()` in parallel against the local NVMe, lifting read QD from 1 to 8. Acks must be emitted in request-submission order (the head's receiver thread drains FIFO per endpoint), so there's a separate `stff_ack_sender` thread that peeks the FIFO front, waits for the task's read to complete, then sends the ack.

- Knob: `LLAMA_RPC_STFF_POOL` (default 8, 0 disables the pool → falls back to the serial main-thread path).
- The ack-sender thread is what unlocks this: a first-cut design without it deadlocked because the worker's main thread only drained acks when forced to (backpressure or a non-STFF command arriving). Once the head finished dispatching and waited for the last batch of acks, both sides wedged.
- `pread` is thread-safe and independent of fd offset, so the same fd serves the pool without coordination.

Measured: slowest-worker RPC wall 76 s → ~41 s, 1.84×.

### 4. Head-side QD=8 pread pool for local CUDA0 tensors

`src/llama-model-loader.cpp`: the `load_all_data` ring-read path for CUDA0-bound tensors (the head's own layers) had the same QD=1 problem on the head's NVMe. Added a `head_read_pool` that submits `pread` tasks from the ring loop and waits on them in FIFO order before issuing `ggml_backend_tensor_set_async` to the pinned ring buffer.

- Knob: `LLAMA_HEAD_READ_POOL` (default 8, 0 disables → serial `read_raw_unsafe` path).
- POSIX-only (guarded with `#ifndef _WIN32`). Windows keeps the stock serial path.
- `n_buffers` grows from 4 to `pool_n + 2` when the pool is active, so `pool_n` reads can be in flight while a couple of slots are waiting on their HtoD event.

Measured: head ring-read phase 67.5 s → 25 s, 2.7×.

### 5. `tset` mutex on the rpc-server (modest; the dedicated-uploader variant was tried and reverted)

Under pool=8 without coordination, 8 pool threads call `ggml_backend_tensor_set` concurrently. Per-tensor HtoD time grew from ~5 ms (single-stream probe) to ~130 ms. A mutex serialising the tset call across pool threads drops it to ~132 ms — small improvement (~2 s off loader wall), shipped.

A deeper fix — dedicated single uploader thread so the CUDA context never switches threads — was attempted and did deliver per-tensor tset of ~52 ms (3× improvement), but regressed setup time from ~10 s to ~37 s per worker (17 pinned staging slots instead of 9 → more lazy `cudaHostAlloc` / grow events dominating startup) and one worker stalled partway through load with an unresolved backpressure interaction. Reverted. See [`findings.md`](findings.md) §"Refactor F" for the full writeup — future work.

## Quick start

```bash
# Build as usual, deploy binaries to all nodes at the same path.
# On each rpc-server:
LD_LIBRARY_PATH=/path/to/bin /path/to/bin/rpc-server -H <ip> -p 50052

# On the head:
env LLAMA_RPC_PARALLEL_LOAD=1 LLAMA_RPC_PARALLEL_LOAD_ASYNC=1 \
    LLAMA_RPC_LOAD_PROFILE=1 \
    /path/to/bin/llama-server \
      -m /path/to/model.gguf \
      --rpc host1:50052,host2:50052,host3:50052 \
      -ngl 999 --no-mmap -fa on \
      ...
```

For production, wrap this in a supervisor (systemd, runit, etc.) so workers come up before the head attempts to connect.

All four env vars are optional:

- `LLAMA_RPC_PARALLEL_LOAD=1` — turns on the worker-read path (falls back to head-push silently if workers don't support it).
- `LLAMA_RPC_PARALLEL_LOAD_ASYNC=1` — fire-and-forget dispatch + per-endpoint async receiver threads on the head.
- `LLAMA_RPC_LOAD_PROFILE=1` — adds per-worker and per-phase timing to the log (setup / read / tset sums).
- `LLAMA_RPC_STFF_POOL=<N>` — rpc-server-side pool width (default 8, 0 disables).
- `LLAMA_HEAD_READ_POOL=<N>` — head-side pool width for CUDA0 ring-read (default 8, 0 disables).

## Preconditions for the full 11–12× speedup

1. **Multi-node RPC load.** On single-node, only the head-side pool applies; expect ~2.7× on the local ring-read phase, 1.5–2× end-to-end.
2. **Model file at the same absolute path on head and workers.** Replicate to local disk for best results; a shared filesystem at the same mount point works but is latency-bound.
3. **GGUF format.** The file-read protocol is byte-offset-based and maps naturally to GGUF.
4. **`--no-mmap`.** The fast path assumes explicit pread into pinned staging. Mmap was tried and caused a 3× regression.
5. **CUDA workers** (proven). ROCm / Metal / Vulkan should work but are untested; Refactor 4 (head pool) works anywhere POSIX; Refactor 1–3 depend on the worker backend exposing pinned host buffers + async events.
6. **Model big enough to matter.** Fixed-cost startup (RDMA probe, metadata parse, warmup) is ~10–15 s regardless of size. Below ~30 GB models the total-wall gain is muted.

## When it probably *won't* help much

- Single-node, small model, mmap already fast: already memory-speed.
- NFS-shared model with no per-node local copy: QD>1 still helps a bit (2–4× on NFS is typical) but the absolute wall is NFS-bound.
- CPU-only RPC workers: pinned staging is moot (no HtoD to pin against); QD>1 reads still help.

## Backward compatibility

- **Protocol**: adds one new RPC command (`RPC_CMD_SET_TENSOR_FROM_FILE`) and two new proc-address exports (`ggml_backend_rpc_buffer_set_tensor_from_file`, `…_async`, `ggml_backend_rpc_flush_pending_reads`). Old workers that don't advertise the capability get the stock head-push path.
- **Env vars only**: with no env vars set and no worker-read support, behaviour is byte-identical to stock.
- **Inference path unchanged.**

## Code touchpoints (diff vs. `master`)

Five files, +1,554 / −46 lines:

| file | +add | −rm | what |
|---|---:|---:|---|
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | 942 | 2 | pinned staging, `stff_pool`, `stff_ack_sender`, `tset_mu`, `SET_TENSOR_FROM_FILE` command, head-side async dispatcher + receiver, probe instrumentation |
| `src/llama-model-loader.cpp` | 534 | 41 | `head_read_pool` class, pipelined ring-read loop, `launch_load_all_data` wrapper, `flush_pending_rpc_reads` integration, head-load profile instrumentation, `try_worker_read` path |
| `ggml/include/ggml-rpc.h` | 38 | 1 | new proc-address exports, capability bits |
| `src/llama-model-loader.h` | 29 | 0 | `launch_load_all_data` / `flush_pending_rpc_reads` declarations, dispatcher bookkeeping, `files_paths` for the worker-read path |
| `src/llama-model.cpp` | 11 | 2 | call sites that go through `launch_load_all_data` instead of direct `load_all_data` |

Most of the code is in `ggml-rpc.cpp`. Reasonable review order:

1. `ggml-rpc.h` — see the protocol surface.
2. `ggml-rpc.cpp`, `SET_TENSOR_FROM_FILE` server handler + `rpc_server::set_tensor_from_file` — the unit of work.
3. `stff_pool` + `stff_ack_sender` — the concurrency shape.
4. `llama-model-loader.cpp`, `head_read_pool` + the ring-read loop — head-side.
5. `launch_load_all_data` + `flush_pending_rpc_reads` — the orchestration across multiple buffers.

## Known limitations and open items

- The worker tset under `pool=8` is still ~130 ms/tensor (vs 5 ms single-stream). The mutex helps marginally but doesn't fix the underlying "CUDA context state cools when threads alternate" issue. A clean fix is the dedicated-uploader thread — needs eager slot pre-allocation to not regress setup, and there's a pending deadlock-adjacent investigation on one worker specifically. Projected additional win: ~15 s off the cold wall, landing end-to-end around 45 s.
- Head's `flush_pending_rpc_reads` spends ~5 s draining after the dispatcher threads join. Could be overlapped with the head's own ring-read.
- Windows gets only the stock path for Refactor 4 (head pool). All the worker-side refactors work on any POSIX worker.
- Startup overhead (RDMA probe × N workers, 8-shard metadata parse) is ~10–15 s and now a visible fraction of total. Low priority, cheap shaves available.
- `pool_size` and `n_buffers` are env-var tuned; no auto-sizing based on NVMe bandwidth / core count.

## Full diagnostic narrative

See [`findings.md`](findings.md) for the complete story — every refactor iteration, the blind alleys (mmap regression, warm/cold/primed sweep that revealed the Grace-Blackwell UMA cache ceiling, the first-cut QD=8 deadlock, the dedicated-uploader regression), and the measurement sweeps behind every number in this document.

## License

Same as llama.cpp (MIT).
