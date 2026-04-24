# Benchmarks

Raw `llama-server` logs from the cold-load measurements quoted in the top-level [`PARALLEL_RPC_LOADING.md`](../PARALLEL_RPC_LOADING.md).

## Files

| File | What it is |
|---|---|
| `cold-reference.server.log` | Reference cold-load run quoted in the TL;DR speedup table — the 41.7 s loader wall / ~66 s end-to-end number. |
| `cold-pread-validation.server.log` | Post-[`dbea709`](https://github.com/luxemque/llama.cpp/commit/dbea709) cold-load run validating the worker `std::ifstream` → `::pread` conversion. Loader wall 41.81 s, end-to-end ~65 s — flat vs reference, within run-to-run noise. |

Each log includes the full llama.cpp build banner, RDMA probe output, all `LLAMA_RPC_LOAD_PROFILE=1` timing dumps (per-worker read/tset phase breakdowns), the `head load profile — loop wall` lines, and the final "model loaded" / "server is listening" markers.

## Hardware

- **Nodes**: 1 head + 3 RPC workers, all NVIDIA GB10 (Grace-Blackwell). 128 GB LPDDR5X unified memory per node.
- **Storage**: model replicated to local NVMe on every node at the same absolute path.
- **Network**: 200 GbE RDMA (RoCEv2) between nodes.

This is n=1 on hardware — see [`../PARALLEL_RPC_LOADING.md#hardware-and-measurement-notes`](../PARALLEL_RPC_LOADING.md#hardware-and-measurement-notes) for how to interpret that.

## Launch command

Both runs used this exact command on the head node (`$MODEL` pointing at shard 1 of the Kimi-K2.6-Q2 GGUF, `$WORKER_1..3` at the three workers' `IP:50052` endpoints):

```bash
env LLAMA_RPC_LOAD_PROFILE=1 \
    LLAMA_RPC_PARALLEL_LOAD=1 \
    LLAMA_RPC_PARALLEL_LOAD_ASYNC=1 \
    LLAMA_RPC_STFF_POOL=8 \
    LLAMA_HEAD_READ_POOL=8 \
    ./llama-server \
        -m "$MODEL" \
        --rpc "$WORKER_1,$WORKER_2,$WORKER_3" \
        -ngl 999 --host 0.0.0.0 --port 8080 \
        -c 8192 -np 1 -b 512 -ub 512 \
        --no-mmap -fa on --reasoning-format deepseek \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --tensor-split 1,1,1,1 -fit off
```

Workers were started with a matching binary and `LD_LIBRARY_PATH` pointing at the binary's directory (the RPATH points at the head's build tree, which doesn't exist on workers). No env vars on the worker side for these runs.

## Cold-drop recipe

Before each measurement, on every node:

```bash
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

and the rpc-server process was restarted to clear any residual file mappings.

## How to read the timings

- `head load profile — loop wall N.NNN s (head_read_pool=8)` — cumulative wall on the head's local CUDA0 ring-read path. Printed once per GGUF shard; the last value is the total head-side wall.
- `loader wall time: N.NNN s; effective wall throughput: M.M MB/s` — the aggregate loader wall across head + all RPC workers, printed per shard. The largest value is the end-of-load total.
- `RPC load profile (per-worker):` block — per-worker counters: `n_set` (tensors assigned), `n_wread` (tensors served via `SET_TENSOR_FROM_FILE`), `data_GB` (total tensor bytes), `sent_GB` (bytes actually pushed over the socket, ~0 when worker-read is active), `wread_GB` (bytes read from local disk on the worker).
- End-to-end wall is the difference between the `LAUNCH_T0_NS=...` marker (when the launcher wrapper starts) and the `main: model loaded` line. About ~24 s of startup (RDMA probe, metadata parse, warmup) sit between the final loader wall and "model loaded".
