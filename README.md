# Turboload llama.cpp — 10× faster RPC model loading

Research archive documenting a set of changes to [llama.cpp](https://github.com/ggml-org/llama.cpp)'s RPC loading path that cut cold model load time on multi-node clusters by roughly an order of magnitude.

## What

Cold-loading a 318 GB MoE model (Kimi-K2.6 Q2) across three RPC workers drops from **~701 s on unmodified llama.cpp to ~66 s** — same hardware, same model, same `llama-server` command line, only the binary changed. End-to-end speedup ~10.7×.

Inference throughput is unchanged. These refactors only touch model loading.

Five mechanisms account for the speedup:

1. **`SET_TENSOR_FROM_FILE`** — a new RPC command that lets each worker read its tensor shards directly from the local filesystem, instead of the head streaming bytes over the socket.
2. **Pinned host staging on the rpc-server** — a ring of `cudaHostAlloc`'d buffers on each worker so HtoD copies go through pinned memory, not pageable `std::vector`.
3. **Worker QD=8 pread pool + dedicated ack-sender** — eight concurrent `pread` threads per worker, with a separate ack-sender thread so the socket never blocks on in-flight copies.
4. **Async head dispatch + head-side pread pool** — the head no longer blocks on RPC replies per tensor, and the head-local GPU's shard reads go through an 8-way POSIX `pread` pool.
5. **`tset` mutex on the worker pool** — serialising `ggml_backend_tensor_set` across the eight worker threads to avoid CUDA context thrash under concurrency.

## Why

Large MoE models in the 200–600 GB range no longer fit on any single device, so multi-node RPC sharding has become the realistic deployment pattern. But the stock RPC loader pulls tensors serially over a single socket — the head opens the file, reads one shard at a time, and pushes bytes to the target worker before moving on. For a 318 GB model that's ~12 minutes of cold load, with the GPUs idle most of the way.

Twelve minutes matters: every restart after a crash, binary update, or config change blocks serving that long; dev iteration on any loader or backend change costs a twelve-minute round trip per build; failover from one head to another is as slow as a cold start.

The fix is RPC-specific, doesn't touch inference, and is gated behind env vars — which made it seem worth publishing the archive. The failure modes navigated along the way (an mmap regression, a QD=8 deadlock on the first attempt, a dedicated-uploader variant that regressed startup time, and a UMA-memory finding that invalidates "warm page cache makes reads free" intuition on Grace-Blackwell hardware) are the kind of finding that costs each team a week to rediscover independently.

## Prerequisites / assumptions

The changes rely on a specific shape of deployment. If any of these don't hold, the modified loader either won't help or won't work.

- **Model distribution — identical path on every node.** Every node (head + all RPC workers) must be able to open the GGUF at the *same absolute path*. Two working patterns:
  - *Local replication* — `rsync`/`scp` the file to each node under e.g. `/models/foo.gguf`. Fastest; each worker reads its shards off local NVMe.
  - *Shared NFS* — one export, mounted at identical mountpoints on every node (e.g. `/models` on all of them). Simpler ops; cold load is slower because every worker's `pread` goes over the NFS wire.

  Mixed is fine — some nodes local, some via NFS — as long as the path resolves everywhere. Workers open by path string, not by filename, so a worker with the file at a *different* path will fail at load.

- **Format.** GGUF, sharded or not. `-m` points at shard 1, llama.cpp resolves the rest. Other weight formats aren't supported on this path.

- **`--no-mmap` is required.** The loader uses explicit `pread` — the mmap fast-path was tried and regressed cold load ~3×.

- **CUDA backend on workers.** The pinned-staging path lives in the CUDA backend. Other backends fall back transparently to pageable `std::vector` but don't benefit from the staging ring.

- **Matching binaries.** Every node must run the same llama.cpp build — the RPC wire format is version-dependent.

- **Network.** Any TCP-reachable worker address works. Bandwidth and latency matter: the measurements here were taken on 200 GbE RDMA, but the code doesn't require it — plain Ethernet works, proportionally slower.

## How

The changes live in a companion fork of llama.cpp:

> **[luxemque/llama.cpp @ `parallel-rpc-loading`](https://github.com/luxemque/llama.cpp/tree/parallel-rpc-loading)**

Five commits, each a cohesive piece of the change:

| Commit | Piece |
|---|---|
| [`d42cc7d`](https://github.com/luxemque/llama.cpp/commit/d42cc7d) | `ggml-rpc`: `SET_TENSOR_FROM_FILE` + pinned host staging + load profiling |
| [`eb282ee`](https://github.com/luxemque/llama.cpp/commit/eb282ee) | `ggml-rpc`: worker `stff_pool` + dedicated ack-sender + `tset_mu` |
| [`6856c96`](https://github.com/luxemque/llama.cpp/commit/6856c96) | `llama-model-loader`: parallel-RPC load head stack (async dispatch + head pread pool) |
| [`0c322ff`](https://github.com/luxemque/llama.cpp/commit/0c322ff) | Docs: breadcrumb `PARALLEL_RPC_LOADING.md` pointing at this archive |
| [`dbea709`](https://github.com/luxemque/llama.cpp/commit/dbea709) | `ggml-rpc`: worker `pread` conversion + `LLAMA_RPC_MODEL_ROOT` path-root guard |

**For running it** — env vars, the exact `llama-server` command, the speedup table broken out by config — see [`PARALLEL_RPC_LOADING.md`](PARALLEL_RPC_LOADING.md) in this repo.

## Status

The branch is feature-gated behind env vars and has been stable in production on one cluster (1 head + 3 workers, all NVIDIA GB10 / Grace-Blackwell, 200 GbE) since April 2026. It is not upstreamed; an upstream PR against `ggml-org/llama.cpp` will not be opened as this would violate their contribution policy. If you are a maintainer and find this useful... feel free to use the approach/code. Having this in the official distribution would help others who do not have datacenter-class GPU equipment at their hands.

## License

Same as upstream llama.cpp: MIT. See the fork.
