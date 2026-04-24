# turboload.llama.cpp

Research archive and documentation for the parallel-RPC loading work on
[llama.cpp](https://github.com/ggml-org/llama.cpp) — a set of changes that drops
cold model load over llama.cpp's RPC backend from ~12 minutes to ~60 seconds on
a three-worker cluster by making the head dispatch asynchronous and the worker
I/O path pinned + parallel.

The code lives in a companion fork:

> https://github.com/luxemque/llama.cpp/tree/parallel-rpc-loading

This repo holds the *story* of how those changes were arrived at: the
measurement sweeps, the refactors that were tried and rolled back, the
preconditions that matter, and the operational recipes for running it on a
cluster.

---

Curation in progress — more detail is being folded in from the staging
archive. If a section reads thin, that's why.
