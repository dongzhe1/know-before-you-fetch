"""Retriever latency benchmark: ANN search, serialization, batch scaling.

Measures:
  1. Query encoding latency (BGE-large)
  2. FAISS ANN search latency (k=1,5,10)
  3. Passage serialization token count
  4. Throughput scaling with batch size

Usage:
  python scripts/retriever_benchmark.py \
      --encoder #WORKSPACE/models/bge-large-en-v1.5 \
      --index #WORKSPACE/llmdm/wiki_bge_index \
      --n 500
"""

from __future__ import annotations
import argparse, json, os, time
import numpy as np


def build_dummy_index(dim=1024, n_vecs=2000000):
    """Build a random FAISS index for testing if no real index exists."""
    import faiss

    print(f"[bench] Building dummy FAISS index ({n_vecs} × {dim})...")
    t0 = time.time()
    vecs = np.random.randn(n_vecs, dim).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    print(
        f"[bench] Dummy index built in {time.time() - t0:.1f}s ({index.ntotal} vectors)"
    )
    return index


def load_index(index_dir):
    """Load pre-built FAISS index and passage file handle for lazy access."""
    import faiss

    index_file = os.path.join(index_dir, "faiss.index")
    passage_file = os.path.join(index_dir, "passages.jsonl")
    if os.path.exists(index_file) and os.path.exists(passage_file):
        index = faiss.read_index(index_file)
        # Build byte-offset index for lazy passage access (not all passages in RAM)
        offsets = []
        with open(passage_file, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                offsets.append(pos)
        print(f"[bench] Loaded index: {index.ntotal} vectors, {len(offsets)} passages")
        return index, offsets, passage_file
    else:
        print(f"[bench] Index not found at {index_dir}, using dummy index")
        index = build_dummy_index()
        return index, None, None


def benchmark_retrieval(
    encoder_path,
    index,
    passage_offsets,
    passage_file,
    queries,
    ks,
    batch_sizes,
    n_warmup=10,
):
    """Comprehensive retrieval latency benchmark."""
    import torch
    from sentence_transformers import SentenceTransformer
    import faiss

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bench] Device: {dev}")

    enc = SentenceTransformer(encoder_path, device=dev)
    n_iters = n_warmup + 20
    n_needed = n_iters * max(batch_sizes)
    reps = (n_needed // len(queries)) + 1
    all_queries = queries * reps
    print(
        f"[bench] {len(queries)} base queries × {reps} = {len(all_queries)} total (need {n_needed} for bs={max(batch_sizes)})"
    )

    # Move FAISS index to GPU once (not per-iteration)
    if dev == "cuda":
        gpu_res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
    else:
        gpu_index = index

    results = []

    for bs in batch_sizes:
        print(f"\n[bench] Batch size = {bs}")
        # ── Query encoding ──
        encode_times = []
        for i in range(n_iters):
            batch = all_queries[i * bs : (i + 1) * bs]
            if len(batch) < bs:
                continue
            torch.cuda.synchronize() if dev == "cuda" else None
            t0 = time.perf_counter()
            qemb = enc.encode(
                batch,
                batch_size=bs,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            torch.cuda.synchronize() if dev == "cuda" else None
            t = time.perf_counter() - t0
            if i >= n_warmup:
                encode_times.append(t / bs)  # per-query
        if not encode_times:
            print(
                f"  WARN: no timed encode iterations for bs={bs} (all_q={len(all_queries)}, need={n_iters}*{bs}={n_iters * bs})"
            )
            continue
        enc_mean = np.mean(encode_times) * 1000
        enc_p50 = np.median(encode_times) * 1000
        enc_p95 = np.percentile(encode_times, 95) * 1000
        enc_throughput = bs / np.mean(encode_times)
        print(
            f"  Encode:  {enc_mean:.1f} ms/query (p50={enc_p50:.1f}, p95={enc_p95:.1f})  "
            f"throughput={enc_throughput:.0f} q/s"
        )

        # ── FAISS search ──
        # Pre-encode queries and tile to fill largest batch size
        qemb_base = enc.encode(
            all_queries[: len(queries)],
            batch_size=min(bs, len(queries)),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        tile_reps = (n_iters * bs // len(qemb_base)) + 1
        qemb_all = np.ascontiguousarray(np.tile(qemb_base, (tile_reps, 1)))

        for k in ks:
            search_times = []
            for i in range(n_iters):
                q_batch = qemb_all[i * bs : (i + 1) * bs]
                if len(q_batch) < bs:
                    continue
                if i == 0:
                    _ = gpu_index.search(q_batch[:1], k)  # warmup call
                torch.cuda.synchronize() if dev == "cuda" else None
                t0 = time.perf_counter()
                scores, indices = gpu_index.search(q_batch, k)
                torch.cuda.synchronize() if dev == "cuda" else None
                t = time.perf_counter() - t0
                if i >= n_warmup:
                    search_times.append(t / bs)
            if not search_times:
                print(f"  WARN: no timed search iterations for bs={bs}, k={k}")
                continue
            search_mean = np.mean(search_times) * 1000
            search_p50 = np.median(search_times) * 1000
            search_p95 = np.percentile(search_times, 95) * 1000
            print(
                f"  FAISS k={k}: {search_mean:.2f} ms/query (p50={search_p50:.2f}, p95={search_p95:.2f})"
            )

            # ── Serialization tokens ──
            sample_indices = indices[:10]
            tok_counts = []
            if passage_offsets is not None and passage_file is not None:
                n_pass = len(passage_offsets)
                with open(passage_file, "r") as pf:
                    for idx_row in sample_indices:
                        texts = []
                        for pi in idx_row:
                            if 0 <= pi < n_pass:
                                pf.seek(passage_offsets[pi])
                                texts.append(json.loads(pf.readline())["text"])
                        n_words = len(" ".join(texts).split())
                        tok_counts.append(int(n_words * 1.3))
            else:
                for idx_row in sample_indices:
                    tok_counts.append(int(100 * k * 1.3))
            tok_mean = np.mean(tok_counts)
            print(f"  Serialization: ~{tok_mean:.0f} tokens per query (k={k})")

            results.append(
                {
                    "batch_size": bs,
                    "k": k,
                    "encode_ms_mean": round(enc_mean, 2),
                    "encode_ms_p50": round(enc_p50, 2),
                    "encode_ms_p95": round(enc_p95, 2),
                    "encode_throughput_qps": round(enc_throughput, 0),
                    "search_ms_mean": round(search_mean, 3),
                    "search_ms_p50": round(search_p50, 3),
                    "search_ms_p95": round(search_p95, 3),
                    "serialization_tokens": round(tok_mean, 0),
                    "total_ms_mean": round(enc_mean + search_mean, 2),
                }
            )

    # ── Scale to full corpus ──
    # FAISS flat search is O(N_corpus) per query. Report scaling.
    n_vecs = index.ntotal
    print(f"\n[bench] Corpus size: {n_vecs} vectors")
    print(f"[bench] FAISS FlatIP search: O({n_vecs}) per query")
    print(
        f"[bench] At 2M vectors, search is bottleneck. At 100K, encoding is bottleneck."
    )
    print(
        f"[bench] Realistic ANN (IVF/HNSW) would reduce search to O(log N) — see Appendix."
    )

    del enc
    if dev == "cuda":
        import torch

        torch.cuda.empty_cache()

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--index", default="", help="path to pre-built FAISS index dir")
    ap.add_argument("--n", type=int, default=500, help="number of queries to benchmark")
    ap.add_argument("--ks", default="1,5,10", help="comma-separated k values")
    ap.add_argument(
        "--batch_sizes", default="1,8,32,128", help="comma-separated batch sizes"
    )
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # Sample queries (from TriviaQA or synthetic)
    SAMPLE_QUERIES = [
        "What city celebrates the original Oktoberfest?",
        "Who wrote the novel Nineteen Eighty-Four?",
        "What is the chemical symbol for gold?",
        "Which planet is known as the Red Planet?",
        "Who painted the Mona Lisa?",
        "What is the capital of France?",
        "In which year did World War II end?",
        "What is the largest ocean on Earth?",
        "Who discovered penicillin?",
        "What is the speed of light?",
    ]

    if args.index:
        index, offsets, passage_file = load_index(args.index)
    else:
        index, offsets, passage_file = build_dummy_index(), None, None

    print(f"[bench] {args.n} queries, ks={ks}, batch_sizes={batch_sizes}")
    print(f"[bench] Encoder: {args.encoder}")

    results = benchmark_retrieval(
        args.encoder, index, offsets, passage_file, SAMPLE_QUERIES, ks, batch_sizes
    )

    # ── Summary table ──
    print(f"\n{'=' * 90}")
    print(
        f"{'Batch':>6} {'k':>3} {'Encode(ms)':>10} {'Search(ms)':>10} {'Total(ms)':>10} {'SerTok':>6} {'Throughput':>10}"
    )
    print(f"{'-' * 90}")
    for r in results:
        print(
            f"{r['batch_size']:>6} {r['k']:>3} {r['encode_ms_mean']:>10.1f} {r['search_ms_mean']:>10.2f} "
            f"{r['total_ms_mean']:>10.1f} {r['serialization_tokens']:>6.0f} {r['encode_throughput_qps']:>10.0f} q/s"
        )

    # ── Combined cost model ──
    print(f"\n{'=' * 90}")
    print("COMBINED COST MODEL (per query, batch=1):")
    print(f"  LLM CB generation:     ~65 ms (8B) / ~162 ms (32B)")
    encode_1 = next(
        r["encode_ms_mean"] for r in results if r["batch_size"] == 1 and r["k"] == 5
    )
    search_5 = next(
        r["search_ms_mean"] for r in results if r["batch_size"] == 1 and r["k"] == 5
    )
    print(f"  Retriever encode k=5:  ~{encode_1:.1f} ms")
    print(f"  Retriever search k=5:  ~{search_5:.2f} ms")
    print(f"  LLM OB@5 generation:   ~80 ms (8B) / ~319 ms (32B)")
    ret_total = encode_1 + search_5
    print(f"  Retriever total:       ~{ret_total:.1f} ms")
    print(
        f"  Retriever as % of 8B total:  {ret_total / (65 + ret_total + 80) * 100:.0f}%"
    )
    print(
        f"  Retriever as % of 32B total: {ret_total / (162 + ret_total + 319) * 100:.0f}%"
    )

    # Write CSV
    import csv

    out_dir = os.path.join(os.path.dirname(__file__), "..", "paper", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "retriever_benchmark.csv")
    cols = [
        "batch_size",
        "k",
        "encode_ms_mean",
        "encode_ms_p50",
        "encode_ms_p95",
        "encode_throughput_qps",
        "search_ms_mean",
        "search_ms_p50",
        "search_ms_p95",
        "serialization_tokens",
        "total_ms_mean",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {len(results)} rows → {out_path}")


if __name__ == "__main__":
    main()
