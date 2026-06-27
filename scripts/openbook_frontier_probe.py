"""Contribution-level probe for Direction 2 — does a calibrated closed-book-confidence gate actually buy a
cost-quality frontier that DOMINATES fixed-k RAG and the obvious baselines?

The original go/no-go probe (retrieval_decision_probe.py) only proved the PREMISE: closed-book confidence
separates correct/wrong (AUROC ~0.79). That is selective prediction, not the paper. This probe tests the
CONTRIBUTION, which needs a real open-book pass:

  1. closed-book pass  -> per-query {answer, correct, seq_logprob, self_consistency}   (dumped to --dump)
  2. open-book pass     -> dense-retrieve top-k from the question's own candidate pool (bge) + RAG answer,
                           at k_small and k_large; per-query open_correct@k
  3. the gate frontier  -> calibrate P(correct | confidence[, features]) out-of-fold (no leakage); sweep tau:
                           skip (use closed answer) when confident, else retrieve@k. accuracy vs passages-read.
  4. baselines on the same axes: never-retrieve (closed only), always-retrieve fixed-k, random-skip,
                           ORACLE gate (skip iff closed-correct = ceiling), popularity/length heuristic gate,
                           and an Adaptive-RAG-style query-feature classifier (no confidence) at matched cost.
  5. noise-avoidance    -> among closed-CORRECT queries, how often does retrieval FLIP them to wrong?
  6. transfer (optional)-> train the gate on this set, apply to --transfer_data (gate must be dataset-robust).

GO only if the gate frontier beats random + heuristic + adaptive-rag-proxy AND reaches always-retrieve
accuracy at a retrieval rate < 100% (real cost saving). Uses datasets that ship a candidate pool
(hotpotqa / triviaqa_rc from prepare_qa.py) so it runs OFFLINE with NO Wikipedia index. GPU.

Usage:
  python scripts/openbook_frontier_probe.py --data data/hotpotqa.jsonl \
      --model #WORKSPACE/models/Qwen3-8B --encoder #WORKSPACE/models/bge-large-en-v1.5 \
      --n 600 --k_small 1 --k_large 5 --self_k 5 --dump logs/hotpotqa_table.jsonl \
      --transfer_data data/triviaqa_rc.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string

import numpy as np


def _mem_info():
    """Return (current_rss_mb, cgroup_usage_mb, cgroup_limit_mb)."""
    # Current RSS from /proc (not peak)
    rss_mb = -1
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_mb = int(line.split()[1]) / 1024  # KB→MB
                    break
    except Exception:
        pass
    # Cgroup memory (enforced by the container/cgroup runtime)
    cg_usage, cg_limit = -1, -1
    for cg_base in ["/sys/fs/cgroup", "/sys/fs/cgroup/memory"]:
        for usage_file in ["memory.current", "memory.usage_in_bytes"]:
            try:
                with open(os.path.join(cg_base, usage_file)) as f:
                    cg_usage = int(f.read().strip()) / (1024 * 1024)
                break
            except Exception:
                pass
        for limit_file in ["memory.max", "memory.limit_in_bytes"]:
            try:
                with open(os.path.join(cg_base, limit_file)) as f:
                    val = f.read().strip()
                    if val != "max" and int(val) < 2**60:
                        cg_limit = int(val) / (1024 * 1024)
                break
            except Exception:
                pass
        if cg_usage > 0:
            break
    return rss_mb, cg_usage, cg_limit


def _log_mem(tag):
    rss, cg_use, cg_lim = _mem_info()
    lim_str = f"/{cg_lim:.0f}" if cg_lim > 0 else ""
    cg_str = f"  cgroup={cg_use:.0f}{lim_str} MB" if cg_use > 0 else ""
    print(f"[MEM] {tag}: RSS={rss:.0f} MB{cg_str}", flush=True)


def load_jsonl(p, n=0, seed=42):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    if n and n < len(rows):
        rng = np.random.default_rng(seed)
        rows = [rows[i] for i in rng.choice(len(rows), size=n, replace=False)]
    return rows


def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def correct(pred, gold, aliases):
    p = norm(pred)
    gs = [norm(g) for g in [gold] + list(aliases) if g]
    return int(bool(p) and any(p == g or (len(g) >= 3 and g in p) for g in gs))


def build_lm(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if dev == "cuda" else torch.float32),
        device_map=("auto" if dev == "cuda" else None),
    )
    if dev == "cpu":
        model.to(dev)
    model.eval()
    return tok, model, ("qwen3" in path.lower().replace("/", ""))


def chatify(tok, is_q3, content):
    if getattr(tok, "chat_template", None):
        kw = dict(tokenize=False, add_generation_prompt=True)
        if is_q3:
            kw["enable_thinking"] = False
        return tok.apply_chat_template([{"role": "user", "content": content}], **kw)
    return content + "\nAnswer:"


def _free_cuda():
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _gen_adaptive(model, inp, batch, **gen_kw):
    """model.generate that halves the (sub-)batch and retries on CUDA OOM.

    Splits the already-tokenized `inp` along dim 0 into sub-batches of size `cur`;
    on OutOfMemoryError it frees the cache, halves `cur`, and retries from scratch.
    Returns (seqs, scores_list, settled_batch) so the caller can carry the shrunk
    batch into the next chunk (avoids re-discovering the OOM every chunk). Long
    prompts on big models (e.g. open-book k=5 on 32B) self-tune down to a fit.
    """
    import torch

    n = inp["input_ids"].shape[0]
    cur = max(1, batch)
    while True:
        try:
            seqs, scores_list = [], []
            for s in range(0, n, cur):
                sub = {k: v[s : s + cur] for k, v in inp.items()}
                with torch.no_grad():
                    out = model.generate(**sub, **gen_kw)
                if gen_kw.get("return_dict_in_generate"):
                    seqs.append(out.sequences.cpu())
                    if gen_kw.get("output_scores"):
                        # [steps, sub_b, vocab] -> log-probs, kept on CPU to save VRAM
                        scores_list.append(
                            torch.log_softmax(
                                torch.stack(out.scores, 1).float(), -1
                            ).cpu()
                        )
                else:
                    seqs.append(out.cpu())
            return seqs, scores_list, cur
        except torch.OutOfMemoryError:
            _free_cuda()
            if cur == 1:
                raise
            cur = max(1, cur // 2)
            print(f"[oom] retrying generate with batch={cur}", flush=True)


def greedy(tok, model, prompts, batch, max_len, want_lp=False, max_new=24):
    """Greedy decode; returns (preds, mean-token-logprob-or-None). OOM-robust."""
    import torch

    preds, lps = [], []
    eff = max(1, batch)
    for i in range(0, len(prompts), batch):
        ch = prompts[i : i + batch]
        inp = tok(
            ch, return_tensors="pt", padding=True, truncation=True, max_length=max_len
        ).to(model.device)
        seqs, scores_list, eff = _gen_adaptive(
            model,
            inp,
            eff,
            do_sample=False,
            max_new_tokens=max_new,
            pad_token_id=tok.pad_token_id,
            return_dict_in_generate=True,
            output_scores=want_lp,
        )
        plen = inp["input_ids"].shape[1]
        for si, seq in enumerate(seqs):
            gen = seq[:, plen:]
            lp = scores_list[si] if want_lp else None
            for b in range(seq.shape[0]):
                toks = gen[b]
                m = toks != tok.pad_token_id
                preds.append(
                    tok.decode(toks, skip_special_tokens=True).strip().split("\n")[0]
                )
                if want_lp:
                    tlp = lp[b, torch.arange(len(toks)), toks][m[: len(toks)]]
                    lps.append(float(tlp.mean()) if len(tlp) else -20.0)
        del inp
    _free_cuda()
    return preds, (np.array(lps) if want_lp else None)


def self_consistency(tok, model, prompts, batch, max_len, k, temp=0.7):
    """Self-consistency agreement per prompt. OOM-robust (k return-sequences are heavy)."""
    agree = []
    eff = max(1, batch)
    for i in range(0, len(prompts), batch):
        ch = prompts[i : i + batch]
        inp = tok(
            ch, return_tensors="pt", padding=True, truncation=True, max_length=max_len
        ).to(model.device)
        seqs, _, eff = _gen_adaptive(
            model,
            inp,
            eff,
            do_sample=True,
            temperature=temp,
            top_p=0.95,
            num_return_sequences=k,
            max_new_tokens=24,
            pad_token_id=tok.pad_token_id,
        )
        plen = inp["input_ids"].shape[1]
        dec = []
        for seq in seqs:
            dec.extend(tok.batch_decode(seq[:, plen:], skip_special_tokens=True))
        for j in range(len(ch)):
            a = [norm(s.strip().split("\n")[0]) for s in dec[j * k : (j + 1) * k]]
            agree.append(max([a.count(x) for x in set(a)] or [0]) / k)
        del inp
    _free_cuda()
    return np.array(agree)


def retrieve(encoder, rows, k, batch=256):
    """Dense top-k passage indices per row from its own candidate pool (offline, no global index)."""
    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer(encoder, device=dev)
    qemb = enc.encode(
        [r["question"] for r in rows],
        batch_size=batch,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    topk = []
    for i, r in enumerate(rows):
        ctx = r.get("contexts") or []
        if not ctx:
            topk.append([])
            continue
        cemb = enc.encode(
            ctx,
            batch_size=batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        sims = cemb @ qemb[i]
        order = np.argsort(-sims)[:k]
        topk.append([ctx[j] for j in order])
    del enc
    if dev == "cuda":
        torch.cuda.empty_cache()
    return topk


class WikiIndexRetriever:
    """Retrieve passages from a pre-built Wikipedia FAISS index.

    Memory-efficient: FAISS index is mmap'd (not loaded into RAM),
    passages are read on-demand via byte offsets (not kept in memory).
    """

    def __init__(self, index_dir, encoder_path):
        import faiss, torch
        from sentence_transformers import SentenceTransformer

        dev = "cuda" if torch.cuda.is_available() else "cpu"

        _log_mem("  WikiInit: before offset scan")
        self._passage_file = os.path.join(index_dir, "passages.jsonl")
        if not os.path.exists(self._passage_file):
            raise FileNotFoundError(f"passages.jsonl not found in {index_dir}")
        self._offsets = []
        with open(self._passage_file, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                self._offsets.append(pos)
        self._n_passages = len(self._offsets)
        print(f"[wiki_index] Indexed {self._n_passages} passage offsets", flush=True)
        _log_mem("  WikiInit: after offset scan")

        index_file = os.path.join(index_dir, "faiss.index")
        self.index = faiss.read_index(index_file, faiss.IO_FLAG_MMAP)
        print(
            f"[wiki_index] FAISS index: {self.index.ntotal} vectors (mmap)", flush=True
        )
        _log_mem("  WikiInit: after faiss mmap")

        self.encoder = SentenceTransformer(encoder_path, device=dev)
        _log_mem("  WikiInit: after encoder load")
        self.dev = dev

    def _get_passages(self, indices):
        """Read specific passages from disk by byte offset."""
        results = []
        with open(self._passage_file, "r") as f:
            for idx in indices:
                if 0 <= idx < self._n_passages:
                    f.seek(self._offsets[idx])
                    results.append(json.loads(f.readline())["text"])
        return results

    def retrieve(self, questions, k, batch=256):
        """Retrieve top-k passages for each question from the Wikipedia index."""
        import torch

        _log_mem(f"  retrieve(k={k}): before encode")
        qemb = self.encoder.encode(
            questions,
            batch_size=batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        _log_mem(f"  retrieve(k={k}): after encode, before faiss.search")
        scores, indices = self.index.search(qemb, k)
        _log_mem(f"  retrieve(k={k}): after faiss.search")
        topk = []
        for idx_row in indices:
            topk.append(self._get_passages(idx_row))
        return topk

    def cleanup(self):
        import torch

        del self.encoder, self.index
        self._offsets = []
        if self.dev == "cuda":
            torch.cuda.empty_cache()


def build_shared_corpus(encoder, rows, batch=256):
    """Realistic-retrieval mode: pool EVERY row's passages into one shared corpus and retrieve
    each query against the whole pool (so distractors from other questions compete). Returns a
    closure retrieve_k(k)->list[list[str]] and a recall@k helper. This is the credible middle
    ground between per-question oracle pools and a full Wikipedia index."""
    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer(encoder, device=dev)
    # unique passage pool + map passage -> owner row indices (for recall proxy)
    pool, owners = [], {}
    own_sets = []
    for i, r in enumerate(rows):
        s = set()
        for p in r.get("contexts") or []:
            if p not in owners:
                owners[p] = len(pool)
                pool.append(p)
            s.add(owners[p])
        own_sets.append(s)
    print(
        f"[shared] corpus size = {len(pool)} unique passages from {len(rows)} questions",
        flush=True,
    )
    pemb = enc.encode(
        pool,
        batch_size=batch,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    qemb = enc.encode(
        [r["question"] for r in rows],
        batch_size=batch,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    sims = qemb @ pemb.T  # [nq, npool]
    del enc
    if dev == "cuda":
        torch.cuda.empty_cache()

    def retrieve_k(k):
        topk, hit = [], 0
        for i in range(len(rows)):
            order = np.argsort(-sims[i])[:k]
            topk.append([pool[j] for j in order])
            if own_sets[i] & set(order.tolist()):
                hit += 1
        recall = hit / len(rows)
        print(
            f"[shared] recall@{k} (own passage in shared top-k) = {recall:.3f}",
            flush=True,
        )
        return topk

    return retrieve_k


def openbook_prompt(tok, is_q3, q, passages):
    body = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    c = (
        "Use the passages to answer the question with a short answer of a few words.\n"
        f"Passages:\n{body}\n\nQuestion: {q}"
    )
    return chatify(tok, is_q3, c)


def calibrated_oof(X, y):
    """Out-of-fold calibrated P(correct) via CV logistic (no leakage). Returns prob array."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if y.min() == y.max():
        return np.full(len(y), float(y.mean()))
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return cross_val_predict(pipe, X, y, cv=5, method="predict_proba")[:, 1]


def frontier(skip_score, closed_y, open_y, k, n_pts=41):
    """Higher skip_score => more likely to SKIP retrieval. Sweep threshold -> (retr_rate, accuracy)."""
    qs = np.quantile(skip_score, np.linspace(0, 1, n_pts))
    rr, acc = [], []
    for tau in np.unique(np.r_[qs, skip_score.max() + 1]):
        skip = skip_score >= tau
        a = np.where(skip, closed_y, open_y).mean()
        rr.append(float((~skip).mean()))
        acc.append(float(a))
    o = np.argsort(rr)
    return np.array(rr)[o], np.array(acc)[o]


def graded_frontier(
    skip_score, closed_y, oy_small, oy_large, k_small, k_large, n_pts=41
):
    """3-tier Pareto frontier: most-confident queries skip, middle tier gets k_small,
    least-confident gets k_large. X-axis: avg_passages / k_large (same scale as frontier()).
    Strictly dominates the binary frontier because it has an extra degree of freedom."""
    if k_small == k_large:
        return frontier(skip_score, closed_y, oy_large, k_large, n_pts)
    n = len(skip_score)
    order = np.argsort(-skip_score)  # descending: most confident first
    c_y = closed_y[order]
    s_y = oy_small[order]
    l_y = oy_large[order]
    pts = []
    for alpha in np.linspace(0, 1, n_pts):  # fraction that skip
        n_skip = int(round(alpha * n))
        rest = n - n_skip
        for beta in np.linspace(0, 1, n_pts):  # fraction of remaining that get k_small
            n_small = int(round(beta * rest))
            n_large = rest - n_small
            acc = (
                c_y[:n_skip].sum()
                + s_y[n_skip : n_skip + n_small].sum()
                + l_y[n_skip + n_small :].sum()
            ) / n
            avg_psg = (
                (n_small * k_small + n_large * k_large) / n / k_large
            )  # normalized to [0,1]
            pts.append((avg_psg, float(acc)))
    # upper Pareto envelope: at each budget level, keep the max accuracy seen
    pts.sort()
    rr_arr = np.array([p[0] for p in pts])
    ac_arr = np.array([p[1] for p in pts])
    grid = np.linspace(0, 1, 101)
    envelope = np.array(
        [
            ac_arr[rr_arr <= r + 1e-6].max()
            if (rr_arr <= r + 1e-6).any()
            else float("nan")
            for r in grid
        ]
    )
    return grid, envelope


def acc_at(rr, acc, target):
    return float(np.interp(target, rr, acc))


def fr_auc(rr, acc):
    """Area under accuracy-vs-retrieval-rate (higher = better frontier). Trapz over [0,1]."""
    g = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(g, rr, acc), g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--encoder",
        required=True,
        help="dense retriever (bge/e5) for the open-book pass",
    )
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--k_small", type=int, default=1)
    ap.add_argument("--k_large", type=int, default=5)
    ap.add_argument(
        "--k_list",
        default="",
        help="comma-sep extra k values to also run open-book at "
        "and dump open_correct_k{k} (for the budget-depth sweep + harm-vs-k); "
        "e.g. '2,3,10'. k_small/k_large still drive the reported frontier.",
    )
    ap.add_argument("--self_k", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument(
        "--dump", default="", help="write the per-query table to this jsonl"
    )
    ap.add_argument(
        "--transfer_data", default="", help="apply the gate trained here to this set"
    )
    ap.add_argument(
        "--shared_corpus",
        action="store_true",
        help="retrieve from a pooled shared corpus (realistic retrieval), not per-question pools",
    )
    ap.add_argument(
        "--wiki_index",
        default="",
        help="path to pre-built Wikipedia FAISS index directory (from build_wiki_index.py)",
    )
    ap.add_argument(
        "--dump_open_conf",
        action="store_true",
        help="also record the open-book answer's seq-logprob (for selective-RAG abstention)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _log_mem("startup")
    rows = load_jsonl(args.data, args.n, args.seed)
    if not any(r.get("contexts") for r in rows):
        print(
            "[D2+] ERROR: this data has no per-row 'contexts'; use hotpotqa/triviaqa_rc from prepare_qa.py."
        )
        return
    print(
        f"[D2+] {args.data} | n={len(rows)} | model={args.model} | k_small={args.k_small} k_large={args.k_large}"
    )
    _log_mem("after load_jsonl")

    tok, model, is_q3 = build_lm(args.model)
    _log_mem("after build_lm")

    # ---- 1. closed-book ----
    cb_prompts = [
        chatify(
            tok,
            is_q3,
            "Answer the question with a short answer of a few words, using only "
            f"your own knowledge.\n\nQuestion: {r['question']}",
        )
        for r in rows
    ]
    cb_pred, seqlp = greedy(
        tok, model, cb_prompts, args.batch, args.max_len, want_lp=True
    )
    closed_y = np.array(
        [correct(p, r["gold"], r.get("aliases", [])) for p, r in zip(cb_pred, rows)]
    )
    _log_mem("after closed-book greedy")
    selfc = (
        self_consistency(tok, model, cb_prompts, args.batch, args.max_len, args.self_k)
        if args.self_k > 0
        else np.zeros(len(rows))
    )
    print(f"[D2+] closed-book accuracy = {closed_y.mean():.3f}")
    _log_mem("after self-consistency")

    # ---- 2. open-book at k_small / k_large ----
    # Determine retrieval mode (priority: wiki_index > shared_corpus > per-query pool)
    wiki_retr = None
    shared_retr = None
    if args.wiki_index:
        import gc

        gc.collect()
        _free_cuda()
        print(f"[D2+] RETRIEVAL MODE = Wikipedia index ({args.wiki_index})", flush=True)
        _log_mem("before WikiIndexRetriever init")
        wiki_retr = WikiIndexRetriever(args.wiki_index, args.encoder)
        _log_mem("after WikiIndexRetriever init")
        print(
            f"[D2+] Corpus recall proxy: Wikipedia index with {wiki_retr.index.ntotal} passages",
            flush=True,
        )
    elif args.shared_corpus:
        shared_retr = build_shared_corpus(args.encoder, rows)
        print("[D2+] RETRIEVAL MODE = shared pooled corpus (realistic)")
    else:
        print("[D2+] RETRIEVAL MODE = per-question candidate pool (oracle pool)")

    open_y = {}
    open_lp = None
    extra_k = (
        {int(x) for x in args.k_list.split(",") if x.strip()} if args.k_list else set()
    )
    for k in sorted({args.k_small, args.k_large} | extra_k):
        _log_mem(f"before retrieve k={k}")
        if wiki_retr:
            topk = wiki_retr.retrieve([r["question"] for r in rows], k)
            _log_mem(f"after wiki retrieve k={k}")
        elif shared_retr:
            topk = shared_retr(k)
        else:
            topk = retrieve(args.encoder, rows, k)
        ob_prompts = [
            openbook_prompt(tok, is_q3, r["question"], topk[i])
            for i, r in enumerate(rows)
        ]
        want = args.dump_open_conf and k == args.k_large
        import gc

        gc.collect()
        _free_cuda()
        _log_mem(f"before greedy k={k} (want_lp={want})")
        ob_pred, ob_lp = greedy(
            tok, model, ob_prompts, args.batch, args.max_len, want_lp=want
        )
        _log_mem(f"after greedy k={k}")
        oy = np.array(
            [correct(p, r["gold"], r.get("aliases", [])) for p, r in zip(ob_pred, rows)]
        )
        open_y[k] = oy
        if want:
            open_lp = ob_lp
        print(f"[D2+] open-book accuracy @k={k} = {oy.mean():.3f}")

    if wiki_retr:
        _log_mem("before wiki cleanup")
        wiki_retr.cleanup()
        del wiki_retr
        _log_mem("after wiki cleanup")

    # ---- query features (for the heuristic / Adaptive-RAG-style baselines + gate) ----
    qlen = np.array([len(r["question"].split()) for r in rows], dtype=float)
    pop = np.array([float(r.get("popularity", 0.0)) for r in rows])
    has_pop = pop.any()
    feats = np.column_stack([qlen] + ([np.log1p(pop)] if has_pop else []))

    # signals for the gate: confidence (+ features). skip when P(correct) high.
    conf = np.column_stack([seqlp, selfc])
    p_correct = calibrated_oof(
        np.column_stack([conf, feats]), closed_y
    )  # the gate score = P(closed correct)
    from sklearn.metrics import roc_auc_score

    auroc = (
        roc_auc_score(closed_y, p_correct)
        if closed_y.min() != closed_y.max()
        else float("nan")
    )
    print(f"[D2+] gate P(correct) AUROC (oof) = {auroc:.3f}")

    if args.dump:
        with open(args.dump, "w") as f:
            for i, r in enumerate(rows):
                f.write(
                    json.dumps(
                        {
                            "question": r["question"],
                            "gold": r["gold"],
                            "closed_pred": cb_pred[i],
                            "closed_correct": int(closed_y[i]),
                            "seq_logprob": float(seqlp[i]),
                            "self_consistency": float(selfc[i]),
                            "p_correct": float(p_correct[i]),
                            "qlen": float(qlen[i]),
                            "popularity": float(pop[i]),
                            **{f"open_correct_k{k}": int(open_y[k][i]) for k in open_y},
                            **(
                                {f"open_seq_logprob_k{args.k_large}": float(open_lp[i])}
                                if open_lp is not None
                                else {}
                            ),
                        }
                    )
                    + "\n"
                )
        print(f"[D2+] dumped per-query table -> {args.dump}")

    # ---- 3/4. frontier: gate vs baselines, at k_large ----
    K = args.k_large
    oy = open_y[K]
    rng = np.random.default_rng(args.seed)
    methods = {
        "gate (confidence+feat)": frontier(p_correct, closed_y, oy, K),
        "confidence only": frontier(calibrated_oof(conf, closed_y), closed_y, oy, K),
        "adaptive-rag proxy": frontier(
            calibrated_oof(feats, closed_y), closed_y, oy, K
        ),
        "random-skip": frontier(rng.random(len(rows)), closed_y, oy, K),
    }
    if has_pop:
        methods["popularity heuristic"] = frontier(np.log1p(pop), closed_y, oy, K)
    else:
        methods["length heuristic"] = frontier(-qlen, closed_y, oy, K)
    # graded 3-tier (skip / k_small / k_large) — paper's main contribution vs binary skip
    graded_key = f"graded (skip/{args.k_small}/{K})"
    if args.k_small != args.k_large and args.k_small in open_y:
        methods[graded_key] = graded_frontier(
            p_correct, closed_y, open_y[args.k_small], oy, args.k_small, K
        )
    # oracle gate: skip iff closed-correct (the ceiling)
    or_rr = float((~closed_y.astype(bool)).mean())
    or_acc = float(np.where(closed_y.astype(bool), closed_y, oy).mean())

    always_acc = float(oy.mean())
    never_acc = float(closed_y.mean())
    print(
        f"\n=== cost-quality frontier @k={K}  (anchors: never={never_acc:.3f}@0%, always={always_acc:.3f}@100%) ==="
    )
    print(f"  {'method':<28} {'AUC':>6} {'acc@25%':>8} {'acc@50%':>8} {'acc@75%':>8}")
    auc = {}
    for nme, (rr, ac) in methods.items():
        auc[nme] = fr_auc(rr, ac)
        print(
            f"  {nme:<28} {auc[nme]:6.3f} {acc_at(rr, ac, 0.25):8.3f} {acc_at(rr, ac, 0.50):8.3f} {acc_at(rr, ac, 0.75):8.3f}"
        )
    print(
        f"  {'ORACLE gate':<28} {'':>6} (acc={or_acc:.3f} at retr-rate {or_rr * 100:.0f}%)"
    )

    # ---- 5. noise-avoidance: retrieval flipping confident-correct queries to wrong ----
    cc = closed_y.astype(bool)
    if cc.sum():
        flip = float(
            ((oy == 0) & cc).mean() / cc.mean()
        )  # P(open wrong | closed correct)
        ob_on_cc = float(oy[cc].mean())
        print(f"\n=== noise-avoidance ===")
        print(
            f"  among closed-CORRECT queries: open-book acc = {ob_on_cc:.3f} -> retrieval FLIPS {flip * 100:.1f}% to WRONG"
        )
        print(f"  (a skip-the-confident gate avoids exactly these losses)")

    # ---- 6. transfer ----
    if args.transfer_data:
        trows = load_jsonl(args.transfer_data, args.n, args.seed)
        if any(r.get("contexts") for r in trows):
            print(f"\n=== transfer -> {args.transfer_data} (n={len(trows)}) ===")
            tcb = [
                chatify(
                    tok,
                    is_q3,
                    "Answer the question with a short answer of a few words, using only "
                    f"your own knowledge.\n\nQuestion: {r['question']}",
                )
                for r in trows
            ]
            tpred, tlp = greedy(tok, model, tcb, args.batch, args.max_len, want_lp=True)
            tcy = np.array(
                [
                    correct(p, r["gold"], r.get("aliases", []))
                    for p, r in zip(tpred, trows)
                ]
            )
            tsc = (
                self_consistency(tok, model, tcb, args.batch, args.max_len, args.self_k)
                if args.self_k > 0
                else np.zeros(len(trows))
            )
            ttop = retrieve(args.encoder, trows, K)
            top_pred, _ = greedy(
                tok,
                model,
                [
                    openbook_prompt(tok, is_q3, r["question"], ttop[i])
                    for i, r in enumerate(trows)
                ],
                args.batch,
                args.max_len,
            )
            toy = np.array(
                [
                    correct(p, r["gold"], r.get("aliases", []))
                    for p, r in zip(top_pred, trows)
                ]
            )
            # fit gate on SOURCE confidence, apply to TARGET confidence (transfer, no target labels)
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            clf = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000)
            ).fit(conf, closed_y)
            tp = clf.predict_proba(np.column_stack([tlp, tsc]))[:, 1]
            rr, ac = frontier(tp, tcy, toy, K)
            rr_r, ac_r = frontier(rng.random(len(trows)), tcy, toy, K)
            print(
                f"  transferred gate  AUC={fr_auc(rr, ac):.3f}  acc@50%={acc_at(rr, ac, 0.5):.3f}  "
                f"(random-skip AUC={fr_auc(rr_r, ac_r):.3f})"
            )
            print(f"  anchors: never={tcy.mean():.3f}  always@k={K}={toy.mean():.3f}")

    # ---- VERDICT ----
    g_auc = auc["gate (confidence+feat)"]
    beats = all(
        g_auc >= auc[b] + 0.002
        for b in ["random-skip"]
        + (["popularity heuristic"] if has_pop else ["length heuristic"])
    )
    # does the gate reach always-retrieve accuracy below 100% retrieval (real saving)?
    grr, gac = methods["gate (confidence+feat)"]
    saving = gac >= always_acc - 0.005
    save_rr = float(grr[np.argmax(gac >= always_acc - 0.005)]) if saving.any() else 1.0
    print("\n--- VERDICT ---")
    if beats and saving.any() and save_rr < 0.95:
        print(
            f"  GO: gate frontier dominates random+heuristic (AUC {g_auc:.3f}) AND reaches always-retrieve"
        )
        print(
            f"      accuracy ({always_acc:.3f}) at ~{save_rr * 100:.0f}% retrieval -> real cost saving. Build D2."
        )
    elif not beats:
        print(
            f"  NO-GO: gate frontier (AUC {g_auc:.3f}) does NOT beat the cheap baselines -> calibration adds nothing."
        )
    else:
        print(
            f"  PARTIAL: gate beats baselines but needs ~100% retrieval to match fixed-k -> saving is thin."
        )
    print("=================================================")


if __name__ == "__main__":
    main()
