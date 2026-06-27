"""Prepare open-QA data on a LOGIN NODE (internet) for the Direction-2 probes.

  triviaqa   : mandarjoshi/trivia_qa (rc.nocontext) -> {"question","gold","aliases":[...]}
  nq         : google-research-datasets/nq_open      -> {"question","gold","aliases":[...]}
  popqa      : akariasai/PopQA                        -> {... ,"popularity": s_pop}   (no contexts)
  hotpotqa   : hotpotqa/hotpot_qa (distractor)        -> {... ,"contexts":[passage,...]}  (gold+distractors)
  triviaqa_rc: mandarjoshi/trivia_qa (rc)             -> {... ,"contexts":[passage,...]}  (wiki/web evidence)
  msmarco    : ms_marco v1.1 (factoid query types)    -> {... ,"contexts":[passage,...]}  (BM25-retrieved pool)

The *_rc / hotpotqa / msmarco variants ship a per-question candidate passage pool, so the open-book
frontier probe (scripts/openbook_frontier_probe.py) can run a real RAG pass OFFLINE. msmarco is the
second main frontier dataset; it covers different topics from TriviaQA and has per-question passages.

Usage:
  python scripts/prepare_qa.py --task triviaqa    --n 800 --out data/triviaqa.jsonl
  python scripts/prepare_qa.py --task hotpotqa    --n 800 --out data/hotpotqa.jsonl
  python scripts/prepare_qa.py --task triviaqa_rc --n 800 --out data/triviaqa_rc.jsonl
  python scripts/prepare_qa.py --task popqa       --n 800 --out data/popqa.jsonl
  python scripts/prepare_qa.py --task msmarco     --n 800 --out data/msmarco.jsonl
  python scripts/prepare_qa.py --task nq_dpr      --n 800 --out data/nq_dpr.jsonl
  # nq_dpr prereq: wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz -O data/biencoder-nq-dev.json.gz
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _trim(t, max_chars=1200):
    t = " ".join(str(t).split())
    return t[:max_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task",
        choices=[
            "triviaqa",
            "nq",
            "popqa",
            "hotpotqa",
            "triviaqa_rc",
            "msmarco",
            "nq_dpr",
        ],
        default="triviaqa",
    )
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument(
        "--max_ctx", type=int, default=20, help="cap candidate passages per question"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    rng = np.random.default_rng(args.seed)
    rows = []

    if args.task == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=args.split)
        for r in ds:
            a = r.get("answer") or {}
            gold = a.get("value")
            if r.get("question") and gold:
                rows.append(
                    {
                        "question": str(r["question"]),
                        "gold": str(gold),
                        "aliases": [str(x) for x in (a.get("aliases") or [])],
                    }
                )

    elif args.task == "nq":
        ds = load_dataset("google-research-datasets/nq_open", split=args.split)
        for r in ds:
            ans = r.get("answer") or []
            if r.get("question") and ans:
                rows.append(
                    {
                        "question": str(r["question"]),
                        "gold": str(ans[0]),
                        "aliases": [str(x) for x in ans],
                    }
                )

    elif args.task == "popqa":
        # akariasai/PopQA: question, possible_answers (json list), s_pop (subject popularity)
        ds = load_dataset("akariasai/PopQA", split="test")
        for r in ds:
            try:
                aliases = json.loads(r.get("possible_answers") or "[]")
            except Exception:
                aliases = []
            if r.get("question") and aliases:
                rows.append(
                    {
                        "question": str(r["question"]),
                        "gold": str(aliases[0]),
                        "aliases": [str(x) for x in aliases],
                        "popularity": float(r.get("s_pop") or 0.0),
                    }
                )

    elif args.task == "hotpotqa":
        ds = load_dataset(
            "hotpotqa/hotpot_qa", "distractor", split=args.split, trust_remote_code=True
        )
        for r in ds:
            q, gold = r.get("question"), r.get("answer")
            ctx = r.get("context") or {}
            titles = ctx.get("title") or []
            sents = ctx.get("sentences") or []
            passages = [_trim(t + ". " + " ".join(s)) for t, s in zip(titles, sents)]
            passages = [p for p in passages if p][: args.max_ctx]
            if q and gold and passages:
                rows.append(
                    {
                        "question": str(q),
                        "gold": str(gold),
                        "aliases": [str(gold)],
                        "contexts": passages,
                    }
                )

    elif args.task == "nq_dpr":
        # NQ dev set from DPR biencoder data (Facebook AI Research).
        # Prereq (login node): wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz \
        #   -O data/biencoder-nq-dev.json.gz
        # Per-question pool = positive_ctxs + hard_negative_ctxs + negative_ctxs.
        # bge re-ranks this pool; gold passage is always present so open-book can succeed.
        import gzip, pathlib

        # accept both compressed (.json.gz) and plain (.json) versions
        for candidate in (
            "data/biencoder-nq-dev.json.gz",
            "data/biencoder-nq-dev.json",
        ):
            if pathlib.Path(candidate).exists():
                dpr_path = candidate
                break
        opener = gzip.open if dpr_path.endswith(".gz") else open
        with opener(dpr_path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
        for r in data:
            answers = [str(a) for a in (r.get("answers") or []) if a]
            if not answers:
                continue
            all_ctxs = (
                (r.get("positive_ctxs") or [])
                + (r.get("hard_negative_ctxs") or [])
                + (r.get("negative_ctxs") or [])
            )
            passages = [
                _trim(c.get("title", "") + ". " + c.get("text", ""))
                for c in all_ctxs[: args.max_ctx]
                if c.get("text")
            ]
            if r.get("question") and passages:
                rows.append(
                    {
                        "question": str(r["question"]),
                        "gold": answers[0],
                        "aliases": answers,
                        "contexts": passages,
                    }
                )

    elif args.task == "triviaqa_rc":
        # cap shard loading: train split has 78k rows; slice to 5x needed to keep load fast
        cap = min(args.n * 5, 15000) if args.n else None
        load_split = f"{args.split}[:{cap}]" if cap else args.split
        ds = load_dataset("mandarjoshi/trivia_qa", "rc", split=load_split)
        for r in ds:
            a = r.get("answer") or {}
            gold = a.get("value")
            passages = []
            ep = r.get("entity_pages") or {}
            for c in ep.get("wiki_context") or []:
                if c:
                    passages.append(_trim(c))
            sr = r.get("search_results") or {}
            for c in sr.get("search_context") or []:
                if c:
                    passages.append(_trim(c))
            passages = passages[: args.max_ctx]
            if r.get("question") and gold and passages:
                rows.append(
                    {
                        "question": str(r["question"]),
                        "gold": str(gold),
                        "aliases": [str(x) for x in (a.get("aliases") or [])],
                        "contexts": passages,
                    }
                )

    elif args.task == "msmarco":
        # MS-MARCO v1.1: keep short-answer queries (factoid) by filtering on answer length.
        # query_type field exists but casing is unreliable across versions; answer length is robust.
        cap = min(args.n * 6, 30000) if args.n else None
        load_split = f"{args.split}[:{cap}]" if cap else args.split
        ds = load_dataset("microsoft/ms_marco", "v1.1", split=load_split)
        for r in ds:
            ans = [
                a.strip()
                for a in (r.get("answers") or [])
                if a
                and a.strip()
                and "no answer" not in a.strip().lower()
                and len(a.split()) <= 10
            ]
            if not ans:
                continue
            passages = [
                _trim(p) for p in (r.get("passages", {}).get("passage_text") or []) if p
            ]
            passages = [p for p in passages if p][: args.max_ctx]
            if r.get("query") and passages:
                rows.append(
                    {
                        "question": str(r["query"]),
                        "gold": str(ans[0]),
                        "aliases": [str(a) for a in ans],
                        "contexts": passages,
                    }
                )

    if args.n and args.n < len(rows):
        rows = [rows[i] for i in rng.choice(len(rows), size=args.n, replace=False)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    has_ctx = sum(1 for r in rows if r.get("contexts"))
    print(f"wrote {len(rows)} -> {args.out}  (with contexts: {has_ctx})")


if __name__ == "__main__":
    main()
