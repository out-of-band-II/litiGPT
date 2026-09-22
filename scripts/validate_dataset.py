#!/usr/bin/env python3
"""
Check a generated training set before spending GPU time on it.

Catches the failures that are cheap to fix now and expensive to discover after
a training run: personas with too few examples, a cohort so imbalanced the
model will collapse onto the loudest user, markup that survived cleaning, and
prompts that lost their username.

    python scripts/validate_dataset.py --config config.top30.yaml
    python scripts/validate_dataset.py --data-dir data/training --strict
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

CHARS_PER_TOK = 3.6  # Italian text, Llama/Phi BPE ballpark

# Markup that should not survive clean_text.
LEFTOVERS = {
    "html entity": re.compile(r"&(gt|lt|amp|quot|#x?[0-9a-fA-F]+);"),
    "quote marker": re.compile(r"(?m)^[ \t]*>"),
    "bare url": re.compile(r"https?://"),
    "markdown link": re.compile(r"\[[^\]]*\]\(https?://"),
}


class Report:
    def __init__(self, strict: bool):
        self.strict, self.failed = strict, False

    def ok(self, msg):   print(f"  [ ok ] {msg}")
    def info(self, msg): print(f"         {msg}")

    def warn(self, msg):
        print(f"  [warn] {msg}")
        if self.strict:
            self.failed = True

    def fail(self, msg):
        print(f"  [FAIL] {msg}")
        self.failed = True


def load(path: Path):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i}: malformed JSON: {e}") from None
    return rows


def check_split(name, rows, r: Report):
    print(f"\n{name}: {len(rows):,} examples")
    if not rows:
        r.fail(f"{name} is empty")
        return Counter()

    users, lengths, bad_struct, bad_prompt, empty_ctx = Counter(), [], 0, 0, 0
    leftovers = Counter()

    for ex in rows:
        u = ex.get("username")
        msgs = ex.get("messages")
        if not u or not isinstance(msgs, list) or len(msgs) < 3:
            bad_struct += 1
            continue
        users[u] += 1

        roles = [m.get("role") for m in msgs]
        if roles[:3] != ["system", "user", "assistant"]:
            bad_struct += 1
            continue

        system, ctx, resp = (msgs[0].get("content", ""), msgs[1].get("content", ""),
                             msgs[2].get("content", ""))
        # The whole design rests on the persona being named in the prompt.
        if u not in system:
            bad_prompt += 1
        if not ctx.strip():
            empty_ctx += 1
        for label, pat in LEFTOVERS.items():
            if pat.search(resp) or pat.search(ctx):
                leftovers[label] += 1
        lengths.append(len(system) + len(ctx) + len(resp))

    if bad_struct:
        r.fail(f"{bad_struct:,} examples with a malformed message structure")
    else:
        r.ok("every example is system/user/assistant with a username")

    if bad_prompt:
        r.fail(f"{bad_prompt:,} system prompts do not name their user")
    else:
        r.ok("every system prompt names its user")

    if empty_ctx:
        r.warn(f"{empty_ctx:,} examples have empty context")

    for label, n in leftovers.items():
        pct = 100 * n / len(rows)
        (r.warn if pct < 1 else r.fail)(f"{label} survived cleaning in {n:,} examples ({pct:.1f}%)")
    if not leftovers:
        r.ok("no HTML entities, quote markers or raw URLs survived cleaning")

    if lengths:
        lengths.sort()
        tok = [n / CHARS_PER_TOK for n in lengths]
        p50, p95 = tok[len(tok) // 2], tok[int(len(tok) * 0.95)]
        r.info(f"est. tokens/example: median {p50:.0f}, p95 {p95:.0f}, max {tok[-1]:.0f}")
        over = sum(1 for t in tok if t > 512)
        if over:
            r.info(f"{over:,} ({100*over/len(tok):.1f}%) exceed max_seq_length 512 and will be truncated")
    return users


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="read data.training_dir from this config")
    ap.add_argument("--data-dir", help="directory holding train.jsonl / val.jsonl")
    ap.add_argument("--min-per-user", type=int, default=100,
                    help="floor below which a persona has too little data (default: 100)")
    ap.add_argument("--max-imbalance", type=float, default=3.0,
                    help="max tolerated most/least-represented user ratio (default: 3.0)")
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = ap.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif args.config:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        data_dir = Path((cfg.get("data") or {}).get("training_dir", "data/training"))
    else:
        ap.error("pass --config or --data-dir")

    print(f"Validating {data_dir}")
    r = Report(args.strict)

    train_p, val_p = data_dir / "train.jsonl", data_dir / "val.jsonl"
    for p in (train_p, val_p):
        if not p.exists():
            raise SystemExit(f"missing {p} — run the preprocess step first")

    train_users = check_split("train.jsonl", load(train_p), r)
    val_users = check_split("val.jsonl", load(val_p), r)

    print(f"\ncohort: {len(train_users)} users")
    if train_users:
        ranked = train_users.most_common()
        for u, n in ranked[:5]:
            print(f"    {u:<28} {n:>6,}")
        if len(ranked) > 10:
            print(f"    {'...':<28} {len(ranked)-10:>6} more")
        for u, n in ranked[-5:] if len(ranked) > 5 else []:
            print(f"    {u:<28} {n:>6,}")

        lo, hi = ranked[-1][1], ranked[0][1]
        ratio = hi / max(lo, 1)
        if ratio > args.max_imbalance:
            r.warn(f"imbalance {ratio:.1f}x exceeds {args.max_imbalance}x "
                   f"({ranked[0][0]} {hi:,} vs {ranked[-1][0]} {lo:,}) — set data.max_pairs_per_user")
        else:
            r.ok(f"cohort balanced within {ratio:.1f}x")

        thin = [f"{u} ({n})" for u, n in ranked if n < args.min_per_user]
        if thin:
            r.warn(f"{len(thin)} users below {args.min_per_user} examples: {', '.join(thin[:5])}")
        else:
            r.ok(f"every user has at least {args.min_per_user} examples")

        missing = set(train_users) - set(val_users)
        if missing:
            r.warn(f"{len(missing)} users absent from val, so their personas go unmeasured: "
                   f"{', '.join(sorted(missing)[:5])}")
        else:
            r.ok("every user appears in both train and val")

    print("\n" + ("FAILED" if r.failed else "PASSED"))
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
