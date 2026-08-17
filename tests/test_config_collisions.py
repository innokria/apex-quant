#!/usr/bin/env python3
"""
Guard against tensor-type pattern collisions in APEX configs.

llama-quantize matches --tensor-type-file entries with std::regex_search:
unanchored, FIRST MATCH WINS (src/llama-quant.cpp:694). A pattern written for
one tensor can therefore silently capture a different one, producing a quant
that is wrong in a way nothing in the config or the quantize log reveals.

This replays that exact matching against real tensor name lists read out of the
published GGUF headers (tests/fixtures/), and fails if any tensor is matched by
two rules that disagree about the type.

Known-benign ambiguity: the MoE emitter writes unanchored "blk.N.attn_q", which
also matches "blk.N.attn_qkv.weight". Every MoE profile gives both the same
type, so the outcome is identical — but it is one edit away from being a real
bug, which is exactly why this test exists.

Usage: ./tests/test_config_collisions.py
"""

import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(REPO, "tests", "fixtures")

# fixture -> config prefixes to check against that model's tensor names
MODELS = {
    "tl_Laguna-XS-2.1.json":           ["laguna_xs21"],
    "tl_LFM2.5-8B-A1B.json":           ["lfm25_8b"],
    "tl_Nemotron-3-Nano-30B-A3B.json": ["nemotron3_nano_30b"],
    "tl_Qwen3.5-35B-A3B.json":         ["apex", "qwen38_27b"],
}

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def load_rules(path):
    rules = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        pat, qtype = line.rsplit("=", 1)
        rules.append((pat.strip(), qtype.strip().upper()))
    return rules


def main():
    conflicts = 0
    benign = 0
    checked = 0

    for fixture, prefixes in sorted(MODELS.items()):
        fpath = os.path.join(FIXTURES, fixture)
        if not os.path.exists(fpath):
            print(f"{RED}missing fixture {fixture}{RESET}")
            return 1
        tensors = json.load(open(fpath))
        model = fixture[3:-5]

        configs = []
        for p in prefixes:
            configs += sorted(glob.glob(os.path.join(REPO, "configs", f"{p}_*.txt")))
        if not configs:
            continue

        print(f"\n{model}  ({len(tensors)} tensors, {len(configs)} configs)")
        for cfg in configs:
            rules = [(re.compile(p), p, q) for p, q in load_rules(cfg)]
            cfg_conflicts, cfg_benign = [], 0
            for t in tensors:
                hits = [(p, q) for rx, p, q in rules if rx.search(t)]
                if len(hits) > 1:
                    if len({q for _, q in hits}) > 1:
                        cfg_conflicts.append((t, hits))
                    else:
                        cfg_benign += 1
            checked += 1
            conflicts += len(cfg_conflicts)
            benign += cfg_benign
            name = os.path.basename(cfg)
            if cfg_conflicts:
                print(f"  {RED}CONFLICT{RESET}  {name}")
                for t, hits in cfg_conflicts[:5]:
                    print(f"      {t}")
                    for p, q in hits[:4]:
                        marker = "  <-- WINS" if (p, q) == hits[0] else ""
                        print(f"         {p!r} -> {q}{marker}")
            elif cfg_benign:
                print(f"  {YELLOW}ok{RESET}        {name}  ({cfg_benign} same-type multi-matches)")
            else:
                print(f"  {GREEN}ok{RESET}        {name}")

    print("\n" + "=" * 66)
    print(f"configs checked: {checked}")
    print(f"same-type multi-matches (harmless, but fragile): {benign}")
    print(f"conflicting multi-matches (WOULD SHIP A WRONG QUANT): {conflicts}")
    return 1 if conflicts else 0


if __name__ == "__main__":
    sys.exit(main())
