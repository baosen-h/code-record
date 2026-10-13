import json
import sys
import threading
import time
from pathlib import Path

import psutil

# Unbuffered output — so prints appear immediately
sys.stdout.reconfigure(line_buffering=True)

from tests.adapters import run_train_bpe
from tests.common import gpt2_bytes_to_unicode

TRAIN_PATH = Path("data/TinyStoriesV2-GPT4-train.txt")
VOCAB_OUT  = Path("tinystories_vocab.json")
MERGES_OUT = Path("tinystories_merges.txt")

VOCAB_SIZE = 10_000
SPECIAL    = ["<|endoftext|>"]

# ---------- heartbeat thread ----------
_stop = threading.Event()

def heartbeat(start):
    while not _stop.is_set():
        elapsed = time.perf_counter() - start
        rss = psutil.Process().memory_info().rss / 1e9
        print(f"  [heartbeat] elapsed={elapsed:6.1f}s  RSS={rss:5.2f} GB", flush=True)
        _stop.wait(5.0)   # print every 5 seconds

# ---------- main ----------
def main():
    print("=" * 60, flush=True)
    print(f"dataset: {TRAIN_PATH}", flush=True)
    print(f"size:    {TRAIN_PATH.stat().st_size / 1e9:.2f} GB", flush=True)
    print(f"vocab_size: {VOCAB_SIZE}", flush=True)
    print(f"specials:   {SPECIAL}", flush=True)
    print("=" * 60, flush=True)

    proc = psutil.Process()
    rss0 = proc.memory_info().rss

    t0 = time.perf_counter()

    # start heartbeat
    hb = threading.Thread(target=heartbeat, args=(t0,), daemon=True)
    hb.start()

    try:
        print("\n[train] calling run_train_bpe ... (this will take a while)\n", flush=True)
        vocab, merges = run_train_bpe(
            input_path=TRAIN_PATH,
            vocab_size=VOCAB_SIZE,
            special_tokens=SPECIAL,
        )
    finally:
        _stop.set()
        hb.join(timeout=1.0)

    t1 = time.perf_counter()
    rss1 = proc.memory_info().rss

    print("\n" + "=" * 60, flush=True)
    print(f"time:       {t1 - t0:.1f} s", flush=True)
    print(f"RSS before: {rss0 / 1e9:.2f} GB", flush=True)
    print(f"RSS after:  {rss1 / 1e9:.2f} GB", flush=True)
    print(f"vocab size: {len(vocab)}", flush=True)
    print(f"num merges: {len(merges)}", flush=True)
    print("=" * 60, flush=True)

    # sanity checks
    assert len(vocab) == VOCAB_SIZE, f"expected {VOCAB_SIZE}, got {len(vocab)}"
    assert len(merges) == VOCAB_SIZE - 256 - len(SPECIAL)
    assert len(set(vocab.values())) == len(vocab), "duplicate byte sequences!"
    assert b"<|endoftext|>" in vocab.values()

    # longest token
    longest = max(vocab.values(), key=len)
    print(f"\nlongest token: {longest!r}  ({len(longest)} bytes)", flush=True)
    print(f"as text:       {longest.decode('utf-8', errors='replace')!r}", flush=True)

    print("\ntop 10 longest:", flush=True)
    for tok in sorted(vocab.values(), key=len, reverse=True)[:10]:
        print(f"  {len(tok):2d}  {tok!r}", flush=True)

    # serialize (GPT-2 format)
    enc = gpt2_bytes_to_unicode()

    with open(VOCAB_OUT, "w", encoding="utf-8") as f:
        json.dump(
            {"".join(enc[b] for b in tok): idx for idx, tok in vocab.items()},
            f, ensure_ascii=False, indent=2,
        )

    with open(MERGES_OUT, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write("".join(enc[x] for x in a) + " " +
                    "".join(enc[x] for x in b) + "\n")

    print(f"\nwrote {VOCAB_OUT}", flush=True)
    print(f"wrote {MERGES_OUT}", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()