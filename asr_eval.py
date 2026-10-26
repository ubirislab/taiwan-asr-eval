"""Score a CTranslate2 Whisper model on one test set.

usage: python asr_eval.py <model_dir> <set> <language> [limit] [out.jsonl]
  set:      cv_nan | tt_hak | moe_hak | fleurs:<lang_dir>
  language: the Whisper language token passed to the decoder (zh, vi, id, th, tl ...)

Prints one tab-separated summary line; per-utterance results go to out.jsonl.
Chinese-script sets and Thai are scored by character error rate (CER), the
others by word error rate (WER).
"""
import csv, glob, json, os, re, sys, time, unicodedata

from faster_whisper import WhisperModel
from opencc import OpenCC

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
to_traditional = OpenCC("s2twp").convert


def items(name, limit):
    if name in ("cv_nan", "tt_hak"):
        for j in sorted(glob.glob(os.path.join(DATA, name, "*.json")))[:limit]:
            meta = json.load(open(j))
            yield j[:-5] + meta["audio_ext"], meta["sentence"]
    elif name == "moe_hak":
        import pyarrow.parquet as pq
        rows = [r for r in pq.read_table(os.path.join(DATA, "moe_hak.parquet")).to_pylist()
                if len(r["text"]) >= 8]  # example sentences, not single words
        step = max(1, len(rows) // limit)
        out = os.path.join(DATA, "moe_hak")
        os.makedirs(out, exist_ok=True)
        for r in rows[::step][:limit]:
            p = os.path.join(out, f"{r['id']}.audio")
            if not os.path.exists(p):
                open(p, "wb").write(r["path"]["bytes"])
            yield p, r["text"]
    elif name.startswith("fleurs:"):
        d = os.path.join(DATA, "fleurs", name.split(":", 1)[1])
        n = 0
        for r in csv.reader(open(os.path.join(d, "dev.tsv")), delimiter="\t"):
            p = os.path.join(d, "dev", r[1])
            if os.path.exists(p):
                yield p, r[2]
                n += 1
                if n >= limit:
                    return
    else:
        sys.exit(f"unknown set: {name}")


def chars(s):
    # drop romanisation in parentheses, unify to Traditional, keep letters only
    s = to_traditional(re.sub(r"[（(][^）)]*[）)]", "", s))
    return [c for c in s if not unicodedata.category(c).startswith(("P", "Z", "S"))]


def words(s):
    s = unicodedata.normalize("NFC", s.lower())
    return "".join(" " if unicodedata.category(c).startswith("P") else c for c in s).split()


def distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    model_dir, name, lang = sys.argv[1:4]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    out = sys.argv[5] if len(sys.argv) > 5 else os.devnull
    by_char = not name.startswith("fleurs:") or lang == "th"
    model = WhisperModel(model_dir, device="auto", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    err = total = n = 0
    audio = wall = 0.0
    with open(out, "w") as f:
        for path, ref in items(name, limit):
            t0 = time.time()
            segs, info = model.transcribe(path, language=lang, beam_size=5)
            hyp = " ".join(s.text.strip() for s in segs)
            wall += time.time() - t0
            if lang == "th":
                r, h = list("".join(words(ref))), list("".join(words(hyp)))
            elif by_char:
                r, h = chars(ref), chars(hyp)
            else:
                r, h = words(ref), words(hyp)
            e = distance(r, h)
            err += e; total += len(r); audio += info.duration; n += 1
            f.write(json.dumps({"audio": os.path.basename(path), "ref": ref, "hyp": hyp,
                                "errors": e, "ref_len": len(r), "seconds": round(info.duration, 2)},
                               ensure_ascii=False) + "\n")
    metric = "CER" if by_char else "WER"
    print(f"{os.path.basename(os.path.normpath(model_dir))}\t{name}\t{lang}\tn={n}"
          f"\t{metric}={err / max(total, 1):.1%}\taudio={audio:.0f}s\twall={wall:.0f}s"
          f"\tRTF={wall / max(audio, 1):.2f}")


if __name__ == "__main__":
    main()
