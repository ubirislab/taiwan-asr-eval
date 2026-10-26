"""Download the public test sets used by asr_eval.py into ./data.

usage: python fetch_data.py [cv_nan] [tt_hak] [moe_hak] [fleurs:vi_vn ...]

With no arguments every set is downloaded. Each set keeps its own licence;
read the dataset card before using the audio for anything beyond evaluation.
"""
import io, os, sys, tarfile, urllib.request

HF = "https://huggingface.co/datasets"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FLEURS = ["vi_vn", "id_id", "th_th", "fil_ph"]


def fetch(url):
    print("GET", url, file=sys.stderr)
    with urllib.request.urlopen(url) as r:
        return r.read()


def save(path, blob):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(blob)


def untar(blob, dest, mode="r:"):
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode=mode) as t:
        t.extractall(dest, filter="data")


def tt_test(name, dest):
    # Taiwan-Tongues-ASR-CE test splits (WebDataset tar: <key>.json + audio)
    base = f"{HF}/adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-{name}/resolve/main"
    untar(fetch(f"{base}/test/test-000000.tar"), os.path.join(DATA, dest))


def main(sets):
    for s in sets:
        if s == "cv_nan":
            # Common Voice nan-tw clips, as packaged by Taiwan-Tongues
            tt_test("hokkien", "cv_nan")
        elif s == "tt_hak":
            tt_test("hakka", "tt_hak")
        elif s == "moe_hak":
            # Ministry of Education Hakka dictionary audio (Sixian)
            save(os.path.join(DATA, "moe_hak.parquet"), fetch(
                f"{HF}/ouob/hakkadict_210430_sixian/resolve/main/data/"
                "train-00000-of-00001-7d0cc40078812d2c.parquet"))
        elif s.startswith("fleurs:"):
            lang = s.split(":", 1)[1]
            base = f"{HF}/google/fleurs/resolve/main/data/{lang}"
            d = os.path.join(DATA, "fleurs", lang)
            save(os.path.join(d, "dev.tsv"), fetch(f"{base}/dev.tsv"))
            untar(fetch(f"{base}/audio/dev.tar.gz"), d, "r:gz")
        else:
            sys.exit(f"unknown set: {s}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["cv_nan", "tt_hak", "moe_hak"] + [f"fleurs:{l}" for l in FLEURS])
