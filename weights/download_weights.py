#!/usr/bin/env python
"""Download the four demo checkpoints into this folder and verify their SHA-256.

    python weights/download_weights.py           # download missing files, then verify all
    python weights/download_weights.py --check   # verify only
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_URL = ""   # TODO: download location, to be added

FILES = {
    "anthropeconet_mitb2.pth": "a724841749db5f91e74217dbef7fe5f592ebbfc20c11b9939a08b7488b055227",
    "eco_classifier.pth": "ff307f5cc4b0776e7c4e87d204a26b8e59eba32269d647f252396efdeb5cbb12",
    "turbidity_classifier.pth": "2d9927604cebcda6021a9e72576bfb9f4a9ea533ee477f3dbc36cf74b4d3a9a0",
    "enhancer_dcenet.pt": "318b08079c5e8d26ca4e2ba77fb39f6df3fed55ec614830721fdc5e504f883f7",
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only verify the files already present")
    a = ap.parse_args()
    if not a.check:
        missing = [f for f in FILES if not (HERE / f).exists()]
        if missing and not BASE_URL:
            sys.exit("The download link has not been published yet; see weights/README.md.")
        for f in missing:
            print(f"downloading {f} ...", flush=True)
            tmp = HERE / (f + ".part")
            urllib.request.urlretrieve(f"{BASE_URL.rstrip('/')}/{f}", tmp)
            tmp.rename(HERE / f)
    ok = True
    for f, ref in FILES.items():
        p = HERE / f
        if not p.exists():
            print(f"MISSING  {f}"); ok = False
        elif sha256(p) != ref:
            print(f"BAD HASH {f}"); ok = False
        else:
            print(f"ok       {f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
