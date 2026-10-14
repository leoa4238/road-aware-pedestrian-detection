"""Extract user-provided archives without changing archives or overwriting files."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(archive, destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            name = PurePosixPath(info.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or any(":" in p for p in name.parts):
                raise ValueError("Unsafe archive member: " + info.filename)
            target = destination.joinpath(*name.parts).resolve()
            if destination not in target.parents:
                raise ValueError("Archive member escaped destination")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                raise FileExistsError("Refusing to overwrite: " + str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--weights-zip", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for archive, folder in [(args.data_zip, "dataset"), (args.weights_zip, "weights")]:
        digest = sha256(archive)
        count = extract(archive, args.destination / folder)
        if sha256(archive) != digest:
            raise RuntimeError("Source archive changed during extraction")
        records.append({"archive": str(archive.resolve()), "sha256": digest,
                        "extracted_files": count, "destination": str((args.destination / folder).resolve())})
        print(json.dumps(records[-1]), flush=True)
    with (args.destination / "extraction_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2)


if __name__ == "__main__":
    main()
