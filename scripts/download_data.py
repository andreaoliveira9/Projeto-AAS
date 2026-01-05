from __future__ import annotations
import os
import shutil
import zipfile
import urllib.request
import urllib.error
from pathlib import Path


URL = "https://github.com/andreaoliveira9/Projeto-AAS/releases/download/v1.0.0/dataset.zip"


def download_with_resume(url: str, dst: Path, chunk_size: int = 1024 * 1024) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    existing = dst.stat().st_size if dst.exists() else 0

    headers = {"User-Agent": "python-download-script"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as resp:
            status = getattr(resp, "status", None)
            if status not in (200, 206):
                raise RuntimeError(f"Unexpected HTTP status: {status}")

            mode = "ab" if status == 206 and existing > 0 else "wb"
            if mode == "wb" and existing > 0:
                existing = 0

            total = resp.headers.get("Content-Length")
            total_int = int(total) if total and total.isdigit() else None

            downloaded = existing
            with open(dst, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_int is not None:
                        full = existing + total_int if status == 206 else total_int
                        pct = (downloaded / full) * 100 if full else 0
                        print(
                            f"\rDownloading: {downloaded/1e6:.1f} MB ({pct:.1f}%)",
                            end="",
                        )
                    else:
                        print(f"\rDownloading: {downloaded/1e6:.1f} MB", end="")

            print()

    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read(200).decode("utf-8", errors="ignore")
        except Exception:
            pass
        raise RuntimeError(
            f"HTTP error {e.code} while downloading: {e.reason}. {body}".strip()
        ) from e


def unzip_to_data(zip_path: Path, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            name = info.filename

            if (
                name.startswith("__MACOSX/")
                or name.endswith("/.DS_Store")
                or name.endswith(".DS_Store")
            ):
                continue

            if name.endswith("/"):
                continue

            target_path = (data_dir / name).resolve()
            if (
                not str(target_path).startswith(str(data_dir.resolve()) + os.sep)
                and target_path != data_dir.resolve()
            ):
                raise RuntimeError(f"Unsafe path in zip: {name}")

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info, "r") as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

    macosx_dir = data_dir / "__MACOSX"
    if macosx_dir.exists() and macosx_dir.is_dir():
        shutil.rmtree(macosx_dir)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    zip_path = data_dir / "dataset.zip"

    print(f"URL: {URL}")
    print(f"Downloading to: {zip_path}")
    download_with_resume(URL, zip_path)

    if not zipfile.is_zipfile(zip_path):
        head = zip_path.read_bytes()[:200]
        raise RuntimeError(
            f"Downloaded file is not a valid ZIP. Size={zip_path.stat().st_size} bytes. "
            f"First bytes: {head!r}"
        )

    print(f"Unzipping into: {data_dir}")
    unzip_to_data(zip_path, data_dir)

    print(f"Removing zip: {zip_path}")
    zip_path.unlink(missing_ok=True)

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
