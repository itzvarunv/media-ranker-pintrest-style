"""
The vault: image files plus their `<id>.txt` captions.

Titles and descriptions live ONLY in the caption files (the `ranker` table has
no title column), so the web app reads them from here, and uploads write new
captions in exactly the format train.py already parses:

    ASSET ID: <id>
    NICHE: <niche>
    TITLE: <one line>
    DESCRIPTION:
    <free text>
"""
import io
import os
import re

from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VAULT_DIR = os.environ.get("NEUROED_VAULT_DIR") or os.path.join(BASE_DIR, "vault")

UPLOAD_EXTS = {".jpg", ".jpeg", ".png", ".webp"}   # accepted from users
SERVE_EXTS = {".jpg", ".jpeg", ".png"}             # what /vault/<file> will serve
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_SIDE = 2048                                     # stored images are downscaled to this
TITLE_MAX = 150
DESCRIPTION_MAX = 2000

_ID_RE = re.compile(r"^[A-Za-z0-9]{1,32}$")
Image.MAX_IMAGE_PIXELS = 50_000_000                 # refuse decompression bombs


def is_valid_id(asset_id):
    return bool(asset_id) and bool(_ID_RE.match(asset_id))


def image_path(asset_id):
    return os.path.join(VAULT_DIR, f"{asset_id}.jpeg")


def caption_path(asset_id):
    return os.path.join(VAULT_DIR, f"{asset_id}.txt")


def relative_filepath(asset_id):
    """The value stored in ranker.filepath (matches the 150 existing rows)."""
    return f"vault/{asset_id}.jpeg"


def asset_id_from_filepath(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def read_caption(asset_id):
    """Return {'niche', 'title', 'description'}; falls back gracefully if the .txt is missing."""
    fallback = {"niche": "", "title": asset_id, "description": ""}
    if not is_valid_id(asset_id):
        return fallback
    try:
        with open(caption_path(asset_id), "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return fallback

    niche = re.search(r"^NICHE:\s*(.+?)\s*$", text, re.M)
    title = re.search(r"^TITLE:\s*(.+?)\s*$", text, re.M)
    desc = re.search(r"^DESCRIPTION:[ \t]*\n?(.*)\Z", text, re.M | re.S)
    return {
        "niche": niche.group(1) if niche else "",
        "title": title.group(1) if title else asset_id,
        "description": desc.group(1).strip() if desc else "",
    }


def write_caption(asset_id, niche, title, description):
    # A title must stay on one line, otherwise it could smuggle extra
    # "NICHE:"/"ASSET ID:" header lines into the training data.
    title = " ".join((title or "").split())[:TITLE_MAX] or asset_id
    description = (description or "").strip()[:DESCRIPTION_MAX]
    with open(caption_path(asset_id), "w", encoding="utf-8") as f:
        f.write(f"ASSET ID: {asset_id}\nNICHE: {niche}\nTITLE: {title}\nDESCRIPTION:\n{description}\n")
    return title, description


def save_image(data, asset_id):
    """
    Validate `data` (raw bytes) as a real JPEG/PNG/WEBP image, normalise it
    (EXIF rotation, RGB, downscale) and store it as vault/<id>.jpeg.
    Raises ValueError with a user-facing message if it isn't a usable image.
    """
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()                                # cheap integrity check
        img = Image.open(io.BytesIO(data))            # verify() invalidates the object
        if img.format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("Only JPG, PNG and WEBP images are supported.")
        img = ImageOps.exif_transpose(img).convert("RGB")
    except ValueError:
        raise
    except Exception:
        raise ValueError("That file isn't a valid image.")

    img.thumbnail((MAX_SIDE, MAX_SIDE))
    os.makedirs(VAULT_DIR, exist_ok=True)
    path = image_path(asset_id)
    img.save(path, "JPEG", quality=92)
    return path


def delete_asset_files(asset_id):
    """Remove the files of an asset we just created (used to roll back a failed upload)."""
    for path in (image_path(asset_id), caption_path(asset_id)):
        try:
            os.remove(path)
        except OSError:
            pass


def search_ids(query):
    """Asset ids whose caption (title/description/niche) contains every word of `query`."""
    words = [w for w in re.findall(r"\w+", (query or "").lower()) if w]
    if not words:
        return []
    hits = []
    try:
        names = os.listdir(VAULT_DIR)
    except OSError:
        return []
    for name in names:
        if not name.endswith(".txt"):
            continue
        asset_id = name[:-4]
        try:
            with open(os.path.join(VAULT_DIR, name), "r", encoding="utf-8") as f:
                text = f.read().lower()
        except OSError:
            continue
        if all(w in text for w in words):
            hits.append(asset_id)
    return hits
