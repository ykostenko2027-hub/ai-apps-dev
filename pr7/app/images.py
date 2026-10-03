import io
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from PIL import Image, ImageOps

load_dotenv()

UPLOAD_MAX_BYTES = int(float(os.getenv("UPLOAD_MAX_MB", "10")) * 1024 * 1024)
IMAGE_MAX_SIDE = int(os.getenv("IMAGE_MAX_SIDE", "1600"))


class ImageError(Exception):
    pass


@dataclass
class PreparedImage:
    data: bytes
    mime: str
    original: dict
    sent: dict


def prepare(content: bytes) -> PreparedImage:
    if not content:
        raise ImageError("Файл порожній.")
    if len(content) > UPLOAD_MAX_BYTES:
        max_mb = UPLOAD_MAX_BYTES // (1024 * 1024)
        raise ImageError(f"Розмір файлу перевищує ліміт {max_mb} МБ.")
    try:
        img = Image.open(io.BytesIO(content))
        orig_w, orig_h = img.size
    except Exception:
        raise ImageError("Файл не є придатним зображенням.")
    if orig_w < 100 or orig_h < 100:
        raise ImageError("Роздільність зображення занадто мала для обробки документа.")
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    max_side = max(img.width, img.height)
    if max_side > IMAGE_MAX_SIDE:
        scale = IMAGE_MAX_SIDE / max_side
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    sent_bytes = buf.getvalue()
    return PreparedImage(
        data=sent_bytes,
        mime="image/png",
        original={"width": orig_w, "height": orig_h, "bytes": len(content)},
        sent={"width": img.width, "height": img.height, "bytes": len(sent_bytes)},
    )
