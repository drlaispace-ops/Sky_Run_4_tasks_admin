"""Shrink report figures for embedding as data URIs in the published artifact.

The artifact CSP blocks every external host, so each image must be inlined.
Raw figures are ~2.9 MB; base64 inflates by 4/3. This downscales and requantises
them into tier_d/admin/reports/web/ without touching the originals.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

SRC = Path("tier_d/admin/reports")
DST = SRC / "web"

PNG_WIDTH = 1180
GIF_WIDTH = 380
GIF_MAX_FRAMES = 18
GIF_COLORS = 48


def _png(name: str) -> int:
    im = Image.open(SRC / name).convert("RGB")
    if im.width > PNG_WIDTH:
        h = round(im.height * PNG_WIDTH / im.width)
        im = im.resize((PNG_WIDTH, h), Image.LANCZOS)
    # Adaptive palette: these are flat-ish plots, so 256 colours is plenty and
    # cuts size several-fold versus truecolour PNG.
    im = im.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.Dither.NONE)
    out = DST / name
    im.save(out, optimize=True)
    return out.stat().st_size


def _gif(name: str) -> int:
    im = Image.open(SRC / name)
    frames, n = [], getattr(im, "n_frames", 1)
    stride = max(1, n // GIF_MAX_FRAMES)
    for i in range(0, n, stride):
        im.seek(i)
        f = im.convert("RGB")
        h = round(f.height * GIF_WIDTH / f.width)
        frames.append(f.resize((GIF_WIDTH, h), Image.LANCZOS)
                      .quantize(colors=GIF_COLORS, method=Image.MEDIANCUT))
    out = DST / name
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   loop=0, duration=140, optimize=True)
    return out.stat().st_size


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    total_before = total_after = 0
    for p in sorted(SRC.glob("*.png")) + sorted(SRC.glob("*.gif")):
        before = p.stat().st_size
        after = _gif(p.name) if p.suffix == ".gif" else _png(p.name)
        total_before += before
        total_after += after
        print(f"  {p.name:28s} {before/1024:7.0f}K -> {after/1024:6.0f}K")
    print(f"\n  total {total_before/1024:.0f}K -> {total_after/1024:.0f}K "
          f"(~{total_after*4/3/1024:.0f}K as base64)")


if __name__ == "__main__":
    main()
