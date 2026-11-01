#!/usr/bin/env python3
"""
Film-style photo polish (no skin work): faded blacks, gentle contrast and clarity,
warm highlights / cool shadows, highlight glow, light vignette and visible film grain.
Classical image processing only.

Usage:
    .venv/bin/python polish.py INPUT [INPUT ...] [--outdir output]

Writes <name>_polish_subtle/_medium/_strong.jpg (full resolution, original colour
profile, JPEG 96) and <name>_polish_comparison.jpg.
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

# ============================================================================
# STRENGTH SETTINGS
# ============================================================================
PRESETS = {
    #            faded   S-curve   local      warm/cool  colour   highlight  edge     grain std   grain
    #            blacks  contrast  clarity    tint       satur.   glow       vignette (0..1)      size px
    "subtle": dict(FADE=0.020, CONTRAST=0.10, CLARITY=0.10, SPLIT_TONE=0.015, SATURATION=1.03,
                   GLOW=0.08, VIGNETTE=0.10, GRAIN=0.016, GRAIN_SIZE=0.8),
    "medium": dict(FADE=0.035, CONTRAST=0.16, CLARITY=0.15, SPLIT_TONE=0.025, SATURATION=1.05,
                   GLOW=0.12, VIGNETTE=0.15, GRAIN=0.023, GRAIN_SIZE=0.9),
    "strong": dict(FADE=0.050, CONTRAST=0.22, CLARITY=0.20, SPLIT_TONE=0.035, SATURATION=1.07,
                   GLOW=0.16, VIGNETTE=0.20, GRAIN=0.030, GRAIN_SIZE=1.0),
}
JPEG_QUALITY = 96


def gblur(x, s):
    return x if s <= 0.3 else cv2.GaussianBlur(x, (0, 0), s, borderType=cv2.BORDER_REFLECT)

def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)

def luma(rgb):
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)

def load(path):
    im = Image.open(path)
    icc = im.info.get("icc_profile")
    exif = im.getexif()
    exif[0x0112] = 1
    return np.asarray(im.convert("RGB")), icc, exif.tobytes()

def save(path, img, icc, exif):
    u8 = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    kw = dict(quality=JPEG_QUALITY, subsampling=0, optimize=True)
    if icc:
        kw["icc_profile"] = icc
    if exif:
        kw["exif"] = exif
    Image.fromarray(u8).save(path, "JPEG", **kw)


def polish(img, P, seed=11):
    H, W = img.shape[:2]
    D = max(H, W)
    out = img.copy()

    # Local clarity, kept out of the deep shadows so night noise is not amplified.
    Y = luma(out)
    det = Y - gblur(Y, 0.008 * D)
    det = 0.05 * np.tanh(det / 0.05)
    out += (P["CLARITY"] * det * smoothstep(Y, 0.04, 0.18))[..., None]

    # Gentle S-curve on luminance (colour ratios kept), pivoted on the photo's own
    # mid-brightness so a dark night shot gains contrast without getting darker.
    Y = np.clip(luma(out), 1e-4, 1)
    pivot = float(np.clip(np.median(Y[H // 4: 3 * H // 4, W // 4: 3 * W // 4]) * 1.5, 0.15, 0.5))
    s = Y + P["CONTRAST"] * (Y - pivot) * 4 * Y * (1 - Y)
    s = np.clip(s, 0, 1)
    out = out * (s / Y)[..., None]

    # Saturation (gentle, protects already-saturated colours).
    Y = luma(out)[..., None]
    sat = np.abs(out - Y).max(-1, keepdims=True)
    k = 1 + (P["SATURATION"] - 1) * (1 - np.clip(sat * 2, 0, 1))
    out = Y + (out - Y) * k

    # Warm highlights, cool-teal shadows.
    Y = np.clip(luma(out), 0, 1)[..., None]
    t = P["SPLIT_TONE"]
    out += t * smoothstep(Y, 0.45, 0.95) * np.array([1.0, 0.35, -0.6], np.float32)
    out += t * (1 - smoothstep(Y, 0.05, 0.45)) * np.array([-0.5, 0.1, 0.6], np.float32) * 0.8

    # Highlight glow (screen blend of blurred highlights).
    sm = cv2.resize(np.clip(out, 0, 1), (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    hi = sm * smoothstep(luma(sm), 0.65, 0.98)[..., None]
    hi = cv2.resize(gblur(hi, 0.006 * D / 4), (W, H), interpolation=cv2.INTER_LINEAR)
    out = 1 - (1 - np.clip(out, 0, 1)) * (1 - P["GLOW"] * hi)

    # Faded / matte blacks.
    f = P["FADE"]
    out = f + (1 - f) * out

    # Soft vignette.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / np.sqrt(2)
    out *= (1 - P["VIGNETTE"] * smoothstep(r, 0.45, 1.0))[..., None]
    del yy, xx, r

    # Film grain: mostly luminance, slightly clumpy, strongest in the midtones.
    rng = np.random.default_rng(seed)
    n = gblur(rng.standard_normal((H, W), dtype=np.float32), P["GRAIN_SIZE"])
    n /= n.std() + 1e-6
    nc = gblur(rng.standard_normal((H, W, 3), dtype=np.float32), P["GRAIN_SIZE"])
    nc /= nc.std() + 1e-6
    Y = np.clip(luma(out), 0, 1)
    tonal = (0.45 + 2.2 * Y * (1 - Y))[..., None]
    out += P["GRAIN"] * (0.92 * n[..., None] + 0.08 * nc) * tonal
    return np.clip(out, 0, 1)


def comparison(versions, path, crop_center, icc=None):
    names = list(versions)
    H, W = versions[names[0]].shape[:2]
    tw = 900
    th = int(H * tw / W)
    cs = 700
    cx, cy = crop_center
    x0 = int(np.clip(cx - cs // 2, 0, W - cs))
    y0 = int(np.clip(cy - cs // 2, 0, H - cs))
    cols = []
    for name in names:
        u8 = np.clip(versions[name] * 255 + 0.5, 0, 255).astype(np.uint8)
        full = cv2.resize(u8, (tw, th), interpolation=cv2.INTER_AREA)
        crop = cv2.resize(u8[y0:y0 + cs, x0:x0 + cs], (tw, tw), interpolation=cv2.INTER_NEAREST)
        label = np.full((70, tw, 3), 255, np.uint8)
        cv2.putText(label, name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3, cv2.LINE_AA)
        cols.append(np.vstack([label, full, np.full((10, tw, 3), 255, np.uint8), crop]))
    sep = np.full((cols[0].shape[0], 10, 3), 255, np.uint8)
    sheet = cols[0]
    for c in cols[1:]:
        sheet = np.hstack([sheet, sep, c])
    Image.fromarray(sheet).save(path, "JPEG", quality=92, **({"icc_profile": icc} if icc else {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
    ap.add_argument("--crop", help="x,y centre of the 100%% crop in the comparison")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for p in a.inputs:
        name = os.path.splitext(os.path.basename(p))[0]
        rgb, icc, exif = load(p)
        img = rgb.astype(np.float32) / 255
        H, W = img.shape[:2]
        versions = {"original": img}
        for preset, P in PRESETS.items():
            out = polish(img, P)
            dst = os.path.join(a.outdir, f"{name}_polish_{preset}.jpg")
            save(dst, out, icc, exif)
            versions[preset] = out
            print(f"{name}: {preset} -> {dst}")
        c = tuple(map(int, a.crop.split(","))) if a.crop else (W // 2, H // 3)
        comparison(versions, os.path.join(a.outdir, f"{name}_polish_comparison.jpg"), c, icc)


if __name__ == "__main__":
    main()
