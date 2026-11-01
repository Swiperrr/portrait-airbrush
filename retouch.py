#!/usr/bin/env python3
"""
Classical portrait retouch: magazine-style skin finish that still reads as a photograph.

No generative models, no inpainting, no reshaping. MediaPipe is used only to find
the face and to label skin / hair; every pixel change is plain image processing
(frequency separation, guided filter, median healing, dodge & burn, grain).

Usage:
    .venv/bin/python retouch.py INPUT [INPUT ...] [--outdir output] [--debug DIR]

For each input it writes, at full resolution with the original color profile:
    <name>_airbrushed.jpg            the main result (settings below)
    <name>_subtle/_medium/_strong.jpg
    <name>_comparison.jpg            original vs the three versions, full frame + 100% skin crop
"""
import argparse
import os
import shutil

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

# ============================================================================
# STRENGTH SETTINGS  (the main *_airbrushed.jpg uses these; it equals "medium")
# ============================================================================
SMOOTHING_AMOUNT = 0.55    # 0..1  how much blotchy tone/colour is evened (low-frequency layer)
TEXTURE_RETENTION = 0.55   # 0..1  share of pore texture kept on skin (0.3-0.5 looks real)
FINAL_OPACITY = 0.80       # 0..1  retouched skin blended over the original
GRAIN_AMOUNT = 0.010       # film grain, std as a fraction of full range (0.01-0.02)
BLEMISH_HEAL = 1.0         # 0..1  how fully small spots are healed
UNDER_EYE_LIFT = 0.45      # 0..1  share of the under-eye shadow lifted toward the cheek tone
REDNESS_REDUCTION = 0.16   # 0..1  share of excess red removed on cheeks and nose
CLARITY_AMOUNT = 0.20      # local contrast on eyes and hair only
GLOW_AMOUNT = 0.07         # soft glow on highlights
JPEG_QUALITY = 96

PRESETS = {
    "subtle": dict(SMOOTHING_AMOUNT=0.40, TEXTURE_RETENTION=0.65, FINAL_OPACITY=0.65,
                   GRAIN_AMOUNT=0.008, BLEMISH_HEAL=0.85, UNDER_EYE_LIFT=0.30,
                   REDNESS_REDUCTION=0.10, CLARITY_AMOUNT=0.14, GLOW_AMOUNT=0.05),
    "medium": dict(),  # the settings above
    "strong": dict(SMOOTHING_AMOUNT=0.75, TEXTURE_RETENTION=0.45, FINAL_OPACITY=0.88,
                   GRAIN_AMOUNT=0.012, BLEMISH_HEAL=1.0, UNDER_EYE_LIFT=0.60,
                   REDNESS_REDUCTION=0.25, CLARITY_AMOUNT=0.26, GLOW_AMOUNT=0.09),
}

# Geometry, all as fractions of face width so any resolution behaves the same.
FS_RADIUS_FRAC = 0.0075      # frequency-separation blur (0.5-1% of face width)
TONE_FILTER_FRAC = 0.030     # guided-filter radius for evening tone on the low layer
TONE_FILTER_EPS = 8e-4       # guided-filter edge threshold (variance, 0..1 values)
FEATHER_FRAC = 0.030         # mask feather (Gaussian sigma)
HARSH_TEXTURE_CLIP = 0.045   # high-frequency amplitude above this counts as harsh / blemish
EDGE_LO, EDGE_HI = 4.0, 12.0 # contour strength (change per face width) where retouch fades out

HERE = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL = os.path.join(HERE, "models", "face_landmarker.task")
SEG_MODEL = os.path.join(HERE, "models", "selfie_multiclass.tflite")
MODEL_URLS = {
    FACE_MODEL: "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/latest/face_landmarker.task",
    SEG_MODEL: "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
               "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}


def ensure_models():
    """Download Google's MediaPipe model files on first run."""
    import ssl
    import urllib.request
    try:  # python.org builds on macOS ship without root certificates
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    for path, url in MODEL_URLS.items():
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"Downloading {os.path.basename(path)} ...")
            tmp = path + ".part"
            with urllib.request.urlopen(url, context=ctx) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(tmp, path)


ensure_models()

class NoFaceError(ValueError):
    """Raised when the photo has no detectable face."""


# Face Mesh landmark groups
C = vision.FaceLandmarksConnections
def _idx(conns):
    return sorted({c.start for c in conns} | {c.end for c in conns})
EYE_R = _idx(C.FACE_LANDMARKS_RIGHT_EYE) + _idx(C.FACE_LANDMARKS_RIGHT_IRIS)
EYE_L = _idx(C.FACE_LANDMARKS_LEFT_EYE) + _idx(C.FACE_LANDMARKS_LEFT_IRIS)
BROW_R = _idx(C.FACE_LANDMARKS_RIGHT_EYEBROW)
BROW_L = _idx(C.FACE_LANDMARKS_LEFT_EYEBROW)
LIPS = _idx(C.FACE_LANDMARKS_LIPS)
NOSE = _idx(C.FACE_LANDMARKS_NOSE)
OVAL = _idx(C.FACE_LANDMARKS_FACE_OVAL)
LOWER_LID_R = [33, 7, 163, 144, 145, 153, 154, 155, 133]
LOWER_LID_L = [263, 249, 390, 373, 374, 380, 381, 382, 362]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def odd(n):
    n = int(round(n))
    return n + 1 if n % 2 == 0 else max(n, 1)

def gblur(x, sigma):
    if sigma <= 0.3:
        return x
    return cv2.GaussianBlur(x, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)

def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)

def disk(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

def luma(rgb):
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)

def guided(guide, src, r, eps):
    return cv2.ximgproc.guidedFilter(guide.astype(np.float32), src.astype(np.float32), int(r), eps)

def fill_hull(shape, pts, offset):
    m = np.zeros(shape, np.uint8)
    p = (np.asarray(pts) - offset).astype(np.int32)
    cv2.fillConvexPoly(m, cv2.convexHull(p), 255)
    return m


# ----------------------------------------------------------------------------
# load / save
# ----------------------------------------------------------------------------
def load(path):
    im = Image.open(path)
    icc = im.info.get("icc_profile")
    exif = im.getexif()
    exif[0x0112] = 1  # pixels are already upright
    rgb = np.asarray(im.convert("RGB"))
    return rgb, icc, exif.tobytes()

def save(path, img01, icc, exif):
    u8 = np.clip(img01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    kw = dict(quality=JPEG_QUALITY, subsampling=0, optimize=True)
    if icc:
        kw["icc_profile"] = icc
    if exif:
        kw["exif"] = exif
    Image.fromarray(u8).save(path, "JPEG", **kw)


# ----------------------------------------------------------------------------
# 1. analysis: landmarks + skin / hair masks (done once per photo)
# ----------------------------------------------------------------------------
def segment(rgb_u8):
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=SEG_MODEL), output_confidence_masks=True))
    r = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_u8)))
    # 0 background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 accessories
    out = {k: r.confidence_masks[i].numpy_view().astype(np.float32).reshape(rgb_u8.shape[:2]).copy()
           for k, i in (("hair", 1), ("body", 2), ("face", 3), ("acc", 5))}
    seg.close()
    return out

def find_face(rgb_u8, face_skin):
    """Landmarks of the largest face. The landmarker shrinks the frame first, so a small
    face in a big photo can be missed; then look again in a crop around the face skin
    the segmenter found, and finally in overlapping tiles."""
    H, W = rgb_u8.shape[:2]
    fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL), num_faces=4))

    def detect(x0, y0, x1, y1):
        crop = np.ascontiguousarray(rgb_u8[y0:y1, x0:x1])
        res = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
        h, w = crop.shape[:2]
        return [np.array([[p.x * w + x0, p.y * h + y0] for p in lm], np.float32)
                for lm in res.face_landmarks]

    def square(cx, cy, side):
        side = int(min(max(side, 256), max(H, W)))
        x0 = int(np.clip(cx - side / 2, 0, max(0, W - side)))
        y0 = int(np.clip(cy - side / 2, 0, max(0, H - side)))
        return x0, y0, min(W, x0 + side), min(H, y0 + side)

    try:
        faces = detect(0, 0, W, H)
        if not faces:
            blob = (face_skin > 0.5).astype(np.uint8)
            n, lbl, st, cen = cv2.connectedComponentsWithStats(blob, 8)
            order = np.argsort(-st[1:, cv2.CC_STAT_AREA])[:3] + 1 if n > 1 else []
            for i in order:
                if st[i, cv2.CC_STAT_AREA] < 400:
                    break
                side = 3.0 * max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
                faces = detect(*square(*cen[i], side))
                if faces:
                    break
        if not faces:
            for k in (2, 3):
                side = int(2 * max(H, W) / (k + 1))
                for cy in np.linspace(side / 2, H - side / 2, k):
                    for cx in np.linspace(side / 2, W - side / 2, k):
                        faces += detect(*square(cx, cy, side))
                if faces:
                    break
    finally:
        fl.close()
    if not faces:
        raise NoFaceError("No face found in this photo.")
    return max(faces, key=lambda f: np.ptp(f[:, 0]))


def analyze(rgb_u8, img):
    H, W = img.shape[:2]
    # Segmentation on the whole frame (body) ...
    seg = segment(rgb_u8)
    lm = find_face(rgb_u8, seg["face"])
    fw = float(np.ptp(lm[:, 0]))
    fx0, fy0 = lm.min(0)
    fx1, fy1 = lm.max(0)

    # ... and again on a tight face crop: 256px model -> ~8x finer face edges.
    pad = 0.6 * fw
    cx0, cy0 = int(max(0, fx0 - pad)), int(max(0, fy0 - pad))
    cx1, cy1 = int(min(W, fx1 + pad)), int(min(H, fy1 + pad))
    segc = segment(rgb_u8[cy0:cy1, cx0:cx1])
    ch, cw = cy1 - cy0, cx1 - cx0
    ramp = 0.12
    wy = np.clip(np.minimum(np.arange(ch), ch - 1 - np.arange(ch)) / (ramp * ch), 0, 1)
    wx = np.clip(np.minimum(np.arange(cw), cw - 1 - np.arange(cw)) / (ramp * cw), 0, 1)
    wc = np.outer(wy, wx).astype(np.float32)
    for k in seg:
        s = seg[k][cy0:cy1, cx0:cx1]
        seg[k][cy0:cy1, cx0:cx1] = wc * segc[k] + (1 - wc) * s

    skin_seg = np.clip(seg["face"] + seg["body"], 0, 1) * (1 - seg["acc"])
    hair_seg = seg["hair"]

    # Region of interest around all skin, so heavy filters run on fewer pixels.
    ys, xs = np.where(skin_seg > 0.3)
    m = int(0.15 * fw)
    y0, y1 = max(0, ys.min() - m), min(H, ys.max() + m)
    x0, x1 = max(0, xs.min() - m), min(W, xs.max() + m)
    roi = (y0, y1, x0, x1)
    off = np.array([x0, y0], np.float32)
    shp = (y1 - y0, x1 - x0)
    I = img[y0:y1, x0:x1]

    # Snap the coarse mask to real edges (guided-filter matting refinement).
    r_ref = max(4, int(0.012 * fw))
    skin = guided(I, skin_seg[y0:y1, x0:x1], r_ref, 1e-3)
    skin = smoothstep(skin, 0.35, 0.75)
    hair = smoothstep(guided(I, hair_seg[y0:y1, x0:x1], r_ref, 1e-3), 0.35, 0.75)
    face_soft = smoothstep(guided(I, seg["face"][y0:y1, x0:x1], r_ref, 1e-3), 0.35, 0.75)
    del seg, segc, skin_seg

    # Feature exclusions: eyes, brows, lips/teeth, nostrils.
    excl = np.zeros(shp, np.uint8)
    eyes = np.zeros(shp, np.uint8)
    for ids in (EYE_R, EYE_L):
        e = fill_hull(shp, lm[ids], off)
        eyes |= e
        excl |= cv2.dilate(e, disk(0.022 * fw))
    for ids in (BROW_R, BROW_L):
        excl |= cv2.dilate(fill_hull(shp, lm[ids], off), disk(0.025 * fw))
    excl |= cv2.dilate(fill_hull(shp, lm[LIPS], off), disk(0.014 * fw))
    nose = fill_hull(shp, lm[NOSE], off) > 0
    Y = luma(I)
    core = (skin > 0.9) & (excl == 0)
    medY = float(np.median(Y[core & (face_soft > 0.9)]))
    nostril = (gblur(Y, 0.004 * fw) < 0.68 * medY) & nose
    excl |= cv2.dilate(nostril.astype(np.uint8) * 255, disk(0.010 * fw))
    excl_soft = gblur(excl.astype(np.float32) / 255, 0.010 * fw)

    # Colour gate: drop anything whose chroma is far from her skin (ring, fabric).
    lab = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
    ref = core & (face_soft > 0.9)
    ab = lab[..., 1:3][ref]
    mu = ab.mean(0)
    icov = np.linalg.inv(np.cov(ab.T) + np.eye(2) * 1e-3)
    d = lab[..., 1:3] - mu
    dist = np.sqrt(np.einsum("...i,ij,...j->...", d, icov, d))
    gate = 1 - smoothstep(dist, 4.0, 7.0)
    gate = gblur(gate.astype(np.float32), 0.004 * fw)

    skin = skin * (1 - excl_soft) * gate
    # Drop tiny islands.
    hard = (skin > 0.5).astype(np.uint8)
    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(hard, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] > (0.08 * fw) ** 2
    skin = skin * keep[lab_cc]

    # Heavy feather, pulled inside the skin so the soft tail does not reach hair.
    sig = FEATHER_FRAC * fw
    inner = cv2.erode((skin > 0.5).astype(np.uint8), disk(1.0 * sig)).astype(np.float32)
    M = np.clip(gblur(inner, sig) * 1.15, 0, 1) * np.clip(skin * 1.5, 0, 1)

    # Under-eye zones (soft), and a cheek patch below each one for the target tone.
    under = np.zeros(shp, np.float32)
    cheek_ref = []
    for lid in (LOWER_LID_R, LOWER_LID_L):
        p = lm[lid] - off
        n_ = len(p)
        prof = np.sin(np.linspace(0, np.pi, n_))[:, None]
        top = p + np.array([0, 0.012 * fw]) * (0.4 + prof)
        bot = p + np.array([0, 0.095 * fw]) * (0.35 + 0.65 * prof)
        poly = np.concatenate([top, bot[::-1]]).astype(np.int32)
        z = np.zeros(shp, np.uint8)
        cv2.fillPoly(z, [poly], 255)
        under = np.maximum(under, z.astype(np.float32) / 255)
        cpoly = np.concatenate([bot, (bot + [0, 0.07 * fw])[::-1]]).astype(np.int32)
        cz = np.zeros(shp, np.uint8)
        cv2.fillPoly(cz, [cpoly], 255)
        cheek_ref.append((z > 0, cz > 0))
    under = gblur(under, 0.022 * fw) * M

    # Cheek / nose region for redness: face skin, centred on the mid-face.
    oval = fill_hull(shp, lm[OVAL], off).astype(np.float32) / 255
    mid = np.zeros(shp, np.float32)
    c = lm[[1, 50, 280]].mean(0) - off
    cv2.ellipse(mid, (int(c[0]), int(c[1])), (int(0.42 * fw), int(0.30 * fw)), 0, 0, 360, 1, -1)
    red_zone = gblur(mid * oval, 0.06 * fw) * M

    eye_soft = gblur(cv2.dilate(eyes, disk(0.008 * fw)).astype(np.float32) / 255, 0.004 * fw)
    hair_zone = hair * (1 - gblur((skin > 0.3).astype(np.float32), 0.006 * fw))

    return dict(fw=fw, lm=lm, roi=roi, M=M, skin=skin, under=under, cheek_ref=cheek_ref,
                red_zone=red_zone, eye_soft=eye_soft, hair_zone=hair_zone,
                face=(fx0, fy0, fx1, fy1))


# ----------------------------------------------------------------------------
# 2-5. retouch for one set of settings
# ----------------------------------------------------------------------------
def heal_blemishes(I, M, fw, amount, debug=None):
    """Find small dark / red local outliers and blend in a median heal."""
    lab = cv2.cvtColor(np.clip(I, 0, 1), cv2.COLOR_RGB2LAB)
    L8 = np.clip(lab[..., 0] * 2.55, 0, 255).astype(np.uint8)
    a8 = np.clip(lab[..., 1] + 128, 0, 255).astype(np.uint8)
    k = odd(0.045 * fw)
    bgL = cv2.medianBlur(L8, k).astype(np.float32)
    bga = cv2.medianBlur(a8, k).astype(np.float32)
    dark = (bgL - L8) / 255.0
    red = a8.astype(np.float32) - bga
    cand = ((dark > 0.035) | ((red > 4.0) & (dark > 0.008))) & (M > 0.6)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, disk(1))

    n, lbl, st, cen = cv2.connectedComponentsWithStats(cand, 8)
    a_min = np.pi * (0.004 * fw) ** 2      # below: a pore, leave it
    a_small = np.pi * (0.012 * fw) ** 2    # very small isolated spot
    a_max = np.pi * (0.030 * fw) ** 2      # above: not a blemish
    areas = st[:, cv2.CC_STAT_AREA].astype(np.float32)
    idx = np.arange(1, n)
    idx = idx[(areas[idx] >= a_min) & (areas[idx] <= a_max)]
    sel = np.zeros(n, bool)
    if len(idx):
        flat = lbl.ravel()
        mdark = np.bincount(flat, weights=dark.ravel(), minlength=n) / np.maximum(areas, 1)
        mred = np.bincount(flat, weights=red.ravel(), minlength=n) / np.maximum(areas, 1)
        pts = cen[idx]
        dd = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        neigh = (dd < 0.06 * fw).sum(1) - 1        # freckle fields have many neighbours
        for j, i in enumerate(idx):
            if neigh[j] > 3:
                continue
            if areas[i] <= a_small:
                sel[i] = True                     # tiny isolated spot
            elif mred[i] > 5.0 and mdark[i] < 0.09:
                sel[i] = True                     # red bump; dark moles are left alone
    hm = sel[lbl].astype(np.uint8)
    hm = cv2.dilate(hm, disk(0.006 * fw)).astype(np.float32)
    hm = gblur(hm, 0.004 * fw) * amount
    if debug is not None:
        debug["blemishes"] = int(sel.sum())
        debug["heal_mask"] = hm
    if not sel.any():
        return I
    u8 = np.clip(I * 255 + 0.5, 0, 255).astype(np.uint8)
    med = cv2.medianBlur(u8, k).astype(np.float32) / 255
    # Borrow texture from beside the spot so the heal is not a flat patch.
    s = max(1.0, FS_RADIUS_FRAC * fw)
    tex = I - gblur(I, s)
    tex = np.roll(tex, k, axis=1)
    heal = med + 0.8 * tex
    return I + hm[..., None] * (heal - I)


def retouch(img, A, P, seed=7, debug=None):
    fw = A["fw"]
    y0, y1, x0, x1 = A["roi"]
    M = A["M"]
    O = img[y0:y1, x0:x1]

    # 3. blemish cleanup (done first so spots do not survive in the tone layer)
    I = heal_blemishes(O, M, fw, P["BLEMISH_HEAL"], debug)

    # 2. frequency separation
    s = max(1.5, FS_RADIUS_FRAC * fw)
    low = gblur(I, s)
    high = I - low

    # Contour strength from the tone layer: real edges (jaw, nostril, lid, hairline)
    # are protected, so smoothing never crosses them and leaves no halo.
    Yl = gblur(luma(low), 0.5 * s)
    gx = cv2.Sobel(Yl, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(Yl, cv2.CV_32F, 0, 1, ksize=3) / 8
    g = np.sqrt(gx * gx + gy * gy) * fw
    edge = gblur(smoothstep(g, EDGE_LO, EDGE_HI), 0.5 * s)[..., None]

    # Low layer: edge-preserving guided filter evens blotches, keeps shading gradients.
    tone = guided(low, low, max(2, TONE_FILTER_FRAC * fw), TONE_FILTER_EPS)
    low_s = low + P["SMOOTHING_AMOUNT"] * (1 - edge) * (tone - low)

    # High layer: compress harsh peaks, keep a share of the pore texture.
    k = HARSH_TEXTURE_CLIP
    soft_high = k * np.tanh(high / k)
    high_s = edge * high + (1 - edge) * P["TEXTURE_RETENTION"] * soft_high
    R = low_s + high_s
    del low, high, tone, low_s, high_s, soft_high

    # 4. dodge & burn in Lab
    lab = cv2.cvtColor(np.clip(R, 0, 1), cv2.COLOR_RGB2LAB)
    Ls, as_, bs = (gblur(lab[..., i], 0.012 * fw) for i in range(3))
    U = A["under"]
    lift = np.zeros_like(Ls)
    for (uz, cz) in A["cheek_ref"]:
        if cz.sum() < 50 or uz.sum() < 50:
            continue
        cm = cz & (M > 0.5)
        if cm.sum() < 50:
            continue
        refL, refa, refb = (float(np.median(ch[cm])) for ch in (lab[..., 0], lab[..., 1], lab[..., 2]))
        lift = np.maximum(lift, np.clip(refL - Ls, 0, 25))
        zone = gblur(uz.astype(np.float32), 0.03 * fw) * U
        lab[..., 1] += 0.5 * P["UNDER_EYE_LIFT"] * (refa - as_) * zone
        lab[..., 2] += 0.5 * P["UNDER_EYE_LIFT"] * (refb - bs) * zone
    lab[..., 0] += P["UNDER_EYE_LIFT"] * lift * U

    Z = A["red_zone"]
    core = M > 0.8
    med_a = float(np.median(lab[..., 1][core]))
    excess = np.clip(as_ - med_a, 0, None)
    lab[..., 1] -= P["REDNESS_REDUCTION"] * excess * Z
    chroma = 1 - 0.20 * P["REDNESS_REDUCTION"] * Z
    lab[..., 1] *= chroma
    lab[..., 2] *= chroma
    R = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    del lab, Ls, as_, bs

    # 5a. blend retouched skin over the original through the feathered mask
    out = img.copy()
    w = (P["FINAL_OPACITY"] * M)[..., None]
    out[y0:y1, x0:x1] = O + w * (R - O)
    del R

    # 5b. polish: clarity on eyes and hair only
    reg = out[y0:y1, x0:x1]
    Y = luma(reg)
    det = Y - gblur(Y, 0.015 * fw)
    det = 0.06 * np.tanh(det / 0.06)
    fine = Y - gblur(Y, 1.2)
    wc = P["CLARITY_AMOUNT"] * (A["hair_zone"] + A["eye_soft"])
    reg += (wc * det + 1.2 * P["CLARITY_AMOUNT"] * A["eye_soft"] * fine)[..., None]
    out[y0:y1, x0:x1] = reg

    # 5c. soft glow on highlights (screen blend of blurred highlights)
    H, W = out.shape[:2]
    sm = cv2.resize(out, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    hi = sm * smoothstep(luma(sm), 0.72, 0.97)[..., None]
    hi = gblur(hi, 0.004 * max(H, W) / 4)
    hi = cv2.resize(hi, (W, H), interpolation=cv2.INTER_LINEAR)
    out = 1 - (1 - np.clip(out, 0, 1)) * (1 - P["GLOW_AMOUNT"] * hi)
    del hi, sm

    # 5d. fine uniform film grain over the whole frame
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((H, W), dtype=np.float32)
    n = gblur(n, 0.6)
    n /= n.std() + 1e-6
    nc = gblur(rng.standard_normal((H, W, 3), dtype=np.float32), 0.6)
    nc /= nc.std() + 1e-6
    grain = 0.85 * n[..., None] + 0.15 * nc
    del n, nc
    Yo = luma(out)
    tonal = (0.55 + 1.8 * Yo * (1 - Yo))[..., None]   # a little less in deep shadow / clipped white
    out += P["GRAIN_AMOUNT"] * grain * tonal
    return np.clip(out, 0, 1)


# ----------------------------------------------------------------------------
# comparison sheet
# ----------------------------------------------------------------------------
def comparison(versions, A, path, icc=None):
    names = list(versions)
    imgs = [versions[k] for k in names]
    H, W = imgs[0].shape[:2]
    tw = 900
    th = int(H * tw / W)
    fx0, fy0, fx1, fy1 = A["face"]
    fw = fx1 - fx0
    # 100% crop on the cheek / under-eye area
    cs = 900
    cx = int(fx0 + 0.30 * fw)
    cy = int(fy0 + 0.45 * (fy1 - fy0))
    cx0 = int(np.clip(cx - cs // 2, 0, W - cs))
    cy0 = int(np.clip(cy - cs // 2, 0, H - cs))
    cols = []
    for name, im in zip(names, imgs):
        u8 = np.clip(im * 255 + 0.5, 0, 255).astype(np.uint8)
        full = cv2.resize(u8, (tw, th), interpolation=cv2.INTER_AREA)
        crop = u8[cy0:cy0 + cs, cx0:cx0 + cs]
        crop = cv2.resize(crop, (tw, tw), interpolation=cv2.INTER_NEAREST) if cs != tw else crop
        label = np.full((70, tw, 3), 255, np.uint8)
        cv2.putText(label, name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3, cv2.LINE_AA)
        cols.append(np.vstack([label, full, np.full((10, tw, 3), 255, np.uint8), crop]))
    sep = np.full((cols[0].shape[0], 10, 3), 255, np.uint8)
    sheet = cols[0]
    for c in cols[1:]:
        sheet = np.hstack([sheet, sep, c])
    Image.fromarray(sheet).save(path, "JPEG", quality=92, **({"icc_profile": icc} if icc else {}))


# ----------------------------------------------------------------------------
def settings(preset):
    base = dict(SMOOTHING_AMOUNT=SMOOTHING_AMOUNT, TEXTURE_RETENTION=TEXTURE_RETENTION,
                FINAL_OPACITY=FINAL_OPACITY, GRAIN_AMOUNT=GRAIN_AMOUNT, BLEMISH_HEAL=BLEMISH_HEAL,
                UNDER_EYE_LIFT=UNDER_EYE_LIFT, REDNESS_REDUCTION=REDNESS_REDUCTION,
                CLARITY_AMOUNT=CLARITY_AMOUNT, GLOW_AMOUNT=GLOW_AMOUNT)
    base.update(PRESETS.get(preset, {}))
    return base


def process(path, outdir, debug_dir=None):
    name = os.path.splitext(os.path.basename(path))[0]
    rgb, icc, exif = load(path)
    img = rgb.astype(np.float32) / 255.0
    print(f"{name}: {rgb.shape[1]}x{rgb.shape[0]}, analysing")
    A = analyze(rgb, img)
    print(f"  face width {A['fw']:.0f}px, skin ROI {A['roi']}")
    if debug_dir:
        y0, y1, x0, x1 = A["roi"]
        dbg = (np.stack([A["M"]] * 3, -1) * 0.6 + img[y0:y1, x0:x1] * 0.4)
        cv2.imwrite(os.path.join(debug_dir, f"{name}_mask.jpg"),
                    cv2.cvtColor((cv2.resize(dbg, None, fx=0.3, fy=0.3) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    versions = {"original": img}
    for preset in ("subtle", "medium", "strong"):
        dbg = {}
        out = retouch(img, A, settings(preset), debug=dbg)
        p = os.path.join(outdir, f"{name}_{preset}.jpg")
        save(p, out, icc, exif)
        print(f"  {preset}: healed {dbg.get('blemishes', 0)} spots -> {p}")
        versions[preset] = out
    shutil.copyfile(os.path.join(outdir, f"{name}_medium.jpg"), os.path.join(outdir, f"{name}_airbrushed.jpg"))
    comparison(versions, A, os.path.join(outdir, f"{name}_comparison.jpg"), icc)
    return {k: os.path.join(outdir, f"{name}_{k}.jpg")
            for k in ("subtle", "medium", "strong", "airbrushed", "comparison")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--outdir", default=os.path.join(HERE, "output"))
    ap.add_argument("--debug")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.debug:
        os.makedirs(a.debug, exist_ok=True)
    for p in a.inputs:
        try:
            process(p, a.outdir, a.debug)
        except NoFaceError as e:
            print(f"{os.path.basename(p)}: skipped. {e}")


if __name__ == "__main__":
    main()
