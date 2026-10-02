#!/usr/bin/env python3
"""
Sweep the engine's own crop path (Ocr.kt: stripPadFor -> expandQuad -> transformedRegion ->
stripToTensor) over the 30 frozen quads, scoring against ground truth.

This is a port of the PRODUCTION crop, not of ocr_parity's. ocr_parity.build_strips crops tight to
the quad with findHomography and no stripPad, so it cannot answer questions about stripPad - that
option only exists in Ocr.kt. Porting the production path is the only way to sweep it.

Run single-threaded under an address-space cap; ORT's default intra-op pool is one thread per core.
"""
import json
import math
import os
import resource
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TEXT_HEIGHT = 48
PAD_MARGIN = 16
BASE_STRIP_PAD = 4
BASE_FRACTION = 0.12

FIXTURE = os.path.join(HERE, "fixtures", "faithful_boxes.json")
GROUND_TRUTH = os.path.join(HERE, "fixtures", "ground_truth.json")
ALPHABET = os.path.join(ROOT, "engine/src/main/assets/models/alphabet-all-v5.txt")
BUNDLED_MODEL = os.environ.get("MTL_OCR_INT8", "/mnt/c/users/branden/mtl-models/ocr_int8.onnx")


# ── Geometry.kt ──────────────────────────────────────────────────────────────

def cross(o, a, b):
    """Geometry.cross(o, a, b) - the vertex order matters. With the <0f pop condition the
    mirrored form silently collapses a rectangle to its diagonal, which makes minAreaRect
    report h=0 and corrupts the quad that sortPnts sees."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    """Andrew's monotone chain, matching Geometry.convexHull including its <0f pop condition."""
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    if len(pts) < 3:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    lower.pop()
    upper.pop()
    return lower + upper


def min_area_rect(points):
    """Returns (cx, cy, ux, uy, w, h) like Geometry.RotRect, or None."""
    hull = convex_hull(points)
    if len(hull) < 2:
        return None
    best_area = float("inf")
    best = None
    for i in range(len(hull)):
        a = hull[i]
        b = hull[(i + 1) % len(hull)]
        ex, ey = b[0] - a[0], b[1] - a[1]
        length = math.hypot(ex, ey)
        if length < 1e-6:
            continue
        ex, ey = ex / length, ey / length
        us = []
        vs = []
        for p in hull:
            dx, dy = p[0] - a[0], p[1] - a[1]
            us.append(dx * ex + dy * ey)
            vs.append(-dx * ey + dy * ex)
        w = max(us) - min(us)
        h = max(vs) - min(vs)
        area = w * h
        if area < best_area:
            best_area = area
            cu = (min(us) + max(us)) / 2.0
            cv = (min(vs) + max(vs)) / 2.0
            best = (a[0] + cu * ex - cv * ey, a[1] + cu * ey + cv * ex, ex, ey, w, h)
    return best


def rect_corners(r):
    """Geometry.RotRect.corners(), in the engine's exact order: the minor axis is (-uy, ux), so the
    terms are -hh * -uy and +hh * -uy on x, and -hh * ux / +hh * ux on y."""
    cx, cy, ux, uy, w, h = r
    hw, hh = w / 2.0, h / 2.0
    mx, my = -uy, ux
    return [
        (cx - hw * ux - hh * mx, cy - hw * uy - hh * my),
        (cx + hw * ux - hh * mx, cy + hw * uy - hh * my),
        (cx + hw * ux + hh * mx, cy + hw * uy + hh * my),
        (cx - hw * ux + hh * mx, cy - hw * uy + hh * my),
    ]


def rect_expand(r, pad):
    cx, cy, ux, uy, w, h = r
    return (cx, cy, ux, uy, w + 2.0 * pad, h + 2.0 * pad)


# ── Ocr.kt ──────────────────────────────────────────────────────────────────

def sort_pnts(quad):
    """Ocr.sortPnts: longest-edge detection, then corner ordering. Returns (ordered, is_vertical)."""
    q = list(quad)
    n = len(q)
    norms = {}
    for i in range(n):
        for j in range(n):
            norms[i * n + j] = math.hypot(q[i][0] - q[j][0], q[i][1] - q[j][1])
    order = sorted(norms.items(), key=lambda kv: kv[1])
    b0, b1 = order[8][0], order[10][0]
    l0x = q[b0 // n][0] - q[b0 % n][0]
    l0y = q[b0 // n][1] - q[b0 % n][1]
    l1x = q[b1 // n][0] - q[b1 % n][0]
    l1y = q[b1 // n][1] - q[b1 % n][1]
    if l0x * l1x + l0y * l1y < 0.0:
        l0x, l0y = -l0x, -l0y
    is_v = abs((l0x + l1x) / 2.0) <= abs((l0y + l1y) / 2.0)
    if is_v:
        by_y = sorted(q, key=lambda p: p[1])
        first2 = sorted(by_y[:2], key=lambda p: p[0])
        last2 = sorted(by_y[2:], key=lambda p: p[0])
        return [first2[0], first2[1], last2[1], last2[0]], True
    by_x = sorted(q, key=lambda p: p[0])
    ls = sorted(by_x[:2], key=lambda p: p[1])
    rs = sorted(by_x[2:], key=lambda p: p[1])
    return [ls[0], rs[0], rs[1], ls[1]], False


def strip_pad_for(quad, base, fraction):
    if base <= 0:
        return 0
    r = min_area_rect(quad)
    if r is None:
        return base
    shortest = min(r[4], r[5])
    if not math.isfinite(shortest) or shortest <= 0:
        return base
    return max(base, int(round(shortest * fraction)))


def expand_quad(pts, pad, w, h):
    r = min_area_rect(pts)
    if r is None:
        return pts
    out = []
    for p in rect_corners(rect_expand(r, float(pad))):
        out.append((min(max(p[0], 0.0), w - 1.0), min(max(p[1], 0.0), h - 1.0)))
    return out


def transformed_region(page, pts, is_v, th):
    """Ocr.transformedRegion: edge-midpoint ratio, perspective warp to th, rotate CCW if vertical."""
    import cv2
    p1 = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
    p2 = ((pts[2][0] + pts[3][0]) / 2, (pts[2][1] + pts[3][1]) / 2)
    p3 = ((pts[1][0] + pts[2][0]) / 2, (pts[1][1] + pts[2][1]) / 2)
    p4 = ((pts[3][0] + pts[0][0]) / 2, (pts[3][1] + pts[0][1]) / 2)
    ratio = math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / max(math.hypot(p4[0] - p3[0], p4[1] - p3[1]), 1e-6)
    if not is_v:
        h, w = max(th, 2), max(int(round(th / max(ratio, 1e-6))), 2)
    else:
        w, h = max(th, 2), max(int(round(th * ratio)), 2)
    src = np.array(pts, dtype=np.float32)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    region = cv2.warpPerspective(page, m, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    if is_v:
        region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return region


def strip_to_tensor(strip):
    """Ocr.stripToTensor: CHW, (p-127.5)/127.5, plus PAD_MARGIN white columns on the right."""
    sh, sw = strip.shape[:2]
    w = sw + PAD_MARGIN
    canvas = np.full((sh, w, 3), 255.0, dtype=np.float32)
    canvas[:, :sw] = strip[:, :, :3].astype(np.float32)
    f = (canvas - 127.5) / 127.5
    return np.ascontiguousarray(f.transpose(2, 0, 1)[None])


def cer(ref, hyp):
    if ref == hyp:
        return 0.0
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, b in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b))
        prev = cur
    return prev[len(hyp)] / max(len(ref), 1)


def main():
    cap = int(float(os.environ.get("MTL_MEM_CAP_GB", "4")) * (1 << 30))
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    import cv2
    import onnxruntime as ort

    fx = json.load(open(FIXTURE, encoding="utf-8"))
    gt = json.load(open(GROUND_TRUTH, encoding="utf-8"))["texts"]
    page = cv2.imread(os.path.join(ROOT, fx["image_path"]), cv2.IMREAD_COLOR)
    if page is None:
        sys.exit("cannot read page")
    ph, pw = page.shape[:2]

    alphabet = [l.rstrip("\n") for l in open(ALPHABET, encoding="utf-8") if l != ""]
    if not os.path.isfile(BUNDLED_MODEL):
        sys.exit(f"missing model {BUNDLED_MODEL} (override with MTL_OCR_INT8)")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False
    sess = ort.InferenceSession(BUNDLED_MODEL, sess_options=opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    widths = [o.shape[2] for o in sess.get_outputs() if isinstance(o.shape[2], int)]
    if widths and max(widths) != len(alphabet):
        sys.exit(f"logits width {max(widths)} != alphabet {len(alphabet)}")

    def read(quad, pad, fraction):
        ordered, is_v = sort_pnts(quad)
        if pad > 0:
            p = strip_pad_for(ordered, pad, fraction)
            ordered = expand_quad(ordered, p, pw, ph)
            ordered, is_v = sort_pnts(ordered)
        region = transformed_region(page, ordered, is_v, TEXT_HEIGHT)
        x = strip_to_tensor(region)
        outs = sess.run(None, {inp: x})
        logits = max((q for q in outs if q.ndim == 3), key=lambda q: q.shape[-1])[0]
        lp = logits - logits.max(1, keepdims=True)
        lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
        idx = lp.argmax(1)
        out, last = [], 0
        for t in range(len(idx)):
            c = int(idx[t])
            if c != last and c != 0 and c < len(alphabet):
                out.append(alphabet[c])
            last = c
        return "".join(out)

    def run(pad, fraction):
        rc = c = trunc = n = 0
        for idx, box in enumerate(fx["boxes"]):
            got = read(box["quad"], pad, fraction)
            ref = gt[str(idx)]
            if got != ref:
                n += 1
                if ref.startswith(got):
                    trunc += 1
            rc += round(cer(ref, got) * len(ref))
            c += len(ref)
        return rc / c, n, trunc

    print(f"page {fx['image']} {pw}x{ph}  quads {len(fx['boxes'])}  alphabet {len(alphabet)}")
    print(f"\n  production crop path: stripPadFor -> expandQuad -> warp -> rot90 -> PAD_MARGIN")
    print(f"  base = {BASE_STRIP_PAD}, fraction = {BASE_FRACTION}\n")
    print(f"  {'stripPad':>9}{'fraction':>10}{'CER':>9}{'bad':>6}{'trunc':>7}")
    print("  " + "-" * 41)

    ref_cer, ref_bad, ref_trunc = run(BASE_STRIP_PAD, BASE_FRACTION)
    print(f"  {BASE_STRIP_PAD:>9}{BASE_FRACTION:>10}{ref_cer:>9.4f}{ref_bad:>6}{ref_trunc:>7}   <- shipped default")

    rows = []
    for pad in (0, 2, 4, 6, 8, 10, 12):
        for fraction in (0.0, 0.06, 0.12, 0.18, 0.25):
            if pad == BASE_STRIP_PAD and fraction == BASE_FRACTION:
                continue
            c_, b_, t_ = run(pad, fraction)
            rows.append((pad, fraction, c_, b_, t_))
            print(f"  {pad:>9}{fraction:>10}{c_:>9.4f}{b_:>6}{t_:>7}")

    best = min(rows, key=lambda r: (r[2], r[3]))
    print(f"\n  best: stripPad={best[0]} fraction={best[1]} -> CER {best[2]:.4f} "
          f"({best[3]} bad, {best[4]} trunc) vs default {ref_cer:.4f} ({ref_bad} bad, {ref_trunc} trunc)")
    if best[2] < ref_cer:
        print("  NOTE: a single-page sweep can favour a value that regresses elsewhere. Confirm on more pages.")



if __name__ == "__main__":
    main()