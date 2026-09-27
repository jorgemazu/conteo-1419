# MU Immortal - contador HUD + inventario
# 1.4.20
# 1. Bolsa y baul solo si hay una foto donde se ven los dos.
# 2. Si la foto es solo la bolsa, se lee nada mas el cuadro
#    de oro, MUC, bound MUC y diamantes.
# 3. Si hay varias fotos, se usan las de los 10 minutos
#    hacia atras de la ultima. Esos 10 minutos no deciden el HUD.

import argparse, os, sys, time, json, re
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from PIL import Image

try:
    import pytesseract
    for _tess in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.isfile(_tess):
            pytesseract.pytesseract.tesseract_cmd = _tess
            break
    HAS_TESS = True
except Exception:
    HAS_TESS = False

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def find_icon_dir(cli=None):
    cands = []
    if cli:
        cands.append(cli)
    here = script_dir()
    cands += [
        os.path.join(here, "iconos"),
        os.path.join(here, "icons"),
        r"C:\keepalive\conteo\iconos",
    ]
    for d in cands:
        if d and os.path.isdir(d):
            return d
    return os.path.join(here, "iconos")


def parse_shot_stamp(name):
    m = re.search(r"(20\d{6})[-_]?(\d{6})", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except Exception:
        return None


def file_stamp(path):
    st = parse_shot_stamp(os.path.basename(path))
    if st:
        return st
    try:
        return datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return None


def misma_tanda(a, b, minutos=10):
    if not a or not b:
        return False
    return abs(a - b) <= timedelta(minutes=minutos)


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


# ---------- clasificacion de pantalla ----------

def _grid_score(gray):
    """Que tan parecido a una grilla 8x8 de inventario."""
    h, w = gray.shape
    if h < 40 or w < 40:
        return 0.0
    gx = np.abs(np.diff(gray.astype(np.int16), axis=1)).mean()
    gy = np.abs(np.diff(gray.astype(np.int16), axis=0)).mean()
    return float(gx + gy)


def bag_left_frac(img):
    """X relativo donde empieza el panel derecho (mochila)."""
    h, w = img.shape[:2]
    gray = np.array(Image.fromarray(img).convert("L"), dtype=np.int16)
    best_x, best = int(w * 0.62), -1
    y0, y1 = int(h * 0.10), int(h * 0.88)
    for frac in (0.50, 0.54, 0.58, 0.62, 0.66, 0.70, 0.74):
        x = int(w * frac)
        sl = gray[y0:y1, x:min(w - 4, x + max(8, w // 12))]
        sc = _grid_score(sl)
        if sc > best:
            best, best_x = sc, x
    return best_x / float(w)


HUD_TMPL = {
    "0": ("0ff00ff03c3c3c3cf00cf00cf00cf00ff00ff00ff00fc00fc00ff00ff00ff00ff00ff00cf00cf00c3c3c3c3c0ff00ff0","0ff00ff03c3c3c3c300f300f300f300ff003f003f003f003f003f003f003f003300f300f3c0c3c0c0f3c0f3c03c003c0","0ff00ff03c3c3c3c300f300f300f300ff003f003f003f003f003f003f003f003300f300f3c0c3c0c0c3c0c3c03c003c0"),
    "1": ("ffffffff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff","ffffffff07ff07ff001f001f001f001f001f001f001f001f001f001f001f001f001f001f001f001f001f001f001f001f"),
    "2": ("3fe03fe0f87cf87cc01cc01cc01cc01c001c001c0060006001e001e007800780060006003e003e00f800f800ffffffff",),
    "3": ("3fe03fe0c07cc07c001c001c001c007c007c01e001e001e001e0001c001c001f001f001f001f001fc01cc01cffe0ffe0","07fc07fc381c381c381f381f001f001f001c001c01fc01fc001f001f0003000300030003f803f8033e7f3e7f07e007e0"),
    "4": ("007c007c01fc01fc019c019c019c061c061c061c061c381c381cf81ff81fffffffff007f007f007f001c001c001c001c",),
    "5": ("3ffc3ffc380038003800380038003e003e003fe03fe0007c007c001f001f001f001f001c001c001cc07cc07cffe0ffe0",),
    "6": ("0060006001e001e00780078007803f803f803ffc3ffc3e1f3e1ff803f803f803f803f803f803f803381f381f07fc07fc","01e001e0078007803e003e003e003e003e00ffe0ffe0f87cf87cc01fc01fc01fc01fc01fc01fc01ff81cf81c3fe03fe0","0060006001e001e007800780060006003ffc3ffc3fff3ffff803f803f803f803f803f803380338033e1f3e1f01e001e0"),
    "7": ("ffffffff00070007001f001f001f0018001800f800f800e000e007e007e0070007000700070007001800180018001800","ffffffff001f001f001c001c001c001c001c006000600060006001e001e001800180078007800780060006003e003e00"),
    "8": ("07fc07fc3e1f3e1f3803380338033e1c3e1c07fc07fc3ffc3ffc38033803f803f803f803f803f803381f381f07fc07fc",),
    "9": ("07fc07fc381f381ff803f803f803f803f803f803f8033e1f3e1f07ff07ff007c007c007c007c007c01e001e001800180",),
}
HUD_HOLES = {"0": {1}, "1": {0}, "2": {0}, "3": {0}, "4": {1}, "5": {0}, "6": {1}, "7": {0}, "8": {2}, "9": {1}}
_HUD_VECS = None


def _rgb(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img.convert("RGB"))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img


def _gold_mask(img):
    rgb = _rgb(img)
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (r > 145) & (g > 100) & (g < 225) & (b < 155) & ((r - b) > 28) & (r + 20 > g)


def _col_glyphs(mask, y0, y1, min_col):
    band = mask[y0:y1]
    cols = band.sum(0) >= min_col
    out = []
    i = 0
    n = int(cols.shape[0])
    while i < n:
        if not cols[i]:
            i += 1
            continue
        j = i
        while j < n and cols[j]:
            j += 1
        sl = band[:, i:j]
        rows = sl.any(1)
        if rows.any():
            r0 = int(np.argmax(rows))
            r1 = int(len(rows) - np.argmax(rows[::-1]))
            g = sl[r0:r1].astype(np.uint8)
            bh, bw = g.shape
            if bh >= 6 and 1 <= bw <= bh * 1.15:
                out.append({"x0": i, "x1": j, "y0": y0 + r0, "w": bw, "h": bh, "g": g})
        i = j + 1
    return out


def _gold_bands(mask, min_row):
    hot = mask.sum(1) >= min_row
    raw = []
    i = 0
    h = len(hot)
    while i < h:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < h and hot[j]:
            j += 1
        if j - i >= 8:
            raw.append([i, j])
        i = j
    merged = []
    for b in raw:
        if merged and b[0] - merged[-1][1] <= 4:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    return [(a, c) for a, c in merged]


def _groups(glyphs):
    digits = [g for g in glyphs if g["w"] <= g["h"] * 0.92]
    if not digits:
        return []
    med = float(np.median([g["w"] for g in digits]))
    digits = sorted(digits, key=lambda t: t["x0"])
    groups = [[digits[0]]]
    for a, b in zip(digits, digits[1:]):
        if b["x0"] - a["x1"] > max(3, med * 1.35):
            groups.append([b])
        else:
            groups[-1].append(b)
    return [g for g in groups if 1 <= len(g) <= 10]


def _slot_dark(img, glyphs):
    """A la izquierda del numero hay casilla negra, no un icono de la mochila."""
    rgb = _rgb(img)
    y0 = min(t["y0"] for t in glyphs)
    y1 = max(t["y0"] + t["h"] for t in glyphs)
    x1 = min(t["x0"] for t in glyphs)
    h = max(4, y1 - y0)
    x0 = max(0, x1 - int(h * 2.2))
    if x1 - x0 < 4 or y1 <= y0:
        return 0.0
    sl = rgb[y0:y1, x0:x1]
    luma = sl.astype(np.int16).mean(axis=2)
    return float((luma < 48).mean())


def _same_height(groups):
    heights = [float(np.median([t["h"] for t in g])) for g in groups]
    if min(heights) < 6:
        return False
    return (max(heights) / min(heights)) <= 1.25


def _find_money_panel(img):
    mask = _gold_mask(img)
    H = mask.shape[0]
    min_row = 12 if H > 400 else 8
    min_col = 1 if H > 400 else 2
    cands = []
    for y0, y1 in _gold_bands(mask, min_row):
        gr = _groups(_col_glyphs(mask, y0, y1, min_col))
        if len(gr) >= 2:
            cands.append((y0, y1, gr))
    best = None
    for i, (y0, y1, gr) in enumerate(cands):
        for y2, y3, gr2 in cands[i + 1:i + 3]:
            if not (0 < y2 - y1 <= max(36, int(H * 0.06))):
                continue
            for L in gr:
                for R in gr:
                    if R[0]["x0"] < L[-1]["x1"] + 12:
                        continue
                    eL, eR = L[-1]["x1"], R[-1]["x1"]
                    if eR - eL < 40:
                        continue
                    for L2 in gr2:
                        if abs(L2[-1]["x1"] - eL) > 14:
                            continue
                        for R2 in gr2:
                            if abs(R2[-1]["x1"] - eR) > 14:
                                continue
                            if R2[0]["x0"] < L2[-1]["x1"] + 12:
                                continue
                            quad = (L, R, L2, R2)
                            if not _same_height(quad):
                                continue
                            if any(_slot_dark(img, grp) < 0.90 for grp in quad):
                                continue
                            score = len(L) + len(R) + len(L2) + len(R2) + (y0 / float(H)) * 3
                            if best is None or score > best[0]:
                                best = (score, [L, R, L2, R2])
    return None if best is None else best[1]


def _norm_glyph(g, W=16, H=24):
    im = Image.fromarray((g * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
    return (np.array(im) > 80).astype(np.uint8)


def _holes(binimg):
    h, w = binimg.shape
    bg = np.zeros_like(binimg, np.uint8)
    stack = [(y, x) for y in (0, h - 1) for x in range(w)]
    stack += [(y, x) for x in (0, w - 1) for y in range(h)]
    while stack:
        y, x = stack.pop()
        if y < 0 or x < 0 or y >= h or x >= w or bg[y, x] or binimg[y, x]:
            continue
        bg[y, x] = 1
        stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
    seen = bg.copy()
    n = 0
    for y in range(h):
        for x in range(w):
            if binimg[y, x] or seen[y, x]:
                continue
            n += 1
            stack = [(y, x)]
            seen[y, x] = 1
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and not binimg[ny, nx]:
                        seen[ny, nx] = 1
                        stack.append((ny, nx))
    return n


def _slope(n):
    xs = []
    for y, row in enumerate(n):
        idx = np.where(row)[0]
        if len(idx):
            xs.append((y, float(idx.mean())))
    if len(xs) < 4:
        return 0.0
    ys = np.array([a for a, _ in xs], dtype=np.float64)
    xx = np.array([b for _, b in xs], dtype=np.float64)
    A = np.vstack([ys, np.ones(len(ys))]).T
    m, _c = np.linalg.lstsq(A, xx, rcond=None)[0]
    return float(m)


def _vecs():
    global _HUD_VECS
    if _HUD_VECS is not None:
        return _HUD_VECS
    out = {}
    for ch, hexs in HUD_TMPL.items():
        vecs = []
        for hx in hexs:
            bits = bin(int(hx, 16))[2:].zfill(16 * 24)
            vecs.append(np.array([1 if b == "1" else 0 for b in bits], dtype=np.uint8))
        out[ch] = vecs
    _HUD_VECS = out
    return out


def _read_glyphs(glyphs):
    s = ""
    vecs = _vecs()
    for g in glyphs:
        n = _norm_glyph(g["g"])
        v = n.astype(np.int16).ravel()
        hh = _holes(n)
        ratio = g["w"] / float(g["h"])
        if hh >= 2:
            s += "8"
            continue
        if ratio < 0.42 and hh == 0:
            s += "1"
            continue
        best, bch = 9.0, None
        for ch, tpls in vecs.items():
            if hh not in HUD_HOLES.get(ch, ()):
                continue
            if ch == "1" and ratio >= 0.42:
                continue
            sc = min(float(np.abs(v - t.astype(np.int16)).mean()) for t in tpls)
            if sc < best:
                best, bch = sc, ch
        if bch in ("2", "3"):
            sl = _slope(n)
            if bch == "3" and sl <= -0.09:
                bch = "2"
        if not bch or best > 0.42:
            return None
        s += bch
    return s


def hud_panel_visible(img):
    """El cuadro de monedas esta abierto: cuatro numeros alineados a la derecha."""
    try:
        return _find_money_panel(img) is not None
    except Exception:
        return False


def read_hud_best(img):
    """Oro, MUC, diamantes y bound MUC. No usa coordenadas fijas ni Tesseract."""
    try:
        panel = _find_money_panel(img)
    except Exception as exc:
        print("HUD error", exc)
        return {}
    if not panel:
        print("HUD panel no encontrado")
        return {}
    out = {}
    for key, glyphs in zip(("gold", "muc", "diamantes", "boundmuc"), panel):
        raw = _read_glyphs(glyphs)
        if raw and raw.isdigit():
            out[key] = int(raw)
            print("HUD", key, out[key], "raw", raw)
        else:
            print("HUD", key, "vacio", raw)
    return out


def es_baul_y_bolsa(img):
    """Dos grillas grandes: almacen a la izquierda y bolsa a la derecha."""
    h, w = img.shape[:2]
    gray = np.array(Image.fromarray(img).convert("L"))
    y0, y1 = int(h * 0.14), int(h * 0.80)
    derecha = gray[y0:y1, int(w * 0.68):int(w * 0.98)]
    almacen = gray[y0:y1, int(w * 0.30):int(w * 0.62)]
    sr = _grid_score(derecha)
    sa = _grid_score(almacen)
    print("grilla bolsa", round(sr, 1), "grilla baul", round(sa, 1))
    return sr >= 24.0 and sa >= 24.0


def es_solo_bolsa(img):
    if es_baul_y_bolsa(img):
        return False
    return hud_panel_visible(img)


def guess_regions(w, h):
    # en pantalla almacen: bodega centro-izq, mochila derecha
    bag = 0.62
    return {
        "mochila": (int(w * bag), int(h * 0.12), int(w * 0.985), int(h * 0.82)),
        "bodega": (int(w * 0.28), int(h * 0.12), int(w * bag), int(h * 0.82)),
    }


# ---------- iconos / grilla (solo fotos de almacen) ----------

ICON_MAP = [
    ("bless", "BLESS"), ("soul", "SOUL"), ("life", "LIFE"),
    ("chaos", "CHAOS"), ("creation", "CREATION"),
    ("sd_seed", "SD SEED"), ("sd", "SD SEED"),
    ("combo_heart", "COMBO HEART"), ("condor", "CONDOR"),
    ("garuda", "GARUDA"), ("wing_enhance", "WING ENHANCE STONE"),
    ("heavenly_steel", "HEAVENLY STEEL"),
    ("angel_signet", "ANGEL SIGNET"), ("rossy_signet", "ROSSY SIGNET"),
    ("anillos", "ANILLOS"), ("aros", "AROS"), ("collares", "COLLARES"),
    ("fluorite_fire", "Fluorite Fire"), ("fluorite_ice", "Fluorite Ice"),
    ("fluorite_wind", "Fluorite Wind"), ("fluorite_water", "Fluorite Water"),
]


def load_templates(icon_dir):
    tpls = []
    if not icon_dir or not os.path.isdir(icon_dir):
        print("sin iconos", icon_dir)
        return tpls
    files = [fn for fn in os.listdir(icon_dir) if fn.lower().endswith(".png")]
    for fn in files:
        raw = os.path.splitext(fn)[0].lower()
        name = None
        for st, pretty in ICON_MAP:
            if raw == st or raw.startswith(st + "_"):
                name = pretty
                break
        if not name:
            continue
        try:
            im = np.array(Image.open(os.path.join(icon_dir, fn)).convert("RGB"))
            tpls.append((name, im, raw))
        except Exception as e:
            print("no lei icono", fn, e)
    print("iconos:", icon_dir, "templates:", len(tpls))
    return tpls


def ncc_match(panel, tpl):
    ph, pw = panel.shape[:2]
    th, tw = tpl.shape[:2]
    if th >= ph or tw >= pw or th < 4 or tw < 4:
        return 0.0, 0, 0
    if HAS_CV2:
        p = cv2.cvtColor(panel, cv2.COLOR_RGB2GRAY)
        t = cv2.cvtColor(tpl, cv2.COLOR_RGB2GRAY)
        if p.shape[0] <= t.shape[0] or p.shape[1] <= t.shape[1]:
            return 0.0, 0, 0
        res = cv2.matchTemplate(p, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        return float(mx), int(loc[0]), int(loc[1])
    # numpy fallback, recorre cada 4 px
    t = tpl.astype(np.float32)
    t = t - t.mean()
    denom_t = np.sqrt((t * t).sum()) + 1e-6
    best, bx, by = -1.0, 0, 0
    step = max(2, min(th, tw) // 6)
    for y in range(0, ph - th, step):
        for x in range(0, pw - tw, step):
            w = panel[y:y + th, x:x + tw].astype(np.float32)
            w = w - w.mean()
            sc = float((w * t).sum() / (np.sqrt((w * w).sum()) * denom_t + 1e-6))
            if sc > best:
                best, bx, by = sc, x, y
    return best, bx, by


def read_qty_corner(cell):
    """Numero chico arriba-derecha de la celda."""
    h, w = cell.shape[:2]
    crop = cell[0:max(4, int(h * 0.38)), int(w * 0.42):]
    pil = Image.fromarray(crop)
    pil = pil.resize((pil.size[0] * 3, pil.size[1] * 3), Image.NEAREST)
    raw = _tess_digits(Image.fromarray(np.where(np.array(pil.convert("L")) > 150, 255, 0).astype(np.uint8)))
    if not raw:
        return 1
    try:
        n = int(raw)
        if 1 <= n <= 255:
            return n
        if n > 255:
            return 255
    except Exception:
        pass
    return 1


def fluorite_level(name, cell):
    if not name.startswith("Fluorite"):
        return name
    # nivel: digito bajo-izquierda oscuro; default 1
    h, w = cell.shape[:2]
    crop = cell[int(h * 0.55):, 0:int(w * 0.45)]
    raw = _tess_digits(Image.fromarray(crop))
    lvl = 1
    if raw:
        try:
            v = int(raw[0])
            if 1 <= v <= 9:
                lvl = v
            elif v == 0:
                lvl = 10
        except Exception:
            pass
    return "%s %d" % (name, lvl)


def match_region(img, x0, y0, x1, y1, tpls):
    if not tpls:
        return []
    panel = img[y0:y1, x0:x1]
    ph, pw = panel.shape[:2]
    if ph < 16 or pw < 16:
        return []
    rows, cols = 8, 8
    ch, cw = ph // rows, pw // cols
    hits = []
    for r in range(rows):
        for c in range(cols):
            cell = panel[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            if cell.mean() < 18:
                continue
            best, bname = 0.35, None
            for name, tpl, _raw in tpls:
                # escala template a ~70% del lado de celda
                side = max(8, int(min(cell.shape[0], cell.shape[1]) * 0.72))
                t = np.array(Image.fromarray(tpl).resize((side, side), Image.BILINEAR))
                sc, _, _ = ncc_match(cell, t)
                if sc > best:
                    best, bname = sc, name
            if not bname:
                continue
            qty = read_qty_corner(cell)
            item = fluorite_level(bname, cell)
            hits.append((best, item, qty, r, c))
    return hits


def merge_row_counts(photos, tpls, panel_name, region):
    total = defaultdict(int)
    x0, y0, x1, y1 = region
    seen_rows = []
    for path, img in photos:
        h, w = img.shape[:2]
        xa, xb = max(0, x0), min(w, x1)
        ya, yb = max(0, y0), min(h, y1)
        panel = img[ya:yb, xa:xb]
        if panel.size == 0:
            continue
        # firma de filas para no sumar el mismo scroll dos veces
        small = np.array(Image.fromarray(panel).resize((32, 16), Image.BILINEAR).convert("L"))
        sig = small.tobytes()
        if sig in seen_rows:
            print("fila repetida", os.path.basename(path), panel_name)
            continue
        seen_rows.append(sig)
        peaks = match_region(img, xa, ya, xb, yb, tpls)
        for sc, item, qty, r, c in peaks:
            total[item] += qty
            print("  ", panel_name, item, "qty", qty, "score", round(sc, 3), "cell", r, c)
    return dict(total)


def _json_num(o):
    if isinstance(o, dict):
        return {str(k): _json_num(v) for k, v in o.items()}
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o)
    return o


def list_image_files(folder):
    files = []
    for fn in os.listdir(folder):
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            files.append(os.path.join(folder, fn))
    files.sort(key=lambda p: (parse_shot_stamp(os.path.basename(p)) or datetime.min, os.path.basename(p)))
    return files


def tiene_grilla_inventario(img):
    h, w = img.shape[:2]
    gray = np.array(Image.fromarray(img).convert("L"))
    sl = gray[int(h * 0.12):int(h * 0.82), int(w * 0.62):int(w * 0.98)]
    return _grid_score(sl) >= 18.0


def _tanda(files, minutos=10):
    """Ultima foto y las de hasta 10 minutos hacia atras. No hacia adelante."""
    items = []
    for fp in files:
        items.append((file_stamp(fp) or datetime.min, fp))
    items.sort(key=lambda t: (t[0], t[1]))
    if not items:
        return []
    last = items[-1][0]
    if last == datetime.min:
        return [fp for _, fp in items]
    kept = []
    for st, fp in items:
        if st == datetime.min:
            continue
        delta = last - st
        if timedelta(0) <= delta <= timedelta(minutes=minutos):
            kept.append(fp)
    return kept


def scan_folder(folder, tpls, region="auto", max_fotos=2):
    files = list_image_files(folder)
    if not files:
        return {}
    print("carpeta", folder, "archivos", len(files))
    tanda = _tanda(files, 10)
    print("tanda", len(tanda), "de", len(files))
    for fp in tanda:
        print("  en tanda", os.path.basename(fp), file_stamp(fp))

    solo = []
    ambos = []
    for fp in reversed(tanda):
        try:
            img = load_rgb(fp)
        except Exception as e:
            print("no lei", fp, e)
            continue
        if es_baul_y_bolsa(img):
            ambos.append((fp, img))
            print("BAUL+BOLSA", os.path.basename(fp))
            continue
        if hud_panel_visible(img):
            solo.append((fp, img))
            print("SOLO BOLSA", os.path.basename(fp))
            continue
        print("sin cuadro y sin baul", os.path.basename(fp))

    out = {"mochila": {}, "bodega": {}, "currencies": {}}

    # Regla 2: el cuadro sale de la foto mas nueva que sea solo bolsa.
    if solo:
        fp, img = solo[0]
        hud = read_hud_best(img)
        cur = {}
        if "gold" in hud:
            cur["gold"] = hud["gold"]
            cur["oro"] = hud["gold"]
        if "muc" in hud:
            cur["muc"] = hud["muc"]
        if "boundmuc" in hud:
            cur["boundmuc"] = hud["boundmuc"]
            cur["bound_muc"] = hud["boundmuc"]
        if "diamantes" in hud:
            cur["diamantes"] = hud["diamantes"]
            cur["diamond"] = hud["diamantes"]
        out["currencies"] = cur
        print("currencies gold=%s muc=%s diamantes=%s bound=%s foto=%s" % (
            cur.get("gold", 0), cur.get("muc", 0), cur.get("diamantes", 0),
            cur.get("boundmuc", 0), os.path.basename(fp)))
    else:
        print("sin foto de solo bolsa: no hay cuadro de monedas")

    # Regla 1: joyas solo si en la tanda hay una foto con baul y bolsa a la vez.
    do_m = region in ("auto", "mochila")
    do_b = region in ("auto", "bodega")
    if ambos and (do_m or do_b):
        w = ambos[0][1].shape[1]
        h = ambos[0][1].shape[0]
        regions = guess_regions(w, h)
        if do_m:
            print("grilla mochila", len(ambos))
            out["mochila"] = merge_row_counts(ambos, tpls, "mochila", regions["mochila"])
        if do_b:
            print("grilla bodega", len(ambos))
            out["bodega"] = merge_row_counts(ambos, tpls, "bodega", regions["bodega"])
    else:
        print("sin foto de baul y bolsa: no se cuentan joyas")

    return out


def print_table(counts_map):
    rows = []
    keys = set()
    for d in counts_map.values():
        if isinstance(d, dict):
            keys.update(d.keys())
    keys = sorted(keys)
    for k in keys:
        row = {"item": k}
        for name, d in counts_map.items():
            if isinstance(d, dict):
                row[name] = d.get(k, 0)
        rows.append(row)
        print(k, row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imagen", nargs="?")
    ap.add_argument("--region", choices=["mochila", "bodega", "auto"], default="auto")
    ap.add_argument("--iconos", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--folder", default=None)
    ap.add_argument("--max-fotos", dest="max_fotos", type=int, default=2)
    args = ap.parse_args()

    icon_dir = args.iconos or find_icon_dir()
    tpls = load_templates(icon_dir)

    if args.folder:
        counts_map = scan_folder(args.folder, tpls, args.region, getattr(args, "max_fotos", 2))
        print_table(counts_map)
        if args.json:
            payload = {name: _json_num(d) if isinstance(d, dict) else _json_num(d) for name, d in counts_map.items()}
            blob = json.dumps(payload, ensure_ascii=False, default=str)
            print("JSON_START")
            print(blob)
            print("JSON_END")
            side = os.path.join(script_dir(), "last_py.json")
            try:
                with open(side, "w", encoding="utf-8") as fh:
                    fh.write(blob)
                print("json_file:", side)
            except Exception as e:
                print("json_file fallo", e)
        return

    if args.imagen:
        img = load_rgb(args.imagen)
        print("imagen:", args.imagen, img.shape[1], "x", img.shape[0])
        if es_baul_y_bolsa(img):
            print("baul+bolsa")
            regions = guess_regions(img.shape[1], img.shape[0])
            moch = merge_row_counts([(args.imagen, img)], tpls, "mochila", regions["mochila"]) if args.region in ("auto", "mochila") else {}
            bod = merge_row_counts([(args.imagen, img)], tpls, "bodega", regions["bodega"]) if args.region in ("auto", "bodega") else {}
            hud = {}
        elif hud_panel_visible(img):
            print("solo bolsa")
            hud = read_hud_best(img)
            moch, bod = {}, {}
        else:
            print("sin cuadro y sin baul")
            hud, moch, bod = {}, {}, {}
        print(hud)
        if args.json:
            cur = {}
            if "gold" in hud:
                cur["gold"] = hud["gold"]
                cur["oro"] = hud["gold"]
            if "muc" in hud:
                cur["muc"] = hud["muc"]
            if "boundmuc" in hud:
                cur["boundmuc"] = hud["boundmuc"]
                cur["bound_muc"] = hud["boundmuc"]
            if "diamantes" in hud:
                cur["diamantes"] = hud["diamantes"]
                cur["diamond"] = hud["diamantes"]
            payload = {"currencies": _json_num(cur), "mochila": _json_num(moch), "bodega": _json_num(bod)}
            print("JSON_START")
            print(json.dumps(payload, ensure_ascii=False))
            print("JSON_END")


if __name__ == "__main__":
    main()
