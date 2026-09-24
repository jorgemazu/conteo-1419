# MU Immortal - contador HUD + inventario
# 1.4.19
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


HUD_BOXES_1280 = {
    # cuadro de monedas, medido en 1280x720 sobre las fotos del 2026-09-23
    "gold": (568, 630, 674, 652),
    "muc": (778, 630, 862, 652),
    "diamantes": (600, 668, 692, 688),
    "boundmuc": (800, 672, 900, 686),
}
# moneda de oro, a la izquierda del numero. Sirve para saber que es solo bolsa.
COIN_1280 = (488, 622, 552, 658)

_HUD_BANK_TXT = """
0:0ff00ff03c3c3c3c300f300f300f300f300ff003f003f003f003f003f003f003f003300f300f300f3c0c3c0c0f3c0f3c
0:0ff00ff03c3c3c3cf00cf00cf00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00cf00c300c300c3c303c3003c003c0
0:0ff00ff03c0c3c0c300f300f300ff00ff00ff003f003f003f003f003f003f00ff00f300f300f300f3c0c3c0c0ff00ff0
0:0ff00ff03c3c3c3c300c300cf00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00cf00c300c300c3c303c3003c003c0
1:ffffffff07ff07ff001f001f001f001f001f001f001f001f001f001f001f07ff07ff07ff07ff07ff07ff07ff001f001f
1:ffffffff07ff07ff001f001f001f001f001f001f001f001f001f001f001f001f001f001f001f001f001f001f001f001f
1:ffffffff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff
1:ffffffff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff07ff
2:07fc07fc3e1f3e1ff803f803f803f803001f001f001c001c0060006001e001e007800780060006003e003e003fff3fff
2:3ffc3ffcf81cf81cc01fc01f001c001c007c007c0060006001e001e0078007803e003e003e003e00fffcfffc3ffc3ffc
2:1ff81ff8f81ff81fe007e007e007e007e007001f001f0018001800f800f807e007e00700070007001f001f00ff00ff00
2:07fc07fc381f381ff803f80300030003001f001f007c007c0060006001e001e0078007803e003e003fff3fff07ff07ff
3:3ffc3ffc381c381c001f001f001f001c001c007c007c007c007c001f001f00030003000300030003f81ff81f3ffc3ffc
3:3fe03fe0f87cf87c001c001c001c007c007c01e001e001e001e0007c007c001f001f001f001f001fc07cc07cffe0ffe0
3:07fc07fc381c381c381f381f001f001f001c001c01fc01fc001f001f0003000300030003f81ff81f3e7f3e7f07e007e0
3:07fc07fc381c381c381f381f001f001f001c001c01fc01fc001f001f0003000300030003f803f8033e7f3e7f07e007e0
4:007c007c01fc01fc019c019c019c061c061c061c061c381c381cf81ff81fffffffff007f007f007f001c001c001c001c
4:007f007f007f007f01ff01ff01ff079f079f061f061f3e1f3e1f381f381fffffffff007f007f007f001f001f001f001f
5:3ffc3ffc380038003800380038003e003e003fe03fe0007c007c001f001f001f001f001f001f001fc07cc07cffe0ffe0
5:1fff1fff18001800180018001800f800f800fff8fff8001f001f0007000700070007000700070007001f001ffff8fff8
5:3ffc3ffc3e003e00380038003800380038003fe03fe039fc39fc001c001c001f001f001f001f001f001c001cf9fcf9fc
6:0060006001e001e00780078007803f803f803ffc3ffc3e1f3e1ff803f803f803f803f803f803f803381f381f07fc07fc
6:01e001e0078007803e003e003e003e003e00ffe0ffe0f87cf87cc01fc01fc01fc01fc01fc01fc01ff81cf81c3fe03fe0
6:007c007c01e001e0078007803e003e003ffc3ffc3fff3ffff803f803f803f803f803f803380338033e7f3e7f01e001e0
6:01e001e0078007803e003e003e003e003e00ffe0ffe0f87cf87cc01fc01fc01fc01fc01fc01fc01ff81cf81c3ffc3ffc
7:ffffffff001f001f00180018001800f800f800e000e000e000e007e007e0070007001f001f001f0018001800f800f800
7:ffffffff001f001f001c001c001c001c007c007c0060006001e001e00180018007800780060006003e003e0038003800
7:ffffffff00070007001f001f001f0018001800f800f800e000e007e007e0070007001f001f001f001800180018001800
7:ffffffff001f001f00180018001800f800f800e000e000e000e007000700070007001f001f001f0018001800f800f800
8:07fc07fc3e1f3e1f3803380338033e1c3e1c07fc07fc3ffc3ffc38033803f803f803f803f803f803381f381f07fc07fc
8:3fe03fe0f81cf81cf81cf81cf81c387c387c3fe03fe03ffc3ffcf81cf81cc01fc01fc01fc01fc01ff81cf81c3ffc3ffc
8:07fc07fc3e1f3e1f3803380338033e1f3e1f07fc07fc3ffc3ffc38033803f803f803f803f803f803381f381f07fc07fc
9:07fc07fc381f381ff803f803f803f803f803f803f8033e1f3e1f07ff07ff007c007c007c007c007c01e001e001800180
9:07fc07fc3e1f3e1ff803f803f803f803f803f803f8033e1f3e1f07ff07ff007c007c007c007c007c01e001e001800180
"""


def _scale_box(box, w, h):
    x0, y0, x1, y1 = box
    return (
        int(round(x0 * w / 1280.0)),
        int(round(y0 * h / 720.0)),
        int(round(x1 * w / 1280.0)),
        int(round(y1 * h / 720.0)),
    )


def hud_boxes(img):
    h, w = img.shape[:2]
    return {k: _scale_box(b, w, h) for k, b in HUD_BOXES_1280.items()}


def _hud_bank():
    global _HUD_BANK
    if _HUD_BANK is not None:
        return _HUD_BANK
    bank = []
    for line in _HUD_BANK_TXT.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        ch, hx = line.split(":", 1)
        bits = bin(int(hx, 16))[2:].zfill(16 * 24)
        bank.append((ch, np.array([1 if b == "1" else 0 for b in bits], dtype=np.uint8)))
    _HUD_BANK = bank
    return bank


_HUD_BANK = None


def _shift_box(box, dx, dy, w, h):
    x0, y0, x1, y1 = box
    x0, x1 = x0 + dx, x1 + dx
    y0, y1 = y0 + dy, y1 + dy
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return (x0, y0, x1, y1)


def _glyphs_in_box(img, box, thr=125):
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    g = np.array(Image.fromarray(crop).convert("L"))
    g = np.array(Image.fromarray(g).resize((g.shape[1] * 3, g.shape[0] * 3), Image.NEAREST))
    ink = g > thr
    cols = ink.any(axis=0)
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
        sl = ink[:, i:j]
        rows = sl.any(axis=1)
        if rows.any():
            r0 = int(np.argmax(rows))
            r1 = int(len(rows) - np.argmax(rows[::-1]))
            if (r1 - r0) >= 16 and (j - i) >= 8:
                out.append(sl[r0:r1])
        i = j + 1
    return out


def _norm_glyph(g):
    im = Image.fromarray((g.astype(np.uint8) * 255)).resize((16, 24), Image.NEAREST)
    return (np.array(im) > 80).astype(np.uint8).ravel()


def _match_hud_glyph(g):
    v = _norm_glyph(g).astype(np.int16)
    best, bch = 9.0, None
    for ch, tpl in _hud_bank():
        sc = float(np.abs(v - tpl.astype(np.int16)).mean())
        if sc < best:
            best, bch = sc, ch
    if bch is None or best > 0.28:
        return None
    return bch


def _read_box_digits(img, box):
    glyphs = _glyphs_in_box(img, box)
    if not glyphs:
        return None
    chars = []
    for g in glyphs:
        ch = _match_hud_glyph(g)
        if not ch:
            return None
        chars.append(ch)
    return "".join(chars)


def _as_int(digits):
    if not digits or not digits.isdigit():
        return None
    try:
        n = int(digits)
    except Exception:
        return None
    if n < 0 or n > 9_999_999_999:
        return None
    return n


def _tess_digits(pil_img):
    if pil_img is None or not HAS_TESS:
        return None
    try:
        t = pytesseract.image_to_string(
            pil_img,
            config="--psm 7 -c tessedit_char_whitelist=0123456789",
        )
        digits = re.sub(r"\D", "", t or "")
        return digits or None
    except Exception:
        return None


def hud_panel_visible(img):
    """La moneda de oro del cuadro esta en pantalla: es la bolsa, no el almacen."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _scale_box(COIN_1280, w, h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return False
    coin = img[y0:y1, x0:x1].astype(np.int16)
    r, g, b = coin[:, :, 0], coin[:, :, 1], coin[:, :, 2]
    yel = ((r > 140) & (g > 110) & (b < 120) & (r > b + 30)).mean()
    return float(yel) > 0.06


def read_hud(img, shift=(0, 0)):
    h, w = img.shape[:2]
    dx, dy = shift
    boxes = hud_boxes(img)
    out = {}
    for key in ("gold", "muc", "diamantes", "boundmuc"):
        box = _shift_box(boxes[key], dx, dy, w, h)
        if box is None:
            continue
        digits = _read_box_digits(img, box)
        val = _as_int(digits)
        if val is None and HAS_TESS:
            # mismo recorte, por si el glifo no esta en el banco
            crop = img[box[1]:box[3], box[0]:box[2]]
            pil = Image.fromarray(crop).resize((crop.shape[1] * 3, crop.shape[0] * 3), Image.NEAREST)
            gray = np.array(pil.convert("L"))
            bw = np.where(gray > 125, 255, 0).astype(np.uint8)
            raw = _tess_digits(Image.fromarray(bw))
            val = _as_int(raw)
            digits = raw
        if val is not None:
            out[key] = val
            print("HUD", key, val, "raw", digits, "box", box)
        else:
            print("HUD", key, "vacio raw", digits, "box", box)
    return out


def _hud_score(hud):
    if not hud or "gold" not in hud or "muc" not in hud:
        return -1
    return 100 + (1 if "diamantes" in hud else 0) + (1 if "boundmuc" in hud else 0)


def read_hud_best(img):
    """Primero el recorte medido. Si el oro o el MUC fallan, prueba un desplazamiento chico."""
    base = read_hud(img, (0, 0))
    if _hud_score(base) >= 100:
        return base
    h, w = img.shape[:2]
    best, bsc = base, _hud_score(base)
    for dy in (-8, -4, 4, 8):
        for dx in (-8, -4, 4, 8):
            hud = read_hud(img, (dx, dy))
            sc = _hud_score(hud)
            if sc > bsc:
                best, bsc = hud, sc
    return best


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
