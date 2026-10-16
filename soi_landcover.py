"""Rule-based land-cover classification of Survey of India one-inch sheets (1914-15).

classify_map     production classifier: HSV thresholds + morphology
classify_pixels  calibration: thresholds only, per pixel
"""
import itertools

import numpy as np
from scipy.ndimage import binary_closing, binary_opening

# native codes -> harmonised classes
HARMONISE = {0: 6, 1: 1, 2: 3, 3: 2, 4: 4, 5: 7, 255: 255}
HARM_LABELS = {1: "Built-up", 2: "Cropland", 3: "Water", 4: "Natural / vegetated",
               5: "Bare / other", 6: "Unclassified (SoI)", 7: "Grassland",
               255: "No data"}
HARM_COLOURS = {1: (178, 24, 43), 2: (223, 194, 125), 3: (69, 117, 180),
                4: (27, 120, 55), 5: (191, 129, 45), 6: (235, 235, 235),
                7: (166, 219, 160), 255: (255, 255, 255)}
OBSERVED = [1, 2, 3, 4, 5, 6, 7]  # denominator of reported fractions

# hue in degrees, S and V in 0-1; open/close_iter: 3x3 morphology iterations
RULES = dict(
    built=dict(hue_lo=340, hue_hi=25, sat_min=0.16, val_min=0.35,
               open_iter=3, close_iter=2),
    water=dict(hue_lo=180, hue_hi=265, sat_min=0.10, val_min=0.22, val_max=0.88,
               open_iter=1),
    veg=dict(hue_lo=55, hue_hi=145, sat_min=0.11, val_max=0.95, open_iter=2),
    fields=dict(hue_lo=32, hue_hi=55, sat_min=0.125, val_min=0.55, open_iter=3),
    cultivated=dict(hue_lo=32, hue_hi=55, sat_min=0.040, sat_max=0.125,
                    val_min=0.70, open_iter=3),
)


def hsv(rgb):
    """RGB (3, ...) uint8 -> h (degrees), s, v."""
    r, g, b = [c.astype("float32") / 255 for c in rgb]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    v = mx
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    h = np.zeros_like(v)
    m = (mx == r) & (mx > mn); h[m] = (60 * ((g - b) / (mx - mn + 1e-9)) % 360)[m]
    m = (mx == g) & (mx > mn); h[m] = (60 * ((b - r) / (mx - mn + 1e-9)) + 120)[m]
    m = (mx == b) & (mx > mn); h[m] = (60 * ((r - g) / (mx - mn + 1e-9)) + 240)[m]
    return h, s, v


def classify_map(rgb, inside, rules=RULES):
    """Native codes; priority built > water > veg > fields > pale wash > uncoloured."""
    h, s, v = hsv(rgb)
    out = np.zeros(h.shape, "uint8")
    cu = rules["cultivated"]
    out[binary_opening((h > cu["hue_lo"]) & (h < cu["hue_hi"]) &
                       (s > cu["sat_min"]) & (s <= cu["sat_max"]) &
                       (v > cu["val_min"]), iterations=cu["open_iter"])] = 5
    fi = rules["fields"]
    out[binary_opening((h > fi["hue_lo"]) & (h < fi["hue_hi"]) &
                       (s > fi["sat_min"]) & (v > fi["val_min"]),
                       iterations=fi["open_iter"])] = 3
    ve = rules["veg"]
    out[binary_opening((h > ve["hue_lo"]) & (h < ve["hue_hi"]) &
                       (s > ve["sat_min"]) & (v < ve["val_max"]),
                       iterations=ve["open_iter"])] = 4
    wa = rules["water"]
    out[binary_opening((h > wa["hue_lo"]) & (h < wa["hue_hi"]) &
                       (s > wa["sat_min"]) & (v > wa["val_min"]) &
                       (v < wa["val_max"]), iterations=wa["open_iter"])] = 2
    bu = rules["built"]
    out[binary_closing(binary_opening(((h < bu["hue_hi"]) | (h > bu["hue_lo"])) &
                                      (s > bu["sat_min"]) & (v > bu["val_min"]),
                                      iterations=bu["open_iter"]),
                       iterations=bu["close_iter"])] = 1
    out[~inside] = 255
    return out


def harmonise(native):
    out = np.full_like(native, 255)
    for k, v in HARMONISE.items():
        out[native == k] = v
    return out


# --- calibration ---------------------------------------------------------------
NAME2CODE = {"built-up": 1, "water": 2, "fields": 3, "nature/veg": 4,
             "cultivated": 5, "bare/other": 0}
CODES = [1, 2, 3, 5, 4, 0]

V2 = dict(  # rule form searched over
    built=dict(hue_lo=340, hue_hi=25, sat_min=0.20, val_min=0.35),
    water=dict(hue_lo=180, hue_hi=265, sat_min=0.14, val_min=0.22, val_max=0.88),
    veg=dict(hue_lo=55, hue_hi=145, sat_min=0.11, val_max=0.80),
    fields=dict(hue_lo=32, hue_hi=55, sat_min=0.115, val_min=0.55),
    cultivated=dict(hue_lo=32, hue_hi=55, sat_min=0.040, sat_max=0.085, val_min=0.70),
)
GRID_SPLIT = [0.095, 0.105, 0.115, 0.125, 0.135]
GRID_VEGV = [0.80, 0.85, 0.90, 0.95]
GRID_BUILTS = [0.16, 0.18, 0.20, 0.22]
GRID_WATERS = [0.10, 0.12, 0.14, 0.16]
GRID_CULTS = [0.040, 0.055, 0.070, 0.085]


def make_rules(split, vegv, builts, waters, cults):
    R = {k: dict(v) for k, v in V2.items()}
    R["cultivated"]["sat_min"] = cults
    R["cultivated"]["sat_max"] = split
    R["fields"]["sat_min"] = split
    R["veg"]["val_max"] = vegv
    R["built"]["sat_min"] = builts
    R["water"]["sat_min"] = waters
    return R


def all_rule_combinations():
    return [make_rules(*c) for c in itertools.product(
        GRID_SPLIT, GRID_VEGV, GRID_BUILTS, GRID_WATERS, GRID_CULTS)]


def classify_pixels(h, s, v, R):
    out = np.zeros(h.shape, "uint8")
    cu = R["cultivated"]
    out[(h > cu["hue_lo"]) & (h < cu["hue_hi"]) & (s > cu["sat_min"]) &
        (s <= cu["sat_max"]) & (v > cu["val_min"])] = 5
    fi = R["fields"]
    out[(h > fi["hue_lo"]) & (h < fi["hue_hi"]) & (s > fi["sat_min"]) &
        (v > fi["val_min"])] = 3
    ve = R["veg"]
    out[(h > ve["hue_lo"]) & (h < ve["hue_hi"]) & (s > ve["sat_min"]) &
        (v < ve["val_max"])] = 4
    wa = R["water"]
    out[(h > wa["hue_lo"]) & (h < wa["hue_hi"]) & (s > wa["sat_min"]) &
        (v > wa["val_min"]) & (v < wa["val_max"])] = 2
    bu = R["built"]
    out[((h < bu["hue_hi"]) | (h > bu["hue_lo"])) & (s > bu["sat_min"]) &
        (v > bu["val_min"])] = 1
    return out


class Pool:
    """Labelled pixels; macro accuracy over classes in order of first appearance."""

    def __init__(self, h, s, v, truth):
        self.h, self.s, self.v, self.truth = h, s, v, truth
        self.order = []
        for t in truth[np.r_[0, np.flatnonzero(np.diff(truth)) + 1]]:
            if int(t) not in self.order:
                self.order.append(int(t))
        self.total = {c: int((truth == c).sum()) for c in self.order}

    def macro(self, R):
        ok = classify_pixels(self.h, self.s, self.v, R) == self.truth
        return float(np.mean([int(ok[self.truth == c].sum()) / max(self.total[c], 1)
                              for c in self.order]))


def best_rules(pool, combos=None):
    """First rule set with the highest macro accuracy."""
    bm, bR = -1.0, None
    for R in combos or all_rule_combinations():
        m = pool.macro(R)
        if m > bm:
            bm, bR = m, R
    return bR, bm
