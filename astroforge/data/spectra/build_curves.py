"""Builds curves.csv: the built-in spectral library of AstroForge (run: python build_curves.py).

ALL CURVES ARE APPROXIMATIONS, NOT MANUFACTURER DATA. They are smooth piecewise curves drawn through
publicly documented band edges / peaks (manufacturer product pages, published spectral plots, sensor
datasheet plots as reproduced by camera makers). They are good enough for SPCC, whose result depends on
the broad shape of the channel responses, not on percent-level details. Users who own measured curves
can drop them as CSV in ~/.astroforge/spectra (see astroforge/spectra.py) to override these.

Grid: 330-1100 nm, 1 nm. Every column is normalised to a maximum of 1 (filters keep their peak
transmission, e.g. 0.95). Column naming:
  sensor:<name>:R|G|B   full channel response of a colour sensor (CFA dye x silicon QE, like the
                        PixInsight "Sony Color Sensor R/G/B" curves, to be used with an ideal QE)
  qe:<name>             quantum efficiency shape of a sensor (relative, peak 1)
  filter:<name>         transmission of a filter

Sources of the band edges (approximate, from public product information):
  Sony colour CFA (IMX071 / IMX294 class): B peak ~455 nm, G ~530 nm, R ~600-620 nm, all dyes
      transparent above ~800 nm (NIR leak), front-illuminated silicon QE peak ~530 nm.
  Panasonic MN34230 colour (QHY163C / ASI1600MC): similar Bayer dyes, R edge ~585 nm, QE peak ~
      520-550 nm and weaker NIR than the Sony sensor (QHY / ZWO published QE plots).
  MN34230 mono (QHY163M / ASI1600MM): peak QE ~60% at ~530 nm, ~38% at Ha, ~10% at 900 nm.
  Sony BSI mono (IMX455/571 class): flatter, peak ~90% at 500-550 nm, ~80% at Ha.
  UV/IR cut: 400-700 nm (Astronomik L-2 / generic UV-IR cut).
  Optolong L-Pro: passes ~410-540 nm and ~560-580 nm and ~620-700 nm, blocks the Hg 546 nm and the
      Na 589 nm / 598 nm / 616 nm region (Optolong product page curve).
  Optolong L-eNhance: dual/tri band, Hb+OIII band ~480-510 nm and Ha band ~650-662 nm.
  Optolong L-eXtreme: 7 nm at OIII 500.7 nm and Ha 656.3 nm.
  Baader LRGB (CMOS optimised): L 420-685, B 400-510, G 500-575, R 595-690 nm.
  Astronomik Deep-Sky / Typ 2c LRGB: L 400-700, B 410-520, G 490-580, R 590-695 nm.
  Narrowband 7 nm: Ha 656.3, OIII 500.7, SII 672.0 nm (generic, any brand).
"""
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import expit

WL = np.arange(330.0, 1101.0, 1.0)


def knots(pairs):
    x, y = np.array(pairs, float).T
    return np.clip(PchipInterpolator(x, y, extrapolate=True)(np.clip(WL, x[0], x[-1])), 0, None)


def band(lo, hi, peak=0.95, edge=2.0):
    """Smooth top-hat with half-maximum points at lo and hi (nm)."""
    a, b = expit((WL - lo) / edge), expit((hi - WL) / edge)
    return peak * a * b


def nb(center, fwhm, peak=0.92):
    return band(center - fwhm / 2, center + fwhm / 2, peak, edge=max(0.25, fwhm / 12))


def norm(c):
    return c / c.max()


# --- silicon QE shapes (relative)
QE = {
    "sony_fsi": knots([(330, .18), (360, .3), (400, .58), (450, .82), (500, .96), (530, 1.0), (560, .98),
                       (600, .9), (650, .78), (700, .66), (800, .44), (900, .22), (1000, .07), (1100, .0)]),
    "mn34230_mono": knots([(330, .2), (360, .32), (400, .62), (450, .85), (500, .97), (530, 1.0), (560, .97),
                           (600, .86), (656, .66), (700, .55), (800, .33), (900, .16), (1000, .05), (1100, .0)]),
    "imx_mono": knots([(330, .38), (360, .55), (400, .8), (450, .92), (500, .98), (530, 1.0), (600, .97),
                       (656, .9), (700, .82), (800, .6), (900, .35), (1000, .12), (1100, .01)]),
}

# --- Bayer dye transmissions (relative); NIR: dyes become transparent
SONY_DYE = {
    "B": knots([(330, .1), (360, .35), (400, .78), (430, .93), (455, 1.0), (475, .93), (490, .72), (510, .4),
                (530, .17), (550, .07), (580, .03), (650, .02), (720, .03), (760, .1), (800, .45), (850, .85),
                (900, .95), (1100, .95)]),
    "G": knots([(330, .03), (380, .05), (420, .1), (460, .26), (490, .62), (510, .9), (530, 1.0), (550, .95),
                (570, .76), (590, .43), (610, .2), (640, .08), (680, .05), (720, .09), (760, .35), (800, .75),
                (850, .93), (1100, .95)]),
    "R": knots([(330, .03), (450, .02), (530, .02), (560, .08), (580, .45), (600, .9), (620, 1.0), (660, .98),
                (700, .96), (800, .95), (1100, .95)]),
}
PANA_DYE = {
    "B": knots([(330, .08), (360, .3), (400, .75), (435, .95), (460, 1.0), (480, .9), (495, .68), (515, .35),
                (535, .14), (560, .05), (600, .03), (700, .03), (760, .12), (820, .55), (880, .9), (1100, .95)]),
    "G": knots([(330, .03), (400, .06), (450, .18), (480, .45), (505, .85), (535, 1.0), (560, .9), (580, .62),
                (600, .3), (625, .12), (660, .06), (720, .1), (770, .45), (830, .85), (1100, .95)]),
    "R": knots([(330, .03), (500, .02), (555, .04), (575, .25), (590, .7), (610, .97), (630, 1.0), (700, .97),
                (800, .95), (1100, .95)]),
}

SENSORS = {
    "sony_color": {c: norm(SONY_DYE[c] * QE["sony_fsi"]) for c in "RGB"},
    "panasonic_color": {c: norm(PANA_DYE[c] * QE["mn34230_mono"]) for c in "RGB"},
}

FILTERS = {
    "uv_ir_cut": band(400, 700, 0.97, 3.0),
    "l_pro": np.maximum.reduce([band(410, 541, 0.95, 2.0), band(556, 580, 0.88, 1.5),
                                band(622, 700, 0.95, 2.5)]),
    "l_enhance": np.maximum(band(480, 510, 0.9, 1.2), band(650, 662, 0.92, 1.0)),
    "l_extreme": np.maximum(nb(500.7, 7), nb(656.3, 7)),
    "baader_l": band(420, 685, 0.97, 2.5),
    "baader_r": band(595, 690, 0.96, 2.5),
    "baader_g": band(500, 575, 0.95, 2.5),
    "baader_b": band(400, 510, 0.95, 2.5),
    "astronomik_l": band(400, 700, 0.97, 2.5),
    "astronomik_r": band(590, 695, 0.96, 2.5),
    "astronomik_g": band(490, 580, 0.95, 2.5),
    "astronomik_b": band(410, 520, 0.95, 2.5),
    "ha_7nm": nb(656.3, 7),
    "oiii_7nm": nb(500.7, 7),
    "sii_7nm": nb(672.0, 7),
}


def main(out=Path(__file__).with_name("curves.csv")):
    cols = {}
    for s, ch in SENSORS.items():
        for c in "RGB":
            cols[f"sensor:{s}:{c}"] = ch[c]
    for q, v in QE.items():
        cols[f"qe:{q}"] = norm(v)
    for f, v in FILTERS.items():
        cols[f"filter:{f}"] = v
    names = list(cols)
    data = np.column_stack([WL] + [np.clip(cols[n], 0, 1) for n in names])
    header = ("# AstroForge built-in spectral curves - APPROXIMATIONS, not manufacturer data "
              "(see build_curves.py)\nwavelength_nm," + ",".join(names))
    np.savetxt(out, data, delimiter=",", header=header, comments="", fmt=["%.0f"] + ["%.4f"] * len(names))
    print(f"wrote {out} ({len(names)} curves)")


if __name__ == "__main__":
    main()
