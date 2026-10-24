"""Rebuild the compact offline catalogues in this folder from the original downloads.

    python -I astroforge/data/build_catalogs.py <download_dir>

<download_dir> must contain:
  NGC.csv, addendum.csv   OpenNGC (CC-BY-SA-4.0), https://github.com/mattiaverga/OpenNGC  database_files/
  sh2.csv                 Sharpless (1959) HII regions, VizieR VII/20: Sh2,GLon,GLat,Diam,Bright
  bsc.csv                 Yale Bright Star Catalogue 5th ed., VizieR V/50: HR,Name,RAJ2000,DEJ2000,Vmag (Vmag<3)
Outputs deepsky.csv.gz and brightstars.csv next to this script.
"""
import csv
import gzip
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent

# IAU (WGSN) proper names for the stars brighter than V=3, keyed by BSC Bayer designation
STAR_NAMES = {
    "Alp Cet": "Menkar", "Eta Tau": "Alcyone", "Bet Ari": "Sheratan", "Alp Ari": "Hamal", "Iot Aur": "Hassaleh",
    "Bet Aur": "Menkalinan", "Eps Aur": "Almaaz", "Alp Aur": "Capella", "Bet Per": "Algol", "Bet And": "Mirach",
    "Gam1And": "Almach", "Alp Cas": "Schedar", "Bet Cas": "Caph", "Alp Per": "Mirfak", "Del Cas": "Ruchbah",
    "Alp UMi": "Polaris", "Gam1Leo": "Algieba", "Gam UMa": "Phecda", "Bet UMa": "Merak", "Alp UMa": "Dubhe",
    "Bet Gem": "Pollux", "Alp Gem": "Castor", "Alp CrB": "Alphecca", "Alp Boo": "Arcturus", "Eta Boo": "Muphrid",
    "Eps Boo": "Izar", "Bet Her": "Kornephoros", "Gam Dra": "Eltanin", "Bet Dra": "Rastaban", "Eta UMa": "Alkaid",
    "Alp2CVn": "Cor Caroli", "Zet UMa": "Mizar", "Eps UMa": "Alioth", "Bet UMi": "Kochab", "Eps Peg": "Enif",
    "Eps Cyg": "Aljanah", "Bet Peg": "Scheat", "Eta Peg": "Matar", "Gam Cyg": "Sadr", "Alp Cyg": "Deneb",
    "Del Cyg": "Fawaris", "Alp Lyr": "Vega", "Alp Cep": "Alderamin", "Bet Cet": "Diphda", "Bet Aqr": "Sadalsuud",
    "Alp Aqr": "Sadalmelik", "Gam Peg": "Algenib", "Alp Peg": "Markab", "Alp And": "Alpheratz", "Alp Col": "Phact",
    "Bet CMa": "Mirzam", "Alp CMa": "Sirius", "Bet Lep": "Nihal", "Alp Lep": "Arneb", "Kap Ori": "Saiph",
    "Alp CMi": "Procyon", "Bet CMi": "Gomeisa", "Bet Ori": "Rigel", "Bet Eri": "Cursa", "Zet Ori": "Alnitak",
    "Eps Ori": "Alnilam", "Del Ori": "Mintaka", "Gam Ori": "Bellatrix", "Alp Tau": "Aldebaran",
    "Alp Ori": "Betelgeuse", "Gam Gem": "Alhena", "Mu  Gem": "Tejat", "Eps Gem": "Mebsuta", "Bet Tau": "Elnath",
    "Bet Crv": "Kraz", "Del Crv": "Algorab", "Gam Crv": "Gienah", "Alp Vir": "Spica", "Eps Vir": "Vindemiatrix",
    "Alp Leo": "Regulus", "Bet Leo": "Denebola", "Del Leo": "Zosma", "Eps Sgr": "Kaus Australis",
    "Del Sgr": "Kaus Media", "Gam2Sgr": "Alnasl", "Sig Sgr": "Nunki", "Lam Sgr": "Kaus Borealis",
    "Pi  Sgr": "Albaldah", "Eta Oph": "Sabik", "Alp Aql": "Altair", "Gam Aql": "Tarazed", "Del Oph": "Yed Prior",
    "Alp Ser": "Unukalhai", "Bet Oph": "Cebalrai", "Alp Oph": "Rasalhague", "Alp Eri": "Achernar",
    "Alp Phe": "Ankaa", "Gam Eri": "Zaurak", "Bet Car": "Miaplacidus", "Iot Car": "Aspidiske", "Eps Car": "Avior",
    "Lam Vel": "Suhail", "Alp Car": "Canopus", "Zet Pup": "Naos", "Eta CMa": "Aludra", "Eps CMa": "Adhara",
    "Del CMa": "Wezen", "Rho Pup": "Tureis", "Alp Hya": "Alphard", "Alp TrA": "Atria", "Bet Cen": "Hadar",
    "Alp1Cen": "Rigil Kentaurus", "Alp2Cen": "Toliman", "The Sco": "Sargas", "Kap Sco": "Girtab",
    "Lam Sco": "Shaula", "Ups Sco": "Lesath", "Alp Sco": "Antares", "Sig Sco": "Alniyat", "Tau Sco": "Paikauhale",
    "Eps Sco": "Larawag", "Alp1Cru": "Acrux", "Bet Cru": "Mimosa", "Gam Cru": "Gacrux", "Del Cru": "Imai",
    "The Cen": "Menkent", "Pi  Sco": "Fang", "Del Sco": "Dschubba", "Bet1Sco": "Acrab",
    "Alp2Lib": "Zubenelgenubi", "Bet Lib": "Zubeneschamali", "Alp Pav": "Peacock", "Alp Gru": "Alnair",
    "Bet Gru": "Tiaki", "Alp PsA": "Fomalhaut", "Zet Sgr": "Ascella", "Del Cap": "Deneb Algedi",
    "Del Vel": "Alsephina", "Kap Vel": "Markeb",
}
GREEK = {"Alp": "α", "Bet": "β", "Gam": "γ", "Del": "δ", "Eps": "ε", "Zet": "ζ", "Eta": "η", "The": "θ",
         "Iot": "ι", "Kap": "κ", "Lam": "λ", "Mu": "μ", "Nu": "ν", "Xi": "ξ", "Omi": "ο", "Pi": "π", "Rho": "ρ",
         "Sig": "σ", "Tau": "τ", "Ups": "υ", "Phi": "φ", "Chi": "χ", "Psi": "ψ", "Ome": "ω"}


def _deg(s, hours):
    s = s.strip()
    if not s:
        return None
    sign = -1 if s.startswith("-") else 1
    p = [float(v) for v in s.lstrip("+-").split(":")]
    v = p[0] + p[1] / 60 + (p[2] / 3600 if len(p) > 2 else 0)
    return sign * v * (15 if hours else 1)


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _pretty(name):
    m = re.match(r"^(NGC|IC|Mel|Cl|ESO|HCG|PGC|UGC|MWSC|B|C|H|M)0*(\d.*)$", name)
    if not m:
        return name
    cat, rest = m.groups()
    rest = re.sub(r"^0+(?=\d)", "", rest)
    if cat == "M":
        return f"M{int(rest)}"
    if cat in ("C", "B", "H"):
        return f"{cat}{rest}"
    return f"{cat} {rest}"


def deepsky(src):
    rows = []
    for fname in ("NGC.csv", "addendum.csv"):
        with open(src / fname, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter=";"):
                if r["Type"] in ("Dup", "NonEx") or not r["RA"]:
                    continue
                name = r["Name"]
                cat = "NGC" if name.startswith("NGC") else "IC" if name.startswith("IC") else "other"
                messier = int(r["M"]) if r["M"].strip() else ""
                idents = [s.strip() for s in r["Identifiers"].split(",") if s.strip()]
                cald = next((int(s[2:]) for s in idents if re.fullmatch(r"C \d{3}", s)), "")
                if cat == "other" and re.fullmatch(r"C\d{3}", name):
                    cald = int(name[1:])
                sh2 = next((s.replace("SH 2-", "Sh2-") for s in idents if s.startswith("SH 2-")), "")
                mag = _f(r["V-Mag"]) if _f(r["V-Mag"]) is not None else _f(r["B-Mag"])
                rows.append([_pretty(name), cat, r["Type"], f"{_deg(r['RA'], True):.5f}",
                             f"{_deg(r['Dec'], False):.5f}", r["MajAx"], r["MinAx"], r["PosAng"],
                             "" if mag is None else f"{mag:.2f}", messier, cald, sh2,
                             "|".join(s.strip() for s in r["Common names"].split(",") if s.strip())])
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    with open(src / "sh2.csv", encoding="utf-8") as f:
        sh = [r for r in csv.DictReader(f)]
    c = SkyCoord(l=[float(r["GLon"]) for r in sh] * u.deg, b=[float(r["GLat"]) for r in sh] * u.deg,
                 frame="galactic").icrs
    for r, ra, dec in zip(sh, c.ra.deg, c.dec.deg):
        rows.append([f"Sh2-{int(r['Sh2'])}", "Sh2", "HII", f"{ra:.5f}", f"{dec:.5f}", r["Diam"], r["Diam"], "",
                     "", "", "", "", ""])
    with gzip.open(OUT / "deepsky.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=9) as f:
        w = csv.writer(f)
        w.writerow(["id", "cat", "type", "ra", "dec", "maj", "min", "pa", "mag", "messier", "caldwell", "sh2",
                    "common"])
        w.writerows(rows)
    return len(rows)


def stars(src):
    seen, out = set(), []
    with open(src / "bsc.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bayer = r["Name"][3:] if len(r["Name"]) > 3 else ""
            if not bayer.strip() or bayer in seen:
                continue
            seen.add(bayer)
            g = bayer[:3].strip()
            label = STAR_NAMES.get(bayer) or f"{GREEK.get(g, g)}{bayer[3].strip()} {bayer[4:].strip()}"
            out.append([label, bayer.replace("  ", " "), f"{float(r['RAJ2000']):.5f}", f"{float(r['DEJ2000']):.5f}",
                        r["Vmag"]])
    with open(OUT / "brightstars.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "bayer", "ra", "dec", "vmag"])
        w.writerows(out)
    return len(out)


if __name__ == "__main__":
    src = Path(sys.argv[1])
    print("deep-sky objects:", deepsky(src))
    print("bright stars:", stars(src))
