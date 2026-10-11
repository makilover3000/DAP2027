"""Builds, executes and exports the PIREP EDA notebook."""
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter

PROJ = Path(sys.argv[1])
OUT = PROJ / "eda"
(OUT / "figures").mkdir(parents=True, exist_ok=True)

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Preliminary EDA — IEM PIREP Archive, September 2026
**Project:** Predicting Aviation Turbulence Along Flight Routes Using Ensemble Machine Learning

**Purpose:** check whether pilot reports (PIREPs) can supply turbulence labels for a binary
*moderate-or-greater (MOG)* classifier. **No model is trained here.**

**Source:** Iowa Environmental Mesonet PIREP download, 2026-09-01 00:00 → 2026-09-30 23:59 UTC.
""")

code(r"""
import csv, re, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 120)

DATA = Path("../pireps_202609010000_202609302359.csv")
FIG = Path("figures")

# palette (validated reference palette): blue = primary / non-MOG, orange = MOG
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEV_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#eb6834", "#b8431a"]  # none..severe

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlecolor": INK, "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
})

def tidy(ax, grid_axis="y"):
    for axis in (ax.yaxis,) if grid_axis != "x" else ():
        if axis.get_scale() == "linear" and not ax.get_ylabel().startswith(("Latitude", "MOG")):
            axis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
""")

md("## 1. Load, shape, columns and data types")
md("""
The raw file contains free-text reports with **unbalanced double-quote characters** (e.g. `/TB "PRETTY SMOOTH`).
A default `pd.read_csv` fails at line 60,085, so the file is read with `quoting=csv.QUOTE_NONE`
(every line has exactly 11 comma-separated fields, so this is safe).
""")
code(r"""
raw = pd.read_csv(DATA, dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
print("shape:", raw.shape)
stray_quotes = int(raw["REPORT"].str.contains('"').sum())
print("reports containing stray quote characters:", stray_quotes)

df = raw.replace({"None": np.nan, "": np.nan})
df["VALID"] = pd.to_datetime(df["VALID"], format="%Y%m%d%H%M", errors="coerce")
for c in ["FL", "LAT", "LON"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["URGENT"] = df["URGENT"].eq("T")
df.dtypes.to_frame("dtype after parsing")
""")
code(r"""
df.head(5)
""")

md("## 2. Missing values")
code(r"""
miss = pd.DataFrame({"missing": df.isna().sum(), "missing_%": (df.isna().mean() * 100).round(1)})
miss
""")
md("""
`ICING` and `TURBULENCE` are *structurally* missing: they are only filled when the pilot included an `/IC` or `/TB`
group. Missing `TURBULENCE` therefore means **"not reported"**, not "no turbulence".
""")

md("## 3. The TURB field")
code(r"""
tb = df["TURBULENCE"].dropna().str.upper().str.strip()
print("reports with a /TB group:", len(tb), f"({len(tb)/len(df):.1%} of all PIREPs)")
print("distinct TURB strings:", tb.nunique())
tb.value_counts().head(25).to_frame("count")
""")

md("""
## 4. Severity categories
Rules (applied to the upper-cased `TURBULENCE` text; the **maximum** intensity mentioned is kept,
e.g. `CONS LGT OCNL MOD CHOP` → Moderate):

| Category | Matched tokens |
|---|---|
| None | NEG, NIL, NONE, NEGATIVE, SMOOTH / SMTH |
| Light | LGT, LT, LIGHT, LGHT, LITE, `-CHP` / `-TB` |
| Light–Moderate | LGT-MOD, LGT MOD, LT-MOD, LGT TO MOD (range reports) |
| Moderate | MOD, MDT, MODERATE |
| Severe | SEV, SVR, SEVERE, MOD-SEV, EXTRM / EXTREME |
| Unparseable | anything else (e.g. `N`, `+`, `CHOP`, `LLWS`, typos) |
""")
code(r"""
LEVELS = ["None", "Light", "Light-Moderate", "Moderate", "Severe"]
R_LM   = re.compile(r"\b(LGT|LT|LIGHT|LGHT|LITE)\s*(-|TO|/|\s)\s*(MOD|MDT|MODERATE)\b")
R_SEV  = re.compile(r"\b(SEV|SVR|SEVERE|EXTRM|EXTREME|EXTR)\b")
R_MOD  = re.compile(r"\b(MOD|MDT|MODERATE|MODT)\b")
R_LGT  = re.compile(r"\b(LGT|LT|LIGHT|LGHT|LITE|LIT)\b|(^|\s)-\s*(CHP|CHOP|TB|TURB)")
R_NONE = re.compile(r"\b(NEG|NIL|NONE|NEGATIVE|SMTH|SMT)\b|\bSMO+TH")

def severity(s):
    if not isinstance(s, str):
        return np.nan
    s = s.upper()
    found = []
    if R_LM.search(s):
        found.append(2)
        s = R_LM.sub(" ", s)
    if R_SEV.search(s):  found.append(4)
    if R_MOD.search(s):  found.append(3)
    if R_LGT.search(s):  found.append(1)
    if R_NONE.search(s): found.append(0)
    return LEVELS[max(found)] if found else "Unparseable"

df["SEV_CAT"] = df["TURBULENCE"].map(severity)
df["SEV_CAT"] = pd.Categorical(df["SEV_CAT"], categories=LEVELS + ["Unparseable"], ordered=True)

T = df["TURBULENCE"].fillna("").str.upper()
df["IS_CAT"]  = T.str.contains(r"\bCAT\b")
df["IS_CHOP"] = T.str.contains(r"CHOP|CHP")

sev_counts = df["SEV_CAT"].value_counts(sort=False)
sev_tab = pd.DataFrame({"count": sev_counts, "% of TB reports": (sev_counts / sev_counts.sum() * 100).round(1)})
print("check: categorised =", sev_counts.sum(), "| TB reports =", len(tb))
sev_tab
""")
code(r"""
# spot check: 20 random strings and their assigned category
df.loc[df["TURBULENCE"].notna(), ["TURBULENCE", "SEV_CAT"]].sample(20, random_state=7)
""")
code(r"""
usable = int(df["SEV_CAT"].isin(LEVELS).sum())
print(f"usable turbulence observations (intensity parsed): {usable:,} = {usable/len(df):.1%} of all PIREPs, "
      f"{usable/len(tb):.1%} of TB reports")
print("unparseable TB strings:", int((df['SEV_CAT'] == 'Unparseable').sum()))
print("TB reports mentioning CAT (clear-air):", int(df['IS_CAT'].sum()), "| mentioning CHOP:", int(df['IS_CHOP'].sum()))
""")
code(r"""
fig, ax = plt.subplots(figsize=(7.2, 3.6))
vals = sev_counts[LEVELS]
bars = ax.bar(LEVELS, vals, color=SEV_COLORS, width=0.62, edgecolor="white", linewidth=2)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + vals.max()*0.015, f"{v:,}\n({v/vals.sum():.1%})",
            ha="center", va="bottom", fontsize=9, color=INK)
ax.set_ylim(0, vals.max() * 1.25)
ax.set_ylabel("Number of PIREPs")
ax.set_title(f"Fig 1. Reported turbulence intensity (n = {vals.sum():,} parsed PIREPs, Sep 2026)", loc="left")
tidy(ax)
fig.savefig(FIG / "fig1_severity_distribution.png")
plt.show()
""")

md("## 5. Binary target feasibility: 0 = none/light, 1 = moderate-or-greater (MOG)")
code(r"""
lvl = df["SEV_CAT"].astype(object)
neg_or_light = lvl.isin(["None", "Light"])
mog = lvl.isin(["Moderate", "Severe"])
lm  = lvl.eq("Light-Moderate")

conv = {
    "A: LGT-MOD excluded (recommended)": (int(neg_or_light.sum()), int(mog.sum())),
    "B: LGT-MOD counted as 0":            (int((neg_or_light | lm).sum()), int(mog.sum())),
    "C: LGT-MOD counted as 1":            (int(neg_or_light.sum()), int((mog | lm).sum())),
}
bin_tab = pd.DataFrame(conv, index=["class 0", "class 1 (MOG)"]).T
bin_tab["n"] = bin_tab.sum(axis=1)
bin_tab["MOG %"] = (bin_tab["class 1 (MOG)"] / bin_tab["n"] * 100).round(1)
bin_tab["imbalance ratio (0:1)"] = (bin_tab["class 0"] / bin_tab["class 1 (MOG)"]).round(1)
bin_tab
""")
code(r"""
fig, ax = plt.subplots(figsize=(7.6, 2.6))
labels = list(conv.keys())[::-1]
for i, k in enumerate(labels):
    c0, c1 = conv[k]; n = c0 + c1
    ax.barh(i, c0/n, color=BLUE, height=0.55, edgecolor="white", linewidth=2)
    ax.barh(i, c1/n, left=c0/n, color=ORANGE, height=0.55, edgecolor="white", linewidth=2)
    ax.text(0.01, i, f"0: {c0:,} ({c0/n:.0%})", va="center", color="white", fontsize=9, fontweight="bold")
    ax.text(0.99, i, f"MOG: {c1:,} ({c1/n:.0%})", va="center", ha="right", color="white", fontsize=9, fontweight="bold")
ax.set_yticks(range(len(labels)), labels)
ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
ax.set_title("Fig 2. Binary target class balance under three LGT-MOD conventions", loc="left")
tidy(ax, "x")
fig.savefig(FIG / "fig2_class_balance.png")
plt.show()
""")
md("""
**Interpretation.** Within PIREPs that contain a turbulence group, MOG is a minority but *not* extremely rare,
so a binary target is technically feasible. However this balance is **inflated by reporting bias**: pilots are much more
likely to file a PIREP when they encounter turbulence. In real flight time MOG is far rarer, so the class ratio
here should not be taken as the true base rate, and any model will need class weighting / PR-AUC evaluation.
""")

md("## 6. Aircraft types")
code(r"""
ac = df["AIRCRAFT"].fillna("UNKN").str.upper().str.strip()
print("distinct aircraft type strings:", ac.nunique())
print("unknown / blank / generic types:", int(ac.isin(["UNKN", "UNK", "MULTI", "ALL TYPES", "NONE"]).sum()))
print("non-ICAO-looking designators (contain '-' or space or len>4):",
      int(ac.str.contains(r"[-\s]").sum() + (ac.str.len() > 4).sum()))

AIRLINE = r"^(A19|A2|A3|B7|B3|B4|B5|B6|B78|BCS|E1|E2|E7|E9|CRJ|DH8|AT4|AT7|MD|A1|C30J)"
df["AC_CLASS"] = np.where(ac.str.match(AIRLINE), "Airline / transport", "GA / business / other")
print(df["AC_CLASS"].value_counts(normalize=True).round(3))

top = ac.value_counts().head(15)
top_mog = (df.assign(AC=ac)[mog | neg_or_light].groupby("AC")
           .apply(lambda g: mog[g.index].mean(), include_groups=False))
pd.DataFrame({"reports": top, "MOG rate among labelled (%)": (top_mog.reindex(top.index) * 100).round(1)})
""")
code(r"""
lab = df[mog | neg_or_light]
print("MOG rate by aircraft class (convention A):")
print((lab.groupby("AC_CLASS").apply(lambda g: mog[g.index].mean(), include_groups=False) * 100).round(1))
""")

md("## 7. Geographic coverage")
code(r"""
lat, lon = df["LAT"], df["LON"]
region = np.select(
    [lat.isna() | lon.isna(),
     lat.between(24, 50) & lon.between(-125, -66),
     (lat >= 51) & ((lon <= -129) | (lon >= 170)),
     lat.between(18, 23) & lon.between(-161, -154)],
    ["Missing", "CONUS", "Alaska", "Hawaii"], default="Other / suspect")
df["REGION"] = region
print(df["REGION"].value_counts())
print("\nlat range:", lat.min(), "→", lat.max(), "| lon range:", lon.min(), "→", lon.max())
print("lon > 0 (east of dateline / suspect):", int((lon > 0).sum()))
print("\nTop ARTCCs:"); print(df["ATRCC"].value_counts().head(10))
""")
code(r"""
from matplotlib.colors import LinearSegmentedColormap, LogNorm
seq = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
fig, ax = plt.subplots(figsize=(8.6, 4.6))
pts = df[lon.lt(-50) & lat.between(10, 75)]
hb = ax.hexbin(pts["LON"], pts["LAT"], gridsize=(110, 40), extent=(-180, -60, 15, 75),
               cmap=seq, norm=LogNorm(), mincnt=1, linewidths=0.2, edgecolors="white")
cb = fig.colorbar(hb, ax=ax, pad=0.01, shrink=0.85); cb.set_label("PIREPs per cell (log scale)", color=INK2)
cb.outline.set_visible(False)
ax.set_xlim(-180, -60); ax.set_ylim(15, 75)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title(f"Fig 3. PIREP density, Sep 2026 (n = {len(pts):,}; CONUS {int((df.REGION=='CONUS').sum()):,}, "
             f"Alaska {int((df.REGION=='Alaska').sum()):,}, Hawaii {int((df.REGION=='Hawaii').sum()):,})", loc="left", fontsize=10)
tidy(ax, "both")
fig.savefig(FIG / "fig3_geographic_coverage.png")
plt.show()
""")

md("## 8. Time / date distribution")
code(r"""
v = df["VALID"]
print("time span:", v.min(), "→", v.max(), "| unparsable timestamps:", int(v.isna().sum()))
per_day = v.dt.date.value_counts().sort_index()
print(f"reports/day: mean {per_day.mean():.0f}, min {per_day.min()} ({per_day.idxmin()}), max {per_day.max()} ({per_day.idxmax()})")
by_hour = v.dt.hour.value_counts().sort_index()
dow = v.dt.day_name().value_counts().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
print("\nby day of week:"); print(dow)
hour_mog = (df[mog | neg_or_light].groupby(v.dt.hour).apply(lambda g: mog[g.index].mean(), include_groups=False) * 100).round(1)
print("\nMOG rate by UTC hour (%):"); print(hour_mog.to_string())
""")
code(r"""
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
a1.bar(pd.to_datetime(per_day.index), per_day.values, color=BLUE, width=0.75, edgecolor="white", linewidth=1)
a1.set_title("Fig 4a. PIREPs per day (Sep 2026)", loc="left"); a1.set_ylabel("Reports")
a1.set_xlim(pd.Timestamp("2026-08-31 12:00"), pd.Timestamp("2026-09-30 12:00"))
a1.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(bymonthday=[1, 8, 15, 22, 29]))
a1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d %b"))
tidy(a1)
a2.bar(by_hour.index, by_hour.values, color=BLUE, width=0.75, edgecolor="white", linewidth=1)
a2.set_title("Fig 4b. PIREPs by hour (UTC)", loc="left"); a2.set_xlabel("UTC hour  (US daytime ≈ 12–03 UTC)")
a2.set_xticks(range(0, 24, 3))
tidy(a2)
fig.tight_layout()
fig.savefig(FIG / "fig4_time_distribution.png")
plt.show()
""")

md("""
## 9. Altitude
`FL` is already given in feet, but is missing for ~10% of rows. Recovery from the `/FL` group of the raw text:
2–3 digit values (e.g. `/FL80`) are converted ×100 ft; `DURD`/`DURC` (during descent/climb) and `UNKN` cannot be recovered.
Values of 0 ft or > 60,000 ft are treated as unreliable.
""")
code(r"""
fl_txt = df["REPORT"].str.extract(r"/FL\s*([^/]*)")[0].str.strip().str.upper()
recov = pd.to_numeric(fl_txt.where(fl_txt.str.fullmatch(r"\d{1,3}", na=False)), errors="coerce") * 100
df["ALT_FT"] = df["FL"].fillna(recov)
df["ALT_SRC"] = np.where(df["FL"].notna(), "FL column", np.where(recov.notna(), "recovered from text", "missing"))
phase = fl_txt.where(df["ALT_FT"].isna())
print(df["ALT_SRC"].value_counts())
print("\nwhy still missing:"); print(phase.replace({"DRD": "DURD", "DURA": "DURD"}).value_counts().head(6))

bad_alt = df["ALT_FT"].eq(0) | df["ALT_FT"].gt(60000)
print("\naltitude = 0 ft:", int(df["ALT_FT"].eq(0).sum()), "| altitude > 60,000 ft:", int(df["ALT_FT"].gt(60000).sum()))
df.loc[bad_alt, "ALT_FT"] = np.nan
print("\nusable altitude:", int(df["ALT_FT"].notna().sum()), f"({df['ALT_FT'].notna().mean():.1%})")
print(df["ALT_FT"].describe().round(0))

bands = pd.cut(df["ALT_FT"], [0, 10000, 20000, 30000, 60000], labels=["<10k ft", "10–20k ft", "20–30k ft", "≥30k ft"])
lab_mask = mog | neg_or_light
alt_tab = pd.DataFrame({
    "all PIREPs": bands.value_counts(sort=False),
    "labelled (conv. A)": bands[lab_mask].value_counts(sort=False),
    "MOG": bands[mog].value_counts(sort=False),
})
alt_tab["MOG rate %"] = (alt_tab["MOG"] / alt_tab["labelled (conv. A)"] * 100).round(1)
alt_tab
""")
code(r"""
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
edges = np.arange(0, 47000, 1000)
alt_l = df.loc[lab_mask, "ALT_FT"]
a1.hist([alt_l[neg_or_light[alt_l.index]].dropna(), alt_l[mog[alt_l.index]].dropna()], bins=edges, stacked=True,
        color=[BLUE, ORANGE], edgecolor="white", linewidth=0.6, label=["None / light", "Moderate-or-greater"])
a1.set_xlabel("Aircraft altitude (ft)"); a1.set_ylabel("Labelled PIREPs")
a1.xaxis.set_major_formatter(lambda x, _: f"{int(x/1000)}k")
a1.set_title("Fig 5a. Altitude of labelled turbulence reports", loc="left")
a1.legend(frameon=False); tidy(a1)
r = alt_tab["MOG rate %"]
b = a2.bar(r.index.astype(str), r.values, color=ORANGE, width=0.6, edgecolor="white", linewidth=2)
for bb, val, n in zip(b, r.values, alt_tab["labelled (conv. A)"]):
    a2.text(bb.get_x()+bb.get_width()/2, val+0.6, f"{val:.0f}%\n(n={n:,})", ha="center", fontsize=8.5, color=INK)
a2.set_ylim(0, r.max()*1.3); a2.set_ylabel("MOG share of labelled (%)")
a2.set_title("Fig 5b. MOG share by altitude band", loc="left"); tidy(a2)
fig.tight_layout()
fig.savefig(FIG / "fig5_altitude.png")
plt.show()
""")

md("## 10. Malformed / unreliable records")
code(r"""
dups = int(raw.drop(columns="PRODUCT_ID").duplicated().sum())
issues = {
    "Stray quote chars break default CSV parsing": stray_quotes,
    "Duplicate report (same content, different PRODUCT_ID)": dups,
    "Missing lat/lon": int((lat.isna() | lon.isna()).sum()),
    "Longitude > 0 (dateline / suspect)": int((lon > 0).sum()),
    "Missing ARTCC": int(df["ATRCC"].isna().sum()),
    "FL missing in IEM column": int(df["FL"].isna().sum()),
    "…of which unrecoverable (DURD/DURC/UNKN etc.)": int((df["ALT_SRC"] == "missing").sum()),
    "Altitude 0 ft or > 60,000 ft": int(bad_alt.sum()),
    "TB group present but intensity unparseable": int((df["SEV_CAT"] == "Unparseable").sum()),
    "Unknown / generic aircraft type": int(ac.isin(["UNKN", "UNK", "MULTI", "ALL TYPES", "NONE"]).sum()),
    "Urgent PIREP (UUA)": int(df["URGENT"].sum()),
}
iss = pd.DataFrame({"records": issues}); iss["% of all"] = (iss["records"] / len(df) * 100).round(2)
iss
""")
code(r"""
# examples of unparseable TB strings
df.loc[df["SEV_CAT"] == "Unparseable", "TURBULENCE"].value_counts().head(15)
""")

md("## 11. Summary of key numbers")
code(r"""
labelled_A = int((mog | neg_or_light).sum())
K = {
    "total_pireps": len(df),
    "tb_reports": len(tb), "tb_pct": round(len(tb)/len(df)*100, 1),
    "usable": usable, "usable_pct": round(usable/len(df)*100, 1),
    "unparseable": int((df["SEV_CAT"] == "Unparseable").sum()),
    **{f"cat_{k}": int(v) for k, v in sev_counts.items()},
    "labelled_A": labelled_A, "mog_A": int(mog.sum()), "mog_pct_A": round(mog.sum()/labelled_A*100, 1),
    "ratio_A": round(neg_or_light.sum()/mog.sum(), 1),
    "lgt_mod": int(lm.sum()), "severe": int(lvl.eq("Severe").sum()),
    "conus": int((df["REGION"] == "CONUS").sum()), "alaska": int((df["REGION"] == "Alaska").sum()),
    "hawaii": int((df["REGION"] == "Hawaii").sum()),
    "alt_usable_pct": round(df["ALT_FT"].notna().mean()*100, 1),
    "alt_recovered": int((df["ALT_SRC"] == "recovered from text").sum()),
    "alt_median": float(df["ALT_FT"].median()),
    "airline_pct": round((df["AC_CLASS"] == "Airline / transport").mean()*100, 1),
    "aircraft_types": int(ac.nunique()),
    "per_day_mean": round(float(per_day.mean())),
    "mog_rate_by_band": alt_tab["MOG rate %"].astype(float).to_dict(),
}
Path("key_numbers.json").write_text(json.dumps(K, indent=2, default=str))
display(Markdown(f'''
* **{K["total_pireps"]:,}** PIREPs (≈{K["per_day_mean"]:,}/day); **{K["tb_reports"]:,} ({K["tb_pct"]}%)** contain a turbulence group,
  **{K["usable"]:,}** with a parseable intensity.
* Intensity mix: None {K["cat_None"]:,} · Light {K["cat_Light"]:,} · Light–Mod {K["cat_Light-Moderate"]:,} ·
  Moderate {K["cat_Moderate"]:,} · Severe {K["cat_Severe"]:,}.
* Binary target (LGT-MOD excluded): **{K["mog_A"]:,} MOG of {K["labelled_A"]:,} ({K["mog_pct_A"]}%)**, ratio ≈ {K["ratio_A"]}:1.
* Coverage: {K["conus"]:,} CONUS, {K["alaska"]:,} Alaska, {K["hawaii"]:,} Hawaii — essentially US-only.
* Usable altitude for {K["alt_usable_pct"]}% of reports ({K["alt_recovered"]:,} recovered from text); median {K["alt_median"]:,.0f} ft.
* {K["airline_pct"]}% of reports from airline/transport types; {K["aircraft_types"]:,} distinct type strings.
'''))
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(OUT)}}).execute()
nbf.write(nb, OUT / "PIREP_EDA_Sep2026.ipynb")
html, _ = HTMLExporter().from_notebook_node(nb)
(OUT / "PIREP_EDA_Sep2026.html").write_text(html, encoding="utf-8")
print("done")
