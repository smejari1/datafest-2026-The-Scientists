

import argparse
import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})
PALETTE = sns.color_palette("Blues_d", 8)
ACCENT = "#1f6eb5"
WARN   = "#e07b39"
GOOD   = "#3a9e6f"

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "research_output"

# Load only columns needed by downstream analysis to reduce memory pressure.
REQUIRED_COLUMNS = {
    "encounters.csv": [
        "patientdurablekey",
        "encounterkey",
        "isedvisit",
        "ishospitaladmission",
        "isinpatientadmission",
        "isobservation",
        "primarydiagnosiskey",
        "departmentkey",
        "admityear",
    ],
    "social_determinants.csv": [
        "patientdurablekey",
        "domain",
        "answertext",
        "encounterkey",
    ],
    "diagnosis.csv": [
        "diagnosiskey",
        "groupname",
    ],
    "departments.csv": [
        "departmentkey",
        "censustract",
        "departmenttype",
    ],
    "tigercensuscodes.csv": [
        "geoid",
        "centlat",
        "centlon",
    ],
}

def load(data_dir: str, name: str) -> pd.DataFrame:
    requested_path = Path(data_dir) / name
    candidates = [
        requested_path,
        DEFAULT_DATA_DIR / name,
        BASE_DIR / "output" / "execution_strong" / "standardized_datasets" / name,
        BASE_DIR / "output" / "execution" / "standardized_datasets" / name,
    ]

    resolved_path = None
    for candidate in candidates:
        if candidate.exists():
            resolved_path = candidate
            break

    if resolved_path is None:
        raise FileNotFoundError(
            f"Could not find dataset '{name}'. Tried: {', '.join(str(p) for p in candidates)}"
        )

    print(f"  Loading {name} from {resolved_path}...")

    needed = [c.lower() for c in REQUIRED_COLUMNS.get(name.lower(), [])]

    def _read_csv(path: Path, usecols=None, nrows=None) -> pd.DataFrame:
        # With usecols, prefer Python engine first to avoid occasional pandas C-engine index bugs.
        if usecols is not None:
            attempts = [
                {"engine": "python", "on_bad_lines": "skip"},
                {"engine": "c", "low_memory": True},
            ]
        else:
            attempts = [
                {"engine": "c", "low_memory": True},
                {"engine": "python", "on_bad_lines": "skip"},
            ]
        encodings = ["utf-8", "latin1", "cp1252"]

        last_err = None
        for encoding in encodings:
            for opts in attempts:
                try:
                    return pd.read_csv(path, encoding=encoding, usecols=usecols, nrows=nrows, **opts)
                except (pd.errors.ParserError, MemoryError, UnicodeDecodeError, ValueError, OSError, IndexError) as err:
                    last_err = err

        raise last_err if last_err else RuntimeError(f"Failed to read CSV: {path}")

    header_df = _read_csv(resolved_path, usecols=None, nrows=0)
    original_cols = [str(c).strip() for c in header_df.columns]
    norm_to_original = {}
    for col in original_cols:
        norm_to_original.setdefault(col.lower(), col)

    if needed:
        selected_norm = [col for col in needed if col in norm_to_original]
        selected = [norm_to_original[col] for col in selected_norm]
        missing = sorted(set(needed) - set(selected_norm))
        if missing:
            print(f"    Warning: {name} missing expected column(s): {', '.join(missing)}")
        if not selected:
            raise ValueError(f"No required columns found in {name}.")
        df = _read_csv(resolved_path, usecols=selected)
    else:
        df = _read_csv(resolved_path, usecols=None)

    df.columns = df.columns.str.strip().str.lower()
    return df


def save_fig(output_dir: str, subfolder: str, fname: str):
    d = os.path.join(output_dir, subfolder)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, fname)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {path}")


def save_csv(df: pd.DataFrame, output_dir: str, subfolder: str, fname: str):
    d = os.path.join(output_dir, subfolder)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, fname)
    df.to_csv(path, index=False)
    print(f"    Saved → {path}")


def rq1_sdoh_ed_utilization(enc: pd.DataFrame, sdoh: pd.DataFrame, output_dir: str):
    print("\n[RQ1] SDOH burden and ED utilization")
    folder = "rq1_sdoh_ed"

    sdoh_clean = sdoh[["patientdurablekey", "domain", "answertext"]].dropna(subset=["domain"])
    sdoh_clean["domain"] = sdoh_clean["domain"].str.strip()

    domain_pivot = (
        sdoh_clean.groupby(["patientdurablekey", "domain"])
        .size().unstack(fill_value=0)
        .clip(upper=1)
        .add_prefix("screened_")
    )
    domain_pivot.columns = domain_pivot.columns.str.replace(" ", "_").str.lower()

    patient_enc = (
        enc.groupby("patientdurablekey")
        .agg(
            total_encounters=("encounterkey", "count"),
            ed_visits=("isedvisit", lambda x: (x == 1).sum()),
            hospital_admissions=("ishospitaladmission", lambda x: (x == 1).sum()),
            inpatient_admissions=("isinpatientadmission", lambda x: (x == 1).sum()),
        )
        .reset_index()
    )
    patient_enc["any_ed"] = (patient_enc["ed_visits"] > 0).astype(int)

    # Merge
    pt = patient_enc.merge(domain_pivot, on="patientdurablekey", how="inner")
    print(f"  Patients with both encounters + SDOH: {len(pt):,}")

    screened_cols = [c for c in pt.columns if c.startswith("screened_")]
    domain_labels = [c.replace("screened_", "").replace("_", " ").title() for c in screened_cols]

    results = []
    for col, label in zip(screened_cols, domain_labels):
        screened = pt[pt[col] == 1]
        not_screened = pt[pt[col] == 0]
        if len(screened) < 30 or len(not_screened) < 30:
            continue
        rate_s  = screened["any_ed"].mean()
        rate_ns = not_screened["any_ed"].mean()
        _, pval  = stats.fisher_exact([
            [screened["any_ed"].sum(), len(screened) - screened["any_ed"].sum()],
            [not_screened["any_ed"].sum(), len(not_screened) - not_screened["any_ed"].sum()],
        ])
        results.append({
            "domain": label,
            "ed_rate_screened":     round(rate_s * 100, 2),
            "ed_rate_not_screened": round(rate_ns * 100, 2),
            "n_screened":           len(screened),
            "n_not_screened":       len(not_screened),
            "p_value":              round(pval, 4),
            "significant":          pval < 0.05,
        })

    rq1_df = pd.DataFrame(results).sort_values("ed_rate_screened", ascending=False)
    save_csv(rq1_df, output_dir, folder, "ed_rate_by_sdoh_domain.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(rq1_df))
    w = 0.38
    bars1 = ax.bar(x - w/2, rq1_df["ed_rate_screened"],     w, label="Screened for domain",     color=ACCENT, alpha=0.9)
    bars2 = ax.bar(x + w/2, rq1_df["ed_rate_not_screened"], w, label="Not screened for domain",  color=PALETTE[2], alpha=0.75)
    # Star for significant
    for i, (_, row) in enumerate(rq1_df.iterrows()):
        if row["significant"]:
            h = max(row["ed_rate_screened"], row["ed_rate_not_screened"]) + 0.3
            ax.text(i, h, "*", ha="center", fontsize=14, color=WARN)
    ax.set_xticks(x)
    ax.set_xticklabels(rq1_df["domain"], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("ED visit rate (%)")
    ax.set_title("RQ1 — ED visit rate: screened vs not screened per SDOH domain\n* = statistically significant (p<0.05)")
    ax.legend()
    plt.tight_layout()
    save_fig(output_dir, folder, "rq1_ed_rate_by_domain.png")

    rq1_df["odds_ratio"] = rq1_df.apply(
        lambda r: round(
            (r["ed_rate_screened"] / 100 / (1 - r["ed_rate_screened"] / 100 + 1e-9))
            / (r["ed_rate_not_screened"] / 100 / (1 - r["ed_rate_not_screened"] / 100 + 1e-9)),
            3,
        ),
        axis=1,
    )
    save_csv(rq1_df, output_dir, folder, "ed_odds_ratios_by_domain.csv")
    print(f"  Top domain by OR: {rq1_df.iloc[0]['domain']} (OR={rq1_df.iloc[0]['odds_ratio']})")


def rq2_high_utilizers(enc: pd.DataFrame, sdoh: pd.DataFrame, diag: pd.DataFrame, output_dir: str):
    print("\n[RQ2] High-utilizer phenotyping")
    folder = "rq2_high_utilizers"

    pt_counts = enc.groupby("patientdurablekey")["encounterkey"].count().reset_index()
    pt_counts.columns = ["patientdurablekey", "encounter_count"]
    THRESHOLD = pt_counts["encounter_count"].quantile(0.90)
    pt_counts["is_high_utilizer"] = (pt_counts["encounter_count"] >= THRESHOLD).astype(int)
    print(f"  High-utilizer threshold (90th pct): {THRESHOLD:.0f} encounters")
    print(f"  High utilizers: {pt_counts['is_high_utilizer'].sum():,}")

    fig, ax = plt.subplots(figsize=(9, 4))
    data = pt_counts["encounter_count"].clip(upper=200)
    ax.hist(data, bins=80, color=ACCENT, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(THRESHOLD, color=WARN, linewidth=1.5, linestyle="--", label=f"90th pct = {THRESHOLD:.0f}")
    ax.set_xlabel("Encounters per patient (capped at 200)")
    ax.set_ylabel("Number of patients")
    ax.set_title("RQ2 — Distribution of encounter counts per patient")
    ax.legend()
    plt.tight_layout()
    save_fig(output_dir, folder, "rq2_encounter_distribution.png")

    sdoh_agg = (
        sdoh.groupby("patientdurablekey")["domain"]
        .apply(lambda x: list(x.dropna().unique()))
        .reset_index()
        .rename(columns={"domain": "sdoh_domains"})
    )
    sdoh_agg["n_sdoh_domains"] = sdoh_agg["sdoh_domains"].apply(len)

    pt_full = pt_counts.merge(sdoh_agg, on="patientdurablekey", how="left")
    pt_full["n_sdoh_domains"] = pt_full["n_sdoh_domains"].fillna(0).astype(int)
    pt_full["any_sdoh"] = (pt_full["n_sdoh_domains"] > 0).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # Mean SDOH domains
    means = pt_full.groupby("is_high_utilizer")["n_sdoh_domains"].mean()
    axes[0].bar(["Regular", "High utilizer"], means.values, color=[PALETTE[2], ACCENT])
    axes[0].set_title("Mean # SDOH domains screened")
    axes[0].set_ylabel("Mean domains")
    # SDOH screening rate
    rates = pt_full.groupby("is_high_utilizer")["any_sdoh"].mean() * 100
    axes[1].bar(["Regular", "High utilizer"], rates.values, color=[PALETTE[2], ACCENT])
    axes[1].set_title("% of patients with any SDOH screening")
    axes[1].set_ylabel("% screened")
    plt.suptitle("RQ2 — SDOH screening: high vs regular utilizers", fontsize=12)
    plt.tight_layout()
    save_fig(output_dir, folder, "rq2_sdoh_by_utilizer_tier.png")


    high_pt_keys = pt_counts[pt_counts["is_high_utilizer"] == 1]["patientdurablekey"]
    high_enc = enc[enc["patientdurablekey"].isin(high_pt_keys)].copy()

    # Remove -1 sentinel
    high_enc_diag = high_enc[high_enc["primarydiagnosiskey"].notna()]
    high_enc_diag = high_enc_diag[high_enc_diag["primarydiagnosiskey"].astype(str) != "-1"]

    if len(high_enc_diag) > 0 and len(diag) > 0:
        diag_clean = diag.copy()
        diag_clean["diagnosiskey"] = pd.to_numeric(diag_clean["diagnosiskey"], errors="coerce")
        high_enc_diag["primarydiagnosiskey"] = pd.to_numeric(high_enc_diag["primarydiagnosiskey"], errors="coerce")
        merged_diag = high_enc_diag.merge(diag_clean, left_on="primarydiagnosiskey", right_on="diagnosiskey", how="left")
        top_diag = (
            merged_diag["groupname"].value_counts().head(15).reset_index()
        )
        top_diag.columns = ["diagnosis_group", "count"]
        save_csv(top_diag, output_dir, folder, "rq2_top_diagnoses_high_utilizers.csv")

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(top_diag["diagnosis_group"][::-1], top_diag["count"][::-1], color=ACCENT, alpha=0.85)
        ax.set_xlabel("Encounter count")
        ax.set_title("RQ2 — Top diagnosis groups among high utilizers")
        plt.tight_layout()
        save_fig(output_dir, folder, "rq2_top_diagnoses.png")

    # Summary CSV
    summary = pt_full.groupby("is_high_utilizer").agg(
        n_patients=("patientdurablekey", "count"),
        mean_encounters=("encounter_count", "mean"),
        mean_sdoh_domains=("n_sdoh_domains", "mean"),
        pct_any_sdoh=("any_sdoh", "mean"),
    ).reset_index()
    summary["is_high_utilizer"] = summary["is_high_utilizer"].map({0: "Regular", 1: "High utilizer"})
    save_csv(summary, output_dir, folder, "rq2_utilizer_summary.csv")


# ════════════════════════════════════════════════════════════════════════════
# RQ3 — Geography x SDOH overlap
# ════════════════════════════════════════════════════════════════════════════
def rq3_geography_sdoh(enc: pd.DataFrame, sdoh: pd.DataFrame, dept: pd.DataFrame,
                        tiger: pd.DataFrame, output_dir: str):
    print("\n[RQ3] Geography × SDOH overlap")
    folder = "rq3_geography"

    # ── Department encounter volume ──────────────────────────────────────────
    dept_clean = dept.copy()
    dept_clean.columns = dept_clean.columns.str.lower()
    # Rename dup columns if present
    if "location__dup1" in dept_clean.columns:
        dept_clean = dept_clean.rename(columns={"location": "city", "location__dup1": "county"})

    enc_dept = enc.merge(dept_clean[["departmentkey", "censustract", "departmenttype"]],
                         on="departmentkey", how="left")

    # SDOH screening flag per encounter
    sdoh_screened = sdoh[["encounterkey"]].drop_duplicates().copy()
    sdoh_screened["has_sdoh"] = 1
    enc_dept = enc_dept.merge(sdoh_screened, on="encounterkey", how="left")
    enc_dept["has_sdoh"] = enc_dept["has_sdoh"].fillna(0).astype(int)

    # ── Census tract level summary ───────────────────────────────────────────
    tract_summary = (
        enc_dept.groupby("censustract")
        .agg(
            total_encounters=("encounterkey", "count"),
            sdoh_screened=("has_sdoh", "sum"),
            ed_visits=("isedvisit", lambda x: (x == 1).sum()),
        )
        .reset_index()
    )
    tract_summary["sdoh_screen_rate"] = (
        tract_summary["sdoh_screened"] / tract_summary["total_encounters"] * 100
    ).round(2)
    tract_summary["ed_rate"] = (
        tract_summary["ed_visits"] / tract_summary["total_encounters"] * 100
    ).round(2)

    # Merge tiger census lat/lon
    tiger_clean = tiger.copy()
    tiger_clean["geoid"] = tiger_clean["geoid"].astype(str).str.strip()
    tract_summary["censustract_str"] = tract_summary["censustract"].astype(str).str.replace(r"\.0$", "", regex=True)
    geo_df = tract_summary.merge(tiger_clean, left_on="censustract_str", right_on="geoid", how="left")

    save_csv(geo_df, output_dir, folder, "rq3_tract_sdoh_geo.csv")

    # ── Scatter: volume vs SDOH screen rate ─────────────────────────────────
    plot_df = geo_df[geo_df["total_encounters"] >= 50].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#f3f6fb")
    sc = ax.scatter(
        plot_df["total_encounters"],
        plot_df["sdoh_screen_rate"],
        c=plot_df["ed_rate"],
        cmap="viridis",
        alpha=0.9,
        edgecolors="#1f2937",
        linewidths=0.5,
        s=75,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("ED visit rate (%)")
    ax.set_xlabel("Total encounters at census tract")
    ax.set_ylabel("SDOH screening rate (%)")
    ax.set_title("RQ3 — Census tract: encounter volume vs SDOH screening rate\n(color = ED visit rate)")
    plt.tight_layout()
    save_fig(output_dir, folder, "rq3_tract_scatter.png")

    # ── Underscreened tracts (high volume, low screen rate) ─────────────────
    median_screen = plot_df["sdoh_screen_rate"].median()
    median_vol = plot_df["total_encounters"].median()
    underscreened = plot_df[
        (plot_df["sdoh_screen_rate"] < median_screen) &
        (plot_df["total_encounters"] > median_vol)
    ].sort_values("total_encounters", ascending=False)
    save_csv(underscreened, output_dir, folder, "rq3_underscreened_tracts.csv")
    print(f"  Underscreened high-volume tracts: {len(underscreened)}")

    # ── Folium map (if folium available) ────────────────────────────────────
    try:
        import folium
        map_df = geo_df.dropna(subset=["centlat", "centlon"]).copy()
        m = folium.Map(location=[map_df["centlat"].mean(), map_df["centlon"].mean()], zoom_start=9)
        for _, row in map_df.iterrows():
            color = "red" if row["sdoh_screen_rate"] < median_screen else "blue"
            folium.CircleMarker(
                location=[row["centlat"], row["centlon"]],
                radius=max(3, min(12, row["total_encounters"] / 500)),
                color=color, fill=True, fill_opacity=0.6,
                popup=(
                    f"Tract: {row['censustract']}<br>"
                    f"Encounters: {int(row['total_encounters'])}<br>"
                    f"SDOH screen rate: {row['sdoh_screen_rate']}%<br>"
                    f"ED rate: {row['ed_rate']}%"
                ),
            ).add_to(m)
        map_path = os.path.join(output_dir, folder, "rq3_tract_map.html")
        m.save(map_path)
        print(f"    Saved interactive map → {map_path}")
    except ImportError:
        print("  (folium not installed — skipping interactive map; run: pip install folium)")


# ════════════════════════════════════════════════════════════════════════════
# RQ4 — Diagnosis group x social needs
# ════════════════════════════════════════════════════════════════════════════
def rq4_diagnosis_sdoh(enc: pd.DataFrame, sdoh: pd.DataFrame, diag: pd.DataFrame, output_dir: str):
    print("\n[RQ4] Diagnosis group × social needs")
    folder = "rq4_diag_sdoh"

    # Remove -1 sentinel from primarydiagnosiskey
    enc_clean = enc.copy()
    enc_clean["primarydiagnosiskey"] = pd.to_numeric(enc_clean["primarydiagnosiskey"], errors="coerce")
    enc_clean = enc_clean[enc_clean["primarydiagnosiskey"].notna() & (enc_clean["primarydiagnosiskey"] != -1)]

    # Merge diagnoses
    diag_clean = diag.copy()
    diag_clean["diagnosiskey"] = pd.to_numeric(diag_clean["diagnosiskey"], errors="coerce")
    enc_diag = enc_clean.merge(diag_clean[["diagnosiskey", "groupname"]], 
                                left_on="primarydiagnosiskey", right_on="diagnosiskey", how="left")
    enc_diag = enc_diag[enc_diag["groupname"].notna()]

    # Merge SDOH domains
    sdoh_enc = sdoh[["encounterkey", "domain"]].dropna(subset=["domain"])
    sdoh_enc["domain"] = sdoh_enc["domain"].str.strip()
    enc_diag_sdoh = enc_diag.merge(sdoh_enc, on="encounterkey", how="inner")

    # Co-occurrence: diagnosis group × SDOH domain
    co_occur = (
        enc_diag_sdoh.groupby(["groupname", "domain"])
        .size()
        .reset_index(name="count")
    )

    # Keep top 15 diagnosis groups by volume
    top_groups = co_occur.groupby("groupname")["count"].sum().nlargest(15).index
    co_occur_top = co_occur[co_occur["groupname"].isin(top_groups)]

    pivot = co_occur_top.pivot_table(index="groupname", columns="domain", values="count", fill_value=0)

    # Normalize rows to get % of encounters with each domain per diagnosis group
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    save_csv(pivot_pct.reset_index(), output_dir, folder, "rq4_diag_sdoh_cooccurrence_pct.csv")

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(
        pivot_pct,
        cmap="Blues",
        linewidths=0.3,
        linecolor="white",
        annot=True,
        fmt=".1f",
        annot_kws={"size": 7},
        ax=ax,
        cbar_kws={"label": "% of linked encounters"},
    )
    ax.set_title("RQ4 — SDOH domain prevalence by diagnosis group (%)", fontsize=12)
    ax.set_xlabel("SDOH domain")
    ax.set_ylabel("Diagnosis group")
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    save_fig(output_dir, folder, "rq4_diag_sdoh_heatmap.png")

    # ── Top social need per diagnosis group ──────────────────────────────────
    top_domain = pivot_pct.idxmax(axis=1).reset_index()
    top_domain.columns = ["diagnosis_group", "top_sdoh_domain"]
    top_domain["top_sdoh_pct"] = pivot_pct.max(axis=1).values.round(2)
    top_domain = top_domain.sort_values("top_sdoh_pct", ascending=False)
    save_csv(top_domain, output_dir, folder, "rq4_top_sdoh_per_diagnosis.csv")
    print(f"  Diagnosis with highest single SDOH concentration: {top_domain.iloc[0]['diagnosis_group']} "
          f"({top_domain.iloc[0]['top_sdoh_domain']}: {top_domain.iloc[0]['top_sdoh_pct']}%)")


# ════════════════════════════════════════════════════════════════════════════
# RQ5 — Temporal trends in SDOH screening
# ════════════════════════════════════════════════════════════════════════════
def rq5_temporal_sdoh(enc: pd.DataFrame, sdoh: pd.DataFrame, output_dir: str):
    print("\n[RQ5] Temporal trends in SDOH screening")
    folder = "rq5_temporal"

    # Merge year into SDOH via encounters
    enc_year = enc[["encounterkey", "admityear", "isedvisit", "ishospitaladmission"]].copy()
    enc_year["admityear"] = pd.to_numeric(enc_year["admityear"], errors="coerce")
    enc_year = enc_year[enc_year["admityear"].between(2020, 2026)]

    sdoh_year = sdoh.merge(enc_year, on="encounterkey", how="left")
    sdoh_year = sdoh_year[sdoh_year["admityear"].notna()]
    sdoh_year["domain"] = sdoh_year["domain"].str.strip()

    # ── Screening volume by year and domain ─────────────────────────────────
    trend = (
        sdoh_year.groupby(["admityear", "domain"])
        .size()
        .reset_index(name="count")
    )

    # Keep top 8 domains
    top8 = trend.groupby("domain")["count"].sum().nlargest(8).index
    trend_top = trend[trend["domain"].isin(top8)]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = sns.color_palette("tab10", len(top8))
    for (domain, grp), color in zip(trend_top.groupby("domain"), colors):
        grp_sorted = grp.sort_values("admityear")
        ax.plot(grp_sorted["admityear"], grp_sorted["count"], marker="o", linewidth=2,
                label=domain, color=color)
    ax.set_xlabel("Admit year")
    ax.set_ylabel("SDOH screening count")
    ax.set_title("RQ5 — SDOH screening volume by domain over time")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    save_fig(output_dir, folder, "rq5_sdoh_trend_by_domain.png")

    # ── Overall SDOH screen rate per year (# encounters screened / total) ───
    enc_sdoh_flag = enc_year.copy()
    screened_keys = sdoh[["encounterkey"]].drop_duplicates()
    screened_keys["screened"] = 1
    enc_sdoh_flag = enc_sdoh_flag.merge(screened_keys, on="encounterkey", how="left")
    enc_sdoh_flag["screened"] = enc_sdoh_flag["screened"].fillna(0)

    rate_by_year = enc_sdoh_flag.groupby("admityear").agg(
        total=("encounterkey", "count"),
        screened=("screened", "sum"),
    ).reset_index()
    rate_by_year["screen_rate_pct"] = (rate_by_year["screened"] / rate_by_year["total"] * 100).round(2)
    save_csv(rate_by_year, output_dir, folder, "rq5_screen_rate_by_year.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(rate_by_year["admityear"].astype(int), rate_by_year["screen_rate_pct"],
           color=ACCENT, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Year")
    ax.set_ylabel("SDOH screen rate (%)")
    ax.set_title("RQ5 — Overall SDOH screening rate per year")
    for _, row in rate_by_year.iterrows():
        ax.text(int(row["admityear"]), row["screen_rate_pct"] + 0.3, f"{row['screen_rate_pct']}%",
                ha="center", fontsize=9)
    plt.tight_layout()
    save_fig(output_dir, folder, "rq5_overall_screen_rate.png")

    # ── Year-over-year growth by domain ─────────────────────────────────────
    yoy = (
        trend_top.sort_values(["domain", "admityear"])
        .groupby("domain")["count"]
        .pct_change() * 100
    ).round(1)
    trend_top = trend_top.copy()
    trend_top["yoy_growth_pct"] = yoy.values
    save_csv(trend_top, output_dir, folder, "rq5_yoy_growth_by_domain.csv")
    print(f"  Screening rate by year:\n{rate_by_year[['admityear','screen_rate_pct']].to_string(index=False)}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="DataFest 2026 — Research Questions Analysis")
    parser.add_argument("--data_dir",   default=str(DEFAULT_DATA_DIR), help="Path to folder containing CSVs")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Path for output files")
    parser.add_argument("--rq", nargs="*", default=["1","2","3","4","5"],
                        help="Which RQs to run, e.g. --rq 1 3 5")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Data dir:   {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Running RQs: {args.rq}\n")

    # ── Load datasets ────────────────────────────────────────────────────────
    print("Loading datasets...")
    enc  = load(args.data_dir, "encounters.csv")
    sdoh = load(args.data_dir, "social_determinants.csv")
    diag = load(args.data_dir, "diagnosis.csv")
    dept = load(args.data_dir, "departments.csv")
    tiger = load(args.data_dir, "tigercensuscodes.csv")

    # Cast key columns
    for col in ["isedvisit", "ishospitaladmission", "isinpatientadmission", "isobservation"]:
        if col in enc.columns:
            enc[col] = pd.to_numeric(enc[col], errors="coerce").fillna(0).astype(int)

    # ── Run selected research questions ─────────────────────────────────────
    if "1" in args.rq:
        rq1_sdoh_ed_utilization(enc, sdoh, args.output_dir)

    if "2" in args.rq:
        rq2_high_utilizers(enc, sdoh, diag, args.output_dir)

    if "3" in args.rq:
        rq3_geography_sdoh(enc, sdoh, dept, tiger, args.output_dir)

    if "4" in args.rq:
        rq4_diagnosis_sdoh(enc, sdoh, diag, args.output_dir)

    if "5" in args.rq:
        rq5_temporal_sdoh(enc, sdoh, args.output_dir)

    print("\n✓ All done. Check output folder:", args.output_dir)


if __name__ == "__main__":
    main()