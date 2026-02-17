#!/usr/bin/env python3
"""
LANDSCAPE EXPANSION ANALYSIS PIPELINE
-------------------------------------
Core analyses included:
    1. Data loading
    2. Baseline correction (3 modes)
    3. EC50 estimation
    4. Px threshold detection
    5. Activation threshold (first dose exceeding CD69 threshold)
    6. Percent responders across affinities
    7. Logistic regression of activation probability vs Kd
    8. Heatmaps for all experiments
    9. Discrimination power (α) fitting
    10. Unified summary report

Landscape Expansion Definition:
    Memory T cells activate at lower ligand affinities (higher Kd)
    or lower peptide concentrations than naïve T cells.

Author: ChatGPT (rewritten for clarity, modularity, scientific completeness)
"""

# ======================================================================
# IMPORTS
# ======================================================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import expit
from scipy import stats

sns.set_style("ticks")


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = "/Users/maddie/Desktop/Projects/Research/Pettmann_data_analy"
OUT = f"{BASE_DIR}/analysis_outputs"
os.makedirs(OUT, exist_ok=True)

KD_FILE = f"{BASE_DIR}/elife-67092-fig1-data2-v3.csv"

MEMORY_FILES = [
    f'{BASE_DIR}/Figure 2 - 1G4 memory and DC/190722d1 memory CD69 Pos.csv',
    f'{BASE_DIR}/Figure 2 - 1G4 memory and DC/190807d1 memory CD69 Pos.csv',
    f'{BASE_DIR}/Figure 2 - 1G4 memory and DC/190812d4 memory CD69 Pos.csv',
    f'{BASE_DIR}/Figure 2 - 1G4 memory and DC/191019d1 memory CD69 Pos.csv',
    f'{BASE_DIR}/Figure 2 - 1G4 memory and DC/191022d1 memory CD69 Pos.csv',
    f'{BASE_DIR}/Figure 2 - 1G4 memory and DC/191027d1 memory CD69 Pos.csv'
]

NAIVE_FILES = [
    f'{BASE_DIR}/Figure 2 - 1G4 naive and DC/190708d3 naive CD69 Pos.csv',
    f'{BASE_DIR}/Figure 2 - 1G4 naive and DC/190715d1 naive CD69 Pos.csv',
    f'{BASE_DIR}/Figure 2 - 1G4 naive and DC/190715d2 naive CD69 Pos.csv',
    f'{BASE_DIR}/Figure 2 - 1G4 naive and DC/190715d4 naive CD69 Pos.csv',
    f'{BASE_DIR}/Figure 2 - 1G4 naive and DC/190807d2 naive CD69 Pos.csv',
    f'{BASE_DIR}/Figure 2 - 1G4 naive and DC/190816d8 naive CD69 Pos.csv'
]

# Threshold for Px
PX_THRESHOLD = 40
ACTIVATION_THRESHOLD = 20  # %CD69+ for activation


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def four_param_logistic(x, bottom, top, logEC50, hill):
    """Standard 4-parameter logistic dose-response."""
    return bottom + (top - bottom) / (1 + 10**((logEC50 - np.log10(x)) * hill))


def safe_log10(x):
    """Avoid errors for non-positive values."""
    return np.log10(np.clip(x, 1e-12, None))


# ======================================================================
# DATA LOADING
# ======================================================================

def load_kd_table(path):
    kd_df = pd.read_csv(path, encoding="utf-8-sig", skiprows=1)
    kd = {}
    for _, row in kd_df.iterrows():
        name = row.iloc[0].replace("NYE ", "")
        kd[name] = row.iloc[6]  # mean Kd
    return kd


def load_experiment_files(files):
    all_expts = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        date = os.path.basename(f)[:6]

        # melt: concentration, peptide, response
        long = df.melt(id_vars=df.columns[0],
                       var_name="peptide",
                       value_name="response")
        long.rename(columns={df.columns[0]: "conc"}, inplace=True)

        all_expts.append({"file": f, "date": date, "raw": long})
    return all_expts


# ======================================================================
# BASELINE CORRECTIONS
# ======================================================================

def apply_baseline_corrections(long_df):
    """Add corrected columns: bg_sub, normalized, fold_baseline."""
    corrected = long_df.copy()

    corrected["baseline"] = (
        corrected.groupby("peptide")["response"]
        .transform(lambda x: x.iloc[-1])
    )

    corrected["bg_sub"] = corrected["response"] - corrected["baseline"]

    # Normalize 0–100%
    corrected["min"] = corrected.groupby("peptide")["response"].transform("min")
    corrected["max"] = corrected.groupby("peptide")["response"].transform("max")
    corrected["normalized"] = 100 * (corrected["response"] - corrected["min"]) / (
        corrected["max"] - corrected["min"]
    ).replace(0, np.nan)

    corrected["fold_baseline"] = (
        corrected["response"] / corrected["baseline"].replace(0, np.nan)
    )

    return corrected.drop(columns=["min", "max"])


# ======================================================================
# EC50 & Px CALCULATIONS
# ======================================================================

def estimate_ec50(conc, resp):
    """Fit 4PL curve → return EC50 or NaN."""
    try:
        p0 = [resp.min(), resp.max(), np.log10(np.median(conc)), 1]
        popt, _ = curve_fit(four_param_logistic, conc, resp, p0=p0, maxfev=10000)
        return 10 ** popt[2]
    except:
        return np.nan


def compute_px(conc, resp, threshold):
    """Compute Px = concentration at which response crosses threshold."""
    valid = ~np.isnan(resp)
    if resp.max() < threshold:
        return np.nan
    f = interp1d(resp[valid], conc[valid], bounds_error=False)
    try:
        return float(f(threshold))
    except:
        return np.nan


def compute_activation_threshold(conc, resp, cutoff=ACTIVATION_THRESHOLD):
    """Return lowest concentration giving ≥ cutoff activation."""
    above = conc[resp >= cutoff]
    return np.min(above) if len(above) else np.nan


# ======================================================================
# LOGISTIC REGRESSION VS AFFINITY (LANDSCAPE EXPANSION METRIC)
# ======================================================================

def logistic_regression_affinity(kd_values, responses):
    """
    P(activated) = logistic(β0 + β1 log(Kd))
    Lower slope (β1) → more affinity-independent activation → expansion.
    """
    log_kd = safe_log10(np.array(kd_values))
    y = np.array(responses) / 100.0

    def model(X, β0, β1):
        return expit(β0 + β1 * X)

    try:
        popt, _ = curve_fit(model, log_kd, y, p0=[0, -1])
        return popt
    except:
        return (np.nan, np.nan)


# ======================================================================
# HEATMAPS
# ======================================================================
AFFINITY_ORDER = ["9V", "6V", "3Y", "4D", "6T", "4A", "5Y", "5F"]
def plot_heatmap(df, title, save_path):
    # Create a copy to avoid SettingWithCopy warnings
    plot_df = df.copy()
    
    # Filter only for peptides in our list to avoid errors if some are missing
    plot_df = plot_df[plot_df["peptide"].isin(AFFINITY_ORDER)]
    
    # Set the categorical order
    plot_df["peptide"] = pd.Categorical(
        plot_df["peptide"], 
        categories=AFFINITY_ORDER, 
        ordered=True
    )
    
    # Pivot and sort by the categorical index
    pivot = plot_df.pivot(index="peptide", columns="conc", values="response")
    pivot = pivot.reindex(AFFINITY_ORDER) 

    plt.figure(figsize=(10, 8))
    # 'annot=True' is optional if you want to see values in the boxes
    sns.heatmap(pivot, cmap="viridis", cbar_kws={'label': '% CD69+'})
    
    plt.title(title)
    plt.xlabel("Concentration (µM)")
    plt.ylabel("Peptide (Sorted by Affinity: High → Low)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ======================================================================
# DISCRIMINATION POWER α
# ======================================================================

def power_law(x, C, alpha):
    return C * (x ** alpha)


def fit_alpha(kd, px):
    valid = ~(np.isnan(kd) | np.isnan(px))
    if valid.sum() < 3:
        return np.nan, np.nan

    try:
        popt, _ = curve_fit(
            lambda k, C, a: power_law(k, C, a),
            kd[valid], px[valid],
            p0=[1e-3, 1.5]
        )
        return popt[0], popt[1]
    except:
        return np.nan, np.nan


# ======================================================================
# MAIN PER-EXPERIMENT ANALYSIS
# ======================================================================

def analyze_experiment(exp, kd_dict):
    df = exp["corrected"]
    results = []

    for peptide, kd in kd_dict.items():
        sub = df[df["peptide"] == peptide]
        if sub.empty:
            continue

        conc = sub["conc"].values.astype(float)
        resp = sub["response"].values.astype(float)

        ec50 = estimate_ec50(conc, resp)
        px = compute_px(conc, resp, PX_THRESHOLD)
        act_thresh = compute_activation_threshold(conc, resp)

        results.append({
            "peptide": peptide,
            "kd": kd,
            "EC50": ec50,
            "Px40": px,
            "activation_threshold": act_thresh
        })

    return pd.DataFrame(results)


# ======================================================================
# LANDSCAPE EXPANSION DECISION LOGIC
# ======================================================================

def assess_landscape_expansion(summary_mem, summary_naive):
    """
    Landscape expansion if memory cells:
       - Have lower EC50 for high-Kd peptides
       - Have lower Px thresholds
       - Activate at weaker ligands (higher Kd)
       - Logistic regression slope is shallower
    """
    conclusions = []

    # EC50 shift
    if summary_mem["EC50"].median() < summary_naive["EC50"].median():
        conclusions.append("Memory cells show reduced EC50 → expansion.")

    # Px shift
    if summary_mem["Px40"].median() < summary_naive["Px40"].median():
        conclusions.append("Memory cells cross Px(40) at lower concentration → expansion.")

    # Activation of weak ligands
    weak = summary_naive["kd"].quantile(0.8)
    mem_weak = summary_mem[summary_mem["kd"] > weak]["activation_threshold"].notna().mean()
    naive_weak = summary_naive[summary_naive["kd"] > weak]["activation_threshold"].notna().mean()

    if mem_weak > naive_weak:
        conclusions.append("Memory activates more weak-affinity ligands → expansion.")

    return conclusions

def plot_tiled_heatmaps(all_expts, title, save_path, value_to_plot="normalized"):
    """
    Tiles all experiments into a single figure with a shared colorbar.
    value_to_plot can be 'response', 'bg_sub', or 'normalized'.
    """
    num_expts = len(all_expts)
    cols = 3
    rows = (num_expts + cols - 1) // cols 
    
    # Create the figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex=True)
    axes = axes.flatten()

    # Determine max value for the colorbar scale
    v_max = 100 if value_to_plot == "normalized" else 80 
    v_min = 0

    for i, exp in enumerate(all_expts):
        ax = axes[i]
        plot_df = exp["corrected"].copy()
        
        # Apply your Affinity Order
        plot_df = plot_df[plot_df["peptide"].isin(AFFINITY_ORDER)]
        plot_df["peptide"] = pd.Categorical(plot_df["peptide"], categories=AFFINITY_ORDER, ordered=True)

        pivot = plot_df.pivot(index="peptide", columns="conc", values="response").reindex(AFFINITY_ORDER)
        # pivot = plot_df.pivot(index="peptide", columns="conc", values="bg_sub")
        # Plot heatmap without individual colorbars
        sns.heatmap(pivot, ax=ax, cmap="viridis", vmin=v_min, vmax=v_max, cbar=False)
        
        unique_id = os.path.basename(exp['file']).replace(" CD69 Pos.csv", "")
        ax.set_title(unique_id, fontsize=10)
        ax.set_ylabel("")
        ax.set_xlabel("")

    # Add one shared colorbar to the right of the subplots
    mappable = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=v_min, vmax=v_max))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    fig.colorbar(mappable, cax=cbar_ax, label=f"{value_to_plot.capitalize()} Response")

    # Clean up empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Make room for the colorbar
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

# def plot_differential_heatmap(memory_summary, naive_summary, save_path):
#     """
#     Calculates (Mean Memory - Mean Naive) for normalized responses 
#     and plots the difference.
#     """
#     # 1. Group by peptide and concentration to get averages
#     mem_avg = memory_summary.groupby(["peptide", "conc"])["normalized"].mean().reset_index()
#     nai_avg = naive_summary.groupby(["peptide", "conc"])["normalized"].mean().reset_index()

#     # 2. Pivot both into the standard peptide-vs-conc grid
#     mem_pivot = mem_avg.pivot(index="peptide", columns="conc", values="normalized")
#     nai_pivot = nai_avg.pivot(index="peptide", columns="conc", values="normalized")

#     # 3. Reindex to your specific Affinity Order
#     mem_pivot = mem_pivot.reindex(AFFINITY_ORDER)
#     nai_pivot = nai_pivot.reindex(AFFINITY_ORDER)

#     # 4. Subtract Naive from Memory
#     # Positive values (Red) = Memory is more responsive
#     # Negative values (Blue) = Naive is more responsive
#     diff_pivot = mem_pivot - nai_pivot

#     # 5. Plot using a Diverging Colormap (RdBu_r)
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(
#         diff_pivot, 
#         cmap="RdBu_r", 
#         center=0, 
#         annot=True, 
#         fmt=".1f", 
#         cbar_kws={'label': 'Difference in % Activation (Mem - Naive)'}
#     )
    
#     plt.title("Differential Heatmap: Landscape Expansion (Memory - Naive Average)")
#     plt.ylabel("Peptide (Sorted High to Low Affinity)")
#     plt.xlabel("Concentration (µM)")
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()

def plot_differential_heatmap(memory_summary, naive_summary, save_path):
    """
    Calculates (Mean Memory - Mean Naive) using background-subtracted data.
    This isolates the true functional expansion by removing baseline noise.
    """
    # 1. Group by peptide and concentration to get averages of bg_sub
    mem_avg = memory_summary.groupby(["peptide", "conc"])["bg_sub"].mean().reset_index()
    nai_avg = naive_summary.groupby(["peptide", "conc"])["bg_sub"].mean().reset_index()

    # 2. Pivot both into the standard peptide-vs-conc grid
    mem_pivot = mem_avg.pivot(index="peptide", columns="conc", values="bg_sub")
    nai_pivot = nai_avg.pivot(index="peptide", columns="conc", values="bg_sub")

    # 3. Reindex to your specific Affinity Order (9V to 5F)
    mem_pivot = mem_pivot.reindex(AFFINITY_ORDER)
    nai_pivot = nai_pivot.reindex(AFFINITY_ORDER)

    # 4. Subtract Naive from Memory
    # Positive (Red) = Memory induction is stronger
    # Negative (Blue) = Naive induction is stronger
    diff_pivot = mem_pivot - nai_pivot

    # 5. Plot using a Diverging Colormap centered at 0
    plt.figure(figsize=(14, 9))
    sns.heatmap(
        diff_pivot, 
        cmap="RdBu_r", 
        center=0, 
        annot=True, 
        fmt=".1f", 
        linewidths=.5,
        cbar_kws={'label': 'Difference in Signal Induction (% CD69+ Above Baseline)'}
    )
    
    plt.title("Differential Heatmap: Functional Landscape Expansion\n(Avg Memory bg_sub - Avg Naive bg_sub)", fontsize=16)
    plt.ylabel("Peptide (Highest to Lowest Affinity)")
    plt.xlabel("Concentration (µM)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
# ======================================================================
# RUN FULL PIPELINE
# ======================================================================

def main():
    print("\n=== LANDSCAPE EXPANSION ANALYSIS START ===\n")

    kd_dict = load_kd_table(KD_FILE)
    memory = load_experiment_files(MEMORY_FILES)
    naive = load_experiment_files(NAIVE_FILES)
    print(f"Loaded {len(naive)} Naive experiments.")

    for group_name, group in (("memory", memory), ("naive", naive)):
        print(f"Processing {group_name} experiments...")
        for exp in group:
            exp["corrected"] = apply_baseline_corrections(exp["raw"])

    # ------------------------ Compute metrics ------------------------
    memory_results = pd.concat([analyze_experiment(e, kd_dict) for e in memory])
    naive_results  = pd.concat([analyze_experiment(e, kd_dict) for e in naive])

    # ------------------------ Logistic regression ------------------------
    mem_beta = logistic_regression_affinity(memory_results["kd"], memory_results["Px40"].notna())
    naive_beta = logistic_regression_affinity(naive_results["kd"], naive_results["Px40"].notna())

    # ------------------------ Save summaries ------------------------
    memory_results.to_csv(f"{OUT}/memory_summary.csv", index=False)
    naive_results.to_csv(f"{OUT}/naive_summary.csv", index=False)

    # ------------------------ Heatmaps ------------------------
    for exp in memory:
        # Get the filename without the extension for a unique ID
        unique_id = os.path.basename(exp['file']).replace(".csv", "")
        plot_heatmap(exp["corrected"], f"Memory {unique_id}", f"{OUT}/memory_heatmap_{unique_id}.png")
        
    for exp in naive:
        unique_id = os.path.basename(exp['file']).replace(".csv", "")
        plot_heatmap(exp["corrected"], f"Naive {unique_id}", f"{OUT}/naive_heatmap_{unique_id}.png")
    # ------------------------ Landscape expansion decision ------------------------
    conclusions = assess_landscape_expansion(memory_results, naive_results)

    print("Generating tiled master heatmaps...")
    plot_tiled_heatmaps(memory, "1G4 Memory T-Cells: All Experiments", f"{OUT}/MASTER_memory_tiled_base.png")
    plot_tiled_heatmaps(naive, "1G4 Naive T-Cells: All Experiments", f"{OUT}/MASTER_naive_tiled_base.png")

    print("\n=== LANDSCAPE EXPANSION SUMMARY ===")
    for c in conclusions:
        print("•", c)

    print("\nLogistic regression slopes (β1):")
    print(f"  Memory: {mem_beta[1]:.3f}")
    print(f"  Naive : {naive_beta[1]:.3f}")

    print("\nResults saved to:", OUT)
    print("\n=== DONE ===\n")

    # ------------------------ Differential Analysis ------------------------
    # ------------------------ Differential Analysis ------------------------
    print("Generating differential landscape heatmap...")
    
    # Combine all individual corrected dataframes into one big table for Mem and Naive
    full_mem_data = pd.concat([e["corrected"] for e in memory])
    full_naive_data = pd.concat([e["corrected"] for e in naive])
    
    # Pass these FULL dataframes (which have 'conc' and 'normalized') to the function
    plot_differential_heatmap(full_mem_data, full_naive_data, f"{OUT}/DIFFERENTIAL_landscape_heatmap.png")

if __name__ == "__main__":
    main()
