#!/usr/bin/env python3
"""
SIMPLE METRICS - EXACTLY FOLLOWING JOHN'S SUGGESTIONS

Implements three specific metrics John requested:
1. Saturation concentration (for peptides that saturate)
2. Half-maximum response concentration (EC50)
3. Px thresholds (P15, P30, P40) - direct comparison

Author: Madeline
Date: February 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
from scipy.interpolate import interp1d

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

AFFINITY_ORDER = ["9V", "6V", "3Y", "4D", "6T", "4A", "5Y", "5F"]

# ======================================================================
# DATA LOADING
# ======================================================================

def load_kd_values(path):
    kd_df = pd.read_csv(path, encoding="utf-8-sig", skiprows=1)
    kd_dict = {}
    for _, row in kd_df.iterrows():
        name = row.iloc[0].replace("NYE ", "")
        kd_dict[name] = {'kd_mean': row.iloc[6]}
    return kd_dict

def load_and_correct_data(files):
    experiments = []
    for filepath in files:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        
        filename = os.path.basename(filepath)
        date = filename[:6]
        
        # Apply baseline correction
        concentrations = df.iloc[:, 0].values
        corrected_data = {}
        
        for peptide in df.columns[1:]:
            responses = df[peptide].values
            baseline = np.nanmin(responses)
            bg_subtract = responses - baseline
            corrected_data[peptide] = bg_subtract
        
        experiments.append({
            'date': date,
            'filename': filename,
            'concentrations': concentrations,
            'corrected': corrected_data
        })
    
    return experiments

# ======================================================================
# JOHN'S METRIC #1: HALF-MAXIMUM RESPONSE CONCENTRATION (EC50)
# ======================================================================

def hill_equation(x, bottom, top, ec50, hill):
    """4-parameter Hill equation"""
    return bottom + (top - bottom) / (1 + (ec50/x)**hill)

def calculate_ec50(concentrations, responses):
    """
    Fit Hill curve and extract EC50 (half-maximum concentration)
    This is John's "half-maximum response concentration"
    """
    mask = ~np.isnan(responses)
    conc = concentrations[mask]
    resp = responses[mask]
    
    if len(resp) < 4:
        return np.nan, np.nan  # Return EC50 and max response
    
    bottom_guess = np.min(resp)
    top_guess = np.max(resp)
    ec50_guess = np.median(conc)
    
    try:
        popt, _ = curve_fit(
            hill_equation, conc, resp,
            p0=[bottom_guess, top_guess, ec50_guess, 1],
            bounds=([0, 0, conc.min()/10, 0.1], [100, 100, conc.max()*10, 10]),
            maxfev=10000
        )
        return popt[2], popt[1]  # EC50, top (maximum response)
    except:
        return np.nan, np.nan

# ======================================================================
# JOHN'S METRIC #2: SATURATION CONCENTRATION
# ======================================================================

def calculate_saturation_concentration(concentrations, responses, saturation_fraction=0.8):
    """
    Concentration at which response reaches saturation
    John's specific suggestion: "saturation concentration for peptides that reach it"
    
    Definition: Concentration where response = saturation_fraction × max_response
    Default: 80% of maximum (standard definition of saturation)
    """
    mask = ~np.isnan(responses)
    conc = concentrations[mask]
    resp = responses[mask]
    
    if len(resp) < 3:
        return np.nan
    
    max_resp = np.max(resp)
    saturation_level = saturation_fraction * max_resp
    
    # Does curve reach saturation?
    if max_resp < saturation_level:  # Can't saturate if max < saturation
        return np.nan
    
    # Find concentration at saturation level
    try:
        f = interp1d(resp, conc, kind='linear', bounds_error=False, fill_value='extrapolate')
        sat_conc = float(f(saturation_level))
        
        # Sanity check - must be within concentration range
        if sat_conc < conc.min() / 100 or sat_conc > conc.max() * 100:
            return np.nan
            
        return sat_conc
    except:
        return np.nan

# ======================================================================
# JOHN'S METRIC #3: Px THRESHOLDS (DIRECT COMPARISON)
# ======================================================================

def calculate_px(concentrations, responses, threshold):
    """
    Calculate Px: concentration at which response = threshold %
    John's "Thresholds like P15 or P30"
    """
    mask = ~np.isnan(responses)
    conc = concentrations[mask]
    resp = responses[mask]
    
    if len(resp) < 2:
        return np.nan
    if resp.max() < threshold:
        return np.nan  # Never reaches threshold
    if resp.min() > threshold:
        return np.nan  # Always above threshold
    
    try:
        f = interp1d(resp, conc, kind='linear', bounds_error=False, fill_value='extrapolate')
        px = float(f(threshold))
        
        # Sanity check
        if px < conc.min() / 100 or px > conc.max() * 100:
            return np.nan
            
        return px
    except:
        return np.nan

# ======================================================================
# PER-PEPTIDE ANALYSIS (ALL THREE METRICS)
# ======================================================================

def analyze_all_metrics_per_peptide(memory_exps, naive_exps, kd_dict):
    """
    For each peptide, calculate all three of John's suggested metrics
    Compare memory vs naive for each
    """
    print("="*80)
    print("JOHN'S SUGGESTED METRICS - PER-PEPTIDE ANALYSIS")
    print("="*80)
    print()
    
    all_results = []
    
    for peptide in AFFINITY_ORDER:
        if peptide not in kd_dict:
            continue
        
        print(f"\nPeptide: {peptide} (KD = {kd_dict[peptide]['kd_mean']:.1f} µM)")
        print("-"*60)
        
        # Collect all three metrics from all experiments
        mem_ec50s = []
        mem_sats = []
        mem_p15s = []
        mem_p30s = []
        mem_p40s = []
        
        naive_ec50s = []
        naive_sats = []
        naive_p15s = []
        naive_p30s = []
        naive_p40s = []
        
        # Memory experiments
        for exp in memory_exps:
            if peptide in exp['corrected']:
                conc = exp['concentrations']
                resp = exp['corrected'][peptide]
                
                ec50, max_resp = calculate_ec50(conc, resp)
                sat_conc = calculate_saturation_concentration(conc, resp, 0.8)
                p15 = calculate_px(conc, resp, 15)
                p30 = calculate_px(conc, resp, 30)
                p40 = calculate_px(conc, resp, 40)
                
                if not np.isnan(ec50): mem_ec50s.append(ec50)
                if not np.isnan(sat_conc): mem_sats.append(sat_conc)
                if not np.isnan(p15): mem_p15s.append(p15)
                if not np.isnan(p30): mem_p30s.append(p30)
                if not np.isnan(p40): mem_p40s.append(p40)
        
        # Naive experiments
        for exp in naive_exps:
            if peptide in exp['corrected']:
                conc = exp['concentrations']
                resp = exp['corrected'][peptide]
                
                ec50, max_resp = calculate_ec50(conc, resp)
                sat_conc = calculate_saturation_concentration(conc, resp, 0.8)
                p15 = calculate_px(conc, resp, 15)
                p30 = calculate_px(conc, resp, 30)
                p40 = calculate_px(conc, resp, 40)
                
                if not np.isnan(ec50): naive_ec50s.append(ec50)
                if not np.isnan(sat_conc): naive_sats.append(sat_conc)
                if not np.isnan(p15): naive_p15s.append(p15)
                if not np.isnan(p30): naive_p30s.append(p30)
                if not np.isnan(p40): naive_p40s.append(p40)
        
        # Statistical comparisons
        result = {
            'peptide': peptide,
            'kd': kd_dict[peptide]['kd_mean']
        }
        
        # EC50
        if len(mem_ec50s) >= 2 and len(naive_ec50s) >= 2:
            log_mem = np.log10(mem_ec50s)
            log_naive = np.log10(naive_ec50s)
            _, p_ec50 = stats.ttest_ind(log_mem, log_naive)
            
            result['ec50_memory'] = stats.gmean(mem_ec50s)
            result['ec50_naive'] = stats.gmean(naive_ec50s)
            result['ec50_fold'] = result['ec50_naive'] / result['ec50_memory']
            result['ec50_p'] = p_ec50
            result['ec50_n_mem'] = len(mem_ec50s)
            result['ec50_n_naive'] = len(naive_ec50s)
            
            print(f"  EC50: Mem={result['ec50_memory']:.4f} µM, Naive={result['ec50_naive']:.4f} µM, "
                  f"FC={result['ec50_fold']:.2f}×, p={p_ec50:.3f}")
        
        # Saturation concentration
        if len(mem_sats) >= 2 and len(naive_sats) >= 2:
            log_mem = np.log10(mem_sats)
            log_naive = np.log10(naive_sats)
            _, p_sat = stats.ttest_ind(log_mem, log_naive)
            
            result['sat_memory'] = stats.gmean(mem_sats)
            result['sat_naive'] = stats.gmean(naive_sats)
            result['sat_fold'] = result['sat_naive'] / result['sat_memory']
            result['sat_p'] = p_sat
            result['sat_n_mem'] = len(mem_sats)
            result['sat_n_naive'] = len(naive_sats)
            
            print(f"  C80 (saturation): Mem={result['sat_memory']:.4f} µM, Naive={result['sat_naive']:.4f} µM, "
                  f"FC={result['sat_fold']:.2f}×, p={p_sat:.3f}")
        
        # P15
        if len(mem_p15s) >= 2 and len(naive_p15s) >= 2:
            log_mem = np.log10(mem_p15s)
            log_naive = np.log10(naive_p15s)
            _, p_p15 = stats.ttest_ind(log_mem, log_naive)
            
            result['p15_memory'] = stats.gmean(mem_p15s)
            result['p15_naive'] = stats.gmean(naive_p15s)
            result['p15_fold'] = result['p15_naive'] / result['p15_memory']
            result['p15_p'] = p_p15
            result['p15_n_mem'] = len(mem_p15s)
            result['p15_n_naive'] = len(naive_p15s)
            
            print(f"  P15: Mem={result['p15_memory']:.4f} µM, Naive={result['p15_naive']:.4f} µM, "
                  f"FC={result['p15_fold']:.2f}×, p={p_p15:.3f}")
        
        # P30
        if len(mem_p30s) >= 2 and len(naive_p30s) >= 2:
            log_mem = np.log10(mem_p30s)
            log_naive = np.log10(naive_p30s)
            _, p_p30 = stats.ttest_ind(log_mem, log_naive)
            
            result['p30_memory'] = stats.gmean(mem_p30s)
            result['p30_naive'] = stats.gmean(naive_p30s)
            result['p30_fold'] = result['p30_naive'] / result['p30_memory']
            result['p30_p'] = p_p30
            result['p30_n_mem'] = len(mem_p30s)
            result['p30_n_naive'] = len(naive_p30s)
            
            print(f"  P30: Mem={result['p30_memory']:.4f} µM, Naive={result['p30_naive']:.4f} µM, "
                  f"FC={result['p30_fold']:.2f}×, p={p_p30:.3f}")
        
        # P40
        if len(mem_p40s) >= 2 and len(naive_p40s) >= 2:
            log_mem = np.log10(mem_p40s)
            log_naive = np.log10(naive_p40s)
            _, p_p40 = stats.ttest_ind(log_mem, log_naive)
            
            result['p40_memory'] = stats.gmean(mem_p40s)
            result['p40_naive'] = stats.gmean(naive_p40s)
            result['p40_fold'] = result['p40_naive'] / result['p40_memory']
            result['p40_p'] = p_p40
            result['p40_n_mem'] = len(mem_p40s)
            result['p40_n_naive'] = len(naive_p40s)
            
            print(f"  P40: Mem={result['p40_memory']:.4f} µM, Naive={result['p40_naive']:.4f} µM, "
                  f"FC={result['p40_fold']:.2f}×, p={p_p40:.3f}")
        
        all_results.append(result)
    
    return pd.DataFrame(all_results)

#!/usr/bin/env python3
"""
MAXIMUM RESPONSE ANALYSIS
Better metric for low-affinity expansion

For each peptide, compare:
- Maximum response achieved (does it activate strongly?)
- Response at high fixed concentration (cross-reactivity)
- Area under curve (total responsiveness)

These capture MAGNITUDE of response, not just threshold crossing
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_maximum_responses(memory_exps, naive_exps, kd_dict):
    """
    Compare maximum responses for each peptide
    This captures whether cells RESPOND (magnitude), not just threshold
    """
    print("="*80)
    print("MAXIMUM RESPONSE ANALYSIS")
    print("Better metric for low-affinity expansion")
    print("="*80)
    print()
    
    results = []
    
    for peptide in ["9V", "6V", "3Y", "4D", "6T", "4A", "5Y", "5F"]:
        if peptide not in kd_dict:
            continue
        
        mem_maxes = []
        naive_maxes = []
        
        # Collect maximum responses
        for exp in memory_exps:
            if peptide in exp['corrected']:
                max_resp = np.nanmax(exp['corrected'][peptide])
                mem_maxes.append(max_resp)
        
        for exp in naive_exps:
            if peptide in exp['corrected']:
                max_resp = np.nanmax(exp['corrected'][peptide])
                naive_maxes.append(max_resp)
        
        # Compare
        if len(mem_maxes) >= 2 and len(naive_maxes) >= 2:
            t_stat, p_val = stats.ttest_ind(mem_maxes, naive_maxes)
            
            pooled_std = np.sqrt((np.var(mem_maxes) + np.var(naive_maxes)) / 2)
            cohens_d = (np.mean(mem_maxes) - np.mean(naive_maxes)) / pooled_std
            
            result = {
                'peptide': peptide,
                'kd': kd_dict[peptide]['kd_mean'],
                'memory_max': np.mean(mem_maxes),
                'memory_std': np.std(mem_maxes),
                'naive_max': np.mean(naive_maxes),
                'naive_std': np.std(naive_maxes),
                'difference': np.mean(mem_maxes) - np.mean(naive_maxes),
                'p_value': p_val,
                'cohens_d': cohens_d,
                'n_memory': len(mem_maxes),
                'n_naive': len(naive_maxes)
            }
            
            results.append(result)
            
            sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{peptide:4s} (KD={kd_dict[peptide]['kd_mean']:7.1f}): "
                  f"Mem={np.mean(mem_maxes):5.1f}% vs Naive={np.mean(naive_maxes):5.1f}%, "
                  f"Δ={np.mean(mem_maxes)-np.mean(naive_maxes):+5.1f}%, "
                  f"p={p_val:.3f}, d={cohens_d:+.2f} {sig}")
    
    results_df = pd.DataFrame(results)
    
    print()
    print(f"Significant (p<0.05): {sum(results_df['p_value'] < 0.05)}/{len(results_df)} peptides")
    print(f"Large effect (|d|>0.8): {sum(abs(results_df['cohens_d']) > 0.8)}/{len(results_df)} peptides")
    print()
    
    # Test for pattern with affinity
    corr, p_corr = stats.spearmanr(results_df['kd'], results_df['difference'])
    print(f"Correlation (KD vs enhancement): ρ={corr:.3f}, p={p_corr:.3f}")
    if corr > 0 and p_corr < 0.05:
        print("✓ Positive correlation: Memory enhancement STRONGER for low-affinity!")
        print("  This IS landscape expansion signature!")
    
    return results_df



# ======================================================================
# SUMMARY STATISTICS
# ======================================================================

def summarize_results(results_df):
    """
    Summarize how many metrics show significant expansion
    """
    print("\n" + "="*80)
    print("SUMMARY: JOHN'S METRICS")
    print("="*80)
    print()
    
    metrics = ['ec50', 'sat', 'p15', 'p30', 'p40']
    metric_names = {
        'ec50': 'EC50 (half-max)',
        'sat': 'C80 (saturation)',
        'p15': 'P15 (15% threshold)',
        'p30': 'P30 (30% threshold)',
        'p40': 'P40 (40% threshold)'
    }
    
    for metric in metrics:
        p_col = f'{metric}_p'
        fold_col = f'{metric}_fold'
        
        if p_col in results_df.columns:
            p_values = results_df[p_col].dropna()
            fold_values = results_df[fold_col].dropna()
            
            n_total = len(p_values)
            n_sig = sum(p_values < 0.05)
            n_marginal = sum((p_values >= 0.05) & (p_values < 0.10))
            
            mean_fold = fold_values.mean() if len(fold_values) > 0 else np.nan
            
            print(f"{metric_names[metric]:20s}: {n_sig}/{n_total} peptides p<0.05, "
                  f"{n_marginal} marginal, mean FC={mean_fold:.2f}×")
    
    print()
    
    # Overall significance
    all_p_values = []
    for metric in metrics:
        p_col = f'{metric}_p'
        if p_col in results_df.columns:
            all_p_values.extend(results_df[p_col].dropna().tolist())
    
    if len(all_p_values) > 0:
        n_sig_total = sum(np.array(all_p_values) < 0.05)
        n_total = len(all_p_values)
        
        print(f"Overall: {n_sig_total}/{n_total} peptide×metric combinations significant ({n_sig_total/n_total*100:.0f}%)")
        print()
        
        if n_sig_total >= 5:
            print("✓✓ STRONG evidence - multiple metrics across multiple peptides significant")
        elif n_sig_total >= 3:
            print("✓ GOOD evidence - several significant findings")
        elif n_sig_total >= 1:
            print("✓ MODERATE evidence - at least some significant results")
        else:
            print("~ WEAK evidence - trends but no significance")
    
    return

# ======================================================================
# VISUALIZATION
# ======================================================================

def plot_metric_comparison(results_df, save_path):
    """
    Visualize all metrics across peptides
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metrics = [
        ('ec50', 'EC50 (µM)', 'Half-Maximum Concentration'),
        ('sat', 'C80 (µM)', 'Saturation Concentration (80% max)'),
        ('p15', 'P15 (µM)', '15% Activation Threshold'),
        ('p30', 'P30 (µM)', '30% Activation Threshold'),
        ('p40', 'P40 (µM)', '40% Activation Threshold')
    ]
    
    for idx, (metric, ylabel, title) in enumerate(metrics):
        ax = axes[idx]
        
        mem_col = f'{metric}_memory'
        naive_col = f'{metric}_naive'
        p_col = f'{metric}_p'
        
        if mem_col not in results_df.columns:
            ax.axis('off')
            continue
        
        # Get data
        plot_data = results_df[['peptide', mem_col, naive_col, p_col, 'kd']].dropna()
        
        if len(plot_data) == 0:
            ax.axis('off')
            continue
        
        x_pos = np.arange(len(plot_data))
        width = 0.35
        
        # Plot bars
        ax.bar(x_pos - width/2, plot_data[mem_col], width, 
               label='Memory', color='#e74c3c', alpha=0.7, edgecolor='black')
        ax.bar(x_pos + width/2, plot_data[naive_col], width,
               label='Naive', color='#3498db', alpha=0.7, edgecolor='black')
        
        # Add significance markers
        for i, (_, row) in enumerate(plot_data.iterrows()):
            if row[p_col] < 0.01:
                y_pos = max(row[mem_col], row[naive_col]) * 1.1
                ax.text(i, y_pos, '**', ha='center', fontsize=14, fontweight='bold')
            elif row[p_col] < 0.05:
                y_pos = max(row[mem_col], row[naive_col]) * 1.1
                ax.text(i, y_pos, '*', ha='center', fontsize=14, fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_data['peptide'], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_xlabel('Peptide (High to Low Affinity)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Summary panel
    ax = axes[5]
    ax.axis('off')
    
    # Count significant results
    summary_text = "SUMMARY:\n\n"
    
    for metric, _, name in metrics:
        p_col = f'{metric}_p'
        if p_col in results_df.columns:
            p_vals = results_df[p_col].dropna()
            n_sig = sum(p_vals < 0.05)
            n_total = len(p_vals)
            
            sig_marker = "✓✓" if n_sig >= 4 else "✓" if n_sig >= 2 else "~"
            summary_text += f"{sig_marker} {name}:\n   {n_sig}/{n_total} peptides p<0.05\n\n"
    
    ax.text(0.1, 0.9, summary_text, fontsize=11, family='monospace',
            verticalalignment='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle("John's Suggested Metrics: Per-Peptide Comparison\n" +
                "(Lower values = earlier activation = expansion)",
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ======================================================================
# MAIN ANALYSIS
# ======================================================================

def main():
    print("\n" + "="*80)
    print("IMPLEMENTING JOHN'S SUGGESTED METRICS")
    print("="*80)
    print()
    print("Metrics:")
    print("1. EC50 (half-maximum response concentration)")
    print("2. Saturation concentration (80% of max)")
    print("3. Px thresholds (P15, P30, P40)")
    print()
    
    # Load data
    kd_dict = load_kd_values(KD_FILE)
    memory_exps = load_and_correct_data(MEMORY_FILES)
    naive_exps = load_and_correct_data(NAIVE_FILES)
    print(f"✓ Loaded {len(memory_exps)} memory, {len(naive_exps)} naive experiments\n")
    
    # Analyze all metrics per peptide
    results_df = analyze_all_metrics_per_peptide(memory_exps, naive_exps, kd_dict)
    analyze_maximum_responses(memory_exps, naive_exps, kd_dict)
    
    # Summarize
    summarize_results(results_df)
    
    # Visualize
    print("\nCreating comparison plots...")
    plot_metric_comparison(results_df, save_path=f'{OUT}/johns_metrics_comparison.png')
    print(f"✓ Saved johns_metrics_comparison.png\n")
    
    # Save data
    results_df.to_csv(f'{OUT}/johns_metrics_per_peptide.csv', index=False)
    print(f"✓ Saved johns_metrics_per_peptide.csv\n")
    
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()
    
    return results_df

if __name__ == "__main__":
    results = main()