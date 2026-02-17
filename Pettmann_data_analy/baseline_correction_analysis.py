"""
BASELINE-CORRECTED DISCRIMINATION POWER ANALYSIS

This script corrects for differential baseline CD69+ expression between
memory and naive T cells before calculating discrimination power.

Key improvements:
1. Background subtraction (uses lowest concentration as baseline)
2. Dynamic range normalization 
3. Comparison to uncorrected analysis
4. Assessment of baseline impact on conclusions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import seaborn as sns

sns.set_style("ticks")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = '/Users/maddie/Desktop/Projects/Research/Pettmann_data_analy'
OUTPUT_DIR = f'{BASE_DIR}/outputs'

KD_FILE = f'{BASE_DIR}/elife-67092-fig1-data2-v3.csv'
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

# ============================================================================
# BASELINE CORRECTION FUNCTIONS
# ============================================================================

def apply_baseline_corrections(df):
    """
    Apply multiple baseline correction strategies to dose-response data
    
    Returns dictionary with:
    - 'raw': Original data
    - 'bg_subtract': Baseline subtraction (min value subtracted)
    - 'normalized': 0-100% normalized to dynamic range
    - 'fold_baseline': Fold-change over baseline
    """
    concentrations = df.iloc[:, 0].values
    corrected_data = {
        'concentrations': concentrations,
        'raw': {},
        'bg_subtract': {},
        'normalized': {},
        'fold_baseline': {}
    }
    
    for peptide in df.columns[1:]:
        responses = df[peptide].values
        
        # Raw data
        corrected_data['raw'][peptide] = responses
        
        # Method 1: Background subtraction (subtract minimum)
        baseline = np.nanmin(responses)
        bg_subtract = responses - baseline
        corrected_data['bg_subtract'][peptide] = bg_subtract
        
        # Method 2: Normalize to 0-100% of dynamic range
        min_val = np.nanmin(responses)
        max_val = np.nanmax(responses)
        if max_val > min_val:
            normalized = 100 * (responses - min_val) / (max_val - min_val)
        else:
            normalized = responses * 0  # All zeros if no dynamic range
        corrected_data['normalized'][peptide] = normalized
        
        # Method 3: Fold-change over baseline
        if baseline > 0:
            fold_baseline = responses / baseline
        else:
            fold_baseline = responses  # Can't divide by zero
        corrected_data['fold_baseline'][peptide] = fold_baseline
    
    return corrected_data

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_kd_values(kd_file):
    kd_df = pd.read_csv(kd_file, encoding='utf-8-sig', skiprows=1)
    kd_dict = {}
    for idx, row in kd_df.iterrows():
        name = row.iloc[0].replace('NYE ', '')
        kd_dict[name] = {
            'kd_mean': row.iloc[6],
            'kd_sd': row.iloc[7],
            'sequence': row.iloc[1]
        }
    return kd_dict

def load_data_with_corrections(memory_files, naive_files):
    """Load data and apply baseline corrections"""
    data = {'memory': [], 'naive': []}
    
    for file in memory_files:
        df = pd.read_csv(file, encoding='utf-8-sig')
        filename = file.split('/')[-1]
        date = filename[:6]
        
        corrected = apply_baseline_corrections(df)
        
        data['memory'].append({
            'date': date,
            'file': filename,
            'raw_data': df,
            'corrected': corrected
        })
    
    for file in naive_files:
        df = pd.read_csv(file, encoding='utf-8-sig')
        filename = file.split('/')[-1]
        date = filename[:6]
        
        corrected = apply_baseline_corrections(df)
        
        data['naive'].append({
            'date': date,
            'file': filename,
            'raw_data': df,
            'corrected': corrected
        })
    
    return data

def calculate_px(concentrations, responses, threshold_pct):
    """Calculate Px from dose-response"""
    mask = ~(np.isnan(concentrations) | np.isnan(responses))
    conc = concentrations[mask]
    resp = responses[mask]
    
    if len(resp) < 2 or resp.max() < threshold_pct or resp.min() > threshold_pct:
        return np.nan
    
    log_conc = np.log10(conc)
    f = interp1d(resp, log_conc, kind='linear', 
                 bounds_error=False, fill_value='extrapolate')
    log_px = f(threshold_pct)
    return 10**log_px

def log_power_law(log_kd, log_C, alpha):
    return log_C + alpha * log_kd

def fit_discrimination_power(px_values, kd_values):
    mask = ~(np.isnan(px_values) | np.isnan(kd_values))
    px = px_values[mask]
    kd = kd_values[mask]
    
    if len(px) < 3:
        return None, None, np.nan
    
    log_px = np.log10(px)
    log_kd = np.log10(kd)
    
    try:
        popt, pcov = curve_fit(log_power_law, log_kd, log_px, 
                               p0=[-3, 1.5], maxfev=10000)
        log_px_pred = log_power_law(log_kd, popt[0], popt[1])
        ss_res = np.sum((log_px - log_px_pred) ** 2)
        ss_tot = np.sum((log_px - np.mean(log_px)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        return popt, pcov, r_squared
    except:
        return None, None, np.nan

# ============================================================================
# BASELINE IMPACT ANALYSIS
# ============================================================================

def analyze_baselines(data):
    """
    Analyze baseline CD69+ levels and their impact
    """
    print("="*80)
    print("BASELINE CD69+ ANALYSIS")
    print("="*80)
    print()
    
    baseline_data = []
    
    for condition, replicates in data.items():
        for rep in replicates:
            df = rep['raw_data']
            concentrations = df.iloc[:, 0].values
            
            # Get baseline (lowest concentration)
            baseline_conc = concentrations[-1]  # Lowest concentration
            
            for peptide in df.columns[1:]:
                baseline_response = df[peptide].iloc[-1]
                max_response = df[peptide].max()
                dynamic_range = max_response - baseline_response
                
                baseline_data.append({
                    'condition': condition,
                    'experiment': rep['date'],
                    'peptide': peptide,
                    'baseline': baseline_response,
                    'maximum': max_response,
                    'dynamic_range': dynamic_range
                })
    
    baseline_df = pd.DataFrame(baseline_data)
    
    # Summary statistics
    print("BASELINE CD69+ LEVELS:")
    print("-"*80)
    for condition in ['memory', 'naive']:
        baselines = baseline_df[baseline_df['condition'] == condition]['baseline'].values
        print(f"{condition.upper()}:")
        print(f"  Mean baseline: {np.mean(baselines):.2f}% ± {np.std(baselines):.2f}%")
        print(f"  Median baseline: {np.median(baselines):.2f}%")
        print(f"  Range: [{np.min(baselines):.1f}%, {np.max(baselines):.1f}%]")
        print()
    
    # Statistical test
    mem_baselines = baseline_df[baseline_df['condition'] == 'memory']['baseline'].values
    naive_baselines = baseline_df[baseline_df['condition'] == 'naive']['baseline'].values
    t_stat, p_val = stats.ttest_ind(mem_baselines, naive_baselines)
    
    print(f"Statistical test (memory vs naive baseline):")
    print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
    if p_val < 0.05:
        print("  → Baselines are SIGNIFICANTLY DIFFERENT!")
        print("  → This confounds dose-response analysis!")
    print()
    
    # Dynamic range comparison
    print("DYNAMIC RANGE:")
    print("-"*80)
    for condition in ['memory', 'naive']:
        dranges = baseline_df[baseline_df['condition'] == condition]['dynamic_range'].values
        print(f"{condition.upper()}:")
        print(f"  Mean dynamic range: {np.mean(dranges):.2f}% ± {np.std(dranges):.2f}%")
        print()
    
    return baseline_df

# ============================================================================
# CORRECTED DISCRIMINATION POWER ANALYSIS
# ============================================================================

def fit_alpha_with_correction(data, kd_dict, threshold, correction_method='bg_subtract'):
    """
    Fit discrimination power using baseline-corrected data
    
    correction_method: 'raw', 'bg_subtract', 'normalized', or 'fold_baseline'
    """
    results = []
    
    for condition, replicates in data.items():
        for rep in replicates:
            corrected = rep['corrected']
            concentrations = corrected['concentrations']
            
            px_list = []
            kd_list = []
            
            for peptide in kd_dict.keys():
                if peptide not in corrected[correction_method]:
                    continue
                
                # Get corrected responses
                responses = corrected[correction_method][peptide]
                
                # Calculate Px
                px = calculate_px(concentrations, responses, threshold)
                
                if not np.isnan(px):
                    px_list.append(px)
                    kd_list.append(kd_dict[peptide]['kd_mean'])
            
            # Fit α
            if len(px_list) >= 3:
                popt, pcov, r2 = fit_discrimination_power(
                    np.array(px_list), np.array(kd_list)
                )
                
                if popt is not None:
                    results.append({
                        'condition': condition,
                        'experiment': rep['date'],
                        'correction': correction_method,
                        'n_peptides': len(px_list),
                        'alpha': popt[1],
                        'alpha_se': np.sqrt(pcov[1, 1]) if pcov is not None else np.nan,
                        'C': 10**popt[0],
                        'log_C': popt[0],
                        'r_squared': r2
                    })
    
    return pd.DataFrame(results)

def compare_correction_methods(data, kd_dict, threshold=40):
    """
    Compare results across different baseline correction methods
    """
    print("="*80)
    print(f"COMPARING BASELINE CORRECTION METHODS (P{threshold} threshold)")
    print("="*80)
    print()
    
    methods = ['raw', 'bg_subtract', 'normalized']
    results_dict = {}
    
    for method in methods:
        alpha_df = fit_alpha_with_correction(data, kd_dict, threshold, method)
        
        if len(alpha_df) == 0:
            print(f"{method}: No experiments fit")
            continue
        
        mem_alphas = alpha_df[alpha_df['condition'] == 'memory']['alpha'].values
        naive_alphas = alpha_df[alpha_df['condition'] == 'naive']['alpha'].values
        
        if len(mem_alphas) > 0 and len(naive_alphas) > 0:
            t_stat, p_val = stats.ttest_ind(mem_alphas, naive_alphas)
            pooled_std = np.sqrt((np.var(mem_alphas) + np.var(naive_alphas)) / 2)
            cohens_d = (np.mean(mem_alphas) - np.mean(naive_alphas)) / pooled_std
            
            results_dict[method] = {
                'method': method,
                'n_memory': len(mem_alphas),
                'n_naive': len(naive_alphas),
                'memory_mean': np.mean(mem_alphas),
                'memory_std': np.std(mem_alphas),
                'naive_mean': np.mean(naive_alphas),
                'naive_std': np.std(naive_alphas),
                'delta_alpha': np.mean(mem_alphas) - np.mean(naive_alphas),
                't_pval': p_val,
                'cohens_d': cohens_d,
                'alpha_df': alpha_df
            }
            
            print(f"{method.upper()}:")
            print(f"  Memory: α = {np.mean(mem_alphas):.3f} ± {np.std(mem_alphas):.3f} (n={len(mem_alphas)})")
            print(f"  Naive:  α = {np.mean(naive_alphas):.3f} ± {np.std(naive_alphas):.3f} (n={len(naive_alphas)})")
            print(f"  Δα = {np.mean(mem_alphas) - np.mean(naive_alphas):.3f}")
            print(f"  p = {p_val:.3f}, d = {cohens_d:.2f}")
            print()
    
    return results_dict

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_baseline_correction_comparison(results_dict, baseline_df, save_path=None):
    """
    Visualize impact of baseline correction
    """
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)
    
    colors = {'memory': '#e74c3c', 'naive': '#3498db'}
    methods = list(results_dict.keys())
    
    # Panel A: Baseline levels by condition
    ax_a = fig.add_subplot(gs[0, 0])
    
    for condition in ['memory', 'naive']:
        baselines = baseline_df[baseline_df['condition'] == condition]['baseline'].values
        
        # Violin plot
        parts = ax_a.violinplot([baselines], positions=[0 if condition == 'memory' else 1],
                               widths=0.6, showmeans=True, showmedians=True)
        
        for pc in parts['bodies']:
            pc.set_facecolor(colors[condition])
            pc.set_alpha(0.7)
        
        # Individual points
        x_pos = 0 if condition == 'memory' else 1
        x_jitter = np.random.normal(x_pos, 0.03, len(baselines))
        ax_a.scatter(x_jitter, baselines, s=30, alpha=0.5, 
                    color=colors[condition], edgecolors='black', linewidths=0.5)
    
    # Statistical test
    mem_base = baseline_df[baseline_df['condition'] == 'memory']['baseline'].values
    naive_base = baseline_df[baseline_df['condition'] == 'naive']['baseline'].values
    t_stat, p_val = stats.ttest_ind(mem_base, naive_base)
    
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(['Memory', 'Naive'], fontsize=13, fontweight='bold')
    ax_a.set_ylabel('Baseline CD69+ (%)', fontsize=12, fontweight='bold')
    ax_a.set_title(f'A. Baseline Levels\np={p_val:.4f}', fontsize=14, fontweight='bold')
    ax_a.grid(True, alpha=0.3, axis='y')
    
    # Panel B: Dynamic range comparison
    ax_b = fig.add_subplot(gs[0, 1])
    
    for condition in ['memory', 'naive']:
        drange = baseline_df[baseline_df['condition'] == condition]['dynamic_range'].values
        
        parts = ax_b.violinplot([drange], positions=[0 if condition == 'memory' else 1],
                               widths=0.6, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[condition])
            pc.set_alpha(0.7)
        
        x_pos = 0 if condition == 'memory' else 1
        x_jitter = np.random.normal(x_pos, 0.03, len(drange))
        ax_b.scatter(x_jitter, drange, s=30, alpha=0.5,
                    color=colors[condition], edgecolors='black', linewidths=0.5)
    
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(['Memory', 'Naive'], fontsize=13, fontweight='bold')
    ax_b.set_ylabel('Dynamic Range (%)', fontsize=12, fontweight='bold')
    ax_b.set_title('B. Response Dynamic Range', fontsize=14, fontweight='bold')
    ax_b.grid(True, alpha=0.3, axis='y')
    
    # Panel C: α values across correction methods
    ax_c = fig.add_subplot(gs[0, 2:])
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    mem_means = [results_dict[m]['memory_mean'] for m in methods]
    mem_stds = [results_dict[m]['memory_std'] for m in methods]
    naive_means = [results_dict[m]['naive_mean'] for m in methods]
    naive_stds = [results_dict[m]['naive_std'] for m in methods]
    
    ax_c.bar(x_pos - width/2, mem_means, width, yerr=mem_stds,
            label='Memory', color=colors['memory'], alpha=0.7, capsize=5,
            edgecolor='black', linewidth=1.5)
    ax_c.bar(x_pos + width/2, naive_means, width, yerr=naive_stds,
            label='Naive', color=colors['naive'], alpha=0.7, capsize=5,
            edgecolor='black', linewidth=1.5)
    
    ax_c.axhline(2, color='red', linestyle='--', alpha=0.3, label='α=2 (Pettmann)')
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=11)
    ax_c.set_ylabel('Discrimination Power (α)', fontsize=13, fontweight='bold')
    ax_c.set_xlabel('Correction Method', fontsize=13, fontweight='bold')
    ax_c.set_title('C. Effect of Baseline Correction on α', fontsize=15, fontweight='bold')
    ax_c.legend(fontsize=11)
    ax_c.grid(True, alpha=0.3, axis='y')
    
    # Panel D: p-values by correction method
    ax_d = fig.add_subplot(gs[1, 0])
    
    pvals = [results_dict[m]['t_pval'] for m in methods]
    bars = ax_d.bar(range(len(methods)), pvals, alpha=0.7,
                   color=['darkred' if p < 0.05 else 'orange' if p < 0.10 else 'gray'
                          for p in pvals],
                   edgecolor='black', linewidth=1.5)
    
    ax_d.axhline(0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')
    ax_d.axhline(0.10, color='orange', linestyle='--', linewidth=1.5, label='p=0.10')
    ax_d.set_xticks(range(len(methods)))
    ax_d.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=10)
    ax_d.set_ylabel('p-value', fontsize=12, fontweight='bold')
    ax_d.set_title('D. Statistical Significance', fontsize=14, fontweight='bold')
    ax_d.legend(fontsize=10)
    ax_d.grid(True, alpha=0.3, axis='y')
    
    # Panel E: Effect sizes
    ax_e = fig.add_subplot(gs[1, 1])
    
    effect_sizes = [results_dict[m]['cohens_d'] for m in methods]
    bars = ax_e.bar(range(len(methods)), effect_sizes, alpha=0.7,
                   color=['darkblue' if abs(d) > 1.2 else 'steelblue' if abs(d) > 0.8 else 'lightblue'
                          for d in effect_sizes],
                   edgecolor='black', linewidth=1.5)
    
    ax_e.axhline(0, color='black', linestyle='-', linewidth=1)
    ax_e.axhline(0.8, color='gray', linestyle='--', alpha=0.5)
    ax_e.axhline(-0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
    ax_e.set_xticks(range(len(methods)))
    ax_e.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=10)
    ax_e.set_ylabel("Cohen's d", fontsize=12, fontweight='bold')
    ax_e.set_title('E. Effect Sizes', fontsize=14, fontweight='bold')
    ax_e.legend(fontsize=10)
    ax_e.grid(True, alpha=0.3, axis='y')
    
    # Panel F: Example dose-response curve showing correction
    ax_f = fig.add_subplot(gs[1, 2:])
    
    # Pick a representative memory and naive experiment
    mem_rep = data['memory'][0]
    naive_rep = data['naive'][0]
    peptide = '6T'  # Mid-affinity peptide
    
    # Plot raw
    conc_mem = mem_rep['corrected']['concentrations']
    conc_naive = naive_rep['corrected']['concentrations']
    
    ax_f.semilogx(conc_mem, mem_rep['corrected']['raw'][peptide], 
                 'o-', color=colors['memory'], linewidth=2, markersize=8,
                 label='Memory (raw)', alpha=0.7)
    ax_f.semilogx(conc_naive, naive_rep['corrected']['raw'][peptide],
                 's-', color=colors['naive'], linewidth=2, markersize=8,
                 label='Naive (raw)', alpha=0.7)
    
    # Plot corrected
    ax_f.semilogx(conc_mem, mem_rep['corrected']['bg_subtract'][peptide],
                 'o--', color=colors['memory'], linewidth=2, markersize=6,
                 label='Memory (baseline-corrected)', alpha=0.9)
    ax_f.semilogx(conc_naive, naive_rep['corrected']['bg_subtract'][peptide],
                 's--', color=colors['naive'], linewidth=2, markersize=6,
                 label='Naive (baseline-corrected)', alpha=0.9)
    
    ax_f.axhline(threshold, color='red', linestyle=':', linewidth=2, 
                alpha=0.5, label=f'P{threshold} threshold')
    ax_f.set_xlabel('Peptide Concentration (μM)', fontsize=12, fontweight='bold')
    ax_f.set_ylabel('CD69+ cells (%)', fontsize=12, fontweight='bold')
    ax_f.set_title(f'F. Example: Peptide {peptide} Baseline Correction', 
                  fontsize=14, fontweight='bold')
    ax_f.legend(fontsize=9, loc='best')
    ax_f.grid(True, alpha=0.3)
    
    # Panel G: Comparison table
    ax_g = fig.add_subplot(gs[2, :])
    ax_g.axis('off')
    
    comparison_text = f"""
BASELINE CORRECTION IMPACT SUMMARY (P{threshold} threshold)

METHOD COMPARISON:
"""
    
    for method in methods:
        if method in results_dict:
            r = results_dict[method]
            comparison_text += f"""
{method.upper()}:
  Sample size: n={r['n_memory']} vs n={r['n_naive']}
  Memory α: {r['memory_mean']:.3f} ± {r['memory_std']:.3f}
  Naive α:  {r['naive_mean']:.3f} ± {r['naive_std']:.3f}
  Δα = {r['delta_alpha']:.3f}, p = {r['t_pval']:.3f}, d = {r['cohens_d']:.2f}
  Result: {'SIGNIFICANT' if r['t_pval'] < 0.05 else 'MARGINAL' if r['t_pval'] < 0.10 else 'NOT SIGNIFICANT'}
"""
    
    # Add baseline statistics
    mem_base_mean = baseline_df[baseline_df['condition'] == 'memory']['baseline'].mean()
    naive_base_mean = baseline_df[baseline_df['condition'] == 'naive']['baseline'].mean()
    
    comparison_text += f"""

BASELINE COMPARISON:
  Memory baseline: {mem_base_mean:.2f}% ± {baseline_df[baseline_df['condition'] == 'memory']['baseline'].std():.2f}%
  Naive baseline:  {naive_base_mean:.2f}% ± {baseline_df[baseline_df['condition'] == 'naive']['baseline'].std():.2f}%
  Difference: {mem_base_mean - naive_base_mean:.2f}%
  t-test: p = {stats.ttest_ind(baseline_df[baseline_df['condition'] == 'memory']['baseline'].values, baseline_df[baseline_df['condition'] == 'naive']['baseline'].values)[1]:.4f}

INTERPRETATION:
  {'✓ Baseline correction CHANGES conclusions significantly' if abs(results_dict.get('raw', {}).get('delta_alpha', 0) - results_dict.get('bg_subtract', {}).get('delta_alpha', 0)) > 0.3 else
   '✓ Baseline correction has MODERATE impact on results' if abs(results_dict.get('raw', {}).get('delta_alpha', 0) - results_dict.get('bg_subtract', {}).get('delta_alpha', 0)) > 0.1 else
   '✓ Results are ROBUST to baseline correction'}
  
  {'✓ Memory cells have significantly elevated baseline' if p_val < 0.05 else '~ Memory cells have moderately elevated baseline'}
  
RECOMMENDATION:
  {'Use BASELINE-CORRECTED analysis for publication' if abs(results_dict.get('raw', {}).get('delta_alpha', 0) - results_dict.get('bg_subtract', {}).get('delta_alpha', 0)) > 0.2 else
   'Results are similar; either raw or corrected acceptable'}
  
  Justification: Memory cells are pre-activated (higher basal CD69), which
  confounds dose-response if not corrected. Background subtraction isolates
  peptide-specific response from basal activation state.
"""
    
    ax_g.text(0.05, 0.95, comparison_text, fontsize=10, family='monospace',
             verticalalignment='top', transform=ax_g.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main_baseline_corrected_analysis():
    """
    Run complete baseline-corrected analysis
    """
    print("="*80)
    print("BASELINE-CORRECTED DISCRIMINATION POWER ANALYSIS")
    print("="*80)
    print()
    
    # Load data
    print("Loading data...")
    kd_dict = load_kd_values(KD_FILE)
    data = load_data_with_corrections(MEMORY_FILES, NAIVE_FILES)
    print(f"✓ Loaded {len(data['memory'])} memory and {len(data['naive'])} naive experiments\n")
    
    # Analyze baselines
    baseline_df = analyze_baselines(data)
    
    # Compare correction methods
    results_dict = compare_correction_methods(data, kd_dict, threshold=40)
    
    # Visualization
    print("Creating visualization...")
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig = plot_baseline_correction_comparison(results_dict, baseline_df,
                                              save_path=f'{OUTPUT_DIR}/baseline_correction_analysis.png')
    print(f"✓ Saved baseline_correction_analysis.png\n")
    
    # Save results
    baseline_df.to_csv(f'{OUTPUT_DIR}/baseline_statistics.csv', index=False)
    
    comparison_df = pd.DataFrame([
        {
            'method': m,
            'n_memory': r['n_memory'],
            'n_naive': r['n_naive'],
            'memory_alpha': r['memory_mean'],
            'naive_alpha': r['naive_mean'],
            'delta_alpha': r['delta_alpha'],
            'p_value': r['t_pval'],
            'cohens_d': r['cohens_d']
        }
        for m, r in results_dict.items()
    ])
    comparison_df.to_csv(f'{OUTPUT_DIR}/correction_method_comparison.csv', index=False)
    print(f"✓ Saved CSV files\n")
    
    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    return results_dict, baseline_df

if __name__ == "__main__":
    results_dict, baseline_df = main_baseline_corrected_analysis()
