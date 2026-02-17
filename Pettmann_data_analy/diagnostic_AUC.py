"""
DIAGNOSTIC AND ALTERNATIVE METRICS ANALYSIS

This script:
1. Shows AUC values and plots them
2. Investigates why experiments fail P15 calculation
3. Tries alternative activation thresholds (P5, P10, P20, P25)
4. Provides diagnostic plots for failed experiments
5. Recommends best approach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style("ticks")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_kd_values(kd_file):
    """Load and parse KD values"""
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

def load_pettmann_data(files_dict):
    """Load Pettmann source data files"""
    data = {}
    for condition, files in files_dict.items():
        data[condition] = []
        for file in files:
            df = pd.read_csv(file, encoding='utf-8-sig')
            filename = file.split('/')[-1]
            date = filename[:6]
            day = filename.split('d')[1].split('_')[0] if 'd' in filename else '0'
            data[condition].append({
                'date': date,
                'day': day,
                'data': df,
                'file': filename
            })
    return data

def calculate_px(concentrations, responses, threshold_pct):
    """
    Calculate Px: concentration that elicits X% activation
    More flexible than just P15
    """
    mask = ~(np.isnan(concentrations) | np.isnan(responses))
    conc = concentrations[mask]
    resp = responses[mask]
    
    if len(resp) < 2:
        return np.nan, "too_few_points"
    if resp.max() < threshold_pct:
        return np.nan, f"max_response_too_low_{resp.max():.1f}"
    if resp.min() > threshold_pct:
        return np.nan, f"min_response_too_high_{resp.min():.1f}"
    
    log_conc = np.log10(conc)
    f = interp1d(resp, log_conc, kind='linear', 
                 bounds_error=False, fill_value='extrapolate')
    log_px = f(threshold_pct)
    return 10**log_px, "success"

# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

def diagnose_missing_experiments(data_dict, kd_dict, threshold=15):
    """
    Diagnose why experiments fail to produce P15 values
    """
    print("="*80)
    print(f"DIAGNOSING MISSING EXPERIMENTS (P{threshold} threshold)")
    print("="*80)
    print()
    
    all_results = []
    
    for condition, replicates in data_dict.items():
        for rep in replicates:
            df = rep['data']
            concentrations = df.iloc[:, 0].values
            
            experiment_results = {
                'condition': condition,
                'experiment': rep['date'],
                'file': rep['file'],
                'total_peptides': len(df.columns) - 1,
                'valid_p15': 0,
                'too_low': 0,
                'too_high': 0,
                'too_few_points': 0,
                'valid_peptides': []
            }
            
            for peptide in df.columns[1:]:
                if peptide not in kd_dict:
                    continue
                
                responses = df[peptide].values
                px, reason = calculate_px(concentrations, responses, threshold)
                
                if reason == "success":
                    experiment_results['valid_p15'] += 1
                    experiment_results['valid_peptides'].append(peptide)
                elif "too_low" in reason:
                    experiment_results['too_low'] += 1
                elif "too_high" in reason:
                    experiment_results['too_high'] += 1
                elif reason == "too_few_points":
                    experiment_results['too_few_points'] += 1
            
            all_results.append(experiment_results)
    
    results_df = pd.DataFrame(all_results)
    
    print("SUMMARY BY EXPERIMENT:")
    print("-"*80)
    for _, row in results_df.iterrows():
        status = "✓ CAN FIT" if row['valid_p15'] >= 3 else "✗ CANNOT FIT"
        print(f"{status} {row['condition']:8s} {row['experiment']} ({row['file']})")
        print(f"        Valid P{threshold}: {row['valid_p15']}/{row['total_peptides']} peptides")
        if row['valid_p15'] > 0:
            print(f"        Peptides: {', '.join(row['valid_peptides'])}")
        if row['too_low'] > 0:
            print(f"        Too low (max < {threshold}%): {row['too_low']} peptides")
        if row['too_high'] > 0:
            print(f"        Too high (min > {threshold}%): {row['too_high']} peptides")
        print()
    
    return results_df

def try_alternative_thresholds(data_dict, kd_dict):
    """
    Try different activation thresholds to see which works best
    """
    print("="*80)
    print("TESTING ALTERNATIVE ACTIVATION THRESHOLDS")
    print("="*80)
    print()
    
    thresholds = [5, 10, 15, 20, 25, 30]
    results = []
    
    for threshold in thresholds:
        n_experiments_fit = 0
        total_p15_values = 0
        
        for condition, replicates in data_dict.items():
            for rep in replicates:
                df = rep['data']
                concentrations = df.iloc[:, 0].values
                valid_count = 0
                
                for peptide in df.columns[1:]:
                    if peptide not in kd_dict:
                        continue
                    
                    responses = df[peptide].values
                    px, reason = calculate_px(concentrations, responses, threshold)
                    
                    if reason == "success":
                        valid_count += 1
                        total_p15_values += 1
                
                if valid_count >= 3:  # Need at least 3 peptides to fit α
                    n_experiments_fit += 1
        
        results.append({
            'threshold': threshold,
            'experiments_fit': n_experiments_fit,
            'total_valid_values': total_p15_values,
            'pct_experiments': n_experiments_fit / 12 * 100
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()
    
    best = results_df.iloc[results_df['experiments_fit'].argmax()]
    print(f"RECOMMENDATION: Use P{int(best['threshold'])} threshold")
    print(f"  → Fits {int(best['experiments_fit'])}/12 experiments ({best['pct_experiments']:.0f}%)")
    print(f"  → Generates {int(best['total_valid_values'])} valid Px values")
    print()
    
    return results_df

# ============================================================================
# VISUALIZE FAILED EXPERIMENTS
# ============================================================================

def plot_failed_experiments(data_dict, kd_dict, threshold=15):
    """
    Plot dose-response curves for experiments that failed to fit
    """
    print("="*80)
    print("PLOTTING FAILED EXPERIMENTS")
    print("="*80)
    print()
    
    failed_experiments = []
    
    # Identify failed experiments
    for condition, replicates in data_dict.items():
        for rep in replicates:
            df = rep['data']
            concentrations = df.iloc[:, 0].values
            valid_count = 0
            
            for peptide in df.columns[1:]:
                if peptide not in kd_dict:
                    continue
                responses = df[peptide].values
                px, reason = calculate_px(concentrations, responses, threshold)
                if reason == "success":
                    valid_count += 1
            
            if valid_count < 3:
                failed_experiments.append(rep)
    
    if len(failed_experiments) == 0:
        print("No failed experiments to plot!")
        return
    
    print(f"Plotting {len(failed_experiments)} failed experiments...\n")
    
    n_failed = len(failed_experiments)
    fig, axes = plt.subplots(n_failed, 1, figsize=(12, 4*n_failed))
    if n_failed == 1:
        axes = [axes]
    
    for idx, rep in enumerate(failed_experiments):
        ax = axes[idx]
        df = rep['data']
        concentrations = df.iloc[:, 0].values
        
        for peptide in df.columns[1:]:
            if peptide not in kd_dict:
                continue
            
            responses = df[peptide].values
            mask = ~np.isnan(responses)
            
            ax.semilogx(concentrations[mask], responses[mask], 
                       'o-', alpha=0.7, label=peptide, markersize=6)
        
        ax.axhline(threshold, color='red', linestyle='--', alpha=0.5, 
                  linewidth=2, label=f'P{threshold} threshold')
        ax.set_xlabel('Peptide Concentration (μM)', fontsize=11)
        ax.set_ylabel('CD69+ cells (%)', fontsize=11)
        ax.set_title(f"{rep['file']} - FAILED TO FIT", 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    plt.tight_layout()
    return fig

# ============================================================================
# IMPROVED AUC ANALYSIS AND VISUALIZATION
# ============================================================================

def comprehensive_auc_analysis(data_dict, kd_dict):
    """
    Complete AUC analysis with detailed statistics and visualization
    """
    print("="*80)
    print("COMPREHENSIVE AREA UNDER CURVE (AUC) ANALYSIS")
    print("="*80)
    print()
    
    auc_results = []
    
    # Calculate AUC for each peptide in each experiment
    for peptide in kd_dict.keys():
        peptide_data = {
            'peptide': peptide,
            'kd': kd_dict[peptide]['kd_mean'],
            'memory_aucs': [],
            'naive_aucs': []
        }
        
        for condition, replicates in data_dict.items():
            for rep in replicates:
                df = rep['data']
                
                if peptide not in df.columns:
                    continue
                
                concentrations = df.iloc[:, 0].values
                responses = df[peptide].values
                
                # Calculate AUC in log space
                mask = ~np.isnan(responses)
                if mask.sum() >= 2:
                    log_conc = np.log10(concentrations[mask])
                    resp = responses[mask]
                    auc = np.trapz(resp, log_conc)
                    
                    if condition == 'memory':
                        peptide_data['memory_aucs'].append(auc)
                    else:
                        peptide_data['naive_aucs'].append(auc)
        
        # Statistical comparison
        if len(peptide_data['memory_aucs']) >= 2 and len(peptide_data['naive_aucs']) >= 2:
            mem_aucs = np.array(peptide_data['memory_aucs'])
            naive_aucs = np.array(peptide_data['naive_aucs'])
            
            t_stat, t_pval = stats.ttest_ind(mem_aucs, naive_aucs)
            u_stat, u_pval = stats.mannwhitneyu(mem_aucs, naive_aucs, 
                                               alternative='two-sided')
            
            # Effect size
            pooled_std = np.sqrt((np.var(mem_aucs) + np.var(naive_aucs)) / 2)
            cohens_d = (np.mean(mem_aucs) - np.mean(naive_aucs)) / pooled_std if pooled_std > 0 else np.nan
            
            auc_results.append({
                'peptide': peptide,
                'kd': kd_dict[peptide]['kd_mean'],
                'memory_n': len(mem_aucs),
                'naive_n': len(naive_aucs),
                'memory_auc_mean': np.mean(mem_aucs),
                'memory_auc_std': np.std(mem_aucs),
                'naive_auc_mean': np.mean(naive_aucs),
                'naive_auc_std': np.std(naive_aucs),
                'fold_change': np.mean(mem_aucs) / np.mean(naive_aucs),
                't_stat': t_stat,
                't_pval': t_pval,
                'u_pval': u_pval,
                'cohens_d': cohens_d
            })
    
    auc_df = pd.DataFrame(auc_results)
    auc_df = auc_df.sort_values('kd')
    
    # Print results
    print("AUC COMPARISON BY PEPTIDE:")
    print("-"*80)
    for _, row in auc_df.iterrows():
        sig_marker = "***" if row['t_pval'] < 0.001 else "**" if row['t_pval'] < 0.01 else "*" if row['t_pval'] < 0.05 else "ns"
        print(f"{row['peptide']:4s} (KD={row['kd']:7.1f} μM):")
        print(f"  Memory: {row['memory_auc_mean']:7.1f} ± {row['memory_auc_std']:6.1f} (n={row['memory_n']})")
        print(f"  Naive:  {row['naive_auc_mean']:7.1f} ± {row['naive_auc_std']:6.1f} (n={row['naive_n']})")
        print(f"  FC = {row['fold_change']:.3f}, p = {row['t_pval']:.3f} {sig_marker}, d = {row['cohens_d']:.2f}")
        print()
    
    # Test for affinity-dependent pattern
    if len(auc_df) > 2:
        corr, pval = stats.spearmanr(np.log10(auc_df['kd']), auc_df['fold_change'])
        print("LANDSCAPE EXPANSION TEST:")
        print("-"*80)
        print(f"Correlation (log KD vs fold-change): ρ = {corr:.3f}, p = {pval:.3f}")
        if corr < 0:
            print("→ NEGATIVE correlation: Memory enhancement STRONGER for low-affinity")
            print("  This SUPPORTS landscape expansion!")
        else:
            print("→ POSITIVE correlation: Memory enhancement STRONGER for high-affinity")
        print()
    
    return auc_df

def plot_auc_results(auc_df, save_path=None):
    """
    Create comprehensive AUC visualization
    """
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    colors = {'memory': '#e74c3c', 'naive': '#3498db'}
    
    # Panel 1: AUC by peptide (sorted by KD)
    ax1 = fig.add_subplot(gs[0, :2])
    
    x_pos = np.arange(len(auc_df))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, auc_df['memory_auc_mean'], width,
                    yerr=auc_df['memory_auc_std'], label='Memory',
                    color=colors['memory'], alpha=0.7, capsize=5)
    bars2 = ax1.bar(x_pos + width/2, auc_df['naive_auc_mean'], width,
                    yerr=auc_df['naive_auc_std'], label='Naive',
                    color=colors['naive'], alpha=0.7, capsize=5)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(auc_df['peptide'], fontsize=11)
    ax1.set_ylabel('Area Under Curve (AUC)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Peptide (sorted by KD)', fontsize=13, fontweight='bold')
    ax1.set_title('Full Dose-Response: Area Under Curve Comparison', 
                 fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add significance markers
    for i, (_, row) in enumerate(auc_df.iterrows()):
        if row['t_pval'] < 0.05:
            y_pos = max(row['memory_auc_mean'] + row['memory_auc_std'],
                       row['naive_auc_mean'] + row['naive_auc_std']) + 5
            if row['t_pval'] < 0.001:
                marker = '***'
            elif row['t_pval'] < 0.01:
                marker = '**'
            else:
                marker = '*'
            ax1.text(i, y_pos, marker, ha='center', fontsize=16, fontweight='bold')
    
    # Panel 2: Fold-change vs KD (LANDSCAPE EXPANSION TEST)
    ax2 = fig.add_subplot(gs[0, 2])
    
    ax2.scatter(auc_df['kd'], auc_df['fold_change'], s=200, alpha=0.7,
               c='purple', edgecolors='black', linewidths=2)
    
    for _, row in auc_df.iterrows():
        ax2.text(row['kd']*1.1, row['fold_change'], row['peptide'],
                fontsize=10, fontweight='bold')
    
    # Fit trend line
    log_kd = np.log10(auc_df['kd'].values)
    log_fc = np.log10(auc_df['fold_change'].values)
    slope, intercept = np.polyfit(log_kd, log_fc, 1)
    kd_range = np.logspace(np.log10(auc_df['kd'].min()), 
                          np.log10(auc_df['kd'].max()), 100)
    fc_fit = 10**(slope * np.log10(kd_range) + intercept)
    ax2.plot(kd_range, fc_fit, 'k--', alpha=0.5, linewidth=2,
            label=f'Trend: slope={slope:.2f}')
    
    ax2.axhline(1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('KD (μM) - Lower = Higher Affinity', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Fold-Change (Memory/Naive)', fontsize=11, fontweight='bold')
    ax2.set_title('Landscape Expansion Test\n(Negative slope = supports)', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Add correlation
    corr, pval = stats.spearmanr(np.log10(auc_df['kd']), auc_df['fold_change'])
    ax2.text(0.05, 0.95, f'ρ = {corr:.3f}\np = {pval:.3f}',
            transform=ax2.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='top')
    
    # Panel 3: Effect sizes
    ax3 = fig.add_subplot(gs[1, :])
    
    bars = ax3.barh(range(len(auc_df)), auc_df['cohens_d'], 
                    alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Color by significance
    for i, (_, row) in enumerate(auc_df.iterrows()):
        if row['t_pval'] < 0.05:
            bars[i].set_color('red')
        else:
            bars[i].set_color('gray')
    
    ax3.set_yticks(range(len(auc_df)))
    ax3.set_yticklabels(auc_df['peptide'], fontsize=11)
    ax3.axvline(0, color='black', linestyle='-', linewidth=1.5)
    ax3.axvline(0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
    ax3.axvline(-0.8, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel("Cohen's d (positive = memory > naive)", fontsize=12, fontweight='bold')
    ax3.set_title("Effect Sizes by Peptide (Red = significant p<0.05)", 
                 fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# ============================================================================
# MAIN DIAGNOSTIC ANALYSIS
# ============================================================================

def main_diagnostic():
    """Run comprehensive diagnostic analysis"""
    
    print("="*80)
    print("COMPREHENSIVE DIAGNOSTIC AND ALTERNATIVE METRICS ANALYSIS")
    print("="*80)
    print()
    
    # Load data
    # kd_dict = load_kd_values('/home/maddie/Desktop/Projects/Research/Pettmann_data_analy/elife-67092-fig1-data2-v3.csv')
    kd_dict = load_kd_values('/Users/maddie/Desktop/Projects/Research/Pettmann_data_analy/elife-67092-fig1-data2-v3.csv')
    
    # data_files = {
    #     'memory': [
    #         '/home/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 memory and DC/190722d1 memory CD69 Pos.csv',
    #         '/home/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 memory and DC/190807d1 memory CD69 Pos.csv',
    #         '/home/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 memory and DC/190812d4 memory CD69 Pos.csv',
    #         '/home/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 memory and DC/191019d1 memory CD69 Pos.csv',
    #         '/home/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 memory and DC/191022d1 memory CD69 Pos.csv',
    #         '/home/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 memory and DC/191027d1 memory CD69 Pos.csv'
    #     ],
    #     'naive': [
    #         '/home/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 naive and DC/190708d3 naive CD69 Pos.csv',
    #         '/home/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 naive and DC/190715d1 naive CD69 Pos.csv',
    #         '/home/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 naive and DC/190715d2 naive CD69 Pos.csv',
    #         '/home/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 naive and DC/190715d4 naive CD69 Pos.csv',
    #         '/home/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 naive and DC/190807d2 naive CD69 Pos.csv',
    #         '/home/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 naive and DC/190816d8 naive CD69 Pos.csv'
    #     ]
    # }
    data_files = {
        'memory': [
            '/Users/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 memory and DC/190722d1 memory CD69 Pos.csv',
            '/Users/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 memory and DC/190807d1 memory CD69 Pos.csv',
            '/Users/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 memory and DC/190812d4 memory CD69 Pos.csv',
            '/Users/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 memory and DC/191019d1 memory CD69 Pos.csv',
            '/Users/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 memory and DC/191022d1 memory CD69 Pos.csv',
            '/Users/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 memory and DC/191027d1 memory CD69 Pos.csv'
        ],
        'naive': [
            '/Users/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 naive and DC/190708d3 naive CD69 Pos.csv',
            '/Users/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 naive and DC/190715d1 naive CD69 Pos.csv',
            '/Users/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 naive and DC/190715d2 naive CD69 Pos.csv',
            '/Users/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 naive and DC/190715d4 naive CD69 Pos.csv',
            '/Users/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 naive and DC/190807d2 naive CD69 Pos.csv',
            '/Users/maddie/Desktop/Projects/Research/Pettmann_data_analy/Figure 2 - 1G4 naive and DC/190816d8 naive CD69 Pos.csv'
        ]
    }
    
    data = load_pettmann_data(data_files)
    
    # 1. Diagnose missing experiments
    diag_df = diagnose_missing_experiments(data, kd_dict, threshold=15)
    
    # 2. Try alternative thresholds
    threshold_df = try_alternative_thresholds(data, kd_dict)
    
    # 3. Plot failed experiments
    fig1 = plot_failed_experiments(data, kd_dict, threshold=15)
    if fig1:
        fig1.savefig('outputs/failed_experiments_diagnostic.png', dpi=300, bbox_inches='tight')
        print("✓ Saved failed experiments plot\n")
    
    # 4. Comprehensive AUC analysis
    auc_df = comprehensive_auc_analysis(data, kd_dict)
    
    # 5. Plot AUC results
    fig2 = plot_auc_results(auc_df, save_path='outputs/comprehensive_auc_analysis.png')
    print("✓ Saved comprehensive AUC plot\n")
    
    # Save results
    auc_df.to_csv('outputs/auc_detailed_results.csv', index=False)
    threshold_df.to_csv('outputs/threshold_comparison.csv', index=False)
    print("✓ Saved CSV files\n")
    
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()
    print("Based on this analysis:")
    print()
    print("1. USE AUC INSTEAD OF P15:")
    print("   - AUC uses full curve information")
    print("   - Not sensitive to single threshold")
    print("   - Works for ALL experiments")
    print("   - More statistical power")
    print()
    print("2. IF USING Px, TRY LOWER THRESHOLD:")
    best_threshold = threshold_df.iloc[threshold_df['experiments_fit'].argmax()]
    print(f"   - P{int(best_threshold['threshold'])} gives best coverage")
    print(f"   - Fits {int(best_threshold['experiments_fit'])}/12 experiments")
    print()
    print("3. EXCLUDE PROBLEMATIC PEPTIDES:")
    print("   - Some peptides consistently fail to cross threshold")
    print("   - Consider peptide-by-peptide analysis instead of α fitting")
    print()
    print("4. FOCUS ON COMPLEMENTARY ANALYSES:")
    print("   - AUC comparison (done above)")
    print("   - Per-peptide statistical tests")
    print("   - Effect sizes rather than just p-values")
    print()
    
    return auc_df, threshold_df

if __name__ == "__main__":
    auc_df, threshold_df = main_diagnostic()