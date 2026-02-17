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

# Add this to your analysis!