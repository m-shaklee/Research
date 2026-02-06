
import plotly.graph_objs as go
import numpy as np
import os

def activation_probability_concentration(
    KD_uM, N, tau_total, L0, R0, threshold_CN=1.0, kon=1e5
):
    kp = 1.0 / tau_total
    KD_M = KD_uM * 1e-6
    koff = KD_M * kon

    term = L0 + R0 + koff / kon
    disc = np.maximum(term**2 - 4 * L0 * R0, 0)
    C_tot = (term - np.sqrt(disc)) / 2
    C_N = C_tot * (1 + koff / kp) ** (-N)

    return np.clip(1 - np.exp(-C_N / threshold_CN), 0, 1)


def find_KD_at_threshold(N, tau, L0, R0, P_threshold=0.1):
    """
    Find the maximum KD that achieves at least P_threshold activation probability
    """
    KD_search = np.logspace(-1, 3, 1000)  # 0.1 to 1000 µM
    P_values = activation_probability_concentration(KD_search, N, tau, L0, R0)
    
    # Find where probability drops below threshold
    valid_idx = np.where(P_values >= P_threshold)[0]
    
    if len(valid_idx) == 0:
        return 0  # Can't activate anything
    
    return KD_search[valid_idx[-1]]  # Maximum KD that meets threshold


def calculate_affinity_range_expansion(
    N_ref=2.67,
    tau_ref=2.8,
    N_mem=None,
    tau_mem=None,
    L0=200,
    R0=200,
    P_threshold=0.1
):
    """
    Calculate the fold expansion in detectable affinity range
    """
    # Find maximum KD for naive cells
    KD_max_naive = find_KD_at_threshold(N_ref, tau_ref, L0, R0, P_threshold)
    
    # Find maximum KD for memory cells
    KD_max_memory = find_KD_at_threshold(N_mem, tau_mem, L0, R0, P_threshold)
    
    # Calculate expansion
    fold_expansion = KD_max_memory / KD_max_naive
    percent_expansion = (fold_expansion - 1) * 100
    
    return {
        'KD_max_naive': KD_max_naive,
        'KD_max_memory': KD_max_memory,
        'fold_expansion': fold_expansion,
        'percent_expansion': percent_expansion
    }


def find_minimal_changes_for_negsel_activation(
    KD_negsel=170,  # negative selection threshold
    KD_baseline=50,  # what the naive cell could activate
    threshold_P=0.5,  # activation threshold
    N_ref=2.67,
    tau_ref=2.8,
    L0=200,
    R0=200
):
    """
    Find minimal N and tau reductions that enable activation of 
    antigens ABOVE negative selection threshold (KD > 170 µM)
    while maintaining same activation probability as baseline
    """
    
    # Step 1: Get the target activation probability from baseline
    P_target = activation_probability_concentration(
        KD_baseline, N_ref, tau_ref, L0, R0
    )
    
    print(f"Target: Activate KD={KD_negsel} µM with P={P_target:.1%}")
    print(f"(same activation as naive cells achieve for KD={KD_baseline} µM)\n")
    
    # Step 2: Grid search for minimal parameter changes
    N_values = np.linspace(0.5, N_ref, 150)
    tau_values = np.linspace(0.5, tau_ref, 150)
    
    solutions = []
    
    for N_test in N_values:
        for tau_test in tau_values:
            # Check if these parameters activate the neg-selected antigen
            P_negsel = activation_probability_concentration(
                KD_negsel, N_test, tau_test, L0, R0
            )
            
            # Does it achieve target activation?
            if P_negsel >= P_target * 0.95:  # within 5% of target
                delta_N = N_ref - N_test
                delta_tau = tau_ref - tau_test
                total_change = delta_N + delta_tau
                
                solutions.append({
                    'N': N_test,
                    'tau': tau_test,
                    'delta_N': delta_N,
                    'delta_tau': delta_tau,
                    'total_change': total_change,
                    'P_achieved': P_negsel
                })
    
    if not solutions:
        print(f"❌ No parameter combinations found that activate KD={KD_negsel} µM")
        return None, None
    
    # Find minimal total change
    solutions.sort(key=lambda x: x['total_change'])
    best = solutions[0]
    
    print("✓ Minimal parameter changes found:")
    print(f"  N: {N_ref:.3f} → {best['N']:.3f} (ΔN = {best['delta_N']:.3f})")
    print(f"  τ: {tau_ref:.3f} → {best['tau']:.3f} (Δτ = {best['delta_tau']:.3f})")
    print(f"  Total reduction: {best['total_change']:.3f}")
    print(f"  P(activation): {best['P_achieved']:.1%}")
    
    # Calculate affinity range expansion
    expansion = calculate_affinity_range_expansion(
        N_ref=N_ref,
        tau_ref=tau_ref,
        N_mem=best['N'],
        tau_mem=best['tau'],
        L0=L0,
        R0=R0,
        P_threshold=0.1
    )
    
    print(f"\n📊 Affinity Range Expansion Analysis:")
    print(f"  Max KD (naive, P≥10%): {expansion['KD_max_naive']:.1f} µM")
    print(f"  Max KD (memory, P≥10%): {expansion['KD_max_memory']:.1f} µM")
    print(f"  Fold expansion: {expansion['fold_expansion']:.2f}×")
    print(f"  Percent expansion: {expansion['percent_expansion']:.1f}%")
    
    return best, solutions


# Test across range of neg-selected antigens
print("="*60)
print("FINDING MINIMAL CHANGES TO ACTIVATE NEG-SELECTED ANTIGENS")
print("="*60)

all_results = []

for KD_target in [175, 200, 250, 300]:
    print(f"\n{'='*60}")
    best, solutions = find_minimal_changes_for_negsel_activation(
        KD_negsel=KD_target,
        KD_baseline=100
    )
    
    if best:
        all_results.append({
            'KD_target': KD_target,
            'best_params': best
        })

# Summary across all targets
if all_results:
    print("\n" + "="*60)
    print("SUMMARY: Affinity Range Expansion Across Targets")
    print("="*60)
    
    expansions = []
    for result in all_results:
        expansion = calculate_affinity_range_expansion(
            N_mem=result['best_params']['N'],
            tau_mem=result['best_params']['tau'],
            P_threshold=0.1
        )
        expansions.append(expansion['percent_expansion'])
    
    avg_expansion = np.mean(expansions)
    min_expansion = np.min(expansions)
    max_expansion = np.max(expansions)
    
    print(f"\nAcross all target KDs (175-300 µM):")
    print(f"  Average expansion: {avg_expansion:.1f}%")
    print(f"  Range: {min_expansion:.1f}% - {max_expansion:.1f}%")
    print(f"\n✓ These parameter reductions predict {min_expansion:.0f}-{max_expansion:.0f}% expansion")
    print(f"  in detectable affinity range, providing quantitative support for")
    print(f"  antigenic landscape expansion.")