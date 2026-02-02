def find_minimal_changes_grid(KD_target, threshold_P=0.5):
    """
    Grid search to find smallest N and tau reduction
    """
    N_ref = 2.67
    tau_ref = 2.8
    L0 = 50
    R0 = 50
    
    # Create fine grid
    N_values = np.linspace(0.5, N_ref, 100)
    tau_values = np.linspace(0.5, tau_ref, 100)
    
    min_change = np.inf
    best_params = None
    
    for N_test in N_values:
        for tau_test in tau_values:
            P = activation_probability_concentration(
                KD_target, N_test, tau_test, L0, R0
            )
            
            if P >= threshold_P:  # Achieves activation
                # Calculate total parameter reduction
                total_reduction = (N_ref - N_test) + (tau_ref - tau_test)
                
                if total_reduction < min_change:
                    min_change = total_reduction
                    best_params = (N_test, tau_test, P)
    
    if best_params:
        N_opt, tau_opt, P_opt = best_params
        print(f"\nMinimal changes for KD = {KD_target} µM:")
        print(f"  N: {N_ref:.2f} → {N_opt:.2f} (Δ = {N_ref - N_opt:.2f})")
        print(f"  τ: {tau_ref:.2f} → {tau_opt:.2f} (Δ = {tau_ref - tau_opt:.2f})")
        print(f"  Total reduction: {min_change:.2f}")
        print(f"  P(activation): {P_opt:.1%}")
        
        return N_opt, tau_opt, P_opt
    else:
        print(f"No solution found for KD = {KD_target} µM")
        return None

# Test for different targets below negative selection
for KD in [165, 150, 130, 100, 70]:
    find_minimal_changes_grid(KD)