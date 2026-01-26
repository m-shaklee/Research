import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import os

# =========================
# Models
# =========================

def activation_probability_simple(KD_uM, N, tau_total, kon=1e5):
    """
    Vectorized version accepting numpy arrays or scalars.
    """
    KD = np.asarray(KD_uM, dtype=float)
    KD_M = KD * 1e-6
    koff = KD_M * kon
    # avoid division by zero for KD==0 -> set extremely small
    koff = np.where(koff == 0, 1e-300, koff)
    tau_b = 1.0 / koff
    tau_step = tau_total / N
    p = (tau_b / (tau_b + tau_step)) ** N
    p = np.clip(p, 0.0, 1.0)
    return p

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


def find_equivalent_KD(KD_threshold, N_new, tau_new, N_old, tau_old, model='simple', **kwargs):
    """
    Find the KD that produces the same activation probability as KD_threshold
    with old parameters (N_old, tau_old) when using new parameters (N_new, tau_new)
    """
    # Get target probability with threshold KD and old parameters
    if model == 'simple':
        P_target = activation_probability_simple(KD_threshold, N_old, tau_old)
    else:
        P_target = activation_probability_concentration(
            KD_threshold, N_old, tau_old, 
            kwargs.get('L0', 50), kwargs.get('R0', 50)
        )
    
    # Search for KD with new parameters that gives same probability
    KD_search = np.logspace(-2, 3, 2000)  # 0.01 to 1000 µM
    
    if model == 'simple':
        P_search = activation_probability_simple(KD_search, N_new, tau_new)
    else:
        P_search = activation_probability_concentration(
            KD_search, N_new, tau_new,
            kwargs.get('L0', 50), kwargs.get('R0', 50)
        )
    
    # Find closest match
    idx = np.argmin(np.abs(P_search - P_target))
    return KD_search[idx]


# =========================
# Dash app
# =========================

app = dash.Dash(__name__)

app.layout = html.Div([

    html.H2("Kinetic Proofreading: Memory Cell Priming (Reduced KP Parameters)"),
    
    html.Div([
        html.H4("Model Equations"),
        html.Div(id='equation-display', style={
            'padding': '15px',
            'backgroundColor': '#f0f0f0',
            'borderRadius': '5px',
            'fontFamily': 'monospace',
            'marginBottom': '20px'
        })
    ]),

    dcc.RadioItems(
        id='kp_model',
        options=[
            {'label': 'Simplified KP', 'value': 'simple'},
            {'label': 'KP with concentration', 'value': 'concentration'}
        ],
        value='simple',
        inline=True
    ),

    html.Br(),

    html.Label("τ range (integration time) - Memory has LOWER τ"),
    dcc.RangeSlider(
        id='tau_range', min=0.5, max=10, step=0.25,
        value=[0.5, 2.0],
        marks={i: str(i) for i in range(0, 11, 2)}
    ),

    html.Label("N range (proofreading depth) - Memory has LOWER N"),
    dcc.RangeSlider(
        id='N_range', min=1, max=8, step=1,
        value=[1, 2],
        marks={i: str(i) for i in range(1, 9)}
    ),

    html.Br(),
    html.H4("Reference Parameters (Naive State - Higher KP)"),
    
    html.Label("Reference N (proofreading steps) - Naive"),
    dcc.Slider(
        id='N_ref',
        min=1, max=8, step=1, value=4,
        marks={i: str(i) for i in range(1, 9)}
    ),
    
    html.Label("Reference τ (integration time) - Naive"),
    dcc.Slider(
        id='tau_ref',
        min=0.5, max=10, step=0.25, value=4.0,
        marks={i: str(i) for i in range(0, 11, 2)}
    ),

    html.Label("Activation threshold KD (µM) - for heatmaps"),
    dcc.Slider(
        id='KD_threshold',
        min=1, max=300, step=1, value=50,
        marks={1: '1', 50: '50', 100: '100', 300: '300'}
    ),

    html.Br(),
    html.H4("Additional Parameters"),

    html.Label("Ligand concentration L₀ (concentration KP only)"),
    dcc.Slider(id='L0', min=1, max=200, step=5, value=50),

    html.Label("Receptor count R₀"),
    dcc.Slider(id='R0', min=1, max=200, step=5, value=50),

    html.Br(),

    html.Div([
        html.H4("Single-Cell Activation Probability"),
        dcc.Graph(id='activation-curve', style={'height': '600px'})
    ]),

    html.Div([
        html.H4("Quantitative Summary"),
        html.Div(id='quantitative-summary', style={
            'padding': '15px',
            'backgroundColor': '#e8f4f8',
            'borderRadius': '5px',
            'marginTop': '10px',
            'marginBottom': '20px'
        })
    ]),

    dcc.Graph(id='KD-heatmap', style={'height': '600px'}),
    dcc.Graph(id='KD-difference', style={'height': '600px'}),
    dcc.Graph(id='KD-range-size', style={'height': '600px'})
])

# =========================
# Callbacks
# =========================

@app.callback(
    Output('equation-display', 'children'),
    Input('kp_model', 'value')
)
def update_equations(model):
    if model == 'simple':
        return html.Div([
            html.P([
                html.Strong("Simplified Kinetic Proofreading Model"),
                html.Br(),
                "Single ligand binding model, no concentration dependence"
            ]),
            html.P([
                html.Strong("Step 1: "), "Calculate k", html.Sub("off"), " from K", html.Sub("D")
            ], style={'marginTop': '10px'}),
            html.P([
                "k", html.Sub("off"), " = K", html.Sub("D"), " × k", html.Sub("on"),
                html.Span(" (where k", style={'marginLeft': '10px'}), html.Sub("on"), " = 10⁵ M⁻¹s⁻¹)"
            ], style={'marginLeft': '20px'}),
            html.P([
                html.Strong("Step 2: "), "Calculate bound time and step time"
            ], style={'marginTop': '10px'}),
            html.P([
                "τ", html.Sub("b"), " = 1 / k", html.Sub("off"),
                html.Span(" (time ligand stays bound)", style={'marginLeft': '10px', 'fontStyle': 'italic', 'fontSize': '12px'})
            ], style={'marginLeft': '20px'}),
            html.P([
                "τ", html.Sub("step"), " = τ / N",
                html.Span(" (time per proofreading step)", style={'marginLeft': '10px', 'fontStyle': 'italic', 'fontSize': '12px'})
            ], style={'marginLeft': '20px'}),
            html.P([
                html.Strong("Step 3: "), "Calculate activation probability"
            ], style={'marginTop': '10px'}),
            html.P([
                html.Strong("P(activation) = "),
                "(τ", html.Sub("b"), " / (τ", html.Sub("b"), " + τ", html.Sub("step"), "))", html.Sup("N")
            ], style={'fontSize': '16px', 'marginTop': '5px', 'marginLeft': '20px', 'backgroundColor': '#ffffcc', 'padding': '5px'}),
            html.P([
                "Interpretation: Ligand must remain bound long enough (τ", html.Sub("b"), ") ",
                "to complete all N proofreading steps (each taking τ", html.Sub("step"), ")"
            ], style={'fontSize': '12px', 'fontStyle': 'italic', 'marginTop': '10px', 'color': '#555'})
        ])
    else:
        return html.Div([
            html.P([
                html.Strong("Concentration-Dependent Kinetic Proofreading Model"),
                html.Br(),
                "From Pettmann et al. (2021) - Equations 2 & 3"
            ]),
            html.P([
                "k", html.Sub("p"), " = 1/τ"
            ], style={'marginTop': '10px'}),
            html.P([
                "C", html.Sub("tot"), " = [L₀ + R₀ + k", html.Sub("off"), "/k", html.Sub("on"), 
                " - √((L₀ + R₀ + k", html.Sub("off"), "/k", html.Sub("on"), ")² - 4L₀R₀)] / 2"
            ]),
            html.P([
                "C", html.Sub("N"), " = C", html.Sub("tot"), " × (1 + k", html.Sub("off"), 
                "/k", html.Sub("p"), ")", html.Sup("-N")
            ]),
            html.P([
                html.Strong("P(activation) = "),
                "1 - exp(-C", html.Sub("N"), "/C", html.Sub("N,threshold"), ")"
            ], style={'fontSize': '16px', 'marginTop': '10px'}),
            html.P([
                "where L₀ = ligand concentration, R₀ = receptor count, ",
                "C", html.Sub("N,threshold"), " = 1.0, k", html.Sub("on"), " = 10⁵ M⁻¹s⁻¹"
            ], style={'fontSize': '12px', 'fontStyle': 'italic', 'marginTop': '10px'})
        ])

@app.callback(
    Output('activation-curve', 'figure'),
    Output('quantitative-summary', 'children'),
    Output('KD-heatmap', 'figure'),
    Output('KD-difference', 'figure'),
    Output('KD-range-size', 'figure'),
    Input('kp_model', 'value'),
    Input('tau_range', 'value'),
    Input('N_range', 'value'),
    Input('N_ref', 'value'),
    Input('tau_ref', 'value'),
    Input('KD_threshold', 'value'),
    Input('L0', 'value'),
    Input('R0', 'value')
)
def update_all_plots(model, tau_range, N_range, N_ref, tau_ref, KD_threshold, L0, R0):
    
    # ============================================
    # PART 1: ACTIVATION PROBABILITY VS AFFINITY
    # ============================================
    
    KD_range = np.logspace(-1, 3, 200)  # 0.1 to 1000 µM
    
    # Reference (naive) parameters - HIGH KP (more selective)
    if model == 'simple':
        P_ref = activation_probability_simple(KD_range, N_ref, tau_ref)
    else:
        P_ref = activation_probability_concentration(KD_range, N_ref, tau_ref, L0, R0)
    
    # Memory parameters - use the MIN from the ranges for primed/reduced KP
    N_mem = N_range[0]  # MINIMUM N from range (less proofreading)
    tau_mem = tau_range[0]  # MINIMUM tau from range (faster response)
    
    if model == 'simple':
        P_mem = activation_probability_simple(KD_range, N_mem, tau_mem)
    else:
        P_mem = activation_probability_concentration(KD_range, N_mem, tau_mem, L0, R0)
    
    # Create activation probability plot
    fig_activation = go.Figure()
    
    fig_activation.add_trace(go.Scatter(
        x=KD_range,
        y=P_ref,
        mode='lines',
        name=f'Naive (N={N_ref}, τ={tau_ref})',
        line=dict(width=3, color='blue')
    ))
    
    fig_activation.add_trace(go.Scatter(
        x=KD_range,
        y=P_mem,
        mode='lines',
        name=f'Memory - Primed (N={N_mem}, τ={tau_mem})',
        line=dict(width=3, color='red')
    ))
    
    # Add threshold line
    fig_activation.add_hline(
        y=0.5, 
        line_dash="dash", 
        line_color="gray",
        annotation_text="50% activation"
    )
    
    # Add negative selection threshold
    fig_activation.add_vline(
        x=170,
        line_dash="dot",
        line_color="orange",
        annotation_text="Neg. selection (170 µM)"
    )
    
    fig_activation.update_layout(
        title="Activation Probability vs Antigen Affinity (Naive vs Memory - Reduced KP)",
        xaxis_title="K<sub>D</sub> (µM)",
        yaxis_title="P(activation)",
        xaxis_type="log",
        xaxis_range=[-1, 3],
        template="plotly_white",
        height=600,
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    # ============================================
    # PART 2: QUANTITATIVE SUMMARY
    # ============================================
    
    # Calculate activation probability AT THE THRESHOLD KD
    idx_threshold = np.argmin(np.abs(KD_range - KD_threshold))
    P_ref_threshold = P_ref[idx_threshold]
    P_mem_threshold = P_mem[idx_threshold]
    
    # Find KD values at 50% activation
    idx_ref = np.argmin(np.abs(P_ref - 0.5))
    KD_50_ref = KD_range[idx_ref]
    
    idx_mem = np.argmin(np.abs(P_mem - 0.5))
    KD_50_mem = KD_range[idx_mem]
    
    fold_change_50 = KD_50_mem / KD_50_ref
    
    # Find KD values at 10% activation
    idx_ref_10 = np.argmin(np.abs(P_ref - 0.1))
    KD_10_ref = KD_range[idx_ref_10]
    
    idx_mem_10 = np.argmin(np.abs(P_mem - 0.1))
    KD_10_mem = KD_range[idx_mem_10]
    
    fold_change_10 = KD_10_mem / KD_10_ref
    
    # Check if negative selection threshold is crossed
    P_ref_170 = P_ref[np.argmin(np.abs(KD_range - 170))]
    P_mem_170 = P_mem[np.argmin(np.abs(KD_range - 170))]
    
    # Parameter changes (now showing REDUCTION)
    delta_N = N_mem - N_ref  # Should be negative
    delta_tau = tau_mem - tau_ref  # Should be negative
    fold_tau = tau_mem / tau_ref  # Should be < 1
    
    summary = html.Div([
        html.H5("Quantitative Predictions (Memory Cell Priming):", style={'marginBottom': '10px'}),
        
        html.P([
            html.Strong("Parameter changes (Memory is PRIMED - reduced KP): "),
            f"ΔN = {delta_N} steps, Δτ = {delta_tau:.2f} s ({fold_tau:.2f}×)",
            html.Br(),
            html.Span("→ Memory cells have REDUCED proofreading = MORE permissive", 
                     style={'color': 'green', 'fontWeight': 'bold', 'fontStyle': 'italic'})
        ]),
        
        html.P([
            html.Strong(f"Activation at threshold KD ({KD_threshold} µM): "),
            f"P(activation) = {P_ref_threshold:.1%} (naive) → {P_mem_threshold:.1%} (memory)",
            html.Br(),
            html.Span(f"→ {P_mem_threshold/P_ref_threshold:.1f}× increase in activation probability" if P_ref_threshold > 0 else "→ Memory enables activation where naive cannot", 
                     style={'color': 'green', 'fontWeight': 'bold'}),
            html.Br(),
            html.Span(
                f"This means at KD={KD_threshold} µM, memory cells are {P_mem_threshold/P_ref_threshold:.1f}× more likely to activate" if P_ref_threshold > 0 else f"At KD={KD_threshold} µM, only memory cells can respond",
                style={'fontStyle': 'italic', 'fontSize': '13px'})
        ]),
        
        html.P([
            html.Strong("Affinity range expansion (50% activation threshold): "),
            f"K_D,50% = {KD_50_ref:.1f} µM (naive) → {KD_50_mem:.1f} µM (memory)",
            html.Br(),
            html.Span(f"→ {fold_change_50:.1f}× expansion in acceptable KD range", 
                     style={'color': 'green' if fold_change_50 > 1 else 'red', 'fontWeight': 'bold'})
        ]),
        
        html.P([
            html.Strong("Affinity range expansion (10% activation threshold): "),
            f"K_D,10% = {KD_10_ref:.1f} µM (naive) → {KD_10_mem:.1f} µM (memory)",
            html.Br(),
            html.Span(f"→ {fold_change_10:.1f}× expansion", 
                     style={'color': 'green' if fold_change_10 > 1 else 'red', 'fontWeight': 'bold'})
        ]),
        
        html.P([
            html.Strong("Negative selection threshold (170 µM): "),
            f"P(activation) = {P_ref_170:.1%} (naive) → {P_mem_170:.1%} (memory)",
            html.Br(),
            html.Span(
                "⚠️ Memory parameters enable response to neg. selected antigens!" 
                if P_mem_170 > 0.1 and P_ref_170 < 0.1 
                else "✓ Both below activation threshold" 
                if P_mem_170 < 0.1 
                else "Note: Check if this crosses your activation threshold",
                style={'fontStyle': 'italic', 
                       'color': 'red' if (P_mem_170 > 0.1 and P_ref_170 < 0.1) else 'green'}
            )
        ]),
        
        html.Hr(),
        
        html.P([
            html.Strong("Key insight: "),
            f"At the threshold antigen (KD={KD_threshold} µM), memory cells show {P_mem_threshold:.1%} activation vs {P_ref_threshold:.1%} for naive cells. ",
            f"Reducing N by {abs(delta_N)} steps and τ by {(1-fold_tau)*100:.0f}% ",
            f"enables {P_mem_threshold/P_ref_threshold:.1f}× stronger response to the same antigen." if P_ref_threshold > 0 else f"enables response where naive cells cannot activate."
        ], style={'fontSize': '14px', 'backgroundColor': '#d4edda', 'padding': '10px', 'borderRadius': '5px'})
    ])

    # ============================================
    # PART 3: HEATMAPS
    # ============================================

    tau_vals = np.linspace(tau_range[0], tau_range[1], 80)
    N_vals = np.arange(N_range[0], N_range[1] + 1)
    
    KD_equiv = np.zeros((len(tau_vals), len(N_vals)))

    # Calculate equivalent KD for each parameter combination
    for i, tau in enumerate(tau_vals):
        for j, N in enumerate(N_vals):
            KD_equiv[i, j] = find_equivalent_KD(
                KD_threshold, N, tau, N_ref, tau_ref,
                model=model, L0=L0, R0=R0
            )

    # Log-scale for visualization
    log_KD_equiv = np.log10(np.clip(KD_equiv, 1e-3, None))

    # ---- Heatmap 1: Maximum activatable KD ----
    fig1 = go.Figure(go.Heatmap(
        x=N_vals,
        y=tau_vals,
        z=log_KD_equiv,
        colorscale='Viridis',
        colorbar=dict(title='log10(KD µM)'),
        zmin=np.nanmin(log_KD_equiv),
        zmax=np.nanmax(log_KD_equiv)
    ))
    fig1.update_layout(
        title=f"Equivalent KD with same activation as {KD_threshold} µM (Naive: N={N_ref}, τ={tau_ref})",
        xaxis_title="Proofreading steps (N) - Lower = More Primed",
        yaxis_title="Integration time (τ) - Lower = Faster Response",
        template="plotly_white",
        height=600,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    # ---- Heatmap 2: New activations above negative selection threshold ----
    neg_cutoff = 170  # µM
    log_neg_cutoff = np.log10(neg_cutoff)
    
    # Only show regions where equivalent KD exceeds negative selection
    new_activation = np.where(KD_equiv >= neg_cutoff, log_KD_equiv, np.nan)

    fig2 = go.Figure(go.Heatmap(
        x=N_vals,
        y=tau_vals,
        z=new_activation,
        colorscale='Reds',
        colorbar=dict(title=f'log10(KD ≥ {neg_cutoff} µM)'),
        zmin=log_neg_cutoff,
        zmax=np.nanmax(log_KD_equiv)
    ))
    fig2.update_layout(
        title=f"New activations above negative selection ({neg_cutoff} µM) - Danger Zone",
        xaxis_title="Proofreading steps (N) - Lower = More Primed",
        yaxis_title="Integration time (τ) - Lower = Faster Response",
        template="plotly_white",
        height=600,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    # ---- Heatmap 3: Fold change from threshold ----
    fold_change = KD_equiv / KD_threshold
    log_fold_change = np.log10(fold_change)
    
    fig3 = go.Figure(go.Heatmap(
        x=N_vals,
        y=tau_vals,
        z=log_fold_change,
        colorscale='Cividis',
        colorbar=dict(title='log10(Fold change)'),
        zmin=np.nanmin(log_fold_change),
        zmax=np.nanmax(log_fold_change)
    ))
    fig3.update_layout(
        title=f"Fold expansion of antigen landscape (relative to {KD_threshold} µM baseline)",
        xaxis_title="Proofreading steps (N) - Lower = More Primed",
        yaxis_title="Integration time (τ) - Lower = Faster Response",
        template="plotly_white",
        height=600,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    return fig_activation, summary, fig1, fig2, fig3


# =========================
# Run
# =========================

port = int(os.environ.get("PORT", 8080))
# if __name__ == "__main__":
#     app.run_server(host="0.0.0.0", port=port)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)