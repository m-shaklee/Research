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
    KD_M = KD_uM * 1e-6
    koff = KD_M * kon
    tau_b = 1.0 / koff
    tau_step = tau_total / N
    return np.clip((tau_b / (tau_b + tau_step)) ** N, 0, 1)

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

    html.H2("Kinetic Proofreading: Antigenic Reach & Memory Gain"),
    
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

    html.Label("τ range (integration time)"),
    dcc.RangeSlider(
        id='tau_range', min=0.5, max=10, step=0.25,
        value=[1.0, 6.0]
    ),

    html.Label("N range (proofreading depth)"),
    dcc.RangeSlider(
        id='N_range', min=1, max=8, step=1,
        value=[1, 6],
        marks={i: str(i) for i in range(1, 9)}
    ),

    html.Br(),
    html.H4("Reference Parameters (Original State)"),
    
    html.Label("Reference N (proofreading steps)"),
    dcc.Slider(
        id='N_ref',
        min=1, max=8, step=1, value=2,
        marks={i: str(i) for i in range(1, 9)}
    ),
    
    html.Label("Reference τ (integration time)"),
    dcc.Slider(
        id='tau_ref',
        min=0.5, max=10, step=0.25, value=2.0,
        marks={i: str(i) for i in range(0, 11, 2)}
    ),

    html.Label("Activation threshold KD (µM)"),
    dcc.Slider(
        id='KD_threshold',
        min=1, max=300, step=1, value=100,
        marks={1: '1', 50: '50', 100: '100', 300: '300'}
    ),

    html.Br(),
    html.H4("Additional Parameters"),

    html.Label("Ligand concentration L₀ (concentration KP only)"),
    dcc.Slider(id='L0', min=1, max=200, step=5, value=50),

    html.Label("Receptor count R₀"),
    dcc.Slider(id='R0', min=1, max=200, step=5, value=50),

    html.Br(),

    dcc.Graph(id='KD-heatmap', style={'height': '1000px'}),
    dcc.Graph(id='KD-difference', style={'height': '1000px'}),
    dcc.Graph(id='KD-range-size', style={'height': '1000px'})
])

# =========================
# Callback for equations
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
                "Assumes single ligand binding, no concentration dependence"
            ]),
            html.P([
                "τ", html.Sub("b"), " = 1/k", html.Sub("off"), " = 1/(K", html.Sub("D"), " × k", html.Sub("on"), ")"
            ], style={'marginTop': '10px'}),
            html.P([
                "τ", html.Sub("step"), " = τ / N"
            ]),
            html.P([
                html.Strong("P(activation) = "),
                "(τ", html.Sub("b"), " / (τ", html.Sub("b"), " + τ", html.Sub("step"), "))", html.Sup("N")
            ], style={'fontSize': '16px', 'marginTop': '10px'}),
            html.P([
                "where k", html.Sub("on"), " = 10⁵ M⁻¹s⁻¹ (fixed)"
            ], style={'fontSize': '12px', 'fontStyle': 'italic', 'marginTop': '10px'})
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


# =========================
# Callback for plots
# =========================

@app.callback(
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
def update_plots(model, tau_range, N_range, N_ref, tau_ref, KD_threshold, L0, R0):

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
        title=f"Equivalent KD with same activation as {KD_threshold} µM (N={N_ref}, τ={tau_ref})",
        xaxis_title="Proofreading steps (N)",
        yaxis_title="Integration time (τ)",
        template="plotly_white",
        height=1000,
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
        title=f"New activations above negative selection ({neg_cutoff} µM)",
        xaxis_title="Proofreading steps (N)",
        yaxis_title="Integration time (τ)",
        template="plotly_white",
        height=1000,
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
        title=f"Fold expansion of antigen landscape (relative to {KD_threshold} µM)",
        xaxis_title="Proofreading steps (N)",
        yaxis_title="Integration time (τ)",
        template="plotly_white",
        height=1000,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    return fig1, fig2, fig3


# =========================
# Run
# =========================

port = int(os.environ.get("PORT", 8080))
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=port)