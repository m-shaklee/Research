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


def KD_bounds_simple(N, tau, prob_thresh=0.5):
    KD_vals = np.logspace(-2, 3, 1000)  # 0.01 - 1000 µM
    P = activation_probability_simple(KD_vals, N, tau)
    idx = np.where(P >= prob_thresh)[0]
    if len(idx) == 0:
        return np.nan, np.nan
    return KD_vals[idx[0]], KD_vals[idx[-1]]

def KD_bounds_concentration(N, tau, L0, R0, prob_thresh=0.5):
    KD_vals = np.logspace(-2, 3, 1000)
    P = activation_probability_concentration(KD_vals, N, tau, L0, R0)
    idx = np.where(P >= prob_thresh)[0]
    if len(idx) == 0:
        return np.nan, np.nan
    return KD_vals[idx[0]], KD_vals[idx[-1]]

def KD_bounds_simple_for_slider(N, tau, KD_slider):
    """
    Given a slider KD value (µM), find all KD values
    that would have an activation probability >=
    P(KD_slider)
    """
    KD_vals = np.logspace(-2, 3, 1000)  # 0.01 - 1000 µM
    # Activation probability at the slider KD
    P_thresh = activation_probability_simple(KD_slider, N, tau)
    P = activation_probability_simple(KD_vals, N, tau)
    idx = np.where(P >= P_thresh)[0]
    if len(idx) == 0:
        return np.nan, np.nan
    return KD_vals[idx[0]], KD_vals[idx[-1]]

def KD_bounds_concentration_for_slider(N, tau, L0, R0, KD_slider):
    KD_vals = np.logspace(-2, 3, 1000)
    P_thresh = activation_probability_concentration(KD_slider, N, tau, L0, R0)
    P = activation_probability_concentration(KD_vals, N, tau, L0, R0)
    idx = np.where(P >= P_thresh)[0]
    if len(idx) == 0:
        return np.nan, np.nan
    return KD_vals[idx[0]], KD_vals[idx[-1]]

# =========================
# KD threshold finders
# =========================

def KD_threshold_simple(N, tau, prob_thresh=0.5):
    KD_vals = np.logspace(-2, 3, 1000)
    P = activation_probability_simple(KD_vals, N, tau)
    idx = np.where(P >= prob_thresh)[0]
    return KD_vals[idx[0]] if len(idx) > 0 else np.nan

def KD_threshold_concentration(N, tau, L0, R0, prob_thresh=0.5):
    KD_vals = np.logspace(-2, 3, 1000)
    P = activation_probability_concentration(KD_vals, N, tau, L0, R0)
    idx = np.where(P >= prob_thresh)[0]
    return KD_vals[idx[0]] if len(idx) > 0 else np.nan

# =========================
# Dash app
# =========================

app = dash.Dash(__name__)

app.layout = html.Div([

    html.H2("Kinetic Proofreading: Antigenic Reach & Memory Gain"),

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

    html.Label("Activation threshold KD (µM)"),
    dcc.Slider(
        id='KD_threshold',
        min=1, max=300, step=1, value=100,
        marks={1: '1', 50: '50', 100: '100', 300: '300'}
    ),

    html.Label("Ligand concentration L₀ (concentration KP only)"),
    dcc.Slider(id='L0', min=1, max=200, step=5, value=50),

    html.Label("Receptor count R₀"),
    dcc.Slider(id='R0', min=1, max=200, step=5, value=50),

    html.Br(),

    dcc.Graph(id='KD-heatmap', style={'height': '500px'}),
    dcc.Graph(id='KD-difference', style={'height': '500px'})
])

# =========================
# Callback
# =========================

# =========================
# KD bounds finders
# =========================

def KD_bounds_simple(N, tau, prob_thresh=0.5):
    KD_vals = np.logspace(-2, 3, 1000)  # 0.01 - 1000 µM
    P = activation_probability_simple(KD_vals, N, tau)
    idx = np.where(P >= prob_thresh)[0]
    if len(idx) == 0:
        return np.nan, np.nan
    return KD_vals[idx[0]], KD_vals[idx[-1]]

def KD_bounds_concentration(N, tau, L0, R0, prob_thresh=0.5):
    KD_vals = np.logspace(-2, 3, 1000)
    P = activation_probability_concentration(KD_vals, N, tau, L0, R0)
    idx = np.where(P >= prob_thresh)[0]
    if len(idx) == 0:
        return np.nan, np.nan
    return KD_vals[idx[0]], KD_vals[idx[-1]]

# =========================
# Dash callback
# =========================

@app.callback(
    Output('KD-heatmap', 'figure'),
    Output('KD-difference', 'figure'),
    Input('kp_model', 'value'),
    Input('tau_range', 'value'),
    Input('N_range', 'value'),
    Input('KD_threshold', 'value'),  # slider in µM
    Input('L0', 'value'),
    Input('R0', 'value')
)
def update_plots(model, tau_range, N_range, KD_slider, L0, R0):

    tau_vals = np.linspace(tau_range[0], tau_range[1], 80)
    N_vals = np.arange(N_range[0], N_range[1] + 1)

    KD_lower = np.zeros((len(tau_vals), len(N_vals)))
    KD_upper = np.zeros((len(tau_vals), len(N_vals)))

    # Compute KD bounds dynamically based on slider
    for i, tau in enumerate(tau_vals):
        for j, N in enumerate(N_vals):
            if model == 'simple':
                lower, upper = KD_bounds_simple_for_slider(N, tau, KD_slider)
            else:
                lower, upper = KD_bounds_concentration_for_slider(N, tau, L0, R0, KD_slider)
            KD_lower[i, j] = lower
            KD_upper[i, j] = upper

    # Activation map: slider KD is within the KD bounds
    activation_map = ((KD_lower <= KD_slider) & (KD_upper >= KD_slider)).astype(int)

    # ---- Heatmap 1: Maximum activatable KD ----
    fig1 = go.Figure(
        data=go.Heatmap(
            x=N_vals,
            y=tau_vals,
            z=np.log10(np.nan_to_num(KD_upper, nan=1e-6)),
            colorscale='Viridis',
            colorbar=dict(title='log10(KD upper µM)')
        )
    )
    fig1.update_layout(
        title="Maximum activatable KD (upper bound)",
        xaxis_title="Proofreading steps (N)",
        yaxis_title="Integration time (τ)",
        template="plotly_white"
    )

    # ---- Heatmap 2: Activation region & new activations ----
    neg_cutoff = 170  # µM
    new_activation_map = np.zeros_like(KD_upper)
    new_activation_map[np.where(KD_upper >= neg_cutoff)] = KD_upper[np.where(KD_upper >= neg_cutoff)]

    fig2 = go.Figure()
    # Base activation region (green)
    fig2.add_trace(go.Heatmap(
        x=N_vals,
        y=tau_vals,
        z = np.log10(KD_upper),
        # z=activation_map,
        colorscale=[[0, 'white'], [1, 'green']],
        showscale=False,
        name='Activated'
    ))
    # Overlay new activations (red)
    fig2.add_trace(go.Heatmap(
        x=N_vals,
        y=tau_vals,
        z = np.log10(KD_upper),
        # z=new_activation_map,
        colorscale='Reds',
        colorbar=dict(title='KD > 170 µM'),
        zmin=neg_cutoff,
        zmax=np.nanmax(KD_upper)
    ))

    fig2.update_layout(
        title=f"Activation region (KD ≥ {KD_slider} µM) & new activations (> {neg_cutoff} µM)",
        xaxis_title="Proofreading steps (N)",
        yaxis_title="Integration time (τ)",
        template="plotly_white"
    )

    return fig1, fig2


# =========================
# Run
# =========================

port = int(os.environ.get("PORT", 8080))
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=port)
