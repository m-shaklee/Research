import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import os
import numpy as np

def kp_activation_probability_concentration(KD_uM, L0, R0, N, kp, kon=1e5):
    KD_M = KD_uM * 1e-6
    koff = KD_M * kon
    term = L0 + R0 + koff / kon
    disc = np.maximum(term**2 - 4 * L0 * R0, 0)
    C_tot = (term - np.sqrt(disc)) / 2
    C_N = C_tot * (1 + koff / kp) ** (-N)
    return C_N

def activation_probability_simple_KP(KD_uM, N=2.67, tau_total=2.8, kon=1e5):
    """Return activation probability per Pettmann et al. (2021)."""
    KD_M = KD_uM * 1e-6
    koff = KD_M * kon
    tau_b = 1.0 / koff
    tau_step = tau_total / N
    # return float(np.clip((tau_b / (tau_b + tau_step)) ** N, 0.0, 1.0))
    return np.clip((tau_b / (tau_b + tau_step)) ** N, 0.0, 1.0)

def activation_probability_from_CN(C_N, threshold=1.0):
    return 1 - np.exp(-C_N / threshold)

def find_responsive_affinities(KD_start, L0, R0, N, kp, memory_gain, threshold=0.5):
    """Return ligand affinities (uM) that result in activation probability >= threshold"""
    kp_eff = kp * memory_gain
    KD_range = np.logspace(-2, 3, 500)  # from 0.01 uM to 1000 uM
    C_N_vals = kp_activation_probability_concentration(KD_range, L0, R0, N, kp_eff)
    P_vals = activation_probability_from_CN(C_N_vals)
    
    responsive_KD = KD_range[P_vals >= threshold]
    return responsive_KD, P_vals, KD_range

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Memory-dependent activation phase diagram"),

    dcc.RadioItems(
        id='kp_model',
        options=[
            {'label': 'Simplified KP', 'value': 'simple'},
            {'label': 'KP with concentration', 'value': 'concentration'}
        ],
        value='simple',
        inline=True
    ),

    html.Label("Test ligand affinity KD (µM)"),
    dcc.Slider(id='KD_test', min=1, max=300, step=1, value=100),

    html.Label("Activation threshold"),
    dcc.Slider(id='activation_threshold', min=0.1, max=0.9, step=0.05, value=0.5),

    html.Label("τ range (total integration time)"),
    dcc.RangeSlider(
        id='tau_range',
        min=0.5, max=10, step=0.25,
        value=[1.0, 6.0]
    ),

    html.Label("N range (proofreading steps)"),
    dcc.RangeSlider(
        id='N_range',
        min=1, max=8, step=1,
        value=[1, 6],
        marks={i: str(i) for i in range(1, 9)}
    ),

    html.Label("Ligand concentration L₀ (for concentration KP)"),
    dcc.Slider(id='L0', min=1, max=200, step=5, value=50),

    html.Label("Receptor count R₀"),
    dcc.Slider(id='R0', min=1, max=200, step=5, value=50),

    dcc.Graph(id='phase-diagram', style={'height': '700px'})
])


from dash.dependencies import Input, Output
import plotly.graph_objs as go

@app.callback(
    Output('phase-diagram', 'figure'),
    Input('kp_model', 'value'),
    Input('KD_test', 'value'),
    Input('activation_threshold', 'value'),
    Input('tau_range', 'value'),
    Input('N_range', 'value'),
    Input('L0', 'value'),
    Input('R0', 'value'),
)
def update_phase_diagram(model, KD_test, threshold, tau_range, N_range, L0, R0):

    tau_vals = np.linspace(tau_range[0], tau_range[1], 100)
    N_vals = np.arange(N_range[0], N_range[1] + 1)

    activation = np.zeros((len(tau_vals), len(N_vals)))

    for i, tau in enumerate(tau_vals):
        for j, N in enumerate(N_vals):

            if model == 'simple':
                P = activation_probability_simple_KP(KD_test, N, tau)

            else:
                kp = 1.0 / tau  # map integration time to kp
                C_N = kp_activation_probability_concentration(
                    KD_test, L0, R0, N, kp
                )
                P = activation_probability_from_CN(C_N, threshold=1.0)

            activation[i, j] = P >= threshold

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=N_vals,
        y=tau_vals,
        z=activation,
        colorscale='Greens',
        colorbar=dict(title='Activated')
    ))

    fig.update_layout(
        title=f"Activation phase diagram (KD = {KD_test} µM)",
        xaxis_title="Proofreading steps (N)",
        yaxis_title="Integration time (τ)",
        template="plotly_white"
    )

    return fig


port = int(os.environ.get("PORT", 8080))
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=port)
