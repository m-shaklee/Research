import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import os
import numpy as np

def kp_activation_probability(KD_uM, L0, R0, N, kp, kon=1e5):
    KD_M = KD_uM * 1e-6
    koff = KD_M * kon
    term = L0 + R0 + koff / kon
    disc = np.maximum(term**2 - 4 * L0 * R0, 0)
    C_tot = (term - np.sqrt(disc)) / 2
    C_N = C_tot * (1 + koff / kp) ** (-N)
    return C_N

def activation_probability(C_N, threshold=1.0):
    return 1 - np.exp(-C_N / threshold)

def find_responsive_affinities(KD_start, L0, R0, N, kp, memory_gain, threshold=0.5):
    """Return ligand affinities (uM) that result in activation probability >= threshold"""
    kp_eff = kp * memory_gain
    KD_range = np.logspace(-2, 3, 500)  # from 0.01 uM to 1000 uM
    C_N_vals = kp_activation_probability(KD_range, L0, R0, N, kp_eff)
    P_vals = activation_probability(C_N_vals)
    
    responsive_KD = KD_range[P_vals >= threshold]
    return responsive_KD, P_vals, KD_range

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Kinetic Proofreading: Altered Affinity Range"),

    html.Div([
        html.Label("Number of Proofreading Steps (N)"),
        dcc.Slider(id='N', min=1, max=8, step=1, value=4,
                   marks={i: str(i) for i in range(1, 9)}),

        html.Label("Base Proofreading Rate (kp)"),
        dcc.Slider(id='kp', min=0.1, max=10, step=0.1, value=1.0),

        html.Label("Memory Gain (effective speed-up)"),
        dcc.Slider(id='memory_gain', min=0.5, max=5, step=0.1, value=1.0),

        html.Label("Ligand Concentration (L0)"),
        dcc.Slider(id='L0', min=1, max=200, step=5, value=50),

        html.Label("Receptor Count (R0)"),
        dcc.Slider(id='R0', min=1, max=200, step=5, value=50),

        html.Label("Starting Affinity (KD_start, uM)"),
        dcc.Slider(id='KD_start', min=0.01, max=1000, step=0.1, value=1.0, 
                   marks={0.01: '0.01', 1: '1', 100: '100', 1000: '1000'}),
    ], style={'width': '45%', 'display': 'inline-block'}),

    dcc.Graph(id='activation-graph', style={'height': '700px'})
])

@app.callback(
    Output('activation-graph', 'figure'),
    [Input('N', 'value'),
     Input('kp', 'value'),
     Input('memory_gain', 'value'),
     Input('L0', 'value'),
     Input('R0', 'value'),
     Input('KD_start', 'value')]
)
def update_activation(N, kp, memory_gain, L0, R0, KD_start):
    responsive_KD, P_vals, KD_range = find_responsive_affinities(
        KD_start, L0, R0, N, kp, memory_gain, threshold=0.5
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=KD_range,
        y=P_vals,
        mode='lines',
        name='Activation Probability',
    ))

    if len(responsive_KD) > 0:
        fig.add_trace(go.Scatter(
            x=[responsive_KD[0], responsive_KD[-1]],
            y=[0.5, 0.5],
            mode='lines+markers',
            name='Activation Threshold',
            line=dict(color='red', dash='dash')
        ))

    fig.update_layout(
        title=f"Activation Probability vs Ligand Affinity (KD_start={KD_start} uM)",
        xaxis_title="Ligand Affinity KD (uM)",
        yaxis_title="Activation Probability",
        xaxis_type='log',
        template="plotly_white"
    )

    return fig

port = int(os.environ.get("PORT", 8080))
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=port)
