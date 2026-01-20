import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import os

import numpy as np

def kp_activation_probability(
    KD_uM,
    L0,
    R0,
    N,
    kp,
    kon=1e5
):
    KD_M = KD_uM * 1e-6
    koff = KD_M * kon

    term = L0 + R0 + koff / kon
    disc = np.maximum(term**2 - 4 * L0 * R0, 0)
    C_tot = (term - np.sqrt(disc)) / 2

    C_N = C_tot * (1 + koff / kp) ** (-N)
    return C_N


def activation_probability(C_N, threshold=1.0):
    return 1 - np.exp(-C_N / threshold)


def compute_pareto_metrics(
    N,
    kp,
    memory_gain,
    L0,
    R0,
    KD_agonist=1.0,
    KD_self=100.0
):
    kp_eff = kp * memory_gain

    C_agonist = kp_activation_probability(
        KD_agonist, L0, R0, N, kp_eff
    )
    C_self = kp_activation_probability(
        KD_self, L0, R0, N, kp_eff
    )

    P_agonist = activation_probability(C_agonist)
    P_self = activation_probability(C_self)

    specificity = P_agonist / (P_self + 1e-6)
    speed = kp_eff  # proxy for decision speed

    return speed, specificity, P_agonist, P_self


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Kinetic Proofreading Pareto Front"),

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
    ], style={'width': '45%', 'display': 'inline-block'}),

    dcc.Graph(id='pareto-graph', style={'height': '700px'})
])


@app.callback(
    Output('pareto-graph', 'figure'),
    [Input('N', 'value'),
     Input('kp', 'value'),
     Input('memory_gain', 'value'),
     Input('L0', 'value'),
     Input('R0', 'value')]
)
def update_pareto(N, kp, memory_gain, L0, R0):

    kp_vals = np.linspace(0.1, kp * 3, 50)

    speeds = []
    specs = []

    for kp_i in kp_vals:
        speed, spec, _, _ = compute_pareto_metrics(
            N, kp_i, memory_gain, L0, R0
        )
        speeds.append(speed)
        specs.append(spec)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=speeds,
        y=specs,
        mode='lines+markers',
        name='Pareto Front'
    ))

    fig.update_layout(
        title="Speed–Specificity Trade-off",
        xaxis_title="Decision Speed (effective kp)",
        yaxis_title="Specificity (agonist / self)",
        template="plotly_white"
    )

    return fig


port = int(os.environ.get("PORT", 8080))

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=port)
