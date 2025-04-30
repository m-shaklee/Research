#Date: 04/30/25
import dash 
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
from dash import State
import io
import base64
import pandas as pd


def system_no_STM(t, y, alpha, beta, delta, delta_N, delta_STM, c_N, c_STM):
    S, I, TN, STM, cost, costS, costI = y  # Unpack state variables
    
    S = max(S, 0)
    I = max(I, 0)
    TN = max(TN, 0)
    STM = 0

    dN = c_N*delta_N
    dSTM = 0
    # Compute derivatives
    chgTN = 0
    # chgSTM = alpha * I
    chgSTM = 0 
    dI = beta * S * I - delta * I - delta_N * TN * I - delta_STM * STM * I
    dS = -beta * S * I - dN * TN * S - dSTM * STM * S
    
    # Compute instantaneous cost
    cost_I = -delta * I - delta_N * TN * I 
    cost_S = -dN * TN * S 
    dCost = np.abs(cost_I + cost_S)  # Total cost at this step
    dCost_S = np.abs(cost_S)
    dCost_I = np.abs(cost_I)
    return dS, dI,chgTN,chgSTM,dCost,dCost_S,dCost_I

def system(t, y, alpha, beta, delta, delta_N, delta_STM, c_N, c_STM):
    S, I, TN, STM, cost, costS, costI = y  # Unpack state variables
    
    S = max(S, 0)
    I = max(I, 0)
    TN = max(TN, 0)
    STM = max(STM, 0)
    STM = 0 if STM < 1e-10 else STM

    dN = c_N*delta_N
    dSTM = c_STM*delta_STM
    # Compute derivatives
    chgTN = -alpha * I
    # chgSTM = alpha * I
    chgSTM = 0 if (alpha == 0 or I < 1e-12) else alpha * I
    dI = beta * S * I - delta * I - delta_N * TN * I - delta_STM * STM * I
    dS = -beta * S * I - dN * TN * S - dSTM * STM * S
    
    # Compute instantaneous cost
    cost_I = -delta * I - delta_N * TN * I - delta_STM * STM * I
    cost_S = -dN * TN * S - dSTM * STM * S
    dCost = np.abs(cost_I + cost_S)  # Total cost at this step
    dCost_S = np.abs(cost_S)
    dCost_I = np.abs(cost_I)
    return dS, dI,chgTN,chgSTM,dCost,dCost_S,dCost_I

# Time span
t_span = (0, 21)  # Simulate from t=0 to t=50
t_eval = np.linspace(0, 21, 1000)  # Time points for evaluation
# Create the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("no STM system timescale adjusted"),

    html.Div([
        # LEFT COLUMN
        html.Div([
            html.Label("S0:"),
            dcc.Slider(id='S0', min=0, max=5*10**8, step=10*2, value=4*10**8,
                       marks={round(i, 1): f"{i:.1f}" for i in np.linspace(0,  5*10**8, 6)}),
            html.Label("I0:"),
            dcc.Slider(id='I0', min=0, max=100, step=10, value=100,
                       marks={round(i, 2): f"{i:.2f}" for i in np.linspace(0,  100, 6)}),
            html.Label("TN0:"),
            dcc.Slider(id='TN0', min=0, max=2*10**8, step=1, value=4*10**7,
                       marks={round(i, 2): f"{i:.2f}" for i in np.linspace(0,  2*10**8, 6)}),
            html.Label("STM0:"),
            dcc.Slider(id='STM0', min=0, max=1*10**8, step=1, value=0,
                       marks={round(i, 2): f"{i:.2f}" for i in np.linspace(0,  1*10**8, 6)}),

            # Insert LaTeX-style equations
            html.Div([
                dcc.Markdown("dTN/dt=-α*I"),
                dcc.Markdown("dSTM/dt=α*I"),
                dcc.Markdown("dI/dt=β\*S\*I-δ\*I-δN\*TN\*I-δSTM\*STM\*I"),
                dcc.Markdown("dS/dt=-β\*S\*I-dN\*TN\*S-dSTM\*STM\*S")
            ], style={'marginTop': '30px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # RIGHT COLUMN (sliders only)
        html.Div([
            html.Label("α: TN to STM"),
            dcc.Slider(id='alpha', min=0.001, max=1, step=0.01, value=0.3,
                       marks={round(i, 2): f"{i:.2f}" for i in np.linspace(0.001, 1, 6)}),
            html.Label("β: Infectivity (e-9)"),
            dcc.Slider(id='beta', min=0, max=10, step=0.5, value=5.5,
                       marks={round(i, 2): f"{i:.2f}" for i in np.linspace(0, 10, 6)}),
            html.Label("δ: Infected cell death (e-7)"),
            dcc.Slider(id='delta', min=0, max=1, step=0.01, value=0.01,
                       marks={round(i, 2): f"{i:.2f}" for i in np.linspace(0, 1, 6)}),
            html.Label("δ_TN: Infected cell death by TN (e-7)"),
            dcc.Slider(id='delta_N', min=0, max=1, step=0.01, value=0.02,
                       marks={round(i, 2): f"{i:.2f}" for i in np.linspace(0, 1, 6)}),
            html.Label("δ_STM: Infected cell death by STM (e-7)"),
            dcc.Slider(id='delta_STM',min=0, max=15, step=0.01, value=10,
                       marks={round(i, 2): f"{i:.2f}" for i in np.linspace(0, 15, 6)}),
            html.Label("c_N: Proportion of susceptible cell death by TN to infected cell death"),
            dcc.Slider(id='c_N', min=0, max=1, step=0.01, value=0.005,
                       marks={round(i, 4): f"{i:.4f}" for i in np.linspace(0, 1, 6)}),
            html.Label("c_STM: Proportion of susceptible cell death by STM to infected cell death"),
            dcc.Slider(id='c_STM', min=0, max=2, step=0.01, value=0.001,
                       marks={round(i, 4): f"{i:.4f}" for i in np.linspace(0, 2, 6)}),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ]),

    html.Br(),
    # dcc.Graph(id='output-graph')
    html.Label("System Mode:"),
    dcc.RadioItems(
    id='system_mode',
    options=[
        {'label': 'With STM', 'value': 'with_stm'},
        {'label': 'Without STM (α = 0)', 'value': 'no_stm'}
    ],
    value='with_stm',
    labelStyle={'display': 'inline-block', 'margin-right': '20px'}
),
    html.Button("Download CSV", id="download-btn", n_clicks=0),
    dcc.Download(id="download-csv"),
    dcc.Graph(id='output-graph', style={'height': '2000px', 'width': '100%'})
])
# Callback to update the plot
@app.callback(
    Output("download-csv", "data"),
    Input("download-btn", "n_clicks"),
    State('S0', 'value'),
    State('I0', 'value'),
    State('TN0', 'value'),
    State('STM0', 'value'),
    State('alpha', 'value'),
    State('beta', 'value'),
    State('delta', 'value'),
    State('delta_N', 'value'),
    State('delta_STM', 'value'),
    State('c_N', 'value'),
    State('c_STM', 'value'),
    State('system_mode', 'value'),
    prevent_initial_call=True
)
def download_csv(n_clicks, S0, I0, TN0, STM0, alpha, beta, delta, delta_N, delta_STM, c_N, c_STM, system_mode):
    # Rerun simulation to get results
    y0 = [S0, I0, TN0, STM0, 0, 0, 0]
    beta_adj = beta * 1e-9
    delta_adj = delta * 1e-7
    delta_N_adj = delta_N * 1e-7
    delta_STM_adj = delta_STM * 1e-7

    system_func = system if system_mode == 'with_stm' else system_no_STM
    sol = solve_ivp(system_func, t_span, y0, args=(alpha, beta_adj, delta_adj, delta_N_adj, delta_STM_adj, c_N, c_STM), t_eval=t_eval, method='Radau')

    t_values = sol.t
    S_values, I_values, TN_values, STM_values, cumulative_cost, S_cost, I_cost = sol.y

    # Prepare metadata as a header string
    metadata = [
        f"S0: {S0}", f"I0: {I0}", f"TN0: {TN0}", f"STM0: {STM0}",
        f"alpha: {alpha}", f"beta: {beta_adj}", f"delta: {delta_adj}",
        f"delta_N: {delta_N_adj}", f"delta_STM: {delta_STM_adj}",
        f"c_N: {c_N}", f"c_STM: {c_STM}", f"system_mode: {system_mode}"
    ]
    
    # Create CSV content in-memory
    output = io.StringIO()
    output.write("# Simulation Parameters\n")
    for line in metadata:
        output.write(f"# {line}\n")

    # Create DataFrame
    df = pd.DataFrame({
        "Time": t_values,
        "Susceptible (S)": S_values,
        "Infected (I)": I_values,
        "TN": TN_values,
        "STM": STM_values,
        "Cumulative Cost": cumulative_cost,
        "S Cost": S_cost,
        "I Cost": I_cost
    })
    df.to_csv(output, index=False)
    output.seek(0)
    filename_metadata = f"S{S0}_I{I0}_TN{TN0}_STM{STM0}_a{alpha}_b{beta_adj}_d{delta_adj}_dN{delta_N_adj}_dSTM{delta_STM_adj}_cN{c_N}_cSTM{c_STM}_{system_mode}"
    filename_metadata = filename_metadata.replace('.', 'p')  # replace decimal points to avoid issues
    filename = f"simulation_{filename_metadata}.csv"
    return dcc.send_string(output.getvalue(), filename=filename)
@app.callback(
    Output('output-graph', 'figure'),
    [Input('S0', 'value'),Input('I0', 'value'),Input('TN0', 'value'),Input('STM0', 'value'),Input('alpha', 'value'), Input('beta', 'value'), Input('delta', 'value'),
     Input('delta_N', 'value'), Input('delta_STM', 'value'), Input('c_N', 'value'),
     Input('c_STM', 'value'),Input('system_mode', 'value')]
)
# def update_graph(alpha, beta, delta, delta_N, delta_STM, dN, dSTM,S0,I0,TN0,STM0):
def update_graph(S0, I0, TN0, STM0, alpha, beta, delta, delta_N, delta_STM, c_N, c_STM, system_mode):
    y0=[S0,I0,TN0,STM0,0,0,0]
    beta=beta*1e-9
    delta=delta*1e-7
    delta_N=delta_N*1e-7
    delta_STM=delta_STM*1e-7

    system_func = system if system_mode == 'with_stm' else system_no_STM

    sol = solve_ivp(system_func, t_span, y0, args=(alpha, beta, delta, delta_N, delta_STM, c_N, c_STM),
                     t_eval=t_eval, method='Radau')
    
    # Extract solutions
    t_values = sol.t
    # X, c = sol.y
    S_values, I_values, TN_values, STM_values, cumulative_cost, S_cost, I_cost = sol.y

    title_text = 'Dynamics of Infection Over Time'
    sub_text1=(f'Final cumulative cost: {np.round(np.sum(cumulative_cost),2)} S0 = {y0[0]}, I0 = {y0[1]}, TN0 = {y0[2]}, STM0 = {y0[3]}')
    sub_text2=(f' α={alpha}, β={beta}, δ={delta}, δ_N={delta_N}, δ_STM={delta_STM}, c_N={c_N}, c_STM={c_STM}')
    



    colors = {
    "S": "#1f77b4",       # Blue
    "cost": "#ff7f0e",       # Orange
    "TN": "#2ca02c",      # Green
    "I": "#d62728",     # Red
    "STM": "#9467bd"     # Purple
}

    # Create subplot with two rows: linear and log scale
    figure = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        # subplot_titles=("Linear Scale", "Logarithmic Scale (y-axis)")
    )

    # Plot on linear scale (row=1)
    figure.add_trace(go.Scatter(x=t_values, y=S_values, name="Susceptible (S)", line=dict(width=8, color=colors["S"])), row=1, col=1)
    figure.add_trace(go.Scatter(x=t_values, y=I_values, name="Infected (I)", line=dict(width=8, color=colors["I"])), row=1, col=1)
    figure.add_trace(go.Scatter(x=t_values, y=TN_values, name="TN", line=dict(width=4, color=colors["TN"])), row=1, col=1)
    figure.add_trace(go.Scatter(x=t_values, y=STM_values, name="STM", line=dict(width=4, color=colors["STM"])), row=1, col=1)
    figure.add_trace(go.Scatter(x=t_values, y=cumulative_cost, name="cost", line=dict(width=4, color=colors["cost"])), row=1, col=1)

    # Plot same data on log scale (row=2), hide legend
    figure.add_trace(go.Scatter(x=t_values, y=S_values, showlegend=False, line=dict(width=8, color=colors["S"])), row=2, col=1)
    figure.add_trace(go.Scatter(x=t_values, y=I_values, showlegend=False, line=dict(width=8, color=colors["I"])), row=2, col=1)
    figure.add_trace(go.Scatter(x=t_values, y=TN_values, showlegend=False, line=dict(width=4, color=colors["TN"])), row=2, col=1)
    figure.add_trace(go.Scatter(x=t_values, y=STM_values, showlegend=False, line=dict(width=4, color=colors["STM"])), row=2, col=1)
    figure.add_trace(go.Scatter(x=t_values, y=cumulative_cost, showlegend=False, line=dict(width=4, color=colors["cost"])), row=2, col=1)

    # Set log scale for the second subplot y-axis
    figure.update_yaxes(type="log", row=2, col=1)

    # Layout
    figure.update_layout(
        height=2000,
        title_font=dict(size=24),
        margin=dict(t=250),
        # xaxis_title="Time",
        yaxis_title="Cells/Cost (linear)",
        xaxis2_title="Days",
        yaxis2_title="Cells/Cost (log)",
        xaxis_title_font=dict(size=28),
        yaxis_title_font=dict(size=28),
        xaxis2_title_font=dict(size=28),
        yaxis2_title_font=dict(size=28),
        xaxis=dict(tickfont=dict(size=20)),
        yaxis=dict(tickfont=dict(size=20)),
        xaxis2=dict(tickfont=dict(size=20)),
        yaxis2=dict(tickfont=dict(size=20)),
        legend=dict(font=dict(size=24)),
        template="plotly_white"
    )

    # Add annotations above the plot
    figure.add_annotation(
        x=0.5, y=1.08, xref="paper", yref="paper", showarrow=False,
        text=title_text, font=dict(size=28), align="center"
    )
    figure.add_annotation(
        x=0.5, y=1.05, xref="paper", yref="paper", showarrow=False,
        text=sub_text1, font=dict(size=24), align="center"
    )
    figure.add_annotation(
        x=0.5, y=1.02, xref="paper", yref="paper", showarrow=False,
        text=sub_text2, font=dict(size=24), align="center"
    )
    return figure
port = int(os.environ.get('PORT',8080))
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=port)