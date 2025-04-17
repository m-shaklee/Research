import dash 
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
import os

def system(t, y, alpha, beta, delta, delta_N, delta_STM, c_N, c_STM):
    S, I, TN, STM, cost, costS, costI = y  # Unpack state variables
    
    S = max(S, 0)
    I = max(I, 0)
    TN = max(TN, 0)
    STM = max(STM, 0)
    
    dN = c_N*delta_N
    dSTM = c_STM*delta_STM
    # Compute derivatives
    chgTN = -alpha * I
    chgSTM = alpha * I
    dI = beta * S * I - delta * I - delta_N * TN * I - delta_STM * STM * I
    dS = -beta * S * I - dN * TN * S - dSTM * STM * S
    
    # Compute instantaneous cost
    cost_I = -delta * I - delta_N * TN * I - delta_STM * STM * I
    cost_S = -dN * TN * S - dSTM * STM * S
    dCost = np.abs(cost_I + cost_S)  # Total cost at this step
    dCost_S = np.abs(cost_S)
    dCost_I = np.abs(cost_I)
    return [dS, dI, chgTN, chgSTM, dCost, dCost_S, dCost_I]


# Time span
t_span = (0, 21)  # Simulate from t=0 to t=50
t_eval = np.linspace(0, 21, 2000)  # Time points for evaluation
# Create the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("STM system timescale adjusted"),

    html.Div([
        # LEFT COLUMN
        html.Div([
            html.Label("S0:"),
            dcc.Slider(id='S0', min=0, max=5*10**8, step=10*2, value=4*10**8,
                       marks={round(i, 2): f"{i:.2f}" for i in np.linspace(0,  5*10**8, 6)}),
            html.Label("I0:"),
            dcc.Slider(id='I0', min=0, max=1*10**8, step=10*2, value=10**6,
                       marks={round(i, 2): f"{i:.2f}" for i in np.linspace(0,  1*10**8, 6)}),
            html.Label("TN0:"),
            dcc.Slider(id='TN0', min=0, max=2*10**8, step=1, value=1*10**8,
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
            dcc.Slider(id='alpha', min=0, max=0.1, step=0.01, value=0.01,
                       marks={round(i, 2): f"{i:.2f}" for i in np.linspace(0, 0.1, 6)}),
            html.Label("β: Infectivity"),
            # dcc.Slider(id='beta', min=0, max=5*10**-6, step=0.1*10**-6, value=1.5*10**-6,
            #            marks={i: f"{i:.0e}" for i in np.linspace(0, 5e-6, 6)}),
            dcc.Slider(
                id='beta',
                min=0,
                max=5e-6,
                step=1e-7,
                value=1.5e-6,
                marks={i: f"{i:.0e}" for i in np.linspace(0, 5e-6, 6)}
            )
            html.Label("δ: Infected cell death"),
            dcc.Slider(id='delta', min=0, max=1*10**-6, step=0.1*10**-6, value=0.5*10**-6,
                       marks={i: f"{i:.0e}" for i in np.linspace(0, 1e-6, 6)}),
            html.Label("δ_TN: Infected cell death by TN"),
            dcc.Slider(id='delta_N', min=0, max=1*10**-6, step=0.1*10**-6, value=0.5*10**-6,
                       marks={i: f"{i:.0e}" for i in np.linspace(0, 1e-6, 6)}),
            html.Label("δ_STM: Infected cell death by STM"),
            dcc.Slider(id='delta_STM', min=0, max=1*10**-6, step=0.1*10**-6, value=0.5*10**-6,
                       marks={i: f"{i:.0e}" for i in np.linspace(0, 1e-6, 6)}),
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
    dcc.Graph(id='output-graph', style={'height': '1000px', 'width': '100%'})
])
# Callback to update the plot
@app.callback(
    Output('output-graph', 'figure'),
    [Input('S0', 'value'),Input('I0', 'value'),Input('TN0', 'value'),Input('STM0', 'value'),Input('alpha', 'value'), Input('beta', 'value'), Input('delta', 'value'),
     Input('delta_N', 'value'), Input('delta_STM', 'value'), Input('c_N', 'value'),
     Input('c_STM', 'value')]
)
# def update_graph(alpha, beta, delta, delta_N, delta_STM, dN, dSTM,S0,I0,TN0,STM0):
def update_graph(S0, I0, TN0, STM0, alpha, beta, delta, delta_N, delta_STM, c_N, c_STM):
    y0=[S0,I0,TN0,STM0,0,0,0]
    sol = solve_ivp(system, t_span, y0, args=(alpha, beta, delta, delta_N, delta_STM, c_N, c_STM),
                     t_eval=t_eval, method='RK45')
    
    # Extract solutions
    t_values = sol.t
    # X, c = sol.y
    S_values, I_values, TN_values, STM_values, cumulative_cost, S_cost, I_cost = sol.y

    title_text = 'Dynamics of Infection Over Time'
    sub_text1=(f'Final cumulative cost: {np.round(np.sum(cumulative_cost),2)} S0 = {y0[0]}, I0 = {y0[1]}, TN0 = {y0[2]}, STM0 = {y0[3]}')
    sub_text2=(f' α={alpha}, β={beta}, δ={delta}, δ_N={delta_N}, δ_STM={delta_STM}, c_N={c_N}, c_STM={c_STM}')
    

    # Create the plot
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=t_values, y=S_values,name="Susceptible (S)",line=dict(width=8)))
    figure.add_trace(go.Scatter(x=t_values, y=I_values, name="Infected (I)",line=dict(width=8)))
    figure.add_trace(go.Scatter(x=t_values, y=TN_values, name="TN",line=dict(width=4)))
    figure.add_trace(go.Scatter(x=t_values, y=STM_values, name="STM",line=dict(width=4)))
    # figure.add_trace(go.Scatter(x=t_values, y=cumulative_cost, name="cost",line=dict(width=4)))
    # figure.add_trace(go.Scatter(x=t_values, y=S_cost, name="S cost",line=dict(width=4)))
    # figure.add_trace(go.Scatter(x=t_values, y=I_cost, name="I cost",line=dict(width=4)))

    # sub_text1=(f'Final cumulative cost: {np.round(np.sum(cumulative_cost),2)} S0 = {y0[0]}, I0 = {y0[1]}, TN0 = {y0[2]}, STM0 = {y0[3]}\n α={alpha}, β={beta}, δ={delta}, δ_N={delta_N}, δ_STM={delta_STM}, dN={dN}, dSTM={dSTM}')



    figure.add_annotation(
        x=0.5, y=1.08, xref="paper", yref="paper", showarrow=False,
        text=sub_text1, font=dict(size=24), align="center"
    )
    figure.add_annotation(
        x=0.5, y=1.03, xref="paper", yref="paper", showarrow=False,
        text=sub_text2, font=dict(size=24), align="center"
    )
    figure.add_annotation(
        x=0.5, y=1.13, xref="paper", yref="paper", showarrow=False,
        text=title_text, font=dict(size=28), align="center"
    )


    figure.update_layout(
    # title="Dynamics of Infection Over Time",
    # yaxis_type="log",
    title_font=dict(size=24),  # Adjust title font size
    xaxis_title="Time",
    yaxis_title="Cells/Cost (log)",
    xaxis_title_font=dict(size=28),  # Adjust x-axis title font size
    yaxis_title_font=dict(size=28),  # Adjust y-axis title font size
    xaxis=dict(tickfont=dict(size=20)),  # Adjust x-axis tick labels font size
    yaxis=dict(tickfont=dict(size=20)),  # Adjust y-axis tick labels font size
    legend=dict(
        font=dict(size=28)  # Change the font size of the legend text
    ),
    template="plotly_white"
)
    return figure
port = int(os.environ.get('PORT',8080))
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=port)