import os
from dotenv import load_dotenv

import mlflow
from mlflow.tracking import MlflowClient

import torch

import dash
from dash import dcc, html, Input, Output, State, ctx


# -----------------------
# Setup
# -----------------------
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
client = MlflowClient()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

app = dash.Dash(__name__)

# -----------------------
# Model cache
# -----------------------
MODEL_CACHE = {}

def get_model(uri):
    if uri not in MODEL_CACHE:
        m = mlflow.pyfunc.load_model(uri)
        model = m.unwrap_python_model().model

        # move to device
        if hasattr(model, "to"):
            model = model.to(DEVICE)

        MODEL_CACHE[uri] = model

    return MODEL_CACHE[uri]

# -----------------------
# Helpers
# -----------------------
def list_model_names():
    return sorted([m.name for m in client.search_registered_models()])


def get_versions(model_name):
    if not model_name:
        return []

    try:
        versions = client.search_model_versions(f"name = '{model_name}'")
        return sorted(
            [v.version for v in versions],
            key=lambda x: int(x),
            reverse=True
        )
    except Exception:
        return []

# -----------------------
# Layout
# -----------------------
app.layout = html.Div(
    style={
        "display": "flex",
        "height": "100vh",
        "background": "linear-gradient(to right, #4facfe, #00c6ff)"
    },
    children=[

        # -------- LEFT PANEL --------
        html.Div(
            style={
                "width": "30%",
                "padding": "20px",
                "background": "linear-gradient(to bottom, #5dade2, #3498db)",
                "color": "white"
            },
            children=[
                html.H3("Controls"),

                html.Label("Model A"),
                dcc.Dropdown(
                    id="model-a-name",
                    options=[{"label": n, "value": n} for n in list_model_names()],
                    style={
                        "color": "black",
                        "borderRadius": "8px",
                        "boxShadow": "0px 4px 12px rgba(0,0,0,0.2)",
                    },
                ),
                dcc.Dropdown(
                    id="model-a-version",
                    style={
                        "color": "black",
                        "borderRadius": "8px",
                        "boxShadow": "0px 4px 12px rgba(0,0,0,0.2)",
                    },
                ),

                html.Br(),

                html.Label("Model B (Optional)"),
                dcc.Dropdown(
                    id="model-b-name",
                    options=[{"label": n, "value": n} for n in list_model_names()],
                    style={
                        "color": "black",
                        "borderRadius": "8px",
                        "boxShadow": "0px 4px 12px rgba(0,0,0,0.2)",
                    },
                ),
                dcc.Dropdown(
                    id="model-b-version",
                    style={
                        "color": "black",
                        "borderRadius": "8px",
                        "boxShadow": "0px 4px 12px rgba(0,0,0,0.2)",
                    },
                ),

                html.Br(),

                html.Label("Temperature"),
                dcc.Input(
                    id="temperature",
                    type="number",
                    value=0.8,
                    step=0.1,
                    style={
                        "color": "black",
                        "background": "white",
                        "borderRadius": "8px",
                        "boxShadow": "0px 4px 12px rgba(0,0,0,0.2)",
                    },
                ),
                html.Label("Top-k"),
                dcc.Input(
                    id="top_k",
                    type="number",
                    value=50,
                    style={
                        "color": "black",
                        "background": "white",
                        "borderRadius": "8px",
                        "boxShadow": "0px 4px 12px rgba(0,0,0,0.2)",
                    },
                ),

                html.Label("Top-p"),
                dcc.Input(
                    id="top_p",
                    type="number",
                    value=0.95,
                    step=0.05,
                    style={
                        "color": "black",
                        "background": "white",
                        "borderRadius": "8px",
                        "boxShadow": "0px 4px 12px rgba(0,0,0,0.2)",
                    },
                ),
                html.Label("Max Tokens"),
                dcc.Input(
                    id="max_tokens",
                    type="number",
                    value=64,
                    style={
                        "color": "black",
                        "background": "white",
                        "borderRadius": "8px",
                        "boxShadow": "0px 4px 12px rgba(0,0,0,0.2)",
                    },
                ),
                html.Br(),
                html.Br(),

                html.Button("Generate", id="generate")
            ]
        ),

        # -------- RIGHT PANEL --------
        html.Div(
            style={
                "width": "70%",
                "padding": "20px",
                "background": "linear-gradient(to bottom, #85c1e9, #aed6f1)"
            },
            children=[

                html.H2("MLflow GPT Dashboard", style={"color": "white"}),

                html.Label("Prompt", style={"color": "white"}),
                dcc.Input(
                    id="prompt",
                    type="text",
                    placeholder="Enter prompt and press Enter...",
                    style={
                        "width": "100%",
                        "height": "50px",
                        "background": "white",
                        "color": "black",
                        "borderRadius": "8px",
                        "boxShadow": "0px 4px 12px rgba(0,0,0,0.2)",
                    },
                    debounce=False
                ),

                html.Br(),

                html.H3("Outputs", style={"color": "white"}),

                html.Div([
                    html.Div([
                        html.H4("Model A"),
                        dcc.Loading(
                            html.Div(
                                id="output-a",
                                style={
                                    "background": "white",
                                    "padding": "10px",
                                    "borderRadius": "8px",
                                    "boxShadow": "0px 4px 12px rgba(0,0,0,0.2)",
                                }
                            )
                        )
                    ], style={"width": "48%"}),
                    html.Div([
                        html.H4("Model B"),
                        dcc.Loading(
                            html.Div(
                                id="output-b",
                                style={
                                    "background": "white",
                                    "padding": "10px",
                                    "borderRadius": "8px",
                                    "boxShadow": "0px 4px 12px rgba(0,0,0,0.2)",
                                }
                            )
                        )
                    ], style={"width": "48%"})
                ], style={"display": "flex", "justifyContent": "space-between"}),

                html.Br(),

                html.H3("History", style={"color": "white"}),
                dcc.Store(id="history", data=[]),
                html.Div(
                    id="history-display",
                    style={
                        "background": "white",
                        "padding": "10px",
                        "borderRadius": "8px",
                        "boxShadow": "0px 4px 12px rgba(0,0,0,0.2)",
                    }
                )
            ]
        )
    ]
)

# -----------------------
# Populate versions
# -----------------------
@app.callback(
    Output("model-a-version", "options"),
    Output("model-a-version", "value"),
    Input("model-a-name", "value")
)
def update_versions_a(name):
    versions = get_versions(name)
    opts = [{"label": v, "value": v} for v in versions]
    return opts, (versions[0] if versions else None)


@app.callback(
    Output("model-b-version", "options"),
    Output("model-b-version", "value"),
    Input("model-b-name", "value")
)
def update_versions_b(name):
    versions = get_versions(name)
    opts = [{"label": v, "value": v} for v in versions]
    return opts, (versions[0] if versions else None)

# -----------------------
# Generation callback
# -----------------------
@app.callback(
    Output("output-a", "children"),
    Output("output-b", "children"),
    Output("history", "data"),
    Input("generate", "n_clicks"),
    Input("prompt", "n_submit"),
    State("model-a-name", "value"),
    State("model-a-version", "value"),
    State("model-b-name", "value"),
    State("model-b-version", "value"),
    State("prompt", "value"),
    State("temperature", "value"),
    State("top_k", "value"),
    State("top_p", "value"),
    State("max_tokens", "value"),
    State("history", "data"),
    prevent_initial_call=True
)
def generate(n_clicks, n_submit, a_name, a_ver, b_name, b_ver, prompt, temp, top_k, top_p, max_tokens, history):

    # 🚨 prevents accidental execution on load
    if not ctx.triggered_id:
        raise dash.exceptions.PreventUpdate

    if not a_name or not a_ver:
        return "Select Model A", "", history

    params = dict(
        max_new_tokens=max_tokens,
        temperature=temp,
        top_k=top_k,
        top_p=top_p
    )

    # Model A
    uri_a = f"models:/{a_name}/{a_ver}"
    model_a = get_model(uri_a)
    res_a = model_a.generate_from_prompts([prompt], **params)[0]["completion"]

    # Model B
    res_b = ""
    if b_name and b_ver:
        uri_b = f"models:/{b_name}/{b_ver}"
        model_b = get_model(uri_b)
        res_b = model_b.generate_from_prompts([prompt], **params)[0]["completion"]

    history.append({"prompt": prompt, "A": res_a, "B": res_b})

    return res_a, res_b, history

# -----------------------
# History display
# -----------------------
@app.callback(
    Output("history-display", "children"),
    Input("history", "data")
)
def render(history):
    if not history:
        return "No history yet."

    return [
        html.Div([
            html.P(f"Prompt: {h['prompt']}"),
            html.P(f"A: {h['A'][:200]}"),
            html.P(f"B: {h['B'][:200]}"),
            html.Hr()
        ])
        for h in reversed(history[-10:])
    ]

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)