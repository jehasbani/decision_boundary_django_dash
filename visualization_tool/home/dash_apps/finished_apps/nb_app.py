import time
# import dash
from django_plotly_dash import DjangoDash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

import home.dash_apps.utils.dash_reusable_components as drc
import home.dash_apps.utils.figures as figs
from home.dash_apps.utils.ml_resources import Model, Dataset

app = DjangoDash(
    'GaussianNB',
)

"""
meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
"""
# server = app.server


app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        # Change App Name here
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "Gaussian Naive Bayes Explorer",
                                    href="https://github.com/plotly/dash-svm",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),
                        html.A(
                            id="banner-logo",
                            children=[
                                html.Img(src=app.get_asset_url("dash_apps-logo-new.png"))
                            ],
                            href="https://plot.ly/products/dash/",
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    # className="row",
                    children=[
                        html.Div(
                            # className="three columns",
                            id="left-column",
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Select Dataset",
                                            id="dropdown-select-dataset",
                                            options=[
                                                {"label": "Moons", "value": "moons"},
                                                {
                                                    "label": "Linearly Separable",
                                                    "value": "linear",
                                                },
                                                {
                                                    "label": "Circles",
                                                    "value": "circles",
                                                },
                                            ],
                                            clearable=False,
                                            searchable=False,
                                            value="moons",
                                        ),
                                        drc.NamedSlider(
                                            name="Sample Size",
                                            id="slider-dataset-sample-size",
                                            min=100,
                                            max=500,
                                            step=100,
                                            marks={
                                                str(i): str(i)
                                                for i in [100, 200, 300, 400, 500]
                                            },
                                            value=300,
                                        ),
                                        drc.NamedSlider(
                                            name="Noise Level",
                                            id="slider-dataset-noise-level",
                                            min=0,
                                            max=1,
                                            marks={
                                                i / 10: str(i / 10)
                                                for i in range(0, 11, 2)
                                            },
                                            step=0.1,
                                            value=0.2,
                                        ),
                                    ],
                                ),
                                drc.Card(
                                    id="button-card",
                                    children=[
                                        drc.NamedSlider(
                                            name="Threshold",
                                            id="slider-threshold",
                                            min=0,
                                            max=1,
                                            value=0.5,
                                            step=0.01,
                                        ),
                                        html.Button(
                                            "Reset Threshold",
                                            id="button-zero-threshold",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="div-graphs",
                            children=dcc.Graph(
                                id="graph-sklearn-gbn",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                    )
                                ),
                            ),
                        ),
                    ],
                )
            ],
        ),
    ]
)


@app.callback(
    Output("slider-threshold", "value"),
    [Input("button-zero-threshold", "n_clicks")],
    [State("graph-sklearn-gbn", "figure")],
)
def reset_threshold_center(n_clicks, figure):
    if n_clicks:
        Z = np.array(figure["data"][0]["z"])
        value = -Z.min() / (Z.max() - Z.min())
    else:
        value = 0.4959986285375595
    return value


@app.callback(
    Output("div-graphs", "children"),
    [
        Input("dropdown-select-dataset", "value"),
        Input("slider-dataset-noise-level", "value"),
        Input("slider-threshold", "value"),
        Input("slider-dataset-sample-size", "value"),
    ],
)
def update_gnb_graph(
    dataset_name,
    noise,
    threshold,
    sample_size,
):
    t_start = time.time()
    h = 0.3  # step size in the mesh

    # Data Pre-processing
    data = Dataset(name=dataset_name, n_samples=sample_size, noise=noise)
    X = StandardScaler().fit_transform(data.X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, data.y, test_size=0.4, random_state=42
    )

    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5
    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


    # Train SVM
    gnb = Model(name='Naive Bayes')
    clf = gnb.clf
    clf.fit(X_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    prediction_figure = figs.serve_prediction_plot(
        model=clf,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        Z=Z,
        xx=xx,
        yy=yy,
        mesh_step=h,
        threshold=threshold,
    )

    roc_figure = figs.serve_roc_curve(model=clf, X_test=X_test, y_test=y_test)

    confusion_figure = figs.serve_pie_confusion_matrix(
        model=clf, X_test=X_test, y_test=y_test, Z=Z, threshold=threshold
    )

    return [
        html.Div(
            id="gnb-graph-container",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="graph-sklearn-gbn", figure=prediction_figure),
                style={"display": "none"},
            ),
        ),
        html.Div(
            id="graphs-container",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(id="graph-line-roc-curve", figure=roc_figure),
                ),
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(
                        id="graph-pie-confusion-matrix", figure=confusion_figure
                    ),
                ),
            ],
        ),
    ]


"""
# For Django this is commented out
# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
"""