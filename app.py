import dash
import gdown
import base64
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dash import dcc
from dash import html
from dash import Input, Output, State
import plotly.graph_objs as go

# Inicializar la aplicación Dash
app = dash.Dash(__name__)

# Definir el servidor
server = app.server

# Definir el layout de la aplicación
app.layout = html.Div([
    html.H1('Mi primera aplicación Dash en Heroku'),
    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': 'Montreal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    )
])

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
