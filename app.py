import dash
import gdown
import base64
import io
import threading
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dash import dcc
from dash import html
from dash import Input, Output, State
import plotly.graph_objs as go

# Función para descargar y guardar archivos desde Google Drive
def download_and_save(nombre, file_id):
    try:
        # Descargar el archivo desde Google Drive
        output_file = nombre
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        gdown.download(url, output_file, quiet=False)

        # Leer el archivo descargado
        df = pd.read_excel(output_file)

        return df

    except Exception as e:
        print(f'Error al descargar y guardar el archivo {nombre}: {str(e)}')
        return None

# Lista de archivos para descargar desde Google Drive
archivos = [
    ('CasosCancer.xlsx', '1oRB3DMP1NtnnwfQcaYHo9a3bUcbQfB5U'),
    ('CasosDiabetes.xlsx', '1xHYonZp8RbPYCE9kihc3IthwOtgVNi1P'),
    ('CasosHipertensionArterial.xlsx', '1_jue36lk4iJim6btVh_tSUkR0i_QGeIk'),
    ('CasosObesidad.xlsx', '19aVPGne2nPm7_I0L9i_csyEBRw9geGea'),
    ('CasosNeumonia.xlsx', '1tK7dDEo1b7gWn-KHl1qE_WL62ztrygHw'),
    ('CasosChagas.xlsx', '1kAXyvg1cvLtl7w8a6D1AijMwFLJiialT'),
    ('CasosVIH.xlsx', '1xmnFEOBzaIZa3Ah4daAVEMo4HeLCVyZK'),
    ('CasosEstadoNutricional.xlsx', '1G8k9bqzJop0dSgFjigeVrzVQiuHuUFUp'),
    ('CasosEmbarazoAdolescente.xlsx', '1WGjRPOdiKjbblojvO96WpkfSITvbpvsH'),
    ('CasosConsultaExterna.xlsx', '1iA8HOY1nCGd62dqL1RU3MMgitXKT1a4q')
]

# Función para descargar todos los archivos en un hilo separado
def descargar_archivos():
    for nombre, file_id in archivos:
        download_and_save(nombre, file_id)

# Descargar archivos en un hilo separado
descarga_thread = threading.Thread(target=descargar_archivos)
descarga_thread.start()

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
                {'x': [1, 2, 3], 'y': [8, 5, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': 'Montreal'},
            ],
            'layout': {
                'title': 'ACTUALIZACION PAGINA'
            }
        }
    )
])

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
