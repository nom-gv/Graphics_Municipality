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

# Funciones para obtener datos de los archivos
def get_casos_cancer():
    df_c_cancer = pd.read_excel('CasosCancer.xlsx', sheet_name="CANCER-C")
    df_g_cancer = pd.read_excel('CasosCancer.xlsx', sheet_name="CANCER-G")
    df_pc_cancer = pd.read_excel('CasosCancer.xlsx', sheet_name="CANCER-PC")
    df_sc_cancer = pd.read_excel('CasosCancer.xlsx', sheet_name="CANCER-SC")
    return df_c_cancer, df_g_cancer, df_pc_cancer, df_sc_cancer

# Función para mostrar los DataFrames en HTML
def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

# Definir el layout de la aplicación
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        dcc.Link('Home', href='/'),
        ' | ',
        dcc.Link('Cancer', href='/cancer'),
        ' | ',
        dcc.Link('Diabetes', href='/diabetes'),
        ' | ',
        dcc.Link('Hipertension', href='/hipertension')
    ], style={'padding': '20px', 'background-color': '#f0f0f0', 'text-align': 'center'}),
    html.Div(id='page-content')
])

# Callback para actualizar el contenido según la URL
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/cancer':
        df_c_cancer, df_g_cancer, df_pc_cancer, df_sc_cancer = get_casos_cancer()
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Cancer'),
            html.H3('Datos de CANCER-C:'),
            generate_table(df_c_cancer),
            html.H3('Datos de CANCER-G:'),
            generate_table(df_g_cancer),
            html.H3('Datos de CANCER-PC:'),
            generate_table(df_pc_cancer),
            html.H3('Datos de CANCER-SC:'),
            generate_table(df_sc_cancer)
        ])
    elif pathname == '/diabetes':
        df_c_diabetes, df_g_diabetes, df_pc_diabetes, df_sc_diabetes = get_casos_diabetes()
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Diabetes'),
            html.H3('Datos de DIABETES-C:'),
            generate_table(df_c_diabetes),
            html.H3('Datos de DIABETES-G:'),
            generate_table(df_g_diabetes),
            html.H3('Datos de DIABETES-PC:'),
            generate_table(df_pc_diabetes),
            html.H3('Datos de DIABETES-SC:'),
            generate_table(df_sc_diabetes)
        ])
    elif pathname == '/hipertension':
        df_c_hipertension, df_g_hipertension, df_pc_hipertension, df_sc_hipertension = get_casos_hipertension()
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Hipertensión'),
            html.H3('Datos de HIPERTENSION-C:'),
            generate_table(df_c_hipertension),
            html.H3('Datos de HIPERTENSION-G:'),
            generate_table(df_g_hipertension),
            html.H3('Datos de HIPERTENSION-PC:'),
            generate_table(df_pc_hipertension),
            html.H3('Datos de HIPERTENSION-SC:'),
            generate_table(df_sc_hipertension)
        ])
    else:
        return html.Div([
            html.H1('Mi primera aplicación Dash en Heroku' + pathname),
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

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
