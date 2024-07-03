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

# Define las funciones para descargar y leer los datos
def get_casos_cancer():
    df_c_cancer = pd.read_excel('CasosCancer.xlsx', sheet_name="CANCER-C")
    df_g_cancer = pd.read_excel('CasosCancer.xlsx', sheet_name="CANCER-G")
    df_pc_cancer = pd.read_excel('CasosCancer.xlsx', sheet_name="CANCER-PC")
    df_sc_cancer = pd.read_excel('CasosCancer.xlsx', sheet_name="CANCER-SC")
    return df_c_cancer, df_g_cancer, df_pc_cancer, df_sc_cancer

def get_casos_diabetes():
    df_c_diabetes = pd.read_excel('CasosDiabetes.xlsx', sheet_name="DIABETES-C")
    df_g_diabetes = pd.read_excel('CasosDiabetes.xlsx', sheet_name="DIABETES-G")
    df_pc_diabetes = pd.read_excel('CasosDiabetes.xlsx', sheet_name="DIABETES-PC")
    df_sc_diabetes = pd.read_excel('CasosDiabetes.xlsx', sheet_name="DIABETES-SC")
    return df_c_diabetes, df_g_diabetes, df_pc_diabetes, df_sc_diabetes

def get_casos_hipertension():
    df_c_hipertension = pd.read_excel('CasosHipertensionArterial.xlsx', sheet_name="HIPERTENSION-C")
    df_g_hipertension = pd.read_excel('CasosHipertensionArterial.xlsx', sheet_name="HIPERTENSION-G")
    df_pc_hipertension = pd.read_excel('CasosHipertensionArterial.xlsx', sheet_name="HIPERTENSION-PC")
    df_sc_hipertension = pd.read_excel('CasosHipertensionArterial.xlsx', sheet_name="HIPERTENSION-SC")
    return df_c_hipertension, df_g_hipertension, df_pc_hipertension, df_sc_hipertension

def get_casos_obesidad():
    df_c_obesidad = pd.read_excel('CasosObesidad.xlsx', sheet_name="OBESIDAD-C")
    df_g_obesidad = pd.read_excel('CasosObesidad.xlsx', sheet_name="OBESIDAD-G")
    df_pc_obesidad = pd.read_excel('CasosObesidad.xlsx', sheet_name="OBESIDAD-PC")
    df_sc_obesidad = pd.read_excel('CasosObesidad.xlsx', sheet_name="OBESIDAD-SC")
    return df_c_obesidad, df_g_obesidad, df_pc_obesidad, df_sc_obesidad

def get_casos_neumonia():
    df_c_neumonia = pd.read_excel('CasosNeumonia.xlsx', sheet_name="NEUMONIA-C")
    df_g_neumonia = pd.read_excel('CasosNeumonia.xlsx', sheet_name="NEUMONIA-G")
    df_pc_neumonia = pd.read_excel('CasosNeumonia.xlsx', sheet_name="NEUMONIA-PC")
    df_sc_neumonia = pd.read_excel('CasosNeumonia.xlsx', sheet_name="NEUMONIA-SC")
    return df_c_neumonia, df_g_neumonia, df_pc_neumonia, df_sc_neumonia

def get_casos_chagas():
    df_c_chagas = pd.read_excel('CasosChagas.xlsx', sheet_name="CHAGAS-C")
    df_g_chagas = pd.read_excel('CasosChagas.xlsx', sheet_name="CHAGAS-G")
    df_pc_chagas = pd.read_excel('CasosChagas.xlsx', sheet_name="CHAGAS-PC")
    df_sc_chagas = pd.read_excel('CasosChagas.xlsx', sheet_name="CHAGAS-SC")
    return df_c_chagas, df_g_chagas, df_pc_chagas, df_sc_chagas

def get_casos_vih():
    df_c_vih = pd.read_excel('CasosVIH.xlsx', sheet_name="VIH-C")
    df_g_vih = pd.read_excel('CasosVIH.xlsx', sheet_name="VIH-G")
    df_pc_vih = pd.read_excel('CasosVIH.xlsx', sheet_name="VIH-PC")
    df_sc_vih = pd.read_excel('CasosVIH.xlsx', sheet_name="VIH-SC")
    return df_c_vih, df_g_vih, df_pc_vih, df_sc_vih

def get_casos_nutricion():
    df_c_obesidad = pd.read_excel('CasosEstadoNutricional.xlsx', sheet_name="OBESIDAD-C")
    df_g_obesidad = pd.read_excel('CasosEstadoNutricional.xlsx', sheet_name="OBESIDAD-G")
    df_pc_obesidad = pd.read_excel('CasosEstadoNutricional.xlsx', sheet_name="OBESIDAD-PC")
    df_sc_obesidad = pd.read_excel('CasosEstadoNutricional.xlsx', sheet_name="OBESIDAD-SC")
    
    df_c_sobrepeso = pd.read_excel('CasosEstadoNutricional.xlsx', sheet_name="SOBREPESO-C")
    df_g_sobrepeso = pd.read_excel('CasosEstadoNutricional.xlsx', sheet_name="SOBREPESO-G")
    df_pc_sobrepeso = pd.read_excel('CasosEstadoNutricional.xlsx', sheet_name="SOBREPESO-PC")
    df_sc_sobrepeso = pd.read_excel('CasosEstadoNutricional.xlsx', sheet_name="SOBREPESO-SC")
    
    df_c_bajopeso = pd.read_excel('CasosEstadoNutricional.xlsx', sheet_name="BAJOPESO-C")
    df_g_bajopeso = pd.read_excel('CasosEstadoNutricional.xlsx', sheet_name="BAJOPESO-G")
    df_pc_bajopeso = pd.read_excel('CasosEstadoNutricional.xlsx', sheet_name="BAJOPESO-PC")
    df_sc_bajopeso = pd.read_excel('CasosEstadoNutricional.xlsx', sheet_name="BAJOPESO-SC")
    return df_c_obesidad, df_g_obesidad, df_pc_obesidad, df_sc_obesidad, df_c_sobrepeso, df_g_sobrepeso, df_pc_sobrepeso, df_sc_sobrepeso, df_c_bajopeso, df_g_bajopeso, df_pc_bajopeso, df_sc_bajopeso

def get_casos_embarazo():
    df_c_embarazo = pd.read_excel('CasosEmbarazoAdolescente.xlsx', sheet_name="EMBARAZO-C")
    df_g_embarazo = pd.read_excel('CasosEmbarazoAdolescente.xlsx', sheet_name="EMBARAZO-G")
    df_pc_embarazo = pd.read_excel('CasosEmbarazoAdolescente.xlsx', sheet_name="EMBARAZO-PC")
    df_sc_embarazo = pd.read_excel('CasosEmbarazoAdolescente.xlsx', sheet_name="EMBARAZO-SC")
    return df_c_embarazo, df_g_embarazo, df_pc_embarazo, df_sc_embarazo

def get_casos_consulta():
    df_c_consulta = pd.read_excel('CasosConsultaExterna.xlsx', sheet_name="CONSULTAS-C")
    df_g_consulta = pd.read_excel('CasosConsultaExterna.xlsx', sheet_name="CONSULTAS-G")
    df_pc_consulta = pd.read_excel('CasosConsultaExterna.xlsx', sheet_name="CONSULTAS-PC")
    df_sc_consulta = pd.read_excel('CasosConsultaExterna.xlsx', sheet_name="CONSULTAS-SC")
    return df_c_consulta, df_g_consulta, df_pc_consulta, df_sc_consulta

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

# Define el layout de la aplicación
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.H2("Menú"),
        html.Ul([
            html.Li(dcc.Link('Cancer', href='/cancer')),
            html.Li(dcc.Link('Diabetes', href='/diabetes')),
            html.Li(dcc.Link('Hipertensión Arterial', href='/hipertension')),
            html.Li(dcc.Link('Obesidad', href='/obesidad')),
            html.Li(dcc.Link('Neumonia', href='/neumonia')),
            html.Li(dcc.Link('Chagas', href='/chagas')),
            html.Li(dcc.Link('VIH', href='/vih')),
            html.Li(dcc.Link('Nutrición Embarazo', href='/nutricion')),
            html.Li(dcc.Link('Embarazo Adolescente', href='/adolescentes')),
            html.Li(dcc.Link('Consultas Externas', href='/consultas')),
        ], className='menu')
    ], className='menu-column'),
    html.Div([
        #html.H1(""),
        html.Div(id='page-content')
    ], className='content-column'),
    html.Div(id='btn-calcular', style={'display': 'none'}),  # Div oculto para generar el botón
], className='container')

# Definir opciones de dataframes
opciones_dataframes = [
    {'label': 'Camiri', 'value': 'Camiri'},
    {'label': 'Gutierrez', 'value': 'Gutierrez'},
    {'label': 'Cordillera', 'value': 'Cordillera'},
    {'label': 'Santa Cruz', 'value': 'Santa Cruz'}
]

# Define el layout de la página de cálculo
calculo_layout = html.Div([
    html.Div([
        html.Div([
            html.H4('Población Camiri'),
            html.Div([
                html.P('Mujeres: '),
                html.Label('2019:'),
                dcc.Input(id='input-m1', type='number', value=21308, style={'width': '80px'}),
                html.Label('2020:'),
                dcc.Input(id='input-m2', type='number', value=21577, style={'width': '80px'}),
                html.Label('2021:'),
                dcc.Input(id='input-m3', type='number', value=21835, style={'width': '80px'}),
                html.Label('2022:'),
                dcc.Input(id='input-m4', type='number', value=22081, style={'width': '80px'}),
                html.Label('2023:'),
                dcc.Input(id='input-m5', type='number', value=22315, style={'width': '80px'}),
            ]),
            html.Div([
                html.P('Hombres: '),
                html.Label('2019:'),
                dcc.Input(id='input-h1', type='number', value=19891, style={'width': '80px'}),
                html.Label('2020:'),
                dcc.Input(id='input-h2', type='number', value=20115, style={'width': '80px'}),
                html.Label('2021:'),
                dcc.Input(id='input-h3', type='number', value=20329, style={'width': '80px'}),
                html.Label('2022:'),
                dcc.Input(id='input-h4', type='number', value=20532, style={'width': '80px'}),
                html.Label('2023:'),
                dcc.Input(id='input-h5', type='number', value=20724, style={'width': '80px'}),
            ]),
        ], className='population-section'),
        html.Div([
            html.H4('Población Gutierrez'),
            html.Div([
                html.P('Mujeres: '),
                html.Label('2019:'),
                dcc.Input(id='input-m1-2', type='number', value=7692, style={'width': '80px'}),
                html.Label('2020:'),
                dcc.Input(id='input-m2-2', type='number', value=7786, style={'width': '80px'}),
                html.Label('2021:'),
                dcc.Input(id='input-m3-2', type='number', value=7876, style={'width': '80px'}),
                html.Label('2022:'),
                dcc.Input(id='input-m4-2', type='number', value=7962, style={'width': '80px'}),
                html.Label('2023:'),
                dcc.Input(id='input-m5-2', type='number', value=8044, style={'width': '80px'}),
            ]),
            html.Div([
                html.P('Hombres: '),
                html.Label('2019:'),
                dcc.Input(id='input-h1-2', type='number', value=7719, style={'width': '80px'}),
                html.Label('2020:'),
                dcc.Input(id='input-h2-2', type='number', value=7803, style={'width': '80px'}),
                html.Label('2021:'),
                dcc.Input(id='input-h3-2', type='number', value=7883, style={'width': '80px'}),
                html.Label('2022:'),
                dcc.Input(id='input-h4-2', type='number', value=7959, style={'width': '80px'}),
                html.Label('2023:'),
                dcc.Input(id='input-h5-2', type='number', value=8030, style={'width': '80px'}),

            ]),
        ], className='population-section'),
    ], className='substrac-section'),
    
    html.Div([
        html.Div([
            html.H4('Población Cordillera'),
            html.Div([
                html.P('Mujeres: '),
                html.Label('2019:'),
                dcc.Input(id='input-m1-3', type='number', value=67693, style={'width': '80px'}),
                html.Label('2020:'),
                dcc.Input(id='input-m2-3', type='number', value=68365, style={'width': '80px'}),
                html.Label('2021:'),
                dcc.Input(id='input-m3-3', type='number', value=69000, style={'width': '80px'}),
                html.Label('2022:'),
                dcc.Input(id='input-m4-3', type='number', value=69595, style={'width': '80px'}),
                html.Label('2023:'),
                dcc.Input(id='input-m5-3', type='number', value=70149, style={'width': '80px'}),
            ]),
            html.Div([
                html.P('Hombres: '),
                html.Label('2019:'),
                dcc.Input(id='input-h1-3', type='number', value=70822, style={'width': '80px'}),
                html.Label('2020:'),
                dcc.Input(id='input-h2-3', type='number', value=71422, style={'width': '80px'}),
                html.Label('2021:'),
                dcc.Input(id='input-h3-3', type='number', value=71981, style={'width': '80px'}),
                html.Label('2022:'),
                dcc.Input(id='input-h4-3', type='number', value=72501, style={'width': '80px'}),
                html.Label('2023:'),
                dcc.Input(id='input-h5-3', type='number', value=72980, style={'width': '80px'}),
            ]),
        ], className='population-section'),
        html.Div([
            html.H4('Población Santa Cruz'),
            html.Div([
                html.P('Mujeres: '),
                html.Label('2019:'),
                dcc.Input(id='input-m1-4', type='number', value=1599058, style={'width': '80px'}),
                html.Label('2020:'),
                dcc.Input(id='input-m2-4', type='number', value=1631632, style={'width': '80px'}),
                html.Label('2021:'),
                dcc.Input(id='input-m3-4', type='number', value=1663929, style={'width': '80px'}),
                html.Label('2022:'),
                dcc.Input(id='input-m4-4', type='number', value=1695862, style={'width': '80px'}),
                html.Label('2023:'),
                dcc.Input(id='input-m5-4', type='number', value=1727403, style={'width': '80px'}),
            ]),
            html.Div([
                html.P('Hombres: '),
                html.Label('2019:'),
                dcc.Input(id='input-h1-4', type='number', value=1638165, style={'width': '80px'}),
                html.Label('2020:'),
                dcc.Input(id='input-h2-4', type='number', value=1668971, style={'width': '80px'}),
                html.Label('2021:'),
                dcc.Input(id='input-h3-4', type='number', value=1699448, style={'width': '80px'}),
                html.Label('2022:'),
                dcc.Input(id='input-h4-4', type='number', value=1729537, style={'width': '80px'}),
                html.Label('2023:'),
                dcc.Input(id='input-h5-4', type='number', value=1759221, style={'width': '80px'}),
            ]),
        ], className='population-section'),
    ], className='substrac-section'),
    
    html.Div([
        html.Span('Factor'),
        dcc.Input(id='input-factor', type='number', value=10000, style={'width': '80px'})
    ]),
    
    html.H1("Gráficos de Tendencia"),
    html.Label('Grafica a mostrar:'),
    dcc.Dropdown(
        id='dropdown-graphic-type',
        options=[
            {'label': 'Totales', 'value': 't'},
            {'label': 'Por sexo (Comparacion Junto Municipio, Provincia, Departamento)', 'value': 's1'},
            {'label': 'Por sexo (Comparacion Separado Municipio, Provincia, Departamento)', 'value': 's2'},
            {'label': 'Por sexo (Comparacion entre mujeres y hombres)', 'value': 's3'},
            {'label': 'Por edad', 'value': 'e'},
        ],
        value='t'  # Valor inicial seleccionado
    ),
    html.Label('Porcentaje o Incidencias:'),
    dcc.Dropdown(
        id='dropdown-type-percent',
        options=[
            {'label': 'Incidencias', 'value': 'Incidencia'},
            {'label': 'Porcentajes', 'value': 'Porcentaje'},
        ],
        value='Incidencia'  # Valor inicial seleccionado
    ),
    html.Label('Seleccionar dataframes para graficar:'),
    dcc.Dropdown(
        id='dropdown-dataframes',
        options=opciones_dataframes,
        multi=True,
        value=['Santa Cruz', 'Cordillera', 'Camiri']  # Valores iniciales seleccionados
    ),
    html.Div([
        html.Label('Título del gráfico: '),
        dcc.Input(
            id='input-titulo',
            type='text',
            value='Tendencia'
        ),
        html.Label("Tamaño de letra titulo: "),
        dcc.Input(
            id='input-tamaño-titulo',
            type='number',
            value='16'
        )
    ]),
    
    html.Div([
        html.Label('Pie de Pagina: '),
        dcc.Input(
            id='input-pie',
            type='text',
            value='Datos obtenidos de la página del SNIS'
        ),
        html.Label("Tamaño de letra pie: "),
        dcc.Input(
            id='input-tamaño-pie',
            type='number',
            value='10'
        )
    ]),
    
    html.Label('Ubicación de la leyenda:'),
    dcc.Dropdown(
        id='dropdown-legend-loc',
        options=[
            {'label': 'Arriba a la izquierda', 'value': 'upper left'},
            {'label': 'Arriba a la derecha', 'value': 'upper right'},
            {'label': 'Abajo a la izquierda', 'value': 'lower left'},
            {'label': 'Abajo a la derecha', 'value': 'lower right'}
        ],
        value='upper left'  # Valor inicial seleccionado
    ),
    
    html.Div([
        html.Label('Tamaño de letra leyenda: '),
        dcc.Input(
            id='input-tamaño-leyenda',
            type='number',
            value='8',
            style={'width': '80px'}
        ),
        html.Label("Tamaño de letra de Numeros Graficas: "),
        dcc.Input(
            id='input-tamaño-num-grafica',
            type='number',
            value='10',
            style={'width': '80px'}
        )
    ]),
    
    html.Button('Generar Gráfico', id='btn-calcular'),
    html.Div(id='output-data')
])


# Callback para actualizar el contenido según la URL
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/cancer':
        #df_c_cancer, df_g_cancer, df_pc_cancer, df_sc_cancer = get_casos_cancer()
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Cancer'),
        ]), calculo_layout
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
            generate_table(df_sc_diabetes),
            html.P('Hola mundo'+pathname)
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
            generate_table(df_sc_hipertension),
            html.P('Hola mundo'+pathname)
        ])
    else:
        return html.Div([
            html.H1('Mi primera aplicación Dash en Heroku'),
            html.P('Hola mundo'+pathname),
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