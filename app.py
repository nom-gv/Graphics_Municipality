import os
import gdown
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import threading

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

# Inicialización de la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # Definir el servidor para Heroku

# Diseño de la aplicación Dash
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Funciones para obtener datos de los archivos
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

# Otras funciones de obtención de datos para los demás archivos...

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

# Callback para actualizar el contenido según la URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
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
    # Otras rutas y DataFrames correspondientes...

    else:
        return html.Div('Selecciona una opción del menú')

# Iniciar la aplicación Dash
if __name__ == '__main__':
    app.run_server(debug=True)
