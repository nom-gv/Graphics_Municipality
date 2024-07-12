import dash
import requests
import base64
import io
import threading
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dash import dcc
from dash import html
from dash import dash_table
from dash import Input, Output, State
import plotly.graph_objs as go

# Función para descargar y guardar archivos desde Google Drive
# Función para descargar y guardar archivos desde Google Drive usando requests
def download_and_save(nombre, file_id):
    try:
        # Construir la URL de descarga
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        
        # Realizar la solicitud GET
        response = requests.get(url)
        
        # Verificar si la descarga fue exitosa (código 200)
        if response.status_code == 200:
            # Guardar el contenido de la respuesta en un archivo
            with open(nombre, 'wb') as f:
                f.write(response.content)
            
            # Leer el archivo descargado con pandas
            df = pd.read_excel(nombre)
            return df
        
        else:
            print(f'Error al descargar el archivo {nombre}: Código de estado {response.status_code}')
            return None
    
    except Exception as e:
        print(f'Error al descargar y guardar el archivo {nombre}: {str(e)}')
        return None

# Lista de archivos para descargar desde Google Drive
archivos = [
    ('CasosCancer-Guaranis.xlsx', '1oRB3DMP1NtnnwfQcaYHo9a3bUcbQfB5U'),
    ('CasosDiabetes-Guaranis.xlsx', '1xHYonZp8RbPYCE9kihc3IthwOtgVNi1P'),
    ('CasosHipertensionArterial-Guaranis.xlsx', '1_jue36lk4iJim6btVh_tSUkR0i_QGeIk'),
    ('CasosObesidad-Guaranis.xlsx', '19aVPGne2nPm7_I0L9i_csyEBRw9geGea'),
    ('CasosNeumonia-Guaranis.xlsx', '1tK7dDEo1b7gWn-KHl1qE_WL62ztrygHw'),
    ('CasosChagas-Guaranis.xlsx', '1kAXyvg1cvLtl7w8a6D1AijMwFLJiialT'),
    ('CasosVIH-Guaranis.xlsx', '1xmnFEOBzaIZa3Ah4daAVEMo4HeLCVyZK'),
    ('CasosEstadoNutricional-Guaranis.xlsx', '1G8k9bqzJop0dSgFjigeVrzVQiuHuUFUp'),
    ('CasosEmbarazoAdolescente-Guaranis.xlsx', '1WGjRPOdiKjbblojvO96WpkfSITvbpvsH'),
    ('CasosConsultaExterna-Guaranis.xlsx', '1iA8HOY1nCGd62dqL1RU3MMgitXKT1a4q'),
    ('DatosPoblaciones-Guaranis.xlsx','1Tkr9PBQJHAb5m8zq8k-EFl1bvUMvq-5B'),
    ('DatosEspeciales-Guaranis.xlsx','1LJqErF3ZhuMdl8HBfJUQIcsebon6cN5K'),
    ('CasosCancer-Afrobolivianos.xlsx', '1ysmxBKWrHeC3xXmK1RzuL5-eveaNXqj1'),
    ('CasosDiabetes-Afrobolivianos.xlsx', '1L1XoqEI1ysMxq3TTNLgW1Ji5AUGPN4C4'),
    ('CasosHipertensionArterial-Afrobolivianos.xlsx', '1Rha7FxxGEDaJSLG-mjemzRZuS0rwWTLK'),
    ('CasosObesidad-Afrobolivianos.xlsx', '1V3W07eB4HwZOB-Tnn-Q1uU2MFB0hXpbV'),
    ('CasosNeumonia-Afrobolivianos.xlsx', '1dCVGa3sHmhlglO7j5thD0M8WXdrjpNV7'),
    ('CasosChagas-Afrobolivianos.xlsx', '1SgV1pzBKc2_5dCtQ4xDLgm2QNU-9HA5_'),
    ('CasosVIH-Afrobolivianos.xlsx', '11IWn0JXocoZ2Rh0zwbgV32ijI4zytQIN'),
    ('CasosEstadoNutricional-Afrobolivianos.xlsx', '1PGbomzjaufJt6mOSPtC3B7MIyKBymgxn'),
    ('DatosPoblaciones-Afrobolivianos.xlsx','1_j05pCS_IeudCbt7oTm38tzbhYi2cbRc'),
    ('DatosEspeciales-Afrobolivianos.xlsx','1TOHGe0-akhPcUgpFQN4uNFUvVbUeOiKo')
]

# Función para descargar todos los archivos en un hilo separado
def descargar_archivos():
    for nombre, file_id in archivos:
        download_and_save(nombre, file_id)

# Descargar archivos en un hilo separado
descarga_thread = threading.Thread(target=descargar_archivos)
descarga_thread.start()

# Inicializar la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
# Definir el servidor
server = app.server

def get_casos(tipo, comunidad):
    file_map = {
        'cancer': {
            'guarani': 'CasosCancer-Guaranis.xlsx',
            'afroboliviano': 'CasosCancer-Afrobolivianos.xlsx'
        },
        'diabetes': {
            'guarani': 'CasosDiabetes-Guaranis.xlsx',
            'afroboliviano': 'CasosDiabetes-Afrobolivianos.xlsx'
        },
        'hipertension': {
            'guarani': 'CasosHipertensionArterial-Guaranis.xlsx',
            'afroboliviano': 'CasosHipertensionArterial-Afrobolivianos.xlsx'
        },
        'obesidad': {
            'guarani': 'CasosObesidad-Guaranis.xlsx',
            'afroboliviano': 'CasosObesidad-Afrobolivianos.xlsx'
        },
        'neumonia': {
            'guarani': 'CasosNeumonia-Guaranis.xlsx',
            'afroboliviano': 'CasosNeumonia-Afrobolivianos.xlsx'
        },
        'chagas': {
            'guarani': 'CasosChagas-Guaranis.xlsx',
            'afroboliviano': 'CasosChagas-Afrobolivianos.xlsx'
        },
        'vih': {
            'guarani': 'CasosVIH-Guaranis.xlsx',
            'afroboliviano': 'CasosVIH-Afrobolivianos.xlsx'
        },
        'nutricion': {
            'guarani': 'CasosEstadoNutricional-Guaranis.xlsx',
            'afroboliviano': 'CasosEstadoNutricional-Afrobolivianos.xlsx'
        },
        'poblacion': {
            'guarani': 'DatosPoblaciones-Guaranis.xlsx',
            'afroboliviano': 'DatosPoblaciones-Afrobolivianos.xlsx'
        },
        'poblacion-especial': {
            'guarani': 'DatosEspeciales-Guaranis.xlsx',
            'afroboliviano': 'DatosEspeciales-Afrobolivianos.xlsx'
        }
    }
    
    sheets_map = {
        'cancer': {
            'guarani': ["CANCER-C", "CANCER-G", "CANCER-PC", "CANCER-SC"],
            'afroboliviano': ["CANCER-A", "CANCER-I", "CANCER-CO", "CANCER-CP", "CANCER-SY", "CANCER-NY", "CANCER-LP"]
        },
        'diabetes': {
            'guarani': ["DIABETES-C", "DIABETES-G", "DIABETES-PC", "DIABETES-SC"],
            'afroboliviano': ["DIABETES-A", "DIABETES-I", "DIABETES-CO", "DIABETES-CP", "DIABETES-SY", "DIABETES-NY", "DIABETES-LP"]
        },
        'hipertension': {
            'guarani': ["HIPERTENSION-C", "HIPERTENSION-G", "HIPERTENSION-PC", "HIPERTENSION-SC"],
            'afroboliviano': ["HIPERTENSION-A", "HIPERTENSION-I", "HIPERTENSION-CO", "HIPERTENSION-CP", "HIPERTENSION-SY", "HIPERTENSION-NY", "HIPERTENSION-LP"]
        },
        'obesidad': {
            'guarani': ["OBESIDAD-C", "OBESIDAD-G", "OBESIDAD-PC", "OBESIDAD-SC"],
            'afroboliviano': ["OBESIDAD-A", "OBESIDAD-I", "OBESIDAD-CO", "OBESIDAD-CP", "OBESIDAD-SY", "OBESIDAD-NY", "OBESIDAD-LP"]
        },
        'neumonia': {
            'guarani': ["NEUMONIA-C", "NEUMONIA-G", "NEUMONIA-PC", "NEUMONIA-SC"],
            'afroboliviano': ["NEUMONIA-A", "NEUMONIA-I", "NEUMONIA-CO", "NEUMONIA-CP", "NEUMONIA-SY", "NEUMONIA-NY", "NEUMONIA-LP"]
        },
        'chagas': {
            'guarani': ["CHAGAS-C", "CHAGAS-G", "CHAGAS-PC", "CHAGAS-SC"],
            'afroboliviano': ["CHAGAS-A", "CHAGAS-I", "CHAGAS-CO", "CHAGAS-CP", "CHAGAS-SY", "CHAGAS-NY", "CHAGAS-LP"]
        },
        'vih': {
            'guarani': ["VIH-C", "VIH-G", "VIH-PC", "VIH-SC"],
            'afroboliviano': ["VIH-A", "VIH-I", "VIH-CO", "VIH-CP", "VIH-SY", "VIH-NY", "VIH-LP"]
        },
        'nutricion': {
            'guarani': ["OBESIDAD-C", "OBESIDAD-G", "OBESIDAD-PC", "OBESIDAD-SC", "SOBREPESO-C", "SOBREPESO-G", "SOBREPESO-PC", "SOBREPESO-SC", "BAJOPESO-C", "BAJOPESO-G", "BAJOPESO-PC", "BAJOPESO-SC"],
            'afroboliviano': ["OBESIDAD-A", "OBESIDAD-I", "OBESIDAD-CO", "OBESIDAD-CP", "OBESIDAD-SY", "OBESIDAD-NY", "OBESIDAD-LP", "SOBREPESO-A", "SOBREPESO-I", "SOBREPESO-CO", "SOBREPESO-CP", "SOBREPESO-SY", "SOBREPESO-NY", "SOBREPESO-LP", "BAJOPESO-A", "BAJOPESO-I", "BAJOPESO-CO", "BAJOPESO-CP", "BAJOPESO-SY", "BAJOPESO-NY", "BAJOPESO-LP"]
        },
        'poblacion': {
            'guarani': ["POBLACION-C", "POBLACION-G", "POBLACION-PC", "POBLACION-SC"],
            'afroboliviano': ["POBLACION-A", "POBLACION-I", "POBLACION-CO", "POBLACION-CP", "POBLACION-SY", "POBLACION-NY", "POBLACION-LP"]
        },
        'poblacion-especial': {
            'guarani': ["ESPECIALES-C", "ESPECIALES-G", "ESPECIALES-PC", "ESPECIALES-SC"],
            'afroboliviano': ["ESPECIALES-A", "ESPECIALES-I", "ESPECIALES-CO", "ESPECIALES-CP", "ESPECIALES-SY", "ESPECIALES-NY", "ESPECIALES-LP"]
        }
    }
    
    file_path = file_map[tipo][comunidad]
    sheet_names = sheets_map[tipo][comunidad]
    
    dataframes = [pd.read_excel(file_path, sheet_name=sheet) for sheet in sheet_names]
    
    return dataframes

def get_casos_embarazo():
    df_c_embarazo = pd.read_excel('CasosEmbarazoAdolescente-Guaranis.xlsx', sheet_name="EMBARAZO-C")
    df_g_embarazo = pd.read_excel('CasosEmbarazoAdolescente-Guaranis.xlsx', sheet_name="EMBARAZO-G")
    df_pc_embarazo = pd.read_excel('CasosEmbarazoAdolescente-Guaranis.xlsx', sheet_name="EMBARAZO-PC")
    df_sc_embarazo = pd.read_excel('CasosEmbarazoAdolescente-Guaranis.xlsx', sheet_name="EMBARAZO-SC")
    return df_c_embarazo, df_g_embarazo, df_pc_embarazo, df_sc_embarazo

def get_casos_consulta():
    df_c_consulta = pd.read_excel('CasosConsultaExterna-Guaranis.xlsx', sheet_name="CONSULTAS-C")
    df_g_consulta = pd.read_excel('CasosConsultaExterna-Guaranis.xlsx', sheet_name="CONSULTAS-G")
    df_pc_consulta = pd.read_excel('CasosConsultaExterna-Guaranis.xlsx', sheet_name="CONSULTAS-PC")
    df_sc_consulta = pd.read_excel('CasosConsultaExterna-Guaranis.xlsx', sheet_name="CONSULTAS-SC")

    df_c_consulta_2 = pd.read_excel('CasosConsultaExterna-Guaranis.xlsx', sheet_name="SEGUNDO-C")
    return df_c_consulta, df_g_consulta, df_pc_consulta, df_sc_consulta, df_c_consulta_2

def calculate_gender(df, factor, m, h):
    # Población estimada
    total_mujeres = {2019: m[0], 2020: m[1], 2021: m[2], 2022: m[3], 2023: m[4]}
    total_hombres = {2019: h[0], 2020: h[1], 2021: h[2], 2022: h[3], 2023: h[4]}

    # Calcular incidencias
    df['Incidencia'] = df.apply(
        lambda row: (row['Total'] / total_hombres[row['Año']] * factor) if row['Sexo'] == 'Hombre' else (row['Total'] / total_mujeres[row['Año']] * factor),
        axis=1
    ).round().astype(int)
    
    # Calcular los totales para hombres y mujeres
    total_hombres = df[df['Sexo'] == 'Hombre']['Total'].sum()
    total_mujeres = df[df['Sexo'] == 'Mujer']['Total'].sum()

    # Asegurarse de que los totales no sean cero
    total_hombres = total_hombres if total_hombres != 0 else 1
    total_mujeres = total_mujeres if total_mujeres != 0 else 1

    # Calcular el porcentaje y redondear a 2 decimales
    df['Porcentaje'] = df.apply(
        lambda row: (row['Total'] / total_hombres * 100) if row['Sexo'] == 'Hombre' else (row['Total'] / total_mujeres * 100),
        axis=1
    ).round(2)

    return df

def generate_total(df):
    df = df.groupby('Año').sum()
    df_total = df.drop(columns=['Sexo']).reset_index()
    return df_total

def calculate_total(df, factor, p, column):
    poblacion_estimada = {
        2019: p[0],
        2020: p[1],
        2021: p[2],
        2022: p[3],
        2023: p[4]
    }
    df['Incidencia'] = (((df[column]/df['Año'].map(poblacion_estimada)) * factor).round(0)).astype(int)
    suma_total = df[column].sum()
    suma_total = suma_total if suma_total != 0 else 1
    df['Porcentaje'] = (((df[column]/suma_total)) * 100).round(2)
    
    return df

def calculate_total_embarazadas(df, factor, p, column):
    poblacion_estimada = {
        2019: p[0],
        2020: p[1],
        2021: p[2],
        2022: p[3],
        2023: p[4]
    }
    df['Incidencia'] = (((df[column]/df['Año'].map(poblacion_estimada)) * factor).round(0)).astype(int)
    df['Porcentaje'] = (((df[column]/df['Total'])) * 100).round(2)
    
    return df

def calculate_total_consultas(df):
    # Calcular el total del año sumando hombres y mujeres
    df['Total Año'] = df.groupby('Año')['Total'].transform('sum')
    # Calcular el porcentaje para hombres y mujeres por año
    df['Porcentaje'] = (df['Total'] / df['Total Año'] * 100).round(2).astype(int)
    # Eliminar columnas auxiliares si no son necesarias
    df.drop(columns=['Total Año'], inplace=True)
    
    return df

def calculate_table_total_percent(df):
    # Agrupar por año y sexo para obtener los totales
    df_totales = df.groupby(['Año', 'Sexo'])['Total'].sum().unstack()

    # Calcular el total general por año
    df_totales['Total'] = df_totales.sum(axis=1)

    # Calcular los porcentajes
    df_totales['% Hombres'] = (df_totales['Hombre'] / df_totales['Total'] * 100).round(2).astype(str) + '%'
    df_totales['% Mujeres'] = (df_totales['Mujer'] / df_totales['Total'] * 100).round(2).astype(str) + '%'

    # Crear el nuevo DataFrame con los totales y porcentajes
    df_resultado = pd.DataFrame({
        'Año': df_totales.index,
        'Total Hombres': df_totales['Hombre'].astype(str) + ' (' + df_totales['% Hombres'] + ')',
        'Total Mujeres': df_totales['Mujer'].astype(str) + ' (' + df_totales['% Mujeres'] + ')',
        'Total': df_totales['Total']
    }).reset_index(drop=True)
    
    return df_resultado

def calculate_table_total_percent_age(df):
    # Sumar los rangos de edad y crear nuevas columnas
    df['0-9'] = df['< 6'] + df['0-1'] + df['1-4'] + df['5-9']
    df['10-19'] = df['10-14'] + df['15-19']
    
    # Seleccionar las columnas deseadas
    df_totales = df[['Año', 'Sexo', '0-9', '10-19', '20-39', '40-49', '50-59', '60+', 'Total']]
    
    # Calcular los porcentajes y formatear los valores con los porcentajes
    for col in ['0-9', '10-19', '20-39', '40-49', '50-59', '60+']:
        df_totales[col] = df_totales.apply(lambda row: f"{row[col]} ({(row[col] / row['Total'] * 100):.2f}%)", axis=1)
    
    return df_totales

def calculate_percent_second(df):
    # Crear una copia del DataFrame para no modificar el original
    df_resultado = df.copy()
    
    # Crear columnas adicionales para los porcentajes calculados
    df_resultado['Porcentaje Hombres'] = 0.0
    df_resultado['Porcentaje Mujeres'] = 0.0
    
    # Calcular los porcentajes
    for year in df_resultado['Año'].unique():
        total_hombres_anio = df_resultado[df_resultado['Año'] == year]['Total Hombres'].sum()
        total_mujeres_anio = df_resultado[df_resultado['Año'] == year]['Total Mujeres'].sum()
        
        df_resultado.loc[df_resultado['Año'] == year, 'Porcentaje Hombres'] = round(df_resultado['Total Hombres'] / total_hombres_anio * 100,2)
        df_resultado.loc[df_resultado['Año'] == year, 'Porcentaje Mujeres'] = round(df_resultado['Total Mujeres'] / total_mujeres_anio * 100,2)
    
    return df_resultado

def calculate_percent_second_age(df):
    # Seleccionar las columnas deseadas
    df_totales = df[['Año', 'Especialidad', '[H] 0-9', '[H] 10-19', '[H] 20-39', '[H] 40-49', '[H] 50-59', '[H] 60+', '[M] 0-9', '[M] 10-19', '[M] 20-39', '[M] 40-49', '[M] 50-59', '[M] 60+', 'Total']]
    df_resultado = df_totales.copy()

    rangos_edad = ['0-9', '10-19', '20-39', '40-49', '50-59', '60+']
    for rango in rangos_edad:
        df_resultado[f'% [H] {rango}'] = 0.0
        df_resultado[f'% [M] {rango}'] = 0.0

    # Calcular los porcentajes
    for year in df_resultado['Año'].unique():
        for rango in rangos_edad:
            total_hombres_anio = df_resultado[df_resultado['Año'] == year][f'[H] {rango}'].sum()
            total_mujeres_anio = df_resultado[df_resultado['Año'] == year][f'[M] {rango}'].sum()

            if total_hombres_anio > 0:
                df_resultado.loc[df_resultado['Año'] == year, f'% [H] {rango}'] = round(df_resultado[f'[H] {rango}'] / total_hombres_anio * 100, 2)
            if total_mujeres_anio > 0:
                df_resultado.loc[df_resultado['Año'] == year, f'% [M] {rango}'] = round(df_resultado[f'[M] {rango}'] / total_mujeres_anio * 100, 2)

    return df_resultado


def generate_lines_total(df1, df2, df3, x_column, y_column, title, size_title, footer, size_footer, size_legend, size_graph, labels, legend_loc):
    colors = ['#DD6700', '#EA7E1F', '#FFB26F', '#FFCBA6', '#FFE5D1', '#135490', '#1769B5', '#2688E3', '#8FCFFF', '#CDE7FF']
    
    plt.figure(figsize=(6, 5))
    
    años = sorted(df1[x_column].unique())

    line1, = plt.plot(df1[x_column], df1[y_column], color=colors[5], marker='o', linestyle='-', label=labels[0])
    line2, = plt.plot(df2[x_column], df2[y_column], color=colors[6], marker='o', linestyle='-', label=labels[1])
    line3, = plt.plot(df3[x_column], df3[y_column], color=colors[0], marker='o', linestyle='-', label=labels[2])

    if y_column == 'Incidencia':
        # Agrega los números sobre cada punto de la línea de tendencia 1
        for x, y in zip(df1[x_column], df1[y_column]):
            plt.text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
        # Agrega los números sobre cada punto de la línea de tendencia 2
        for x, y in zip(df2[x_column], df2[y_column]):
            plt.text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
        # Agrega los números sobre cada punto de la línea de tendencia 3
        for x, y in zip(df3[x_column], df3[y_column]):
            plt.text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
    else:
        for x, y in zip(df1[x_column], df1[y_column]):
            plt.text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')
        # Agrega los números sobre cada punto de la línea de tendencia 2
        for x, y in zip(df2[x_column], df2[y_column]):
            plt.text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')
        # Agrega los números sobre cada punto de la línea de tendencia 3
        for x, y in zip(df3[x_column], df3[y_column]):
            plt.text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')

    # Configura los ejes y el título
    plt.xlabel(x_column)
    #plt.ylabel(y_column)
    plt.title(title, fontsize=size_title)
    plt.xticks(años)
    plt.legend(loc=legend_loc, fontsize=size_legend)
    #plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 1])
    # Agregar referencia en la parte inferior del gráfico
    plt.figtext(0.5, 0.01, footer, ha='center', va='center', fontsize=size_footer, color='black')
    # Ocultar el eje y
    #plt.yticks([])
    
    # Convertir la gráfica a base64
    tmp_file = io.BytesIO()
    plt.savefig(tmp_file, format='png')
    tmp_file.seek(0)
    plot_base64 = base64.b64encode(tmp_file.getvalue()).decode('utf-8')

    # Mostrar la gráfica en un componente HTML
    return html.Div([
        html.H2(title),
        html.Img(src='data:image/png;base64,{}'.format(plot_base64), id='responsive-graph')
    ]) 
    

# Función para generar la gráfica y mostrarla en base64
def generate_lines_gender(df1, df2, df3, x_column, y_column, title, size_title, footer, size_footer, size_legend, size_graph, labels, legend_loc):
    años = sorted(df1[x_column].unique())

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    
    colors = ['#DD6700', '#EA7E1F', '#FFB26F', '#FFCBA6', '#FFE5D1', '#135490', '#1769B5', '#2688E3', '#8FCFFF', '#CDE7FF']

    # Gráfica para hombres
    ax[0].plot(df1[df1['Sexo'] == 'Hombre'][x_column], df1[df1['Sexo'] == 'Hombre'][y_column], color=colors[5], marker='o', linestyle='-', label=labels[0])
    ax[0].plot(df2[df2['Sexo'] == 'Hombre'][x_column], df2[df2['Sexo'] == 'Hombre'][y_column], color=colors[6], marker='o', linestyle='-', label=labels[1])
    ax[0].plot(df3[df3['Sexo'] == 'Hombre'][x_column], df3[df3['Sexo'] == 'Hombre'][y_column], color=colors[7], marker='o', linestyle='-', label=labels[2])
    ax[0].set_xlabel(x_column)
    ax[0].set_title(f'Tendencia - Hombres')
    ax[0].legend(loc=legend_loc, fontsize=size_legend)
    ax[0].set_xticks(años)

    if y_column == 'Incidencia':
        for df_h in [df1[df1['Sexo'] == 'Hombre'], df2[df2['Sexo'] == 'Hombre'], df3[df3['Sexo'] == 'Hombre']]:
            for x, y in zip(df_h[x_column], df_h[y_column]):
                ax[0].text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
    else:
        for df_h in [df1[df1['Sexo'] == 'Hombre'], df2[df2['Sexo'] == 'Hombre'], df3[df3['Sexo'] == 'Hombre']]:
            for x, y in zip(df_h[x_column], df_h[y_column]):
                ax[0].text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')

    # Gráfica para mujeres
    ax[1].plot(df1[df1['Sexo'] == 'Mujer'][x_column], df1[df1['Sexo'] == 'Mujer'][y_column], color=colors[0], marker='o', linestyle='-', label=labels[0])
    ax[1].plot(df2[df2['Sexo'] == 'Mujer'][x_column], df2[df2['Sexo'] == 'Mujer'][y_column], color=colors[1], marker='o', linestyle='-', label=labels[1])
    ax[1].plot(df3[df3['Sexo'] == 'Mujer'][x_column], df3[df3['Sexo'] == 'Mujer'][y_column], color=colors[2], marker='o', linestyle='-', label=labels[2])
    ax[1].set_xlabel(x_column)
    ax[1].set_title(f'Tendencia - Mujeres')
    ax[1].legend(loc=legend_loc, fontsize=size_legend)
    ax[1].set_xticks(años)

    if y_column == 'Incidencia':
        for df_m in [df1[df1['Sexo'] == 'Mujer'], df2[df2['Sexo'] == 'Mujer'], df3[df3['Sexo'] == 'Mujer']]:
            for x, y in zip(df_m[x_column], df_m[y_column]):
                ax[1].text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
    else:
        for df_m in [df1[df1['Sexo'] == 'Mujer'], df2[df2['Sexo'] == 'Mujer'], df3[df3['Sexo'] == 'Mujer']]:
            for x, y in zip(df_m[x_column], df_m[y_column]):
                ax[1].text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')
                

    # Agregar título principal
    fig.suptitle(title, fontsize=size_title)
    # Ajustar el layout para dar espacio al título principal
    plt.tight_layout(rect=[0, 0, 1, 1])
    # Agregar referencia en la parte inferior del gráfico
    fig.text(0.5, 0.01, footer, ha='center', va='center', fontsize=size_footer, color='black')
    

    # Convertir la gráfica a base64
    tmp_file = io.BytesIO()
    plt.savefig(tmp_file, format='png')
    tmp_file.seek(0)
    plot_base64 = base64.b64encode(tmp_file.getvalue()).decode('utf-8')

    # Mostrar la gráfica en un componente HTML
    return html.Div([
        html.H2(title),
        html.Img(src='data:image/png;base64,{}'.format(plot_base64), style={'width': '100%'})
    ]) 

def generate_lines_separate_gender(df1, df2, df3, x_column, y_column, title, size_title, footer, size_footer, size_legend, size_graph, labels, legend_loc):
    años = sorted(df1[x_column].unique())
    
    colors = ['#DD6700', '#EA7E1F', '#FFB26F', '#FFCBA6', '#FFE5D1', '#135490', '#1769B5', '#2688E3', '#8FCFFF', '#CDE7FF']

    fig, ax = plt.subplots(2, 3, figsize=(14, 8), sharey=True)

    # Gráfica para hombres - Departamento
    ax[0, 0].plot(df3[df3['Sexo'] == 'Hombre'][x_column], df3[df3['Sexo'] == 'Hombre'][y_column], color=colors[0], marker='o', linestyle='-', label=labels[2])
    ax[0, 0].set_xlabel(x_column)
    ax[0, 0].set_title(f'Tendencia Municipal - {labels[2]}')
    ax[0, 0].legend(loc=legend_loc, fontsize=size_legend)
    ax[0, 0].set_xticks(años)

    # Gráfica para hombres - Provincia
    ax[0, 1].plot(df2[df2['Sexo'] == 'Hombre'][x_column], df2[df2['Sexo'] == 'Hombre'][y_column], color=colors[1], marker='o', linestyle='-', label=labels[1])
    ax[0, 1].set_xlabel(x_column)
    ax[0, 1].set_title(f'Tendencia Provincial - {labels[1]}')
    ax[0, 1].legend(loc=legend_loc, fontsize=size_legend)
    ax[0, 1].set_xticks(años)

    # Gráfica para hombres - Municipio
    ax[0, 2].plot(df1[df1['Sexo'] == 'Hombre'][x_column], df1[df1['Sexo'] == 'Hombre'][y_column], color=colors[2], marker='o', linestyle='-', label=labels[0])
    ax[0, 2].set_xlabel(x_column)
    ax[0, 2].set_title(f'Tendencia Departamental - {labels[0]}')
    ax[0, 2].legend(loc=legend_loc, fontsize=size_legend)
    ax[0, 2].set_xticks(años)

    if y_column == 'Incidencia':
        for x, y in zip(df3[df3['Sexo'] == 'Hombre'][x_column], df3[df3['Sexo'] == 'Hombre'][y_column]):
            ax[0, 0].text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
            
        for x, y in zip(df2[df2['Sexo'] == 'Hombre'][x_column], df2[df2['Sexo'] == 'Hombre'][y_column]):
            ax[0, 1].text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
        
        for x, y in zip(df1[df1['Sexo'] == 'Hombre'][x_column], df1[df1['Sexo'] == 'Hombre'][y_column]):
            ax[0, 2].text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
    else:
        for x, y in zip(df3[df3['Sexo'] == 'Hombre'][x_column], df3[df3['Sexo'] == 'Hombre'][y_column]):
            ax[0, 0].text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')
        
        for x, y in zip(df2[df2['Sexo'] == 'Hombre'][x_column], df2[df2['Sexo'] == 'Hombre'][y_column]):
            ax[0, 1].text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')
            
        for x, y in zip(df1[df1['Sexo'] == 'Hombre'][x_column], df1[df1['Sexo'] == 'Hombre'][y_column]):
            ax[0, 2].text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')

    # Gráfica para mujeres - Departamento
    ax[1, 0].plot(df3[df3['Sexo'] == 'Mujer'][x_column], df3[df3['Sexo'] == 'Mujer'][y_column], color=colors[0], marker='o', linestyle='-', label=labels[2])
    ax[1, 0].set_xlabel(x_column)
    ax[1, 0].set_title(f'Tendencia Municipal - {labels[2]}')
    ax[1, 0].legend(loc=legend_loc, fontsize=size_legend)
    ax[1, 0].set_xticks(años)

    # Gráfica para mujeres - Provincia
    ax[1, 1].plot(df2[df2['Sexo'] == 'Mujer'][x_column], df2[df2['Sexo'] == 'Mujer'][y_column], color=colors[1], marker='o', linestyle='-', label=labels[1])
    ax[1, 1].set_xlabel(x_column)
    ax[1, 1].set_title(f'Tendencia Provincial - {labels[1]}')
    ax[1, 1].legend(loc=legend_loc, fontsize=size_legend)
    ax[1, 1].set_xticks(años)

    # Gráfica para mujeres - Municipio
    ax[1, 2].plot(df1[df1['Sexo'] == 'Mujer'][x_column], df1[df1['Sexo'] == 'Mujer'][y_column], color=colors[2], marker='o', linestyle='-', label=labels[0])
    ax[1, 2].set_xlabel(x_column)
    ax[1, 2].set_title(f'Tendencia Departamental - {labels[0]}')
    ax[1, 2].legend(loc=legend_loc, fontsize=size_legend)
    ax[1, 2].set_xticks(años)
    
    if y_column == 'Incidencia':
        for x, y in zip(df3[df3['Sexo'] == 'Mujer'][x_column], df3[df3['Sexo'] == 'Mujer'][y_column]):
            ax[1, 0].text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
            
        for x, y in zip(df2[df2['Sexo'] == 'Mujer'][x_column], df2[df2['Sexo'] == 'Mujer'][y_column]):
            ax[1, 1].text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
        
        for x, y in zip(df1[df1['Sexo'] == 'Mujer'][x_column], df1[df1['Sexo'] == 'Mujer'][y_column]):
            ax[1, 2].text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
    else:
        for x, y in zip(df3[df3['Sexo'] == 'Mujer'][x_column], df3[df3['Sexo'] == 'Mujer'][y_column]):
            ax[1, 0].text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')
        
        for x, y in zip(df2[df2['Sexo'] == 'Mujer'][x_column], df2[df2['Sexo'] == 'Mujer'][y_column]):
            ax[1, 1].text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')
            
        for x, y in zip(df1[df1['Sexo'] == 'Mujer'][x_column], df1[df1['Sexo'] == 'Mujer'][y_column]):
            ax[1, 2].text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')


    # Agregar etiquetas para los ejes y
    fig.text(0.01, 0.75, y_column + ' Hombres', ha='center', va='center', rotation='vertical', fontsize=12, color='black')
    fig.text(0.01, 0.25, y_column + ' Mujeres', ha='center', va='center', rotation='vertical', fontsize=12, color='black')

    # Ocultar el eje y de todas las subgráficas
    #for axs in ax:
    #    for a in axs:
    #        a.set_yticks([])

    fig.suptitle(title, fontsize=size_title)
    plt.tight_layout(rect=[0, 0, 1, 1])
    # Agregar referencia en la parte inferior del gráfico
    fig.text(0.5, 0.01, footer, ha='center', va='center', fontsize=size_footer, color='black')
    

    # Convertir la gráfica a base64
    tmp_file = io.BytesIO()
    plt.savefig(tmp_file, format='png')
    tmp_file.seek(0)
    plot_base64 = base64.b64encode(tmp_file.getvalue()).decode('utf-8')

    # Mostrar la gráfica en un componente HTML
    return html.Div([
        html.H2(title),
        html.Img(src='data:image/png;base64,{}'.format(plot_base64), style={'width': '100%'})
    ])

def generate_lines_separate_nutricion(df1, df2, df3, x_column, y_column, title, size_title, footer, size_footer, size_legend, size_graph, labels, legend_loc):
    años = sorted(df1[x_column].unique())
    
    colors = ['#DD6700', '#EA7E1F', '#FFB26F', '#FFCBA6', '#FFE5D1', '#135490', '#1769B5', '#2688E3', '#8FCFFF', '#CDE7FF']

    fig, ax = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    # Gráfica para hombres - Departamento
    ax[0].plot(df3[x_column], df3[y_column], color=colors[5], marker='o', linestyle='-', label=labels[2])
    ax[0].set_xlabel(x_column)
    ax[0].set_title(f'Tendencia Municipal - {labels[2]}')
    ax[0].legend(loc=legend_loc, fontsize=size_legend)
    ax[0].set_xticks(años)

    # Gráfica para hombres - Provincia
    ax[1].plot(df2[x_column], df2[y_column], color=colors[6], marker='o', linestyle='-', label=labels[1])
    ax[1].set_xlabel(x_column)
    ax[1].set_title(f'Tendencia Provincial - {labels[1]}')
    ax[1].legend(loc=legend_loc, fontsize=size_legend)
    ax[1].set_xticks(años)

    # Gráfica para hombres - Municipio
    ax[2].plot(df1[x_column], df1[y_column], color=colors[7], marker='o', linestyle='-', label=labels[0])
    ax[2].set_xlabel(x_column)
    ax[2].set_title(f'Tendencia Departamental - {labels[0]}')
    ax[2].legend(loc=legend_loc, fontsize=size_legend)
    ax[2].set_xticks(años)

    if y_column == 'Incidencia':
        for x, y in zip(df3[x_column], df3[y_column]):
            ax[0].text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
            
        for x, y in zip(df2[x_column], df2[y_column]):
            ax[1].text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
        
        for x, y in zip(df1[x_column], df1[y_column]):
            ax[2].text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
    else:
        for x, y in zip(df3[x_column], df3[y_column]):
            ax[0].text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')
        
        for x, y in zip(df2[x_column], df2[y_column]):
            ax[1].text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')
            
        for x, y in zip(df1[x_column], df1[y_column]):
            ax[2].text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')
            
    fig.suptitle(title, fontsize=size_title)
    plt.tight_layout(rect=[0, 0, 1, 1])
    # Agregar referencia en la parte inferior del gráfico
    fig.text(0.5, 0.01, footer, ha='center', va='center', fontsize=size_footer, color='black')

    # Convertir la gráfica a base64
    tmp_file = io.BytesIO()
    plt.savefig(tmp_file, format='png')
    tmp_file.seek(0)
    plot_base64 = base64.b64encode(tmp_file.getvalue()).decode('utf-8')

    # Mostrar la gráfica en un componente HTML
    return html.Div([
        html.H2(title),
        html.Img(src='data:image/png;base64,{}'.format(plot_base64), style={'width': '100%'})
    ])

def generate_lines_comparison_gender(df1, df2, df3, x_column, y_column, title, size_title, footer, size_footer, size_legend, size_graph, labels, legend_loc):
    años = sorted(df1[x_column].unique())
    colors = ['#DD6700', '#EA7E1F', '#FFB26F', '#FFCBA6', '#FFE5D1', '#135490', '#1769B5', '#2688E3', '#8FCFFF', '#CDE7FF']

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # Gráfica para Municipio
    ax[0].plot(df3[df3['Sexo'] == 'Hombre'][x_column], df3[df3['Sexo'] == 'Hombre'][y_column], color=colors[5], marker='o', linestyle='-', label=f'Hombres - {labels[2]}')
    ax[0].plot(df3[df3['Sexo'] == 'Mujer'][x_column], df3[df3['Sexo'] == 'Mujer'][y_column], color=colors[0], marker='o', linestyle='-', label=f'Mujeres - {labels[2]}')
    ax[0].set_xlabel(x_column)
    ax[0].set_title(f'Tendencia Municipal - {labels[2]}')
    ax[0].legend(loc=legend_loc, fontsize=size_legend)
    ax[0].set_xticks(años)
        
    # Gráfica para Provincia
    ax[1].plot(df2[df2['Sexo'] == 'Hombre'][x_column], df2[df2['Sexo'] == 'Hombre'][y_column], color=colors[6], marker='o', linestyle='-', label=f'Hombres - {labels[1]}')
    ax[1].plot(df2[df2['Sexo'] == 'Mujer'][x_column], df2[df2['Sexo'] == 'Mujer'][y_column], color=colors[1], marker='o', linestyle='-', label=f'Mujeres - {labels[1]}')
    ax[1].set_xlabel(x_column)
    ax[1].set_title(f'Tendencia Provincial - {labels[1]}')
    ax[1].legend(loc=legend_loc, fontsize=size_legend)
    ax[1].set_xticks(años)
    
    # Gráfica para Departamento
    ax[2].plot(df1[df1['Sexo'] == 'Hombre'][x_column], df1[df1['Sexo'] == 'Hombre'][y_column], color=colors[7], marker='o', linestyle='-', label=f'Hombres - {labels[0]}')
    ax[2].plot(df1[df1['Sexo'] == 'Mujer'][x_column], df1[df1['Sexo'] == 'Mujer'][y_column], color=colors[2], marker='o', linestyle='-', label=f'Mujeres - {labels[0]}')
    ax[2].set_xlabel(x_column)
    ax[2].set_title(f'Tendencia Departamental - {labels[0]}')
    ax[2].legend(loc=legend_loc, fontsize=size_legend)
    ax[2].set_xticks(años)

    if y_column == 'Incidencia':
        for x, y in zip(df3[df3['Sexo'] == 'Hombre'][x_column], df3[df3['Sexo'] == 'Hombre'][y_column]):
            ax[0].text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
        for x, y in zip(df3[df3['Sexo'] == 'Mujer'][x_column], df3[df3['Sexo'] == 'Mujer'][y_column]):
            ax[0].text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
            
        for x, y in zip(df2[df2['Sexo'] == 'Hombre'][x_column], df2[df2['Sexo'] == 'Hombre'][y_column]):
            ax[1].text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
        for x, y in zip(df2[df2['Sexo'] == 'Mujer'][x_column], df2[df2['Sexo'] == 'Mujer'][y_column]):
            ax[1].text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
        
        for x, y in zip(df1[df1['Sexo'] == 'Hombre'][x_column], df1[df1['Sexo'] == 'Hombre'][y_column]):
            ax[2].text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
        for x, y in zip(df1[df1['Sexo'] == 'Mujer'][x_column], df1[df1['Sexo'] == 'Mujer'][y_column]):
            ax[2].text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=size_graph, color='black')
    else:
        for x, y in zip(df3[df3['Sexo'] == 'Hombre'][x_column], df3[df3['Sexo'] == 'Hombre'][y_column]):
            ax[0].text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')
        for x, y in zip(df3[df3['Sexo'] == 'Mujer'][x_column], df3[df3['Sexo'] == 'Mujer'][y_column]):
            ax[0].text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')
            
        for x, y in zip(df2[df2['Sexo'] == 'Hombre'][x_column], df2[df2['Sexo'] == 'Hombre'][y_column]):
            ax[1].text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')
        for x, y in zip(df2[df2['Sexo'] == 'Mujer'][x_column], df2[df2['Sexo'] == 'Mujer'][y_column]):
            ax[1].text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')
            
        for x, y in zip(df1[df1['Sexo'] == 'Hombre'][x_column], df1[df1['Sexo'] == 'Hombre'][y_column]):
            ax[2].text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')
        for x, y in zip(df1[df1['Sexo'] == 'Mujer'][x_column], df1[df1['Sexo'] == 'Mujer'][y_column]):
            ax[2].text(x, y, f"{y:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')

    
    fig.suptitle(title, fontsize=size_title)
    plt.tight_layout(rect=[0, 0, 1, 1])
    # Agregar referencia en la parte inferior del gráfico
    fig.text(0.5, 0.01, footer, ha='center', va='center', fontsize=size_footer, color='black')
    

    # Convertir la gráfica a base64
    tmp_file = io.BytesIO()
    plt.savefig(tmp_file, format='png')
    tmp_file.seek(0)
    plot_base64 = base64.b64encode(tmp_file.getvalue()).decode('utf-8')

    # Mostrar la gráfica en un componente HTML
    return html.Div([
        html.H2(title),
        html.Img(src='data:image/png;base64,{}'.format(plot_base64), style={'width': '100%'})
    ])



def plot_age_percentages(df, m, h, x_column, y_column, title, size_title, footer, size_footer, size_legend, size_graph, legend_loc, size_bar):
    # Calcular porcentajes
    df_percent = df.copy()
    colors = ['#DD6700', '#EA7E1F', '#FFB26F', '#FFCBA6', '#FFE5D1', '#135490', '#1769B5', '#2688E3', '#8FCFFF', '#CDE7FF']

    df_percent['0-9'] = df_percent['< 6'] + df_percent['0-1'] + df_percent['1-4'] + df_percent['5-9']
    df_percent['10-19'] = df_percent['10-14'] + df_percent['15-19']
    
    
    # Calcular los totales para hombres y mujeres
    total_hombres = df[df['Sexo'] == 'Hombre']['Total'].sum()
    total_mujeres = df[df['Sexo'] == 'Mujer']['Total'].sum()
    
    # Asegurarse de que los totales no sean cero
    total_hombres = total_hombres if total_hombres != 0 else 1
    total_mujeres = total_mujeres if total_mujeres != 0 else 1
    
    # Calcular porcentajes
    if y_column == "Porcentaje":
        """df_percent['0-9 %'] = df_percent.apply(
            lambda row: (row['0-9'] / total_hombres * 100) if row['Sexo'] == 'Hombre' else (row['0-9'] / total_mujeres * 100),
            axis=1
        ).round(2)
        df_percent['10-19 %'] = df_percent.apply(
            lambda row: (row['10-19'] / total_hombres * 100) if row['Sexo'] == 'Hombre' else (row['10-19'] / total_mujeres * 100),
            axis=1
        ).round(2)
        df_percent['20-39 %'] = df_percent.apply(
            lambda row: (row['20-39'] / total_hombres * 100) if row['Sexo'] == 'Hombre' else (row['20-39'] / total_mujeres * 100),
            axis=1
        ).round(2)
        df_percent['40-49 %'] = df_percent.apply(
            lambda row: (row['40-49'] / total_hombres * 100) if row['Sexo'] == 'Hombre' else (row['40-49'] / total_mujeres * 100),
            axis=1
        ).round(2)
        df_percent['50-59 %'] = df_percent.apply(
            lambda row: (row['50-59'] / total_hombres * 100) if row['Sexo'] == 'Hombre' else (row['50-59'] / total_mujeres * 100),
            axis=1
        ).round(2)
        df_percent['60+ %'] = df_percent.apply(
            lambda row: (row['60+'] / total_hombres * 100) if row['Sexo'] == 'Hombre' else (row['60+'] / total_mujeres * 100),
            axis=1
        ).round(2)"""
        df_percent['0-9 %'] = df_percent.apply(lambda row: round((row['0-9'] / row['Total']) * 100, 2) if row['Total'] != 0 else 0, axis=1)
        df_percent['10-19 %'] = df_percent.apply(lambda row: round((row['10-19'] / row['Total']) * 100, 2) if row['Total'] != 0 else 0, axis=1)
        df_percent['20-39 %'] = df_percent.apply(lambda row: round((row['20-39'] / row['Total']) * 100, 2) if row['Total'] != 0 else 0, axis=1)
        df_percent['40-49 %'] = df_percent.apply(lambda row: round((row['40-49'] / row['Total']) * 100, 2) if row['Total'] != 0 else 0, axis=1)
        df_percent['50-59 %'] = df_percent.apply(lambda row: round((row['50-59'] / row['Total']) * 100, 2) if row['Total'] != 0 else 0, axis=1)
        df_percent['60+ %'] = df_percent.apply(lambda row: round((row['60+'] / row['Total']) * 100, 2) if row['Total'] != 0 else 0, axis=1)
    else:
        total_mujeres_0_9 = {2019: m[0], 2020: m[1], 2021: m[2], 2022: m[3], 2023: m[4]}
        total_mujeres_10_19 = {2019: m[5], 2020: m[6], 2021: m[7], 2022: m[8], 2023: m[9]}
        total_mujeres_20_39 = {2019: m[10], 2020: m[11], 2021: m[12], 2022: m[13], 2023: m[14]}
        total_mujeres_40_49 = {2019: m[15], 2020: m[16], 2021: m[17], 2022: m[18], 2023: m[19]}
        total_mujeres_50_59 = {2019: m[20], 2020: m[21], 2021: m[22], 2022: m[23], 2023: m[24]}
        total_mujeres_60 = {2019: m[25], 2020: m[26], 2021: m[27], 2022: m[28], 2023: m[29]}

        total_hombres_0_9 = {2019: h[0], 2020: h[1], 2021: h[2], 2022: h[3], 2023: h[4]}
        total_hombres_10_19 = {2019: h[5], 2020: h[6], 2021: h[7], 2022: h[8], 2023: h[9]}
        total_hombres_20_39 = {2019: h[10], 2020: h[11], 2021: h[12], 2022: h[13], 2023: h[14]}
        total_hombres_40_49 = {2019: h[15], 2020: h[16], 2021: h[17], 2022: h[18], 2023: h[19]}
        total_hombres_50_59 = {2019: h[20], 2020: h[21], 2021: h[22], 2022: h[23], 2023: h[24]}
        total_hombres_60 = {2019: h[25], 2020: h[26], 2021: h[27], 2022: h[28], 2023: h[29]}

        df_percent['0-9 %'] = df_percent.apply(
            lambda row: (row['0-9'] / total_hombres_0_9[row['Año']] * 10000) if row['Sexo'] == 'Hombre' else (row['0-9'] / total_mujeres_0_9[row['Año']] * 10000),
            axis=1
        ).round(2)
        df_percent['10-19 %'] = df_percent.apply(
            lambda row: (row['10-19'] / total_hombres_10_19[row['Año']] * 10000) if row['Sexo'] == 'Hombre' else (row['10-19'] / total_mujeres_10_19[row['Año']] * 10000),
            axis=1
        ).round(2)
        df_percent['20-39 %'] = df_percent.apply(
            lambda row: (row['20-39'] / total_hombres_20_39[row['Año']] * 10000) if row['Sexo'] == 'Hombre' else (row['20-39'] / total_mujeres_20_39[row['Año']] * 10000),
            axis=1
        ).round(2)
        df_percent['40-49 %'] = df_percent.apply(
            lambda row: (row['40-49'] / total_hombres_40_49[row['Año']] * 10000) if row['Sexo'] == 'Hombre' else (row['40-49'] / total_mujeres_40_49[row['Año']] * 10000),
            axis=1
        ).round(2)
        df_percent['50-59 %'] = df_percent.apply(
            lambda row: (row['50-59'] / total_hombres_50_59[row['Año']] * 10000) if row['Sexo'] == 'Hombre' else (row['50-59'] / total_mujeres_50_59[row['Año']] * 10000),
            axis=1
        ).round(2)
        df_percent['60+ %'] = df_percent.apply(
            lambda row: (row['60+'] / total_hombres_60[row['Año']] * 10000) if row['Sexo'] == 'Hombre' else (row['60+'] / total_mujeres_60[row['Año']] * 10000),
            axis=1
        ).round()
    
    
    # Filtrar las columnas que contienen '%'
    columns_pct = ['0-9 %', '10-19 %', '20-39 %', '40-49 %', '50-59 %', '60+ %']

    #  Filtrar los datos por sexo
    df_hombres = df_percent[df_percent['Sexo'] == 'Hombre']
    df_mujeres = df_percent[df_percent['Sexo'] == 'Mujer']

    # Configuración de la figura
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharex=True)


    bar_width = size_bar  # Ancho de las barras
    years = df[x_column].unique()

    # Crear gráfico de barras para Hombres
    for i, year in enumerate(years):
        data = df_hombres[df_hombres[x_column] == year][columns_pct].iloc[0]
        positions = np.arange(len(columns_pct)) + i * bar_width
        bars = ax[0].bar(positions, data, color=colors[i+5], width=bar_width, label=str(year), alpha=1)
        if y_column == 'Porcentaje':
            for bar in bars:
                yval = bar.get_height()
                ax[0].text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}', ha='center', va='bottom', fontsize=size_graph)
        else:
            for bar in bars:
                yval = bar.get_height()
                ax[0].text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.0f}', ha='center', va='bottom', fontsize=size_graph)

    ax[0].set_title(y_column+' por Edad - Hombres')
    ax[0].legend(loc=legend_loc, fontsize=size_legend)
    ax[0].set_ylim(0, max(df_hombres[columns_pct].values.max(), df_mujeres[columns_pct].values.max()) * 1.1)
    ax[0].set_xlabel('Edad')

    # Crear gráfico de barras para Mujeres
    for i, year in enumerate(years):
        data = df_mujeres[df_mujeres[x_column] == year][columns_pct].iloc[0]
        positions = np.arange(len(columns_pct)) + i * bar_width
        bars = ax[1].bar(positions, data, color=colors[i], width=bar_width, label=str(year), alpha=1)
        if y_column == 'Porcentaje':
            for bar in bars:
                yval = bar.get_height()
                ax[1].text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}', ha='center', va='bottom', fontsize=size_graph)
        else:
            for bar in bars:
                yval = bar.get_height()
                ax[1].text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.0f}', ha='center', va='bottom', fontsize=size_graph)

    ax[1].set_title(y_column+' por Edad - Mujeres')
    ax[1].legend(loc=legend_loc, fontsize=size_legend)
    ax[1].set_ylim(0, max(df_hombres[columns_pct].values.max(), df_mujeres[columns_pct].values.max()) * 1.1)
    ax[1].set_xlabel('Edad')

    # Ajustar etiquetas del eje x
    ax[1].set_xticks(np.arange(len(columns_pct)) + bar_width * (len(years) - 1) / 2)
    ax[1].set_xticklabels(['0-9', '10-19', '20-39', '40-49', '50-59', '60+'], rotation=0, fontsize=10)

    # Quitar el nombre del eje y
    ax[0].set_ylabel('')
    ax[1].set_ylabel('')

    fig.suptitle(title, fontsize=size_title)
    plt.tight_layout(rect=[0, 0, 1, 1])
    # Agregar referencia en la parte inferior del gráfico
    fig.text(0.5, 0.01, footer, ha='center', va='center', fontsize=size_footer, color='black')
    

    # Convertir la gráfica a base64
    tmp_file = io.BytesIO()
    plt.savefig(tmp_file, format='png')
    tmp_file.seek(0)
    plot_base64 = base64.b64encode(tmp_file.getvalue()).decode('utf-8')

    # Mostrar la gráfica en un componente HTML
    return html.Div([
        html.H2(title),
        html.Img(src='data:image/png;base64,{}'.format(plot_base64), style={'width': '100%'})
    ])

def generate_bars_gender(df1, df2, df3, x_column, y_column, title, size_title, footer, size_footer, size_legend, size_graph, labels, legend_loc):
    años = sorted(df1[x_column].unique())
    bar_width = 0.334
    r1 = range(len(años))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    colors = ['#DD6700', '#EA7E1F', '#FFB26F', '#FFCBA6', '#FFE5D1', '#135490', '#1769B5', '#2688E3', '#8FCFFF', '#CDE7FF']

    # Gráfica para hombres
    ax[0].bar(r1, df1[df1['Sexo'] == 'Hombre'][y_column], color=colors[5], width=bar_width, label=labels[0])
    ax[0].bar(r2, df2[df2['Sexo'] == 'Hombre'][y_column], color=colors[6], width=bar_width, label=labels[1])
    ax[0].bar(r3, df3[df3['Sexo'] == 'Hombre'][y_column], color=colors[7], width=bar_width, label=labels[2])
    ax[0].set_xlabel(x_column)
    ax[0].set_title(f'Tendencia de Porcentaje - Hombres')
    ax[0].legend(loc=legend_loc, fontsize=size_legend)
    ax[0].set_xticks([r + bar_width for r in range(len(años))])
    ax[0].set_xticklabels(años)

    for r, y in zip([r1, r2, r3], [df1[df1['Sexo'] == 'Hombre'][y_column], df2[df2['Sexo'] == 'Hombre'][y_column], df3[df3['Sexo'] == 'Hombre'][y_column]]):
        for x, val in zip(r, y):
            ax[0].text(x, val, f"{val:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')

    # Gráfica para mujeres
    ax[1].bar(r1, df1[df1['Sexo'] == 'Mujer'][y_column], color=colors[0], width=bar_width, label=labels[0])
    ax[1].bar(r2, df2[df2['Sexo'] == 'Mujer'][y_column], color=colors[1], width=bar_width, label=labels[1])
    ax[1].bar(r3, df3[df3['Sexo'] == 'Mujer'][y_column], color=colors[2], width=bar_width, label=labels[2])
    ax[1].set_xlabel(x_column)
    ax[1].set_title(f'Tendencia de Porcentaje - Mujeres')
    ax[1].legend(loc=legend_loc, fontsize=size_legend)
    ax[1].set_xticks([r + bar_width for r in range(len(años))])
    ax[1].set_xticklabels(años)

    for r, y in zip([r1, r2, r3], [df1[df1['Sexo'] == 'Mujer'][y_column], df2[df2['Sexo'] == 'Mujer'][y_column], df3[df3['Sexo'] == 'Mujer'][y_column]]):
        for x, val in zip(r, y):
            ax[1].text(x, val, f"{val:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')

    fig.suptitle(title, fontsize=size_title)
    plt.tight_layout(rect=[0, 0, 1, 1])
    # Agregar referencia en la parte inferior del gráfico
    fig.text(0.5, 0.01, footer, ha='center', va='center', fontsize=size_footer, color='black')

    # Convertir la gráfica a base64
    tmp_file = io.BytesIO()
    plt.savefig(tmp_file, format='png')
    tmp_file.seek(0)
    plot_base64 = base64.b64encode(tmp_file.getvalue()).decode('utf-8')

    # Mostrar la gráfica en un componente HTML
    return html.Div([
        html.H2(title),
        html.Img(src='data:image/png;base64,{}'.format(plot_base64), style={'width': '100%'})
    ])

def generate_bars_separate_gender(df1, df2, df3, x_column, y_column, title, size_title, footer, size_footer, size_legend, size_graph, labels, legend_loc):
    años = sorted(df1[x_column].unique())
    bar_width = 0.9  # Ajuste del ancho de la barra
    r1 = [x for x in range(len(años))]

    fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
    colors = ['#DD6700', '#EA7E1F', '#FFB26F', '#FFCBA6', '#FFE5D1', '#135490', '#1769B5', '#2688E3', '#8FCFFF', '#CDE7FF']

    # Gráfica para hombres - Departamento
    bars1 = ax[0, 0].bar(r1, df1[df1['Sexo'] == 'Hombre'][y_column], color=colors[5], width=bar_width, label=labels[0])
    ax[0, 0].set_xlabel(x_column)
    ax[0, 0].set_title(f'Porcentaje Municipal - {labels[0]}')
    ax[0, 0].legend(loc=legend_loc, fontsize=size_legend)
    ax[0, 0].set_xticks(r1)
    ax[0, 0].set_xticklabels(años)

    for bar in bars1:
        height = bar.get_height()
        ax[0, 0].text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')

    # Gráfica para hombres - Provincia
    bars2 = ax[0, 1].bar(r1, df2[df2['Sexo'] == 'Hombre'][y_column], color=colors[6], width=bar_width, label=labels[1])
    ax[0, 1].set_xlabel(x_column)
    ax[0, 1].set_title(f'Porcentaje Provincial - {labels[1]}')
    ax[0, 1].legend(loc=legend_loc, fontsize=size_legend)
    ax[0, 1].set_xticks(r1)
    ax[0, 1].set_xticklabels(años)

    for bar in bars2:
        height = bar.get_height()
        ax[0, 1].text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')

    # Gráfica para hombres - Municipio
    bars3 = ax[0, 2].bar(r1, df3[df3['Sexo'] == 'Hombre'][y_column], color=colors[7], width=bar_width, label=labels[2])
    ax[0, 2].set_xlabel(x_column)
    ax[0, 2].set_title(f'Porcentaje Departamental - {labels[2]}')
    ax[0, 2].legend(loc=legend_loc, fontsize=size_legend)
    ax[0, 2].set_xticks(r1)
    ax[0, 2].set_xticklabels(años)

    for bar in bars3:
        height = bar.get_height()
        ax[0, 2].text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')

    # Gráfica para mujeres - Departamento
    bars4 = ax[1, 0].bar(r1, df1[df1['Sexo'] == 'Mujer'][y_column], color=colors[0], width=bar_width, label=labels[0])
    ax[1, 0].set_xlabel(x_column)
    ax[1, 0].set_title(f'Porcentaje Municipal - {labels[0]}')
    ax[1, 0].legend(loc=legend_loc, fontsize=size_legend)
    ax[1, 0].set_xticks(r1)
    ax[1, 0].set_xticklabels(años)

    for bar in bars4:
        height = bar.get_height()
        ax[1, 0].text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')

    # Gráfica para mujeres - Provincia
    bars5 = ax[1, 1].bar(r1, df2[df2['Sexo'] == 'Mujer'][y_column], color=colors[1], width=bar_width, label=labels[1])
    ax[1, 1].set_xlabel(x_column)
    ax[1, 1].set_title(f'Porcentaje Provincial - {labels[1]}')
    ax[1, 1].legend(loc=legend_loc, fontsize=size_legend)
    ax[1, 1].set_xticks(r1)
    ax[1, 1].set_xticklabels(años)

    for bar in bars5:
        height = bar.get_height()
        ax[1, 1].text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')

    # Gráfica para mujeres - Municipio
    bars6 = ax[1, 2].bar(r1, df3[df3['Sexo'] == 'Mujer'][y_column], color=colors[2], width=bar_width, label=labels[2])
    ax[1, 2].set_xlabel(x_column)
    ax[1, 2].set_title(f'Porcentaje Departamental - {labels[2]}')
    ax[1, 2].legend(loc=legend_loc, fontsize=size_legend)
    ax[1, 2].set_xticks(r1)
    ax[1, 2].set_xticklabels(años)

    for bar in bars6:
        height = bar.get_height()
        ax[1, 2].text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}%", ha='center', va='bottom', fontsize=size_graph, color='black')

    # Agregar etiquetas para los ejes y
    fig.text(0.01, 0.75, y_column + ' Hombres', ha='center', va='center', rotation='vertical', fontsize=12, color='black')
    fig.text(0.01, 0.25, y_column + ' Mujeres', ha='center', va='center', rotation='vertical', fontsize=12, color='black')

    # Ocultar el eje y de todas las subgráficas
    #for axs in ax:
    #    for a in axs:
    #        a.set_yticks([])

    fig.suptitle(title, fontsize=size_title)
    plt.tight_layout(rect=[0, 0, 1, 1])
    # Agregar referencia en la parte inferior del gráfico
    fig.text(0.5, 0.01, footer, ha='center', va='center', fontsize=size_footer, color='black')
    

    # Convertir la gráfica a base64
    tmp_file = io.BytesIO()
    plt.savefig(tmp_file, format='png')
    tmp_file.seek(0)
    plot_base64 = base64.b64encode(tmp_file.getvalue()).decode('utf-8')

    # Mostrar la gráfica en un componente HTML
    return html.Div([
        html.H2(title),
        html.Img(src='data:image/png;base64,{}'.format(plot_base64), style={'width': '100%'})
    ])

def generate_bars_comparison_gender(df1, df2, df3, x_column, y_column, title, size_title, footer, size_footer, size_legend, size_graph, labels, legend_loc):
    años = sorted(df1[x_column].unique())
    num_años = len(años)
    bar_width = 0.5  # Ancho de las barras

    colors = ['#DD6700', '#EA7E1F', '#FFB26F', '#FFCBA6', '#FFE5D1', '#135490', '#1769B5', '#2688E3', '#8FCFFF', '#CDE7FF']

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # Gráfica para Municipio
    bars1_h = ax[0].bar(np.arange(num_años), df1[df1['Sexo'] == 'Hombre'][y_column], bar_width, color=colors[5], label=f'Hombres - {labels[0]}')
    bars1_m = ax[0].bar(np.arange(num_años) + bar_width, df1[df1['Sexo'] == 'Mujer'][y_column], bar_width, color=colors[0], label=f'Mujeres - {labels[0]}')
    ax[0].set_xlabel(x_column)
    ax[0].set_title(f'Comparación Hombres vs Mujeres - {labels[0]}')
    ax[0].set_xticks(np.arange(num_años) + bar_width / 2)
    ax[0].set_xticklabels(años)
    ax[0].legend(loc=legend_loc, fontsize=size_legend)

    # Mostrar datos sobre las barras
    for bar in bars1_h + bars1_m:
        height = bar.get_height()
        ax[0].text(bar.get_x() + bar.get_width() / 2, height, f'{height}%', ha='center', va='bottom', fontsize=size_graph)

    # Gráfica para Provincia
    bars2_h = ax[1].bar(np.arange(num_años), df2[df2['Sexo'] == 'Hombre'][y_column], bar_width, color=colors[6], label=f'Hombres - {labels[1]}')
    bars2_m = ax[1].bar(np.arange(num_años) + bar_width, df2[df2['Sexo'] == 'Mujer'][y_column], bar_width, color=colors[1], label=f'Mujeres - {labels[1]}')
    ax[1].set_xlabel(x_column)
    ax[1].set_title(f'Comparación Hombres vs Mujeres - {labels[1]}')
    ax[1].set_xticks(np.arange(num_años) + bar_width / 2)
    ax[1].set_xticklabels(años)
    ax[1].legend(loc=legend_loc, fontsize=size_legend)

    # Mostrar datos sobre las barras
    for bar in bars2_h + bars2_m:
        height = bar.get_height()
        ax[1].text(bar.get_x() + bar.get_width() / 2, height, f'{height}%', ha='center', va='bottom', fontsize=size_graph)

    # Gráfica para Departamento
    bars3_h = ax[2].bar(np.arange(num_años), df3[df3['Sexo'] == 'Hombre'][y_column], bar_width, color=colors[7], label=f'Hombres - {labels[2]}')
    bars3_m = ax[2].bar(np.arange(num_años) + bar_width, df3[df3['Sexo'] == 'Mujer'][y_column], bar_width, color=colors[2], label=f'Mujeres - {labels[2]}')
    ax[2].set_xlabel(x_column)
    ax[2].set_title(f'Comparación Hombres vs Mujeres - {labels[2]}')
    ax[2].set_xticks(np.arange(num_años) + bar_width / 2)
    ax[2].set_xticklabels(años)
    ax[2].legend(loc=legend_loc, fontsize=size_legend)

    # Mostrar datos sobre las barras
    for bar in bars3_h + bars3_m:
        height = bar.get_height()
        ax[2].text(bar.get_x() + bar.get_width() / 2, height, f'{height}%', ha='center', va='bottom', fontsize=size_graph)

    fig.suptitle(title, fontsize=size_title)
    plt.tight_layout(rect=[0, 0, 1, 1])
    # Agregar referencia en la parte inferior del gráfico
    fig.text(0.5, 0.01, footer, ha='center', va='center', fontsize=size_footer, color='black')

    # Convertir la gráfica a base64
    tmp_file = io.BytesIO()
    plt.savefig(tmp_file, format='png')
    tmp_file.seek(0)
    plot_base64 = base64.b64encode(tmp_file.getvalue()).decode('utf-8')

    # Mostrar la gráfica en un componente HTML
    return html.Div([
        html.H2(title),
        html.Img(src='data:image/png;base64,{}'.format(plot_base64), style={'width': '100%'})
    ])

plt.switch_backend('Agg')

def plot_top_services_by_year_and_gender(df, title, size_title, footer, size_footer, size_graph):
    years = [2021, 2022, 2023]
    colors = ['#DD6700', '#EA7E1F', '#FFB26F', '#FFCBA6', '#FFE5D1', '#135490', '#1769B5', '#2688E3', '#8FCFFF', '#CDE7FF']
    genders = ['Hombres', 'Mujeres']

    fig, axes = plt.subplots(2, 3, figsize=(16, 12), sharey=True)

    for j, gender in enumerate(genders):
        for i, year in enumerate(years):
            bar_width = 1
            if gender == 'Hombres':
                col_total = 'Total Hombres'
                col_percent = 'Porcentaje Hombres'
            else:
                col_total = 'Total Mujeres'
                col_percent = 'Porcentaje Mujeres'

            filtered_df = df[df['Año'] == year].nlargest(5, col_total)

            ax = axes[j, i]
            if gender == 'Hombres':
                bars = ax.bar(filtered_df['Especialidad'], filtered_df[col_percent], color=colors[5:10], width=bar_width)
            else:
                bars = ax.bar(filtered_df['Especialidad'], filtered_df[col_percent], color=colors[0:5], width=bar_width)

            ax.set_title(f'Top 5 Especialidades para {gender} en {year}')
            ax.set_xlabel('Especialidad')
            ax.set_ylabel(f'Porcentaje {gender}')
            ax.set_xticks(range(len(filtered_df['Especialidad'])))
            ax.set_xticklabels(filtered_df['Especialidad'], rotation=45, ha='right')

            # Añadir los porcentajes sobre cada barra
            for bar, percent in zip(bars, filtered_df[col_percent]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{percent:.2f}%', ha='center', va='bottom', fontsize=size_graph)

    fig.suptitle(title, fontsize=size_title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar rect para dar espacio al título superior
    fig.text(0.5, 0.01, footer, ha='center', va='center', fontsize=size_footer, color='black')

    # Convertir la gráfica a base64
    tmp_file = io.BytesIO()
    plt.savefig(tmp_file, format='png')
    tmp_file.seek(0)
    plot_base64 = base64.b64encode(tmp_file.getvalue()).decode('utf-8')
    plt.close(fig)  # Cerrar explícitamente la figura

    # Mostrar la gráfica en un componente HTML
    return html.Div([
        html.H2(title),
        html.Img(src='data:image/png;base64,{}'.format(plot_base64), style={'width': '100%'})
    ])

def plot_top5_especialidades(df, years, title, size_title, footer, size_footer, size_legend, size_graph, figsize):
    # Grupos de edades y sus etiquetas
    grupos_edad = ['% [H] 0-9', '% [H] 10-19', '% [H] 20-39', '% [H] 40-49', '% [H] 50-59', '% [H] 60+']
    edades = ['0-9', '10-19', '20-39', '40-49', '50-59', '60+']
    
    # Lista para almacenar los componentes HTML de cada gráfica
    graphs = []
    
    for year in years:
        # Filtrar el DataFrame para el año especificado
        df_year = df[df['Año'] == year]

        # Preparar datos para hombres y mujeres
        top5_data_hombres = []
        top5_data_mujeres = []

        for grupo in grupos_edad:
            # Top 5 para hombres
            top5_hombres = df_year[['Especialidad', grupo]].sort_values(by=grupo, ascending=False).head(5)
            top5_hombres = top5_hombres.rename(columns={grupo: 'Porcentaje'})
            top5_hombres['Grupo Edad'] = grupo
            top5_data_hombres.append(top5_hombres)

            # Top 5 para mujeres
            grupo_mujeres = grupo.replace('[H]', '[M]')
            top5_mujeres = df_year[['Especialidad', grupo_mujeres]].sort_values(by=grupo_mujeres, ascending=False).head(5)
            top5_mujeres = top5_mujeres.rename(columns={grupo_mujeres: 'Porcentaje'})
            top5_mujeres['Grupo Edad'] = grupo
            top5_data_mujeres.append(top5_mujeres)

        # Concatenar datos
        top5_df_hombres = pd.concat(top5_data_hombres, ignore_index=True)
        top5_df_mujeres = pd.concat(top5_data_mujeres, ignore_index=True)

        # Mapear los nombres de los grupos de edad
        top5_df_hombres['Grupo Edad'] = top5_df_hombres['Grupo Edad'].map(dict(zip(grupos_edad, edades)))
        top5_df_mujeres['Grupo Edad'] = top5_df_mujeres['Grupo Edad'].map(dict(zip(grupos_edad, edades)))

        # Definir paletas de colores personalizados (azules para hombres y naranjas para mujeres)
        colores_hombres = ['#135490', '#1769B5', '#2688E3', '#8FCFFF', '#CDE7FF', '#6BA8CC', '#4E9AC5', '#3D85B1']
        colores_mujeres = ['#DD6700', '#EA7E1F', '#FFB26F', '#FFCBA6', '#FFE5D1', '#FFD69C', '#FFB473', '#FF9E4C']

        # Crear figura y ejes
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)

        # Gráfico para hombres
        sns.barplot(ax=axes[0], x='Grupo Edad', y='Porcentaje', hue='Especialidad', data=top5_df_hombres,
                    palette=colores_hombres)
        axes[0].set_title(f'Top 5 Especialidades por Grupo de Edad para Hombres en {year}')
        axes[0].set_xlabel('Grupo de Edad')
        axes[0].set_ylabel('Porcentaje')
        axes[0].legend(title='Especialidad', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=size_legend)

        # Anotaciones sobre las barras para hombres
        for p in axes[0].patches:
            if p.get_height() != 0:  # Verificar si el valor no es 0.0
                axes[0].annotate(format(p.get_height(), '.1f'),
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom',
                                textcoords='offset points', fontsize=size_graph)

        # Gráfico para mujeres
        sns.barplot(ax=axes[1], x='Grupo Edad', y='Porcentaje', hue='Especialidad', data=top5_df_mujeres,
                    palette=colores_mujeres)
        axes[1].set_title(f'Top 5 Especialidades por Grupo de Edad para Mujeres en {year}')
        axes[1].set_xlabel('Grupo de Edad')
        axes[1].set_ylabel('Porcentaje')
        axes[1].legend(title='Especialidad', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=size_legend)

        # Añadir valores numéricos sobre las barras para mujeres
        for p in axes[1].patches:
            if p.get_height() != 0:
                axes[1].annotate(format(p.get_height(), '.1f'),
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom',
                                textcoords='offset points', fontsize=size_graph)

        fig.suptitle(title, fontsize=size_title)
        plt.tight_layout(rect=[0, 0, 1, 1])
        # Agregar referencia en la parte inferior del gráfico
        fig.text(0.5, 0.01, footer, ha='center', va='center', fontsize=size_footer, color='black')

        # Convertir la gráfica a base64
        tmp_file = io.BytesIO()
        plt.savefig(tmp_file, format='png')
        tmp_file.seek(0)
        plot_base64 = base64.b64encode(tmp_file.getvalue()).decode('utf-8')
        plt.close(fig)

        # Agregar la gráfica como componente HTML a la lista
        graphs.append(html.Div([
            html.H2(f'Top 5 Especialidades por Grupo de Edad para Hombres y Mujeres en {year}'),
            html.Img(src='data:image/png;base64,{}'.format(plot_base64), style={'width': '100%'})
        ]))
    
    # Retornar todos los gráficos como una lista de componentes HTML
    return graphs

# Define el layout de la aplicación
app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    html.Div([
        html.H2("Menú"),
        html.Ul([
            html.Li([
                "Cancer: ",
                dcc.Link('Guarani', href='/cancer/guarani'), " | ",
                dcc.Link('Afroboliviano', href='/cancer/afroboliviano')
            ]),
            html.Li([
                "Diabetes: ",
                dcc.Link('Guarani', href='/diabetes/guarani'), " | ",
                dcc.Link('Afroboliviano', href='/diabetes/afroboliviano')
            ]),
            html.Li([
                "Hipertensión Arterial: ",
                dcc.Link('Guarani', href='/hipertension/guarani'), " | ",
                dcc.Link('Afroboliviano', href='/hipertension/afroboliviano')
            ]),
            html.Li([
                "Obesidad: ",
                dcc.Link('Guarani', href='/obesidad/guarani'), " | ",
                dcc.Link('Afroboliviano', href='/obesidad/afroboliviano')
            ]),
            html.Li([
                "Neumonia: ",
                dcc.Link('Guarani', href='/neumonia/guarani'), " | ",
                dcc.Link('Afroboliviano', href='/neumonia/afroboliviano')
            ]),
            html.Li([
                "Chagas: ",
                dcc.Link('Guarani', href='/chagas/guarani'), " | ",
                dcc.Link('Afroboliviano', href='/chagas/afroboliviano')
            ]),
            html.Li([
                "VIH: ",
                dcc.Link('Guarani', href='/vih/guarani'), " | ",
                dcc.Link('Afroboliviano', href='/vih/afroboliviano')
            ]),
            html.Li([
                "Nutricion: ",
                dcc.Link('Guarani', href='/nutricion/guarani'), " | ",
                dcc.Link('Afroboliviano', href='/nutricion/afroboliviano')
            ]),
            html.Li(dcc.Link('Embarazo Adolescente', href='/embarazo')),
            html.Li(dcc.Link('Consultas Externas', href='/consultas')),
        ], className='menu')
    ], className='menu-column'),
    html.Div([
        #html.H1(""),
        html.Div(id='page-content')
    ], className='content-column'),
    html.Div(id='btn-calcular', style={'display': 'none'}),  # Div oculto para generar el botón
], className='container')

# Función para crear una tabla Dash DataTable
def create_table(dataframe):
    return dash_table.DataTable(
        #id='table',
        columns=[{"name": i, "id": i} for i in dataframe.columns],
        data=dataframe.to_dict('records'),
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{Sexo} = "Hombre"'
                },
                'backgroundColor': '#CDE7FF',
                'color': 'black'
            },
            {
                'if': {
                    'filter_query': '{Sexo} = "Mujer"'
                },
                'backgroundColor': '#FFE5D1',
                'color': 'black'
            },
        ],
        style_cell={'textAlign': 'center'}
    )

# Definir opciones de dataframes
opciones_dataframes = [
    {'label': 'Camiri', 'value': 'Camiri'},
    {'label': 'Gutierrez', 'value': 'Gutierrez'},
    {'label': 'Cordillera', 'value': 'Cordillera'},
    {'label': 'Santa Cruz', 'value': 'Santa Cruz'}
]

# Define el layout de la página de cálculo
def generate_calculo_layout(type, title, dataframe_defaults):
    opciones_dataframes = []
    if type == 'Guarani':
        opciones_dataframes = [
            {'label': 'Camiri', 'value': 'Camiri'},
            {'label': 'Gutierrez', 'value': 'Gutierrez'},
            {'label': 'Cordillera', 'value': 'Cordillera'},
            {'label': 'Santa Cruz', 'value': 'Santa Cruz'}
        ]
    else:
        opciones_dataframes = [
            {'label': 'La Asunta', 'value': 'La Asunta'},
            {'label': 'Irupana', 'value': 'Irupana'},
            {'label': 'Coroico', 'value': 'Coroico'},
            {'label': 'Coripata', 'value': 'Coripata'},
            {'label': 'Sud Yungas', 'value': 'Sud Yungas'},
            {'label': 'Nor Yungas', 'value': 'Nor Yungas'},
            {'label': 'La Paz', 'value': 'La Paz'}
        ]
        
    return html.Div([ 
        html.H1("Gráficos de Tendencia"),
        html.Div([
            html.Span('Factor'),
            dcc.Input(id='input-factor', type='number', value=10000, style={'width': '80px'})
        ]),
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
            value=dataframe_defaults  # Valores iniciales seleccionados
        ),
        html.Div([
            html.Label('Título del gráfico: '),
            dcc.Input(
                id='input-titulo',
                type='text',
                value=title
            ),
            html.Label("Tamaño de letra titulo: "),
            dcc.Input(
                id='input-tamaño-titulo',
                type='number',
                value='12'
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

def generate_calculo_layout_nutricion(type, title, dataframe_defaults):
    opciones_dataframes = []
    if type == 'Guarani':
        opciones_dataframes = [
            {'label': 'Camiri', 'value': 'Camiri'},
            {'label': 'Gutierrez', 'value': 'Gutierrez'},
            {'label': 'Cordillera', 'value': 'Cordillera'},
            {'label': 'Santa Cruz', 'value': 'Santa Cruz'}
        ]
    else:
        opciones_dataframes = [
            {'label': 'La Asunta', 'value': 'La Asunta'},
            {'label': 'Irupana', 'value': 'Irupana'},
            {'label': 'Coroico', 'value': 'Coroico'},
            {'label': 'Coripata', 'value': 'Coripata'},
            {'label': 'Sud Yungas', 'value': 'Sud Yungas'},
            {'label': 'Nor Yungas', 'value': 'Nor Yungas'},
            {'label': 'La Paz', 'value': 'La Paz'}
        ]

    return html.Div([    
        html.H1("Gráficos de Tendencia"),
        html.Div([
            html.Span('Factor'),
            dcc.Input(id='input-factor', type='number', value=10000, style={'width': '80px'})
        ]),
        html.Label('Estado Nutricional:'),
        dcc.Dropdown(
            id='dropdown-type-nutrition',
            options=[
                {'label': 'Obesidad', 'value': 'o'},
                {'label': 'Sobrepeso', 'value': 's'},
                {'label': 'Desnutricion', 'value': 'd'},
            ],
            value='o'  # Valor inicial seleccionado
        ),
        html.Label('Gráfica a mostrar:'),
        dcc.Dropdown(
            id='dropdown-graphic-type',
            options=[
                {'label': 'Totales Juntos', 'value': 'tj'},
                {'label': 'Totales Separados', 'value': 'ts'},
            ],
            value='tj'  # Valor inicial seleccionado
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
            options=opciones_dataframes,  # Asignar opciones por defecto
            multi=True,
            value=dataframe_defaults  # Valores iniciales seleccionados
        ),
        html.Div([
            html.Label('Título del gráfico: '),
            dcc.Input(
                id='input-titulo',
                type='text',
                value=title  # Título inicial
            ),
            html.Label("Tamaño de letra título: "),
            dcc.Input(
                id='input-tamaño-titulo',
                type='number',
                value='12'
            )
        ]),
        
        html.Div([
            html.Label('Pie de Página: '),
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
            html.Label("Tamaño de letra de Números Gráficas: "),
            dcc.Input(
                id='input-tamaño-num-grafica',
                type='number',
                value='10',
                style={'width': '80px'}
            )
        ]),
        
        html.Button('Generar Gráfico', id='btn-calcular-nutricion'),
        html.Div(id='output-data-nutricion')
    ])

# Define el layout de la página de cálculo
calculo_layout_embarazo = html.Div([    
    html.H1("Gráficos de Tendencia"),
    html.Div([
        html.Span('Factor'),
        dcc.Input(id='input-factor', type='number', value=10000, style={'width': '80px'})
    ]),
    html.Label('Rango de edad:'),
    dcc.Dropdown(
        id='dropdown-type-age',
        options=[
            {'label': '< 15', 'value': 'r1'},
            {'label': '15 - 19', 'value': 'r2'},
            {'label': '< 19', 'value': 'r3'},
        ],
        value='r3'  # Valor inicial seleccionado
    ),
    html.Label('Meses de Embarazo:'),
    dcc.Dropdown(
        id='dropdown-type-mounth',
        options=[
            {'label': '< 5to mes + > 5to mes', 'value': 'm1'},
            {'label': '< 5to mes', 'value': 'm2'},
            {'label': '> 5to mes', 'value': 'm3'},
        ],
        value='m1'  # Valor inicial seleccionado
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
            value='Comparación a nivel departamental, provincial y municipal casos de X'
        ),
        html.Label("Tamaño de letra titulo: "),
        dcc.Input(
            id='input-tamaño-titulo',
            type='number',
            value='12'
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
    
    html.Button('Generar Gráfico', id='btn-calcular-embarazo'),
    html.Div(id='output-data-embarazo')
])

calculo_layout_consultas = html.Div([ 
    html.H1("Gráficos de Tendencia"),
    html.Div([
        html.Span('Factor'),
        dcc.Input(id='input-factor', type='number', value=10000, style={'width': '80px'})
    ]),
    html.Label('Grafica a mostrar:'),
    dcc.Dropdown(
        id='dropdown-graphic-type',
        options=[
            {'label': 'Porcentaje Comparacion Juntos Primer Nivel', 'value': 'pn1_1'},
            {'label': 'Porcentaje Comparacion Separado Primer Nivel', 'value': 'pn1_2'},
            {'label': 'Porcentaje Comparacion Entre Primer Nivel', 'value': 'pn1_3'},
            {'label': 'Por Edad Consultas Primer Nivel', 'value': 'pn1_4'},
            {'label': 'Por Especialidad Segundo Nivel', 'value': 'pn2_1'},
            {'label': 'Por Edad Consultas Segundo Nivel', 'value': 'pn2_2'},
        ],
        value='pn1_1'  # Valor inicial seleccionado
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
            value='Comparación a nivel departamental, provincial y municipal casos de X'
        ),
        html.Label("Tamaño de letra titulo: "),
        dcc.Input(
            id='input-tamaño-titulo',
            type='number',
            value='12'
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
    
    html.Button('Generar Gráfico', id='btn-calcular-consultas'),
    html.Div(id='output-data-consultas')
])

app.title = "Generate Graph Municipality"

# Callback para actualizar el contenido según la URL
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/cancer/guarani':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Cancer'),
            generate_calculo_layout('Guarani', 'Comparación a nivel departamental, provincial y municipal casos de Cancer', ['Santa Cruz', 'Cordillera', 'Camiri'])
        ])
    elif pathname == '/cancer/afroboliviano':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Cancer'),
            generate_calculo_layout('Afroboliviano', 'Comparación a nivel departamental, provincial y municipal casos de Cancer', ['La Paz', 'Nor Yungas', 'Coroico'])
        ])
    if pathname == '/diabetes/guarani':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Diabetes'),
            generate_calculo_layout('Guarani', 'Comparación a nivel departamental, provincial y municipal casos de Diabetes', ['Santa Cruz', 'Cordillera', 'Camiri'])
        ])
    elif pathname == '/diabetes/afroboliviano':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Diabetes'),
            generate_calculo_layout('Afroboliviano', 'Comparación a nivel departamental, provincial y municipal casos de Diabtes', ['La Paz', 'Nor Yungas', 'Coroico'])
        ])
    if pathname == '/hipertension/guarani':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Hipertensión Arterial'),
            generate_calculo_layout('Guarani', 'Comparación a nivel departamental, provincial y municipal casos de Hipertensión', ['Santa Cruz', 'Cordillera', 'Camiri'])
        ])
    elif pathname == '/hipertension/afroboliviano':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Diabetes'),
            generate_calculo_layout('Afroboliviano', 'Comparación a nivel departamental, provincial y municipal casos de Hipertensión', ['La Paz', 'Nor Yungas', 'Coroico'])
        ])
    if pathname == '/obesidad/guarani':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Obesidad'),
            generate_calculo_layout('Guarani', 'Comparación a nivel departamental, provincial y municipal casos de Obesidad', ['Santa Cruz', 'Cordillera', 'Camiri'])
        ])
    elif pathname == '/obesidad/afroboliviano':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Obesidad'),
            generate_calculo_layout('Afroboliviano', 'Comparación a nivel departamental, provincial y municipal casos de Obesidad', ['La Paz', 'Nor Yungas', 'Coroico'])
        ])
    if pathname == '/neumonia/guarani':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Neumonía'),
            generate_calculo_layout('Guarani', 'Comparación a nivel departamental, provincial y municipal casos de Neumonía', ['Santa Cruz', 'Cordillera', 'Camiri'])
        ])
    elif pathname == '/neumonia/afroboliviano':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Neumonía'),
            generate_calculo_layout('Afroboliviano', 'Comparación a nivel departamental, provincial y municipal casos de Neumonía', ['La Paz', 'Nor Yungas', 'Coroico'])
        ])
    if pathname == '/chagas/guarani':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Chagas'),
            generate_calculo_layout('Guarani', 'Comparación a nivel departamental, provincial y municipal casos de Chagas', ['Santa Cruz', 'Cordillera', 'Camiri'])
        ])
    elif pathname == '/chagas/afroboliviano':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Chagas'),
            generate_calculo_layout('Afroboliviano', 'Comparación a nivel departamental, provincial y municipal casos de Chagas', ['La Paz', 'Nor Yungas', 'Coroico'])
        ])
    if pathname == '/vih/guarani':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos VIH'),
            generate_calculo_layout('Guarani', 'Comparación a nivel departamental, provincial y municipal casos de VIH', ['Santa Cruz', 'Cordillera', 'Camiri'])
        ])
    elif pathname == '/vih/afroboliviano':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos VIH'),
            generate_calculo_layout('Afroboliviano', 'Comparación a nivel departamental, provincial y municipal casos de VIH', ['La Paz', 'Nor Yungas', 'Coroico'])
        ])
    elif pathname == '/nutricion/guarani':        
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Nutricion'),
            generate_calculo_layout_nutricion('Guarani', 'Comparación a nivel departamental, provincial y municipal casos de Nutricion Embarazadas', ['Santa Cruz', 'Cordillera', 'Camiri'])
        ])
    elif pathname == '/nutricion/afroboliviano':        
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Nutricion'),
            generate_calculo_layout_nutricion('Afroboliviano', 'Comparación a nivel departamental, provincial y municipal casos de Nutricion Embarazadas', ['La Paz', 'Nor Yungas', 'Coroico'])
        ])
    elif pathname == '/embarazo':
        df_c_embarazo, df_g_embarazo, d1, d2 = get_casos_embarazo()
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Embarazo Adolescente'),
            html.H2('Datos Camiri'),
            create_table(df_c_embarazo),
            html.H2('Datos Gutierrez'),
            create_table(df_g_embarazo),
            calculo_layout_embarazo
        ])
    elif pathname == '/consultas':
        df_c_consulta, df_g_consulta, d1, d2, df_c_consulta_2 = get_casos_consulta()

        df_c_consulta_t1 = calculate_table_total_percent(df_c_consulta)
        df_c_consulta_t2 = calculate_table_total_percent_age(df_c_consulta)
        df_g_consulta_t1 = calculate_table_total_percent(df_g_consulta)
        df_g_consulta_t2 = calculate_table_total_percent_age(df_g_consulta)

        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Embarazo Adolescente'),
            html.H2('Datos Camiri'),
            create_table(df_c_consulta_t1),
            html.P("-----"),
            create_table(df_c_consulta_t2),
            html.H2('Datos Gutierrez'),
            create_table(df_g_consulta_t1),
            html.P("-----"),
            create_table(df_g_consulta_t2),
            calculo_layout_consultas
        ])
    else:
        return html.Div([
            html.H1('Mi primera aplicación Dash en Heroku'),
            html.P('Hola mundo' + pathname),
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

# Callback para realizar el cálculo de incidencias y porcentajes
@app.callback(
    Output('output-data', 'children'),
    [
        Input('btn-calcular', 'n_clicks'),
        Input('dropdown-graphic-type', 'value'),
        Input('dropdown-type-percent', 'value'),
        Input('dropdown-dataframes', 'value'),
        Input('input-titulo', 'value'),
        Input('input-tamaño-titulo', 'value'),
        Input('input-pie', 'value'),
        Input('input-tamaño-pie', 'value'),
        Input('input-tamaño-leyenda', 'value'),
        Input('input-tamaño-num-grafica', 'value'),
        Input('dropdown-legend-loc', 'value')
    ],
    [State('input-factor', 'value'),
     State('url', 'pathname')]  # Capturar el pathname actual
)
def update_output(n_clicks, graphic_type, type_percent, selected_dataframes, titulo, 
                  tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, legend_loc, 
                  factor, pathname):
    if n_clicks:
        try: 
            if tamanio_titulo != None:
                tamanio_titulo = int(tamanio_titulo)
            else:
                tamanio_titulo = 16
            if tamanio_pie != None:
                tamanio_pie = int(tamanio_pie)
            else:
                tamanio_pie = 10
            if tamanio_leyenda != None:
                tamanio_leyenda = int(tamanio_leyenda)
            else:
                tamanio_leyenda = 8
            if tamanio_num_grafica != None:
                tamanio_num_grafica = int(tamanio_num_grafica)
            else:
                tamanio_num_grafica = 10
            
            #fig = go.Figure()
            resultados = []

            partes = pathname.split('/')
            if 'guarani' in pathname:
                df_c, df_g, df_pc, df_sc = get_casos(partes[1], partes[2])
                p_c, p_g, p_pc, p_sc = get_casos('poblacion', partes[2])
                m = p_c[p_c['Sexo'] == 'Mujer']['Total'].tolist()
                h = p_c[p_c['Sexo'] == 'Hombre']['Total'].tolist()
                p = p_c.groupby('Año')['Total'].sum().tolist()
                m_2 = p_g[p_g['Sexo'] == 'Mujer']['Total'].tolist()
                h_2 = p_g[p_g['Sexo'] == 'Hombre']['Total'].tolist()
                p_2 = p_g.groupby('Año')['Total'].sum().tolist()
                m_3 = p_pc[p_pc['Sexo'] == 'Mujer']['Total'].tolist()
                h_3 = p_pc[p_pc['Sexo'] == 'Hombre']['Total'].tolist()
                p_3 = p_pc.groupby('Año')['Total'].sum().tolist()
                m_4 = p_sc[p_sc['Sexo'] == 'Mujer']['Total'].tolist()
                h_4 = p_sc[p_sc['Sexo'] == 'Hombre']['Total'].tolist()
                p_4 = p_sc.groupby('Año')['Total'].sum().tolist()

                df_c_t = generate_total(df_c)
                df_g_t = generate_total(df_g)
                df_pc_t = generate_total(df_pc)
                df_sc_t = generate_total(df_sc)
                
                df_c_t = calculate_total(df_c_t, factor, p, 'Total')
                df_g_t = calculate_total(df_g_t, factor, p_2, 'Total')
                df_pc_t = calculate_total(df_pc_t, factor, p_3, 'Total')
                df_sc_t = calculate_total(df_sc_t, factor, p_4, 'Total')
                
                df_c = calculate_gender(df_c, factor, m, h)
                df_g = calculate_gender(df_g, factor, m_2, h_2)
                df_pc = calculate_gender(df_pc, factor, m_3, h_3)
                df_sc = calculate_gender(df_sc, factor, m_4, h_4)

                df_c.sort_values(by='Año', inplace=True)
                df_g.sort_values(by='Año', inplace=True)
                df_pc.sort_values(by='Año', inplace=True)
                df_sc.sort_values(by='Año', inplace=True)

                dataframes = {
                    'Santa Cruz': df_sc,
                    'Cordillera': df_pc,
                    'Camiri': df_c,
                    'Gutierrez': df_g
                }
                
                dataframes_total = {
                    'Santa Cruz': df_sc_t,
                    'Cordillera': df_pc_t,
                    'Camiri': df_c_t,
                    'Gutierrez': df_g_t
                }

                group_edad = {
                    'Santa Cruz': p_sc,
                    'Cordillera': p_pc,
                    'Camiri': p_c,
                    'Gutierrez': p_g
                }
                
            elif 'afroboliviano' in pathname:
                df_a, df_i, df_co, df_cp, df_sy, df_ny, df_lp = get_casos(partes[1], partes[2])
                p_a, p_i, p_co, p_cp, p_sy, p_ny, p_lp = get_casos('poblacion', partes[2])

                m = p_a[p_a['Sexo'] == 'Mujer']['Total'].tolist()
                h = p_a[p_a['Sexo'] == 'Hombre']['Total'].tolist()
                p = p_a.groupby('Año')['Total'].sum().tolist()
                m_2 = p_i[p_i['Sexo'] == 'Mujer']['Total'].tolist()
                h_2 = p_i[p_i['Sexo'] == 'Hombre']['Total'].tolist()
                p_2 = p_i.groupby('Año')['Total'].sum().tolist()
                m_3 = p_co[p_co['Sexo'] == 'Mujer']['Total'].tolist()
                h_3 = p_co[p_co['Sexo'] == 'Hombre']['Total'].tolist()
                p_3 = p_co.groupby('Año')['Total'].sum().tolist()
                m_4 = p_cp[p_cp['Sexo'] == 'Mujer']['Total'].tolist()
                h_4 = p_cp[p_cp['Sexo'] == 'Hombre']['Total'].tolist()
                p_4 = p_cp.groupby('Año')['Total'].sum().tolist()
                m_5 = p_sy[p_sy['Sexo'] == 'Mujer']['Total'].tolist()
                h_5 = p_sy[p_sy['Sexo'] == 'Hombre']['Total'].tolist()
                p_5 = p_sy.groupby('Año')['Total'].sum().tolist()
                m_6 = p_ny[p_ny['Sexo'] == 'Mujer']['Total'].tolist()
                h_6 = p_ny[p_ny['Sexo'] == 'Hombre']['Total'].tolist()
                p_6 = p_ny.groupby('Año')['Total'].sum().tolist()
                m_7 = p_lp[p_lp['Sexo'] == 'Mujer']['Total'].tolist()
                h_7 = p_lp[p_lp['Sexo'] == 'Hombre']['Total'].tolist()
                p_7 = p_lp.groupby('Año')['Total'].sum().tolist()
                
                df_a_t = generate_total(df_a)
                df_i_t = generate_total(df_i)
                df_co_t = generate_total(df_co)
                df_cp_t = generate_total(df_cp)
                df_sy_t = generate_total(df_sy)
                df_ny_t = generate_total(df_ny)
                df_lp_t = generate_total(df_lp)
                
                df_a_t = calculate_total(df_a_t, factor, p, 'Total')
                df_i_t = calculate_total(df_i_t, factor, p_2, 'Total')
                df_co_t = calculate_total(df_co_t, factor, p_3, 'Total')
                df_cp_t = calculate_total(df_cp_t, factor, p_4, 'Total')
                df_sy_t = calculate_total(df_sy_t, factor, p_5, 'Total')
                df_ny_t = calculate_total(df_ny_t, factor, p_6, 'Total')
                df_lp_t = calculate_total(df_lp_t, factor, p_7, 'Total')
                
                df_a = calculate_gender(df_a, factor, m, h)
                df_i = calculate_gender(df_i, factor, m_2, h_2)
                df_co = calculate_gender(df_co, factor, m_3, h_3)
                df_cp = calculate_gender(df_cp, factor, m_4, h_4)
                df_sy = calculate_gender(df_sy, factor, m_5, h_5)
                df_ny = calculate_gender(df_ny, factor, m_6, h_6)
                df_lp = calculate_gender(df_lp, factor, m_7, h_7)

                df_a.sort_values(by='Año', inplace=True)
                df_i.sort_values(by='Año', inplace=True)
                df_co.sort_values(by='Año', inplace=True)
                df_cp.sort_values(by='Año', inplace=True)
                df_sy.sort_values(by='Año', inplace=True)
                df_ny.sort_values(by='Año', inplace=True)
                df_lp.sort_values(by='Año', inplace=True)
                
                dataframes = {
                    'La Paz': df_lp,
                    'Nor Yungas': df_ny,
                    'Sud Yungas': df_sy,
                    'Coroico': df_co,
                    'Coripata': df_cp,
                    'La Asunta': df_a,
                    'Irupana': df_i
                }
                
                dataframes_total = {
                    'La Paz': df_lp_t,
                    'Nor Yungas': df_ny_t,
                    'Sud Yungas': df_sy_t,
                    'Coroico': df_co_t,
                    'Coripata': df_cp_t,
                    'La Asunta': df_a_t,
                    'Irupana': df_i_t
                }

                group_edad = {
                    'La Paz': p_lp,
                    'Nor Yungas': p_ny,
                    'Sud Yungas': p_sy,
                    'Coroico': p_co,
                    'Coripata': p_cp,
                    'La Asunta': p_a,
                    'Irupana': p_i
                }
                
            if n_clicks > 0:    
                if (len(selected_dataframes) == 3):
                    if graphic_type == 't':
                        resultados.append(generate_lines_total(dataframes_total[selected_dataframes[0]], dataframes_total[selected_dataframes[1]], dataframes_total[selected_dataframes[2]], 'Año', type_percent, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, selected_dataframes, legend_loc))
                        resultados.append(html.H2(f'Datos '+selected_dataframes[2]))
                        resultados.append(create_table(dataframes_total[selected_dataframes[2]]))
                    elif graphic_type == 's1':
                        resultados.append(generate_lines_gender(dataframes[selected_dataframes[0]], dataframes[selected_dataframes[1]], dataframes[selected_dataframes[2]], 'Año', type_percent, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, selected_dataframes, legend_loc))
                    elif graphic_type == 's2':
                        resultados.append(generate_lines_separate_gender(dataframes[selected_dataframes[0]], dataframes[selected_dataframes[1]], dataframes[selected_dataframes[2]], 'Año', type_percent, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, selected_dataframes, legend_loc))
                    elif graphic_type == 's3':
                        resultados.append(generate_lines_comparison_gender(dataframes[selected_dataframes[0]], dataframes[selected_dataframes[1]], dataframes[selected_dataframes[2]], 'Año', type_percent, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, selected_dataframes, legend_loc))
                    else:
                        return html.Div("") 
                    
                    if(len(resultados) < 3):
                        resultados.append(html.H2(f'Datos '+selected_dataframes[2]))
                        resultados.append(create_table(dataframes[selected_dataframes[2]]))
                elif (len(selected_dataframes) == 1):
                    if graphic_type == 'e':
                        p = group_edad[selected_dataframes[0]]
                        df_mujeres = p[p['Sexo'] == 'Mujer']
                        columnas_edad = ['0-9', '10-19', '20-39', '40-49', '50-59', '60+']
                        listas_edad = [df_mujeres[col].tolist() for col in columnas_edad]
                        r_m = sum(listas_edad, [])
                        df_hombres = p[p['Sexo'] == 'Hombre']
                        listas_edad = [df_hombres[col].tolist() for col in columnas_edad]
                        r_h = sum(listas_edad, [])

                        if 'neumonia' in pathname or 'chagas' in pathname:
                            resultados.append(plot_age_percentages(dataframes[selected_dataframes[0]], r_m, r_h, 'Año', type_percent, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, legend_loc, 0.2))
                        else:
                            resultados.append(plot_age_percentages(dataframes[selected_dataframes[0]], r_m, r_h, 'Año', type_percent, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, legend_loc, 0.25))
                        resultados.append(html.H2(f'Datos '+selected_dataframes[0]))
                        resultados.append(create_table(dataframes[selected_dataframes[0]]))
                    else:
                        return html.Div("")
                else:
                    # Si falta algún dataframe seleccionado, retornar un mensaje de error o un div vacío
                    return html.Div("")
                    
                return resultados
            
            return html.Div("")
              
        except Exception as e:
            return html.Div(f'Error: {e}')

# Callback para realizar el cálculo de incidencias y porcentajes
@app.callback(
    Output('output-data-nutricion', 'children'),
    [
        Input('btn-calcular-nutricion', 'n_clicks'),
        Input('dropdown-type-nutrition', 'value'),
        Input('dropdown-graphic-type', 'value'),
        Input('dropdown-type-percent', 'value'),
        Input('dropdown-dataframes', 'value'),
        Input('input-titulo', 'value'),
        Input('input-tamaño-titulo', 'value'),
        Input('input-pie', 'value'),
        Input('input-tamaño-pie', 'value'),
        Input('input-tamaño-leyenda', 'value'),
        Input('input-tamaño-num-grafica', 'value'),
        Input('dropdown-legend-loc', 'value')
    ],
    [State('input-factor', 'value'),
     State('url', 'pathname')]  # Capturar el pathname actual
)
def update_output(n_clicks, type_nutrition, graphic_type, type_percent, selected_dataframes, titulo, 
                  tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, legend_loc, 
                  factor, pathname):
    if n_clicks:
        try:
            if tamanio_titulo != None:
                tamanio_titulo = int(tamanio_titulo)
            else:
                tamanio_titulo = 16
            if tamanio_pie != None:
                tamanio_pie = int(tamanio_pie)
            else:
                tamanio_pie = 10
            if tamanio_leyenda != None:
                tamanio_leyenda = int(tamanio_leyenda)
            else:
                tamanio_leyenda = 8
            if tamanio_num_grafica != None:
                tamanio_num_grafica = int(tamanio_num_grafica)
            else:
                tamanio_num_grafica = 10

            partes = pathname.split('/')
            if 'guarani' in pathname:
                df_c_o, df_g_o, df_pc_o, df_sc_o, df_c_s, df_g_s, df_pc_s, df_sc_s, df_c_d, df_g_d, df_pc_d, df_sc_d = get_casos(partes[1], partes[2])
                # Convertir a listas de poblaciones
                p_c, p_g, p_pc, p_sc = get_casos('poblacion-especial', partes[2])
                p = p_c.groupby('Año')['Embarazos'].sum().tolist()
                p_2 = p_g.groupby('Año')['Embarazos'].sum().tolist()
                p_3 = p_pc.groupby('Año')['Embarazos'].sum().tolist()
                p_4 = p_sc.groupby('Año')['Embarazos'].sum().tolist()

                if type_nutrition == 'o':
                    df_c = calculate_total(df_c_o, factor, p, 'Embarazadas')
                    df_g = calculate_total(df_g_o, factor, p_2, 'Embarazadas')
                    df_pc = calculate_total(df_pc_o, factor, p_3, 'Embarazadas')
                    df_sc = calculate_total(df_sc_o, factor, p_4, 'Embarazadas')
                elif type_nutrition == 's':
                    df_c = calculate_total(df_c_s, factor, p, 'Embarazadas')
                    df_g = calculate_total(df_g_s, factor, p_2, 'Embarazadas')
                    df_pc = calculate_total(df_pc_s, factor, p_3, 'Embarazadas')
                    df_sc = calculate_total(df_sc_s, factor, p_4, 'Embarazadas')
                elif type_nutrition == 'd':
                    df_c = calculate_total(df_c_d, factor, p, 'Embarazadas')
                    df_g = calculate_total(df_g_d, factor, p_2, 'Embarazadas')
                    df_pc = calculate_total(df_pc_d, factor, p_3, 'Embarazadas')
                    df_sc = calculate_total(df_sc_d, factor, p_4, 'Embarazadas')

                # Seleccionar los dataframes según la selección del usuario
                df_c.sort_values(by='Año', inplace=True)
                df_g.sort_values(by='Año', inplace=True)
                df_pc.sort_values(by='Año', inplace=True)
                df_sc.sort_values(by='Año', inplace=True)
                
                dataframes = {
                    'Santa Cruz': df_sc,
                    'Cordillera': df_pc,
                    'Camiri': df_c,
                    'Gutierrez': df_g
                }    
            elif 'afroboliviano' in pathname:
                df_a_o, df_i_o, df_co_o, df_cp_o, df_sy_o, df_ny_o, df_lp_o, df_a_s, df_i_s, df_co_s, df_cp_s, df_sy_s, df_ny_s, df_lp_s, df_a_d, df_i_d, df_co_d, df_cp_d, df_sy_d, df_ny_d, df_lp_d = get_casos(partes[1], partes[2])
                # Convertir a listas de poblaciones
                p_a, p_i, p_co, p_cp, p_sy, p_ny, p_lp = get_casos('poblacion-especial', partes[2])
                p = p_a.groupby('Año')['Embarazos'].sum().tolist()
                p_2 = p_i.groupby('Año')['Embarazos'].sum().tolist()
                p_3 = p_co.groupby('Año')['Embarazos'].sum().tolist()
                p_4 = p_cp.groupby('Año')['Embarazos'].sum().tolist()
                p_5 = p_sy.groupby('Año')['Embarazos'].sum().tolist()
                p_6 = p_ny.groupby('Año')['Embarazos'].sum().tolist()
                p_7 = p_lp.groupby('Año')['Embarazos'].sum().tolist()

                if type_nutrition == 'o':
                    df_a = calculate_total(df_a_o, factor, p, 'Embarazadas')
                    df_i = calculate_total(df_i_o, factor, p_2, 'Embarazadas')
                    df_co = calculate_total(df_co_o, factor, p_3, 'Embarazadas')
                    df_cp = calculate_total(df_cp_o, factor, p_4, 'Embarazadas')
                    df_sy = calculate_total(df_sy_o, factor, p_5, 'Embarazadas')
                    df_ny = calculate_total(df_ny_o, factor, p_6, 'Embarazadas')
                    df_lp = calculate_total(df_lp_o, factor, p_7, 'Embarazadas')
                elif type_nutrition == 's':
                    df_a = calculate_total(df_a_s, factor, p, 'Embarazadas')
                    df_i = calculate_total(df_i_s, factor, p_2, 'Embarazadas')
                    df_co = calculate_total(df_co_s, factor, p_3, 'Embarazadas')
                    df_cp = calculate_total(df_cp_s, factor, p_4, 'Embarazadas')
                    df_sy = calculate_total(df_sy_s, factor, p_5, 'Embarazadas')
                    df_ny = calculate_total(df_ny_s, factor, p_6, 'Embarazadas')
                    df_lp = calculate_total(df_lp_s, factor, p_7, 'Embarazadas')
                elif type_nutrition == 'd':
                    df_a = calculate_total(df_a_d, factor, p, 'Embarazadas')
                    df_i = calculate_total(df_i_d, factor, p_2, 'Embarazadas')
                    df_co = calculate_total(df_co_d, factor, p_3, 'Embarazadas')
                    df_cp = calculate_total(df_cp_d, factor, p_4, 'Embarazadas')
                    df_sy = calculate_total(df_sy_d, factor, p_5, 'Embarazadas')
                    df_ny = calculate_total(df_ny_d, factor, p_6, 'Embarazadas')
                    df_lp = calculate_total(df_lp_d, factor, p_7, 'Embarazadas')

                df_a.sort_values(by='Año', inplace=True)
                df_i.sort_values(by='Año', inplace=True)
                df_co.sort_values(by='Año', inplace=True)
                df_cp.sort_values(by='Año', inplace=True)
                df_sy.sort_values(by='Año', inplace=True)
                df_ny.sort_values(by='Año', inplace=True)
                df_lp.sort_values(by='Año', inplace=True)
                
                dataframes = {
                    'La Paz': df_lp,
                    'Nor Yungas': df_ny,
                    'Sud Yungas': df_sy,
                    'Coroico': df_co,
                    'Coripata': df_cp,
                    'La Asunta': df_a,
                    'Irupana': df_i
                }   

            resultados = []
            if n_clicks > 0:                
                if (len(selected_dataframes) == 3):
                    if graphic_type == 'tj':
                        # Generar y retornar el gráfico con los parámetros seleccionados
                        resultados.append(generate_lines_total(dataframes[selected_dataframes[0]], dataframes[selected_dataframes[1]], dataframes[selected_dataframes[2]], 'Año', type_percent, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, selected_dataframes, legend_loc))
                    elif graphic_type == 'ts':
                        # Generar y retornar el gráfico con los parámetros seleccionados
                        resultados.append(generate_lines_separate_nutricion(dataframes[selected_dataframes[0]], dataframes[selected_dataframes[1]], dataframes[selected_dataframes[2]], 'Año', type_percent, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, selected_dataframes, legend_loc))
                    else:
                        return html.Div("") 
                else:
                    # Si falta algún dataframe seleccionado, retornar un mensaje de error o un div vacío
                    return html.Div("")
            resultados.append(html.H2(f'Datos '+selected_dataframes[2]))
            resultados.append(create_table(dataframes[selected_dataframes[2 ]]))
            return resultados       
        
        except Exception as e:
            return html.Div(f'Error: {e}')
        
# Callback para realizar el cálculo de incidencias y porcentajes
@app.callback(
    Output('output-data-embarazo', 'children'),
    [
        Input('btn-calcular-embarazo', 'n_clicks'),
        Input('dropdown-type-age', 'value'),
        Input('dropdown-type-mounth', 'value'),
        Input('dropdown-type-percent', 'value'),
        Input('dropdown-dataframes', 'value'),
        Input('input-titulo', 'value'),
        Input('input-tamaño-titulo', 'value'),
        Input('input-pie', 'value'),
        Input('input-tamaño-pie', 'value'),
        Input('input-tamaño-leyenda', 'value'),
        Input('input-tamaño-num-grafica', 'value'),
        Input('dropdown-legend-loc', 'value')
    ],
    [State('input-factor', 'value'),
     State('url', 'pathname')]  # Capturar el pathname actual
)
def update_output(n_clicks, type_age, type_mounth, type_percent, selected_dataframes, titulo, 
                  tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, legend_loc, 
                  factor, pathname):
    if n_clicks:
        try:
            # Convertir a listas de poblaciones
            p_c, p_g, p_pc, p_sc = get_casos('poblacion-especial', 'guarani')
            if type_age == 'r1':
                p = p_c.groupby('Año')['10-14'].sum().tolist()
                p_2 = p_g.groupby('Año')['10-14'].sum().tolist()
                p_3 = p_pc.groupby('Año')['10-14'].sum().tolist()
                p_4 = p_sc.groupby('Año')['10-14'].sum().tolist()
            elif type_age == 'r2':
                p = p_c.groupby('Año')['15-19'].sum().tolist()
                p_2 = p_g.groupby('Año')['15-19'].sum().tolist()
                p_3 = p_pc.groupby('Año')['15-19'].sum().tolist()
                p_4 = p_sc.groupby('Año')['15-19'].sum().tolist()
            else:
                p = p_c.groupby('Año')['Adolescentes'].sum().tolist()
                p_2 = p_g.groupby('Año')['Adolescentes'].sum().tolist()
                p_3 = p_pc.groupby('Año')['Adolescentes'].sum().tolist()
                p_4 = p_sc.groupby('Año')['Adolescentes'].sum().tolist()
            
            if tamanio_titulo != None:
                tamanio_titulo = int(tamanio_titulo)
            else:
                tamanio_titulo = 16
            if tamanio_pie != None:
                tamanio_pie = int(tamanio_pie)
            else:
                tamanio_pie = 10
            if tamanio_leyenda != None:
                tamanio_leyenda = int(tamanio_leyenda)
            else:
                tamanio_leyenda = 8
            if tamanio_num_grafica != None:
                tamanio_num_grafica = int(tamanio_num_grafica)
            else:
                tamanio_num_grafica = 10
            
            if pathname == '/embarazo':
                df_c_embarazo, df_g_embarazo, df_pc_embarazo, df_sc_embarazo = get_casos_embarazo()

                if type_mounth == 'm1':
                    df_c_embarazo = df_c_embarazo.groupby('Año').sum().reset_index()
                    df_g_embarazo = df_g_embarazo.groupby('Año').sum().reset_index()
                    df_pc_embarazo = df_pc_embarazo.groupby('Año').sum().reset_index()
                    df_sc_embarazo = df_sc_embarazo.groupby('Año').sum().reset_index()
                elif type_mounth == 'm2':
                    df_c_embarazo = df_c_embarazo[df_c_embarazo['Tipo'] == 'Nuevo < 5'].drop(columns=['Tipo']).reset_index(drop=True)
                    df_g_embarazo = df_g_embarazo[df_g_embarazo['Tipo'] == 'Nuevo < 5'].drop(columns=['Tipo']).reset_index(drop=True)
                    df_pc_embarazo = df_pc_embarazo[df_pc_embarazo['Tipo'] == 'Nuevo < 5'].drop(columns=['Tipo']).reset_index(drop=True)
                    df_sc_embarazo = df_sc_embarazo[df_sc_embarazo['Tipo'] == 'Nuevo < 5'].drop(columns=['Tipo']).reset_index(drop=True)
                else:
                    df_c_embarazo = df_c_embarazo[df_c_embarazo['Tipo'] == 'Nuevo > 5'].drop(columns=['Tipo']).reset_index(drop=True)
                    df_g_embarazo = df_g_embarazo[df_g_embarazo['Tipo'] == 'Nuevo > 5'].drop(columns=['Tipo']).reset_index(drop=True)
                    df_pc_embarazo = df_pc_embarazo[df_pc_embarazo['Tipo'] == 'Nuevo > 5'].drop(columns=['Tipo']).reset_index(drop=True)
                    df_sc_embarazo = df_sc_embarazo[df_sc_embarazo['Tipo'] == 'Nuevo > 5'].drop(columns=['Tipo']).reset_index(drop=True)
                
                if type_age == 'r1':
                    df_c = calculate_total_embarazadas(df_c_embarazo, factor, p, '< 15')
                    df_g = calculate_total_embarazadas(df_g_embarazo, factor, p_2, '< 15')
                    df_pc = calculate_total_embarazadas(df_pc_embarazo, factor, p_3, '< 15')
                    df_sc = calculate_total_embarazadas(df_sc_embarazo, factor, p_4, '< 15')
                elif type_age == 'r2':
                    df_c = calculate_total_embarazadas(df_c_embarazo, factor, p, '15-19')
                    df_g = calculate_total_embarazadas(df_g_embarazo, factor, p_2, '15-19')
                    df_pc = calculate_total_embarazadas(df_pc_embarazo, factor, p_3, '15-19')
                    df_sc = calculate_total_embarazadas(df_sc_embarazo, factor, p_4, '15-19')
                else:
                    df_c = calculate_total_embarazadas(df_c_embarazo, factor, p, '< 19')
                    df_g = calculate_total_embarazadas(df_g_embarazo, factor, p_2, '< 19')
                    df_pc = calculate_total_embarazadas(df_pc_embarazo, factor, p_3, '< 19')
                    df_sc = calculate_total_embarazadas(df_sc_embarazo, factor, p_4, '< 19')
                
                if n_clicks > 0:
                    # Seleccionar los dataframes según la selección del usuario
                    df_c.sort_values(by='Año', inplace=True)
                    df_g.sort_values(by='Año', inplace=True)
                    df_pc.sort_values(by='Año', inplace=True)
                    df_sc.sort_values(by='Año', inplace=True)
                    
                    dataframes = {
                        'Santa Cruz': df_sc,
                        'Cordillera': df_pc,
                        'Camiri': df_c,
                        'Gutierrez': df_g
                    }
                    
                    if (len(selected_dataframes) == 3):
                        return generate_lines_total(dataframes[selected_dataframes[0]], dataframes[selected_dataframes[1]], dataframes[selected_dataframes[2]], 'Año', type_percent, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, selected_dataframes, legend_loc)
                    else:
                        # Si falta algún dataframe seleccionado, retornar un mensaje de error o un div vacío
                        return html.Div("")

            
            return html.Div("")       
        except Exception as e:
            return html.Div(f'Error: {e}')

# Callback para realizar el cálculo de incidencias y porcentajes
@app.callback(
    Output('output-data-consultas', 'children'),
    [
        Input('btn-calcular-consultas', 'n_clicks'),
        Input('dropdown-graphic-type', 'value'),
        Input('dropdown-dataframes', 'value'),
        Input('input-titulo', 'value'),
        Input('input-tamaño-titulo', 'value'),
        Input('input-pie', 'value'),
        Input('input-tamaño-pie', 'value'),
        Input('input-tamaño-leyenda', 'value'),
        Input('input-tamaño-num-grafica', 'value'),
        Input('dropdown-legend-loc', 'value')
    ],
    [State('input-factor', 'value'),
     State('url', 'pathname')]  # Capturar el pathname actual
)
def update_output(n_clicks, graphic_type, selected_dataframes, titulo, tamanio_titulo, 
                  pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, legend_loc, 
                  factor, pathname):
    if n_clicks:
        try:
            if tamanio_titulo != None:
                tamanio_titulo = int(tamanio_titulo)
            else:
                tamanio_titulo = 16
            if tamanio_pie != None:
                tamanio_pie = int(tamanio_pie)
            else:
                tamanio_pie = 10
            if tamanio_leyenda != None:
                tamanio_leyenda = int(tamanio_leyenda)
            else:
                tamanio_leyenda = 8
            if tamanio_num_grafica != None:
                tamanio_num_grafica = int(tamanio_num_grafica)
            else:
                tamanio_num_grafica = 10
            
            if pathname == '/consultas':
             # Determinar qué conjuntos de datos utilizar según la ruta actual (pathname)
                df_c_consulta, df_g_consulta, df_pc_consulta, df_sc_consulta, df_c_consulta_2 = get_casos_consulta()

                df_c = calculate_total_consultas(df_c_consulta)
                df_g = calculate_total_consultas(df_g_consulta)
                df_pc = calculate_total_consultas(df_pc_consulta)
                df_sc = calculate_total_consultas(df_sc_consulta)

                df_c_2 = calculate_percent_second(df_c_consulta_2)
                df_c_2_t = calculate_percent_second_age(df_c_consulta_2)

                if n_clicks > 0:
                    # Seleccionar los dataframes según la selección del usuario
                    df_c.sort_values(by='Año', inplace=True)
                    df_g.sort_values(by='Año', inplace=True)
                    df_pc.sort_values(by='Año', inplace=True)
                    df_sc.sort_values(by='Año', inplace=True)
                    
                    dataframes = {
                        'Santa Cruz': df_sc,
                        'Cordillera': df_pc,
                        'Camiri': df_c,
                        'Gutierrez': df_g
                    }
                    
                    if (len(selected_dataframes) == 3):
                        if graphic_type == 'pn1_1':
                            # Generar y retornar el gráfico con los parámetros seleccionados
                            return generate_bars_gender(dataframes[selected_dataframes[0]], dataframes[selected_dataframes[1]], dataframes[selected_dataframes[2]], 'Año', 'Porcentaje', titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, selected_dataframes, legend_loc)
                        elif graphic_type == 'pn1_2':
                            return generate_bars_separate_gender(dataframes[selected_dataframes[0]], dataframes[selected_dataframes[1]], dataframes[selected_dataframes[2]], 'Año', 'Porcentaje', titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, selected_dataframes, legend_loc)
                        elif graphic_type == 'pn1_3':
                            return generate_bars_comparison_gender(dataframes[selected_dataframes[0]], dataframes[selected_dataframes[1]], dataframes[selected_dataframes[2]], 'Año', 'Porcentaje', titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, selected_dataframes, legend_loc)
                        else:
                            return html.Div("") 
                    elif (len(selected_dataframes) == 1):
                        if graphic_type == 'pn1_4':
                            return plot_age_percentages(dataframes[selected_dataframes[0]], 0, 0, 'Año', 'Porcentaje', titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, legend_loc, 0.2)
                        elif graphic_type == 'pn2_1':
                            return plot_top_services_by_year_and_gender(df_c_2, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_num_grafica)
                        elif graphic_type == 'pn2_2':
                            return plot_top5_especialidades(df_c_2_t, [2021, 2022, 2023], titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, (12,9))
                        else:
                            return html.Div("")
                    else:
                        # Si falta algún dataframe seleccionado, retornar un mensaje de error o un div vacío
                        return html.Div("")
            
            return html.Div("")
              
        except Exception as e:
            return html.Div(f'Error: {e}')

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)