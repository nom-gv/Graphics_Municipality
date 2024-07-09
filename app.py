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
    ('CasosCancer.xlsx', '1oRB3DMP1NtnnwfQcaYHo9a3bUcbQfB5U'),
    ('CasosDiabetes.xlsx', '1xHYonZp8RbPYCE9kihc3IthwOtgVNi1P'),
    ('CasosHipertensionArterial.xlsx', '1_jue36lk4iJim6btVh_tSUkR0i_QGeIk'),
    ('CasosObesidad.xlsx', '19aVPGne2nPm7_I0L9i_csyEBRw9geGea'),
    ('CasosNeumonia.xlsx', '1tK7dDEo1b7gWn-KHl1qE_WL62ztrygHw'),
    ('CasosChagas.xlsx', '1kAXyvg1cvLtl7w8a6D1AijMwFLJiialT'),
    ('CasosVIH.xlsx', '1xmnFEOBzaIZa3Ah4daAVEMo4HeLCVyZK'),
    ('CasosEstadoNutricional.xlsx', '1G8k9bqzJop0dSgFjigeVrzVQiuHuUFUp'),
    ('CasosEmbarazoAdolescente.xlsx', '1WGjRPOdiKjbblojvO96WpkfSITvbpvsH'),
    ('CasosConsultaExterna.xlsx', '1iA8HOY1nCGd62dqL1RU3MMgitXKT1a4q'),
    ('DatosPoblaciones.xlsx','11On6kmZq_frtfNx8Q-mc-Ei3-rVayleH'),
    ('DatosEspeciales.xlsx','1NoaMbxqsDrw3gtya91fnE2TPZo54Dxf6')
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

    df_c_consulta_2 = pd.read_excel('CasosConsultaExterna.xlsx', sheet_name="SEGUNDO-C")
    return df_c_consulta, df_g_consulta, df_pc_consulta, df_sc_consulta, df_c_consulta_2

def get_poblacion():
    df_c_poblacion = pd.read_excel('DatosPoblaciones.xlsx', sheet_name="POBLACION-C")
    df_g_poblacion = pd.read_excel('DatosPoblaciones.xlsx', sheet_name="POBLACION-G")
    df_pc_poblacion = pd.read_excel('DatosPoblaciones.xlsx', sheet_name="POBLACION-PC")
    df_sc_poblacion = pd.read_excel('DatosPoblaciones.xlsx', sheet_name="POBLACION-SC")
    return df_c_poblacion, df_g_poblacion, df_pc_poblacion, df_sc_poblacion

def get_poblacion_especiales():
    df_c_especiales = pd.read_excel('DatosEspeciales.xlsx', sheet_name="ESPECIALES-C")
    df_g_especiales = pd.read_excel('DatosEspeciales.xlsx', sheet_name="ESPECIALES-G")
    df_pc_especiales = pd.read_excel('DatosEspeciales.xlsx', sheet_name="ESPECIALES-PC")
    df_sc_especiales = pd.read_excel('DatosEspeciales.xlsx', sheet_name="ESPECIALES-SC")
    return df_c_especiales, df_g_especiales, df_pc_especiales, df_sc_especiales

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
                                ha='center', va='center',
                                xytext=(0, 10),
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
                                ha='center', va='center',
                                xytext=(0, 10),
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
            html.Li(dcc.Link('Cancer', href='/cancer')),
            html.Li(dcc.Link('Diabetes', href='/diabetes')),
            html.Li(dcc.Link('Hipertensión Arterial', href='/hipertension')),
            html.Li(dcc.Link('Obesidad', href='/obesidad')),
            html.Li(dcc.Link('Neumonia', href='/neumonia')),
            html.Li(dcc.Link('Chagas', href='/chagas')),
            html.Li(dcc.Link('VIH', href='/vih')),
            html.Li(dcc.Link('Nutrición Embarazo', href='/nutricion')),
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
calculo_layout = html.Div([ 
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
    
    html.Button('Generar Gráfico', id='btn-calcular'),
    html.Div(id='output-data')
])

# Define el layout de la página de cálculo
calculo_layout_nutricion = html.Div([    
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
    html.Label('Grafica a mostrar:'),
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
    if pathname == '/cancer':
        df_c_cancer, df_g_cancer, d1, d2 = get_casos_cancer()
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Cancer'),
            html.H2('Datos Camiri'),
            create_table(df_c_cancer),
            html.H2('Datos Gutierrez'),
            create_table(df_g_cancer),
            calculo_layout
        ])
    elif pathname == '/diabetes':
        df_c_diabetes, df_g_diabetes, d1, d2 = get_casos_diabetes()
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Diabetes'),
            html.H2('Datos Camiri'),
            create_table(df_c_diabetes),
            html.H2('Datos Gutierrez'),
            create_table(df_g_diabetes),
            calculo_layout
        ])
    elif pathname == '/hipertension':
        df_c_hipertension, df_g_hipertension, d1, d2 = get_casos_hipertension()
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Hipertensión Arterial'),
            html.H2('Datos Camiri'),
            create_table(df_c_hipertension),
            html.H2('Datos Gutierrez'),
            create_table(df_g_hipertension),
            calculo_layout
        ])
    elif pathname == '/obesidad':
        df_c_obesidad, df_g_obesidad, d1, d2 = get_casos_obesidad()
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Obesidad'),
            html.H2('Datos Camiri'),
            create_table(df_c_obesidad),
            html.H2('Datos Gutierrez'),
            create_table(df_g_obesidad),
            calculo_layout
        ])
    elif pathname == '/neumonia':
        df_c_neumonia, df_g_neumonia, d1, d2 = get_casos_neumonia()
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Neumonía'),
            html.H2('Datos Camiri'),
            create_table(df_c_neumonia),
            html.H2('Datos Gutierrez'),
            create_table(df_g_neumonia),
            calculo_layout
        ])
    elif pathname == '/chagas':
        df_c_chagas, df_g_chagas, d1, d2 = get_casos_chagas()
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Chagas'),
            html.H2('Datos Camiri'),
            create_table(df_c_chagas),
            html.H2('Datos Gutierrez'),
            create_table(df_g_chagas),
            calculo_layout
        ])
    elif pathname == '/vih':
        df_c_vih, df_g_vih, d1, d2 = get_casos_vih()
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos VIH'),
            html.H2('Datos Camiri'),
            create_table(df_c_vih),
            html.H2('Datos Gutierrez'),
            create_table(df_g_vih),
            calculo_layout
        ])
    elif pathname == '/nutricion':
        df_c_o, df_g_o, d1, d2, df_c_s, df_g_s, d3, d4, df_c_d, df_g_d, d5, d6 = get_casos_nutricion()
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Nutricion'),
            html.H2('Datos Camiri'),
            html.H3('Obesidad'),
            create_table(df_c_o),
            html.H3('Sobrepeso'),
            create_table(df_c_s),
            html.H3('Desnutricion'),
            create_table(df_c_d),
            html.H2('Datos Gutierrez'),
            html.H3('Obesidad'),
            create_table(df_g_o),
            html.H3('Sobrepeso'),
            create_table(df_g_s),
            html.H3('Desnutricion'),
            create_table(df_g_d),
            calculo_layout_nutricion
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
            # Convertir a listas de poblaciones
            p_c, p_g, p_pc, p_sc = get_poblacion()
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

            df_mujeres = p_c[p_c['Sexo'] == 'Mujer']
            columnas_edad = ['0-9', '10-19', '20-39', '40-49', '50-59', '60+']
            listas_edad = [df_mujeres[col].tolist() for col in columnas_edad]
            r_m = sum(listas_edad, [])
            df_hombres = p_c[p_c['Sexo'] == 'Hombre']
            listas_edad = [df_hombres[col].tolist() for col in columnas_edad]
            r_h = sum(listas_edad, [])
            
            
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
            
            if pathname != '/nutricion':
             # Determinar qué conjuntos de datos utilizar según la ruta actual (pathname)
                if pathname == '/cancer':
                    df_c_cancer, df_g_cancer, df_pc_cancer, df_sc_cancer = get_casos_cancer()
                    
                    df_c_t = generate_total(df_c_cancer)
                    df_g_t = generate_total(df_g_cancer)
                    df_pc_t = generate_total(df_pc_cancer)
                    df_sc_t = generate_total(df_sc_cancer)
                    
                    df_c_t = calculate_total(df_c_t, factor, p, 'Total')
                    df_g_t = calculate_total(df_g_t, factor, p_2, 'Total')
                    df_pc_t = calculate_total(df_pc_t, factor, p_3, 'Total')
                    df_sc_t = calculate_total(df_sc_t, factor, p_4, 'Total')
                    
                    df_c = calculate_gender(df_c_cancer, factor, m, h)
                    df_g = calculate_gender(df_g_cancer, factor, m_2, h_2)
                    df_pc = calculate_gender(df_pc_cancer, factor, m_3, h_3)
                    df_sc = calculate_gender(df_sc_cancer, factor, m_4, h_4)
                
                        
                elif pathname == '/diabetes':
                    df_c_diabetes, df_g_diabetes, df_pc_diabetes, df_sc_diabetes = get_casos_diabetes()
                    
                    df_c_t = generate_total(df_c_diabetes)
                    df_g_t = generate_total(df_g_diabetes)
                    df_pc_t = generate_total(df_pc_diabetes)
                    df_sc_t = generate_total(df_sc_diabetes)
                    
                    df_c_t = calculate_total(df_c_t, factor, p, 'Total')
                    df_g_t = calculate_total(df_g_t, factor, p_2, 'Total')
                    df_pc_t = calculate_total(df_pc_t, factor, p_3, 'Total')
                    df_sc_t = calculate_total(df_sc_t, factor, p_4, 'Total')
                    
                    df_c = calculate_gender(df_c_diabetes, factor, m, h)
                    df_g = calculate_gender(df_g_diabetes, factor, m_2, h_2)
                    df_pc = calculate_gender(df_pc_diabetes, factor, m_3, h_3)
                    df_sc = calculate_gender(df_sc_diabetes, factor, m_4, h_4)
                    
                    print(factor, p)
                elif pathname == '/hipertension':
                    df_c_hipertension, df_g_hipertension, df_pc_hipertension, df_sc_hipertension = get_casos_hipertension()
                    
                    df_c_t = generate_total(df_c_hipertension)
                    df_g_t = generate_total(df_g_hipertension)
                    df_pc_t = generate_total(df_pc_hipertension)
                    df_sc_t = generate_total(df_sc_hipertension)
                    
                    df_c_t = calculate_total(df_c_t, factor, p, 'Total')
                    df_g_t = calculate_total(df_g_t, factor, p_2, 'Total')
                    df_pc_t = calculate_total(df_pc_t, factor, p_3, 'Total')
                    df_sc_t = calculate_total(df_sc_t, factor, p_4, 'Total')
                    
                    df_c = calculate_gender(df_c_hipertension, factor, m, h)
                    df_g = calculate_gender(df_g_hipertension, factor, m_2, h_2)
                    df_pc = calculate_gender(df_pc_hipertension, factor, m_3, h_3)
                    df_sc = calculate_gender(df_sc_hipertension, factor, m_4, h_4)
                    
                elif pathname == '/obesidad':
                    df_c_obesidad, df_g_obesidad, df_pc_obesidad, df_sc_obesidad = get_casos_obesidad()
                    
                    df_c_t = generate_total(df_c_obesidad)
                    df_g_t = generate_total(df_g_obesidad)
                    df_pc_t = generate_total(df_pc_obesidad)
                    df_sc_t = generate_total(df_sc_obesidad)
                    
                    df_c_t = calculate_total(df_c_t, factor, p, 'Total')
                    df_g_t = calculate_total(df_g_t, factor, p_2, 'Total')
                    df_pc_t = calculate_total(df_pc_t, factor, p_3, 'Total')
                    df_sc_t = calculate_total(df_sc_t, factor, p_4, 'Total')
                    
                    df_c = calculate_gender(df_c_obesidad, factor, m, h)
                    df_g = calculate_gender(df_g_obesidad, factor, m_2, h_2)
                    df_pc = calculate_gender(df_pc_obesidad, factor, m_3, h_3)
                    df_sc = calculate_gender(df_sc_obesidad, factor, m_4, h_4)
                    
                elif pathname == '/neumonia':
                    df_c_neumonia, df_g_neumonia, df_pc_neumonia, df_sc_neumonia = get_casos_neumonia()
                    
                    df_c_t = generate_total(df_c_neumonia)
                    df_g_t = generate_total(df_g_neumonia)
                    df_pc_t = generate_total(df_pc_neumonia)
                    df_sc_t = generate_total(df_sc_neumonia)
                    
                    df_c_t = calculate_total(df_c_t, factor, p, 'Total')
                    df_g_t = calculate_total(df_g_t, factor, p_2, 'Total')
                    df_pc_t = calculate_total(df_pc_t, factor, p_3, 'Total')
                    df_sc_t = calculate_total(df_sc_t, factor, p_4, 'Total')
                    
                    df_c = calculate_gender(df_c_neumonia, factor, m, h)
                    df_g = calculate_gender(df_g_neumonia, factor, m_2, h_2)
                    df_pc = calculate_gender(df_pc_neumonia, factor, m_3, h_3)
                    df_sc = calculate_gender(df_sc_neumonia, factor, m_4, h_4)
                    
                elif pathname == '/chagas':
                    df_c_chagas, df_g_chagas, df_pc_chagas, df_sc_chagas = get_casos_neumonia()
                    
                    df_c_t = generate_total(df_c_chagas)
                    df_g_t = generate_total(df_g_chagas)
                    df_pc_t = generate_total(df_pc_chagas)
                    df_sc_t = generate_total(df_sc_chagas)
                    
                    df_c_t = calculate_total(df_c_t, factor, p, 'Total')
                    df_g_t = calculate_total(df_g_t, factor, p_2, 'Total')
                    df_pc_t = calculate_total(df_pc_t, factor, p_3, 'Total')
                    df_sc_t = calculate_total(df_sc_t, factor, p_4, 'Total')
                    
                    df_c = calculate_gender(df_c_chagas, factor, m, h)
                    df_g = calculate_gender(df_g_chagas, factor, m_2, h_2)
                    df_pc = calculate_gender(df_pc_chagas, factor, m_3, h_3)
                    df_sc = calculate_gender(df_sc_chagas, factor, m_4, h_4)
                    
                elif pathname == '/vih':
                    df_c_vih, df_g_vih, df_pc_vih, df_sc_vih = get_casos_vih()
                    
                    df_c_t = generate_total(df_c_vih)
                    df_g_t = generate_total(df_g_vih)
                    df_pc_t = generate_total(df_pc_vih)
                    df_sc_t = generate_total(df_sc_vih)
                    
                    df_c_t = calculate_total(df_c_t, factor, p, 'Total')
                    df_g_t = calculate_total(df_g_t, factor, p_2, 'Total')
                    df_pc_t = calculate_total(df_pc_t, factor, p_3, 'Total')
                    df_sc_t = calculate_total(df_sc_t, factor, p_4, 'Total')
                    
                    df_c = calculate_gender(df_c_vih, factor, m, h)
                    df_g = calculate_gender(df_g_vih, factor, m_2, h_2)
                    df_pc = calculate_gender(df_pc_vih, factor, m_3, h_3)
                    df_sc = calculate_gender(df_sc_vih, factor, m_4, h_4)
                    
                
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
                    
                    dataframes_total = {
                        'Santa Cruz': df_sc_t,
                        'Cordillera': df_pc_t,
                        'Camiri': df_c_t,
                        'Gutierrez': df_g_t
                    }
                    
                    if (len(selected_dataframes) == 3):
                        if graphic_type == 't':
                            # Generar y retornar el gráfico con los parámetros seleccionados
                            return generate_lines_total(dataframes_total[selected_dataframes[0]], dataframes_total[selected_dataframes[1]], dataframes_total[selected_dataframes[2]], 'Año', type_percent, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, selected_dataframes, legend_loc)
                        elif graphic_type == 's1':
                            # Generar y retornar el gráfico con los parámetros seleccionados
                            return generate_lines_gender(dataframes[selected_dataframes[0]], dataframes[selected_dataframes[1]], dataframes[selected_dataframes[2]], 'Año', type_percent, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, selected_dataframes, legend_loc)
                        elif graphic_type == 's2':
                            # Generar y retornar el gráfico con los parámetros seleccionados
                            return generate_lines_separate_gender(dataframes[selected_dataframes[0]], dataframes[selected_dataframes[1]], dataframes[selected_dataframes[2]], 'Año', type_percent, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, selected_dataframes, legend_loc)
                        elif graphic_type == 's3':
                            return generate_lines_comparison_gender(dataframes[selected_dataframes[0]], dataframes[selected_dataframes[1]], dataframes[selected_dataframes[2]], 'Año', type_percent, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, selected_dataframes, legend_loc)
                        else:
                            return html.Div("") 
                    elif (len(selected_dataframes) == 1):
                        if graphic_type == 'e':
                            if pathname == '/neumonia' or pathname == '/chagas':
                                return plot_age_percentages(dataframes[selected_dataframes[0]], r_m, r_h, 'Año', type_percent, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, legend_loc, 0.2)
                            else:
                                return plot_age_percentages(dataframes[selected_dataframes[0]], r_m, r_h, 'Año', type_percent, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, legend_loc, 0.25)
                        else:
                            return html.Div("")
                    else:
                        # Si falta algún dataframe seleccionado, retornar un mensaje de error o un div vacío
                        return html.Div("")
            
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
            # Convertir a listas de poblaciones
            p_c, p_g, p_pc, p_sc = get_poblacion_especiales()
            p = p_c.groupby('Año')['Embarazos'].sum().tolist()
            p_2 = p_g.groupby('Año')['Embarazos'].sum().tolist()
            p_3 = p_pc.groupby('Año')['Embarazos'].sum().tolist()
            p_4 = p_sc.groupby('Año')['Embarazos'].sum().tolist()
            
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
    
            
            if pathname == '/nutricion':
                df_c_obesidad, df_g_obesidad, df_pc_obesidad, df_sc_obesidad, df_c_sobrepeso, df_g_sobrepeso, df_pc_sobrepeso, df_sc_sobrepeso, df_c_desnutricion, df_g_desnutricion, df_pc_desnutricion, df_sc_desnutricion = get_casos_nutricion()

                if type_nutrition == 'o':
                    df_c = calculate_total(df_c_obesidad, factor, p, 'Embarazadas')
                    df_g = calculate_total(df_g_obesidad, factor, p_2, 'Embarazadas')
                    df_pc = calculate_total(df_pc_obesidad, factor, p_3, 'Embarazadas')
                    df_sc = calculate_total(df_sc_obesidad, factor, p_4, 'Embarazadas')
                elif type_nutrition == 's':
                    df_c = calculate_total(df_c_sobrepeso, factor, p, 'Embarazadas')
                    df_g = calculate_total(df_g_sobrepeso, factor, p_2, 'Embarazadas')
                    df_pc = calculate_total(df_pc_sobrepeso, factor, p_3, 'Embarazadas')
                    df_sc = calculate_total(df_sc_sobrepeso, factor, p_4, 'Embarazadas')
                elif type_nutrition == 'd':
                    df_c = calculate_total(df_c_desnutricion, factor, p, 'Embarazadas')
                    df_g = calculate_total(df_g_desnutricion, factor, p_2, 'Embarazadas')
                    df_pc = calculate_total(df_pc_desnutricion, factor, p_3, 'Embarazadas')
                    df_sc = calculate_total(df_sc_desnutricion, factor, p_4, 'Embarazadas')    
                
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
                        if graphic_type == 'tj':
                            # Generar y retornar el gráfico con los parámetros seleccionados
                            return generate_lines_total(dataframes[selected_dataframes[0]], dataframes[selected_dataframes[1]], dataframes[selected_dataframes[2]], 'Año', type_percent, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, selected_dataframes, legend_loc)
                        elif graphic_type == 'ts':
                            # Generar y retornar el gráfico con los parámetros seleccionados
                            return generate_lines_separate_nutricion(dataframes[selected_dataframes[0]], dataframes[selected_dataframes[1]], dataframes[selected_dataframes[2]], 'Año', type_percent, titulo, tamanio_titulo, pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, selected_dataframes, legend_loc)
                        else:
                            return html.Div("") 
                    else:
                        # Si falta algún dataframe seleccionado, retornar un mensaje de error o un div vacío
                        return html.Div("")
            
            return html.Div("")       
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
            p_c, p_g, p_pc, p_sc = get_poblacion_especiales()
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