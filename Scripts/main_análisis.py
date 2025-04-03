# # Notebook 1: Visualización de Datos
# Introducción
# Este notebook tiene como objetivo principal proporcionar una visión inicial de los datasets que se utilizarán en el proyecto. Aquí cargaremos los datos, exploraremos sus características principales y realizaremos algunas visualizaciones preliminares para familiarizarnos con su contenido. Esto permitirá identificar patrones iniciales y áreas de interés para los siguientes pasos del análisis.
# El proyecto está enfocado en estudiar el suicidio desde diferentes perspectivas: tendencias demográficas, análisis de textos relacionados y patrones observados en redes sociales. Estos datos serán cruciales para desarrollar un modelo predictivo y obtener insights que puedan contribuir a la prevención del suicidio.
# Descripción de los Datasets
# 1. Clasificación Textos Suicidio**
# Este dataset contiene textos clasificados en dos categorías: relacionados con el suicidio y no relacionados. Es ideal para entrenar un modelo de clasificación de textos y explorar patrones lingüísticos.
# 2. Clasificación Tweets Suicidio**
# Tweets clasificados en dos categorías: potencialmente relacionados con el suicidio y no relacionados. Este dataset permite estudiar el comportamiento en redes sociales y entrenar un algoritmo basado en mensajes cortos.
# 3. Demografía Residencia Suicidio**
# Datos demográficos sobre el suicidio, clasificados por lugar de residencia, sexo y grupo de edad. Es útil para análisis estadísticos y comprender tendencias poblacionales en Europa.
# 4. Intención Tweets Suicidio**
# Contiene tweets con intención explícita de suicidio. Este dataset es crucial para entrenar algoritmos que puedan detectar señales de alerta y análisis de contenido emocional.
# 5. Métodos Suicidio Demografía**
# Información sobre los métodos empleados en suicidios, clasificados por grupo demográfico. Útil para análisis detallados sobre los contextos y factores asociados.
# 6. Salud Mental Entorno Laboral**
# Datos sobre salud mental en el trabajo, antecedentes familiares y diagnósticos. Este dataset aporta perspectivas sobre cómo el entorno laboral afecta la salud mental, lo que puede vincularse al análisis del suicidio.
# 7. Suicidio Europa Datos ICD10**
# Estadísticas europeas sobre el suicidio, clasificados según el código ICD-10 (lesión autoinfligida). Este dataset es ideal para estudiar diferencias entre países y análisis más amplios.
# 8. Tasas Suicidio Europa Temporal ICD10**
# Datos temporales sobre las tasas de suicidio en Europa, desglosados por grupo de edad y género. Excelente para detectar tendencias temporales y correlaciones con otros factores.
# Conclusión
# En este notebook, hemos cargado los datasets y explorado sus características principales. Este punto de partida nos permitirá identificar qué aspectos analizar más a fondo en los siguientes pasos del proyecto. En el próximo notebook, nos enfocaremos en la limpieza y preparación de los datos, así como en el análisis exploratorio inicial.
#Importación de librerias recomendadas para el análisis de datos
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
# Carga de datos
data1 = pd.read_csv("data/Clasificación_Textos_Suicidio.csv")
data2 = pd.read_csv("data/Clasificación_Tweets_Suicidio.csv")
data3 = pd.read_csv("data/Demografía_Residencia_Suicidio.csv")
data4 = pd.read_csv("data/Intención_Tweets_Suicidio.csv")
data5 = pd.read_csv("data/Métodos_Suicidio_Demografía.csv")
data6 = pd.read_csv("data/Salud_Mental_Entorno_Laboral.csv")
data7 = pd.read_csv("data/Suicidio_Europa_Datos_ICD10.csv")
data8 = pd.read_csv("data/Tasas_Suicidio_Europa_Temporal_ICD10.csv")
# Función para visualizar las características principales de un dataset
def explorar_dataset(data, titulo):
    
    print(f"===== {titulo} =====")
    print("\nNúmero de Filas y Columnas:")
    print(data.shape)
    print("\nColumnas:")
    print(data.columns)
    print("\nPrimeras 5 Filas:")
    print(data.head())
    print("\nDescripción Estadística:")
    print(data.describe(include='all'))
    print("=" * 50)

# Exploración de los datasets
explorar_dataset(data1, "Clasificación Textos Suicidio")
explorar_dataset(data2, "Clasificación Tweets Suicidio")
explorar_dataset(data3, "Demografía Residencia Suicidio")
explorar_dataset(data4, "Intención Tweets Suicidio")
explorar_dataset(data5, "Métodos Suicidio Demografía")
explorar_dataset(data6, "Salud Mental Entorno Laboral")
explorar_dataset(data7, "Suicidio Europa Datos ICD10")
explorar_dataset(data8, "Tasas Suicidio Europa Temporal ICD10")
# Notebook 2: Limpieza 
#Introducción
#En este notebook, nos enfocaremos en preparar los datasets para el análisis y realizar un estudio exploratorio inicial. Este proceso es fundamental para garantizar que los datos estén limpios, organizados y listos para extraer insights significativos. Además, llevaremos a cabo un análisis exploratorio que nos permitirá identificar patrones, tendencias y posibles áreas de interés para el proyecto.
# Objetivos:
#1.Limpieza de Datos:**
#Tratar valores faltantes y duplicados.
#Estandarizar nombres de columnas y tipos de datos.
#Detectar y gestionar valores atípicos (outliers).
#2.Análisis Exploratorio de Datos (EDA):**
#Explorar distribuciones, correlaciones y patrones en los datos.
#Analizar la composición y relaciones de las variables clave.
#Identificar posibles problemas o sesgos en los datos.
#Este notebook representa un paso crítico hacia la comprensión de los datos y sentará las bases para las siguientes fases del proyecto.
#Estructura del Notebook
#1. Carga de Datos
# Volveremos a cargar los datasets seleccionados para asegurar un punto de partida limpio. Validaremos su contenido y estructura rápidamente para proceder a las siguientes etapas.
#2. Limpieza de Datos
# Aplicaremos un proceso sistemático para garantizar que los datos estén listos para el análisis:
# - Detección y manejo de valores nulos.
# - Normalización de nombres de columnas.
# - Eliminación de duplicados y registros irrelevantes.
# - Conversión de formatos y ajustes en los tipos de datos.
#Nota Final
#La limpieza de datos es fundamental para obtener resultados precisos y significativos. Es importante mantener un enfoque crítico y documentar cada decisión tomada durante este proceso. Este notebook nos permitirá tener una base sólida para el análisis en profundidad y la construcción del modelo.
# Cargar librerias

# Carga de datos
data1 = pd.read_csv("data/Clasificación_Textos_Suicidio.csv")
data2 = pd.read_csv("data/Clasificación_Tweets_Suicidio.csv")
data3 = pd.read_csv("data/Demografía_Residencia_Suicidio.csv")
data4 = pd.read_csv("data/Intención_Tweets_Suicidio.csv")
data5 = pd.read_csv("data/Métodos_Suicidio_Demografía.csv")
data6 = pd.read_csv("data/Salud_Mental_Entorno_Laboral.csv")
data7 = pd.read_csv("data/Suicidio_Europa_Datos_ICD10.csv")
data8 = pd.read_csv("data/Tasas_Suicidio_Europa_Temporal_ICD10.csv")

# Validación inicial
datasets = [data1, data2, data3, data4, data5, data6, data7, data8]
nombres = [
    "Clasificación Textos Suicidio",
    "Clasificación Tweets Suicidio",
    "Demografía Residencia Suicidio",
    "Intención Tweets Suicidio",
    "Métodos Suicidio Demografía",
    "Salud Mental Entorno Laboral",
    "Suicidio Europa Datos ICD10",
    "Tasas Suicidio Europa Temporal ICD10"
]

# Función para validar datasets
def validar_datos(data, nombre):
    print(f"===== Validando {nombre} =====")
    print(f"Dimensiones: {data.shape}")
    print(f"Columnas: {data.columns.tolist()}")
    print(f"Primeras filas:\n{data.head()}")
    print(f"Valores nulos:\n{data.isnull().sum()}")
    print("=" * 50)

# Validar cada dataset
for i in range(len(datasets)):
    validar_datos(datasets[i], nombres[i])
# Función para analizar valores faltantes y mostrar resultados en pantalla
def analizar_valores_faltantes(data, titulo):
 
    print(f"===== Análisis de Valores Faltantes: {titulo} =====")
    # Calcular el porcentaje de valores faltantes por columna
    nulos = data.isnull().mean() * 100
    columnas_con_nulos = nulos[nulos > 0].sort_values(ascending=False)  # Solo columnas con valores faltantes
    
    if len(columnas_con_nulos) > 0:
        print("Columnas con valores faltantes y su porcentaje:")
        print(columnas_con_nulos)
    else:
        print("No hay valores faltantes en este dataset.")
    
    print("=" * 50)

# Aplicamos la función a cada dataset
for i in range(len(datasets)):
    analizar_valores_faltantes(datasets[i], nombres[i])
# Renombrar columnas
data1.rename(columns={
    'Unnamed: 0': 'id',
    'text': 'contenido',
    'class': 'etiqueta'
}, inplace=True)

# Verificar los cambios
print("===== Nombres de Columnas Actualizados =====")
print(data1.columns)

# Mostrar las primeras filas después del cambio
print("\n===== Primeras Filas con Nuevos Nombres =====")
print(data1.head())


# Limpieza de la columna "contenido"
def limpiar_texto(texto):
    texto = texto.lower()  # Convertir a minúsculas
    texto = re.sub(r"[^a-zA-Z0-9\s]", "", texto)  # Eliminar caracteres especiales
    texto = texto.strip()  # Eliminar espacios extra
    return texto

# Aplicar la limpieza a la columna "contenido"
data1['contenido'] = data1['contenido'].apply(limpiar_texto)

# Verificar el resultado
print("\n===== Ejemplo de Textos Limpios =====")
print(data1['contenido'].head())

# Verificar los valores únicos en la columna "etiqueta"
print("\n===== Valores Únicos en Etiqueta =====")
print(data1['etiqueta'].unique())

# Si hay variaciones, las estandarizamos (ejemplo: espacios o inconsistencias)
data1['etiqueta'] = data1['etiqueta'].str.strip().str.lower()

# Verificar nuevamente los valores únicos
print("\n===== Valores Únicos Después de Estandarizar =====")
print(data1['etiqueta'].unique())
# Guardar el dataset limpio
data1.to_csv("Clasificación_Textos_Suicidio_Limpio.csv", index=False, encoding="utf-8")

print("\nEl dataset limpio se ha guardado como 'Clasificación_Textos_Suicidio_Limpio.csv'")

# Revisar la estructura inicial
print("===== Información General =====")
print(data2.info())  # Información del dataset
print("\n===== Primeras Filas =====")
print(data2.head())  # Primeras filas para revisar contenido
print("\n===== Valores Únicos en Cada Columna =====")
for col in data2.columns:
    print(f"{col}: {data2[col].unique()}")
# 1. Eliminar filas con valores faltantes en 'Tweet'
data2 = data2.dropna(subset=['Tweet'])
# 2. Limpiar textos en la columna 'Tweet'
def limpiar_texto(texto):
    texto = texto.lower()  # Convertir a minúsculas
    texto = re.sub(r"&quot;|[^a-zA-Z0-9\s@#]", "", texto)  # Eliminar caracteres especiales
    texto = texto.strip()  # Eliminar espacios al inicio y final
    return texto
data2['Tweet'] = data2['Tweet'].apply(limpiar_texto)
# 3. Estandarizar la columna 'Suicide'
data2['Suicide'] = data2['Suicide'].str.strip().str.lower()
# 4. Renombrar las columnas
data2.rename(columns={
    'Tweet': 'contenido',
    'Suicide': 'etiqueta'
}, inplace=True)
# 5. Validar los cambios
print("===== Primeras Filas Después de la Limpieza =====")
print(data2.head())
print("\n===== Valores Únicos en la Columna Etiqueta =====")
print(data2['etiqueta'].unique())

# Guardar el dataset limpio
data2.to_csv("Clasificación_Tweets_Suicidio_Limpio.csv", index=False, encoding="utf-8")

print("\nEl dataset limpio se ha guardado como 'Clasificación_Tweets_Suicidio_Limpio.csv'")

# Revisar la estructura inicial del dataset
print("===== Información General =====")
print(data3.info())  # Mostrar detalles sobre columnas y tipos de datos
print("\n===== Primeras Filas =====")
print(data3.head())  # Mostrar las primeras filas
print("\n===== Valores Únicos en Cada Columna =====")
for col in data3.columns:
    print(f"{col}: {data3[col].unique()}")

# Dividir la columna única en múltiples columnas
data3 = data3['Comunidades y ciudades autónomas de residencia\tSexo\tEdad\tTotal'].str.split('\t', expand=True)
data3.columns = ['comunidad', 'sexo', 'edad', 'total']  # Renombrar las nuevas columnas

# Convertir la columna 'total' a numérico (eliminando puntos si es necesario)
data3['total'] = data3['total'].str.replace('.', '').astype(float)

# Verificar los cambios
print("===== Primeras Filas Después de Dividir la Columna =====")
print(data3.head())

# Revisar tipos de datos
print("\n===== Tipos de Datos =====")
print(data3.dtypes)

# Mostrar valores únicos en cada columna para detectar inconsistencias
print("\n===== Valores Únicos por Columna =====")
for col in data3.columns:
    print(f"{col}: {data3[col].unique()}")
# Validar valores únicos por columna para detectar inconsistencias
print("\n===== Valores Únicos por Columna =====")
for col in ['comunidad', 'sexo', 'edad']:
    print(f"{col}: {data3[col].unique()}")

# Si identificamos inconsistencias, limpiemos los valores
data3['comunidad'] = data3['comunidad'].str.strip()
data3['sexo'] = data3['sexo'].str.strip().str.lower()  # Convertir a minúsculas
data3['edad'] = data3['edad'].str.strip()

# Volver a mostrar los valores únicos después de la limpieza
print("\n===== Valores Únicos Después de la Limpieza =====")
for col in ['comunidad', 'sexo', 'edad']:
    print(f"{col}: {data3[col].unique()}")

# Separar los rangos de edad en columnas mínimas y máximas 
def separar_edades(edad):
    try:
        if "a" in edad:  # Rango como 'De 15 a 29 años'
            partes = edad.replace("De ", "").replace(" años", "").split(" a ")
            return int(partes[0]), int(partes[1])
        elif "Menores" in edad:  # Categoría como 'Menores de 15 años'
            return 0, 15
        elif "Todas las edades" in edad:  # Caso especial
            return None, None
        else:
            return int(edad), int(edad)  # Caso de un único valor
    except ValueError:
        return None, None  # Manejar casos inesperados

data3[['edad_min', 'edad_max']] = data3['edad'].apply(separar_edades).apply(pd.Series)

# Verificar el resultado
print("\n===== Dataset con Columnas de Edad Mínima y Máxima =====")
print(data3.head())
# Guardar el dataset limpio
data3.to_csv("Demografía_Residencia_Suicidio_Limpio.csv", index=False, encoding="utf-8")

print("\nEl dataset limpio se ha guardado como 'Demografía_Residencia_Suicidio_Limpio.csv'")

# Revisar la estructura inicial del dataset
print("===== Información General =====")
print(data4.info())  
print("\n===== Primeras Filas =====")
print(data4.head())  
print("\n===== Valores Únicos en Cada Columna =====")
for col in data4.columns:
    print(f"{col}: {data4[col].unique()}")

# 1. Renombrar columnas
data4.rename(columns={
    'tweet': 'contenido',
    'intention': 'intención'
}, inplace=True)

# 2. Limpiar el texto en la columna 
def limpiar_texto(texto):
    texto = texto.lower()  
    texto = re.sub(r"[^a-zA-Z0-9\s]", "", texto) 
    texto = texto.strip()  
    return texto

data4['contenido'] = data4['contenido'].apply(limpiar_texto)

# 3. Validar los valores únicos
print("===== Valores Únicos en la Columna Intención =====")
print(data4['intención'].unique())

# 4. Mostrar las primeras filas para confirmar cambios
print("\n===== Primeras Filas Después de la Limpieza =====")
print(data4.head())

# 5. Guardar el dataset limpio
data4.to_csv("Intención_Tweets_Suicidio_Limpio.csv", index=False, encoding="utf-8")

print("\nEl dataset limpio se ha guardado como 'Intención_Tweets_Suicidio_Limpio.csv'")
# Revisar la estructura inicial del dataset
print("===== Información General =====")
print(data5.info()) 
print("\n===== Primeras Filas =====")
print(data5.head())  
print("\n===== Valores Únicos en Cada Columna =====")
for col in data5.columns:
    print(f"{col}: {data5[col].unique()}")

# 1. Renombrar columnas
data5.rename(columns={
    'Comunidades y ciudades autónomas de residencia': 'comunidad',
    'Sexo': 'sexo',
    'Medio empleado': 'método',
    'Total': 'total'
}, inplace=True)

# 2. Validar valores únicos en las columnas relevantes
print("\n===== Valores Únicos en la Columna Sexo =====")
print(data5['sexo'].unique())

print("\n===== Valores Únicos en la Columna Método =====")
print(data5['método'].unique())

# 3. Verificar la columna 'total' 
print("\n===== Tipos de Datos en la Columna Total =====")
print(data5['total'].dtype)

# 4. Mostrar las primeras filas 
print("\n===== Primeras Filas con Cambios Aplicados =====")
print(data5.head())

# 5. Guardar el dataset limpio
data5.to_csv("Métodos_Suicidio_Demografía_Limpio.csv", index=False, encoding="utf-8")

print("\nEl dataset limpio se ha guardado como 'Métodos_Suicidio_Demografía_Limpio.csv'")

# Identificar columnas con valores faltantes
print("===== Porcentaje de Valores Faltantes =====")
faltantes = data6.isnull().mean() * 100
print(faltantes.sort_values(ascending=False))

# 1. Eliminar columnas con >50% de valores faltantes
columnas_a_eliminar = faltantes[faltantes > 50].index.tolist()
data6.drop(columns=columnas_a_eliminar, inplace=True)

print("\n===== Columnas Eliminadas =====")
print(columnas_a_eliminar)

# 2. Imputación para columnas con <50% valores faltantes
for col in faltantes[faltantes <= 50].index:
    if data6[col].dtype == 'object':  # Imputación para columnas categóricas
        data6[col].fillna("Desconocido", inplace=True)
    else:  # Imputación para columnas numéricas
        data6[col].fillna(data6[col].median(), inplace=True)

# 3. Renombrar columnas para mayor claridad
data6.rename(columns={
    'Insurance': 'seguro',
    'Describe Past Experience': 'experiencia_previa',
    'Rate Reaction to Problems': 'reaccion_problemas',
    'Prefer Anonymity': 'preferencia_anonimato',
    'Diagnosis': 'diagnostico',
    'Disorder Notes': 'notas_desorden',
    'Company Size': 'tamano_empresa',
    'Primarily a Tech Employer': 'empleador_tecnologico',
    'Responsible Employer': 'empleador_responsable',
    'Discuss Mental Health Problems': 'discutir_salud_mental',
    'Negative Consequences': 'consecuencias_negativas',
    'Location': 'ubicacion'
}, inplace=True)

print("\n===== Nombres de Columnas Actualizados =====")
print(data6.columns)

# 4. Mostrar las primeras filas después de la limpieza
print("\n===== Primeras Filas del Dataset Limpio =====")
print(data6.head())
# Guardar el dataset limpio
data6.to_csv("Salud_Mental_Entorno_Laboral_Limpio.csv", index=False, encoding="utf-8")

print("\nEl dataset limpio se ha guardado como 'Salud_Mental_Entorno_Laboral_Limpio.csv'")

# 1. Eliminar las columnas completamente vacías
data7.drop(columns=['OBS_FLAG', 'CONF_STATUS'], inplace=True)

# 2. Revisar las columnas restantes
print("\n===== Información General del Dataset =====")
print(data7.info())
print("\n===== Primeras Filas =====")
print(data7.head())
print("\n===== Valores Únicos en Cada Columna =====")
for col in data7.columns:
    print(f"{col}: {data7[col].unique()}")

# 3. Renombrar columnas para mayor claridad
data7.rename(columns={
    'Country': 'pais',
    'Year': 'anio',
    'Suicides': 'suicidios',
    'Population': 'poblacion',
    'Rate': 'tasa'
}, inplace=True)

print("\n===== Nombres de Columnas Actualizados =====")
print(data7.columns)

# 4. Guardar el dataset limpio
data7.to_csv("Suicidio_Europa_Datos_ICD10_Limpio.csv", index=False, encoding="utf-8")

print("\nEl dataset limpio se ha guardado como 'Suicidio_Europa_Datos_ICD10_Limpio.csv'")

# 1. Eliminar columnas con valores faltantes altos
data8.drop(columns=['OBS_FLAG', 'CONF_STATUS'], inplace=True)

# 2. Revisar columnas restantes
print("\n===== Información General =====")
print(data8.info())
print("\n===== Primeras Filas =====")
print(data8.head())
print("\n===== Valores Únicos en Cada Columna =====")
for col in data8.columns:
    print(f"{col}: {data8[col].unique()}")

# 3. Renombrar columnas (opcional, dependiendo de su contenido)
data8.rename(columns={
    'Country': 'pais',
    'Year': 'anio',
    'Age': 'edad',
    'Gender': 'genero',
    'Suicides': 'suicidios',
    'Rate': 'tasa'
}, inplace=True)

print("\n===== Nombres de Columnas Actualizados =====")
print(data8.columns)

# 4. Guardar el dataset limpio
data8.to_csv("Tasas_Suicidio_Europa_Temporal_ICD10_Limpio.csv", index=False, encoding="utf-8")

print("\nEl dataset limpio se ha guardado como 'Tasas_Suicidio_Europa_Temporal_ICD10_Limpio.csv'")
#**Notebook 3: Análisis Exploratorio de Datos (EDA)**
# **Objetivo**
# El objetivo de este notebook es realizar un **Análisis Exploratorio de Datos (EDA)** sobre los datasets que no serán utilizados directamente en nuestro algoritmo predictor de suicidio. Esto nos permitirá:
# - Explorar las características y la calidad de los datos.
# - Extraer patrones, tendencias y distribuciones.
# - Generar métricas, gráficos y KPIs que aporten insights relevantes para la interpretación.
#  **Plan de Trabajo**
# Este notebook estará estructurado en los siguientes pasos:
# 1. **Apertura de Datasets**
#    - Cargar los datasets seleccionados.
#    - Realizar una revisión inicial (estructura, primeras filas).
# 2. **Revisión General**
#    - Verificar información básica como tipos de datos, valores únicos y resúmenes estadísticos.
#    - Identificar posibles relaciones y tendencias generales.
# 3. **Análisis Exploratorio Detallado**
#    - Para cada dataset:
#      1. Identificar variables clave.
#      2. Generar gráficos y métricas específicas.
#      3. Proponer insights basados en los resultados obtenidos.
# 4. **Resultados e Interpretaciones**
#    - Resumir los hallazgos obtenidos durante el análisis.
#    - Proponer hipótesis o preguntas adicionales basadas en los datos.
#  **Datasets Incluidos**
# Los datasets que analizaremos en este notebook son:
# 1. **Demografía Residencia Suicidio**
#    - Incluye información demográfica y de residencia relacionada con casos de suicidio.
# 2. **Métodos Suicidio Demografía**
#    - Proporciona detalles sobre los métodos empleados y su relación con características demográficas.
# 3. **Salud Mental Entorno Laboral**
#    - Explora factores relacionados con la salud mental en contextos laborales.
# 4. **Suicidio Europa Datos ICD10**
#    - Contiene información detallada sobre tasas de suicidio en Europa clasificadas según ICD10.
# 5. **Tasas Suicidio Europa Temporal ICD10**
#    - Presenta datos temporales y geográficos relacionados con tasas de suicidio en Europa.
#  **Enfoque Metodológico**
# Para cada dataset:
# 1. Abrir y revisar las primeras filas usando `head()`.
# 2. Analizar distribuciones y calcular métricas relevantes.
# 3. Generar gráficos (histogramas, boxplots, barplots, etc.).
# 4. Interpretar las visualizaciones y generar conclusiones.

# Lista de datasets renombrados y su ruta
datasets = {
    "Demografía Residencia Suicidio": "data_limpio/Demografía_Residencia_Suicidio_Limpio.csv",
    "Métodos Suicidio Demografía": "data_limpio/Métodos_Suicidio_Demografía_Limpio.csv",
    "Salud Mental Entorno Laboral": "data_limpio/Salud_Mental_Entorno_Laboral_Limpio.csv",
    "Suicidio Europa Datos ICD10": "data_limpio/Suicidio_Europa_Datos_ICD10_Limpio.csv",
    "Tasas Suicidio Europa Temporal ICD10": "data_limpio/Tasas_Suicidio_Europa_Temporal_ICD10_Limpio.csv"
}

# Cargar los datasets en un diccionario de DataFrames
dataframes = {}
for nombre, ruta in datasets.items():
    dataframes[nombre] = pd.read_csv(ruta)
    print(f"===== Dataset: {nombre} =====")
    print(dataframes[nombre].head())  # Mostrar las primeras filas
    print("\n===== Información General =====")
    print(dataframes[nombre].info())  # Mostrar información básica del dataset
    print("\n")


# ### Dataset 1: Demografía Residencia Suicidio.
# EDA del Dataset 1
# Distribución por Rango de Edad:
# Identificaremos cuáles rangos de edad reportan más suicidios.
# Visualizaremos la distribución total y diferenciada por género.
# Distribución por Comunidad:
# Observaremos el número de suicidios registrados en cada comunidad.
# Exploraremos patrones demográficos asociados.
# Análisis Combinado Edad y Género:
# Profundizaremos en cómo se distribuyen los suicidios dentro de los rangos de edad diferenciados por género.

# Seleccionar el dataset 'Demografía Residencia Suicidio'
dataset1 = dataframes['Demografía Residencia Suicidio']

# Calcular límites del rango intercuartil (IQR) para columnas numéricas
print("===== Límites del Rango Intercuartil (IQR) =====")
numeric_columns = dataset1.select_dtypes(include=['number'])  # Seleccionar solo columnas numéricas
Q1 = numeric_columns.quantile(0.25)
Q3 = numeric_columns.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

# Filtrar el dataset para eliminar outliers y la categoría 'Todas las edades'
dataset1_filtrado = dataset1[
    (dataset1['total'] >= limite_inferior) &
    (dataset1['total'] <= limite_superior) &
    (dataset1['edad'] != 'Todas las edades')
]

# Verificar el dataset después de aplicar los filtros
print("===== Información del Dataset Filtrado =====")
print(dataset1_filtrado.info())
print("\n===== Categorías de Edad Restantes =====")
print(dataset1_filtrado['edad'].unique())

# Distribución de Suicidios por Edad y Género

# Gráfico de barras por edad y género
plt.figure(figsize=(12, 6))
orden_edad = sorted(dataset1_filtrado['edad'].unique(), key=lambda x: int(x.split()[1]) if "De" in x else 0)
sns.barplot(
    x='edad', 
    y='total', 
    hue='sexo', 
    data=dataset1_filtrado, 
    ci=None, 
    order=orden_edad,
    palette="Set2"
)
plt.xticks(rotation=45)
plt.title('Distribución de Suicidios por Edad y Género (Sin Outliers)')
plt.xlabel('Rango de Edad')
plt.ylabel('Total de Suicidios')
plt.legend(title='Género')
plt.show()


# Distribución de Suicidios por Comunidad
# Total de suicidios por comunidad
comunidades = dataset1_filtrado.groupby('comunidad')['total'].sum().sort_values(ascending=False)

# Gráfico de barras para comunidades
plt.figure(figsize=(12, 8))
comunidades.plot(kind='bar', color='teal')
plt.title('Distribución de Suicidios por Comunidad (Sin Outliers)')
plt.xlabel('Comunidad')
plt.ylabel('Total de Suicidios')
plt.xticks(rotation=90)
plt.show()

#**Conclusiones: Análisis de Suicidios en España**
#  **Puntos Clave**
# - **Edad:** Los grupos de **50-54 años** y **75-79 años** son los más afectados, posiblemente debido a crisis vitales, problemas de salud o aislamiento social.
# - **Género:** Los hombres presentan una tasa significativamente más alta que las mujeres, lo que podría relacionarse con métodos más letales y menor búsqueda de ayuda psicológica.
# - **Comunidades:** 
#   - **Andalucía**, **Cataluña** y **Comunitat Valenciana** registran el mayor número de suicidios, probablemente debido a su alta población.
#   - **Madrid** tiene cifras absolutas moderadas, pero su tasa ajustada es inferior a la media nacional.
#  **Observaciones**
# - Las tasas de suicidio tienden a ser más altas en hombres y en personas mayores.
# - Es importante priorizar programas de prevención focalizados en los grupos más vulnerables por edad, género y ubicación geográfica.
# ### Métodos Suicidio Demografía
# Dataset 2: Métodos Suicidio Demografía
# Analizar los métodos de suicidio más frecuentes.
# Observar la distribución por género.
# Identificar patrones relevantes por comunidad.
# 1: Métodos de Suicidio Más Frecuentes
# Eliminar los outliers del dataset 'Métodos Suicidio Demografía'
dataset2 = dataframes['Métodos Suicidio Demografía']
# Calcular los límites del rango intercuartil (IQR) para todas las columnas numéricas
for column in dataset2.select_dtypes(include=['number']).columns:
    Q1 = dataset2[column].quantile(0.25)
    Q3 = dataset2[column].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    # Filtrar los valores dentro de los límites
    dataset2 = dataset2[(dataset2[column] >= limite_inferior) & (dataset2[column] <= limite_superior)]
Q3 = dataset2['total'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
# y guardar el dataset filtrado
dataset2 = dataset2[
    (dataset2['total'] >= limite_inferior) &
    (dataset2['total'] <= limite_superior)
]
# Agrupar por método y calcular el total
metodos_frecuentes = dataframes["Métodos Suicidio Demografía"].groupby('método')['total'].sum().sort_values(ascending=False)

# Gráfico de barras para los métodos
plt.figure(figsize=(12, 8))
metodos_frecuentes.plot(kind='bar', color='skyblue')
plt.title('Métodos de Suicidio Más Comunes')
plt.xlabel('Método')
plt.ylabel('Total de Suicidios')
plt.xticks(rotation=90)
plt.show()

# Mostrar los 5 métodos más frecuentes
print("===== Top 5 Métodos de Suicidio =====")
print(metodos_frecuentes.head())

# 2: Métodos por Género
# Filtrar y agrupar por método y género, excluyendo la categoría 'Total'
metodos_por_genero = dataframes["Métodos Suicidio Demografía"][
    dataframes["Métodos Suicidio Demografía"]['método'] != 'Total'
].groupby(['método', 'sexo'])['total'].sum().unstack()

# Crear el gráfico de barras no apilado por género
fig, ax = plt.subplots(figsize=(14, 8))
metodos_por_genero.plot(
    kind='bar', 
    ax=ax, 
    stacked=False,  
    color=sns.color_palette("Set2"),  
    width=0.75
)

# Personalizar el gráfico
ax.set_title('Métodos de Suicidio por Género', fontsize=18, weight='bold')
ax.set_xlabel('Método', fontsize=14, labelpad=10)
ax.set_ylabel('Total de Suicidios', fontsize=14, labelpad=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)  # Rotar etiquetas para mejor legibilidad
ax.legend(title='Género', labels=['Mujeres', 'Hombres'], fontsize=12, title_fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.6)  # Líneas guía horizontales

# Ajustar diseño para evitar solapamientos
plt.tight_layout()

# Mostrar el gráfico
plt.show()
# 3: Métodos por Comunidad
# # **Conclusiones: Métodos Suicidio Demografía**
#  **Puntos Clave**
# - **Método predominante:** El **ahorcamiento, estrangulación y sofocación (X70)** es el método más utilizado en todas las comunidades y géneros. Esto refleja su accesibilidad y prevalencia global.
# - **Diferencias por género:**
#   - Los hombres tienden a utilizar métodos más letales como el ahorcamiento y armas de fuego (X72).
#   - Las mujeres recurren en mayor proporción al envenenamiento con sustancias como psicotrópicos (X61).
# - **Distribución por comunidad:** Las comunidades con mayor número de suicidios (Andalucía, Cataluña y Comunitat Valenciana) también lideran en el uso del método X70, mientras que métodos menos frecuentes tienen una distribución más dispersa.
#  **Observaciones**
# - Las diferencias de género en la elección del método subrayan la necesidad de estrategias preventivas diferenciadas.
# - El ahorcamiento, al ser el método más accesible, destaca como un foco prioritario en las políticas de prevención.
# - Algunas comunidades muestran patrones específicos que podrían correlacionarse con factores sociales, económicos o culturales.
# 3: Salud Mental Entorno Laboral
# Impacto del historial familiar en la búsqueda de tratamiento:
# Analizar cómo la presencia de un historial familiar de enfermedad mental afecta la decisión de buscar tratamiento.
# Relación entre género y tamaño de empresa:
# Explorar si el tamaño de la empresa influye en las percepciones o decisiones respecto a la salud mental, y si hay diferencias por género.
# Impacto del Historial Familiar en la Búsqueda de Tratamiento
# Gráfico de barras para mostrar la relación
plt.figure(figsize=(10, 6))
sns.countplot(
    x="Family History of Mental Illness", 
    hue="Sought Treatment", 
    data=dataframes["Salud Mental Entorno Laboral"],
    palette="Set2"
)
plt.title("Impacto del Historial Familiar en la Búsqueda de Tratamiento")
plt.xlabel("Historial Familiar de Enfermedad Mental")
plt.ylabel("Frecuencia")
plt.legend(title="Buscó Tratamiento", labels=["No", "Sí"])
plt.show()


# Relación entre Género y Tamaño de Empresa

# Gráfico de barras apilado
plt.figure(figsize=(12, 6))
sns.countplot(
    x="tamano_empresa", 
    hue="Gender", 
    data=dataframes["Salud Mental Entorno Laboral"], 
    palette="Set2"
)
plt.title("Distribución por Tamaño de Empresa y Género")
plt.xlabel("Tamaño de Empresa")
plt.ylabel("Frecuencia")
plt.legend(title="Género")
plt.xticks(rotation=45)
plt.show()


#**Conclusiones: Salud Mental Entorno Laboral**
# **Puntos Clave**
# - **Género y tamaño de empresa:**
#   - Los hombres tienen una representación significativamente mayor en empresas grandes (más de 1000 empleados) y medianas (100-500 empleados).
#   - Las mujeres están presentes en todas las categorías, pero su frecuencia es notablemente menor en las empresas más grandes, lo que podría reflejar desigualdades de género en algunos sectores.
# **Empresas pequeñas y medianas:**
#   - En empresas de menor tamaño (1-25 empleados), la participación está más equilibrada entre hombres y mujeres, aunque los hombres siguen siendo mayoría.
#  **Observaciones**
# - Este análisis podría indicar posibles barreras culturales, estructurales o sociales que influyen en la representación de género en empresas grandes, especialmente en lo relacionado con la salud mental.
# - Sería útil explorar programas de sensibilización sobre salud mental que incluyan enfoques específicos para cada género y se adapten al tamaño de la empresa.

# ### 4: Suicidio Europa Datos ICD10
# Comparación por género:
# Observar las diferencias entre hombres y mujeres en las cifras reportadas.
# Distribución geográfica:
# Identificar países con tasas más altas y más bajas.
# Comparación por género
# Gráfico de barras por género y año
plt.figure(figsize=(12, 6))

sns.barplot(
    x='TIME_PERIOD', 
    y='OBS_VALUE', 
    hue='sex', 
    data=dataframes["Suicidio Europa Datos ICD10"], 
    palette="Set2",  # Corrected palette
    ci=None
)
plt.title('Comparación de Suicidios por Género en Europa (Por Año)')
plt.xlabel('Año')
plt.ylabel('Número de Suicidios')
plt.legend(title='Género', labels=['Mujeres', 'Hombres'])
plt.xticks(rotation=45)
plt.show()

# Identificar países líderes en tasas de suicidio

# Filtrar el dataset para excluir "European Union - 27 countries (from 2020)"
dataset_filtrado = dataframes["Suicidio Europa Datos ICD10"][
    dataframes["Suicidio Europa Datos ICD10"]["geo"] != "European Union - 27 countries (from 2020)"
]

# Calcular el promedio de suicidios por país
promedio_pais_filtrado = dataset_filtrado.groupby('geo')['OBS_VALUE'].mean().sort_values(ascending=False)

# Graficar el promedio filtrado
plt.figure(figsize=(12, 6))
promedio_pais_filtrado.plot(kind='bar', color='coral')
plt.title('Promedio de Suicidios por País en Europa (Filtrado)')
plt.xlabel('País')
plt.ylabel('Promedio de Suicidios')
plt.xticks(rotation=90)
plt.show()


#Conclusiones: Suicidio Europa Datos ICD10**
#  **Puntos Clave**
# - **Diferencias por género:**
#   - Los hombres presentan tasas de suicidio consistentemente más altas que las mujeres a lo largo de los años y en todos los países.
#  **Promedio por país:**
#   - Países como **Alemania** y **Francia** tienen tasas promedio significativamente más altas, destacando como áreas prioritarias para la intervención.
#   - Países con tasas bajas incluyen **Malta** y **Chipre**, lo que podría vincularse a factores culturales o sociales específicos.
# **Observaciones**
# - Las tendencias evidencian la necesidad de enfoques regionales y de género en las estrategias de prevención del suicidio.
# Tasas de Suicidio en Europa Temporal
# Tendencias Temporales por País:
# Analizar cómo han evolucionado las tasas de suicidio en los países europeos a lo largo de los años.
# Comparación entre Países:
# Identificar los países con las tasas más altas y más bajas de suicidio durante un periodo específico.
# Distribución por Género:
# Observar cómo se distribuyen las tasas de suicidio entre hombres y mujeres, y si han cambiado a lo largo del tiempo.
# Tendencias Temporales por País

# Gráfico de líneas: Tendencias temporales por país
plt.figure(figsize=(12, 6))
sns.lineplot(
    x="TIME_PERIOD", 
    y="OBS_VALUE", 
    hue="geo", 
    data=dataframes["Tasas Suicidio Europa Temporal ICD10"], 
    palette="tab20"
)
plt.title('Evolución de las Tasas de Suicidio en Europa por País')
plt.xlabel('Año')
plt.ylabel('Tasa de Suicidio')
plt.legend(title='País', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Obtener tasas promedio por país
tasas_promedio_pais = dataframes["Tasas Suicidio Europa Temporal ICD10"].groupby("geo")["OBS_VALUE"].mean().sort_values(ascending=False)

# Gráfico de barras
plt.figure(figsize=(12, 6))
tasas_promedio_pais.plot(kind='bar', color='lightblue')
plt.title('Promedio de Tasas de Suicidio por País en Europa')
plt.xlabel('País')
plt.ylabel('Tasa Promedio')
plt.xticks(rotation=90)
plt.show()

#**Conclusiones: Tasas de Suicidio en Europa Temporal**
# **Puntos Clave**
# 1. **Tendencias Temporales por País:**
#    - Países como **Lituania** y **Hungría** destacan como líderes en las tasas de suicidio a lo largo de los años, reflejando un desafío continuo en estas regiones.
#    - Otros países, como **España** y **Malta**, presentan tasas consistentemente bajas, lo cual podría relacionarse con factores culturales, sociales o la efectividad de políticas preventivas.
#    - En general, algunos países muestran una tendencia descendente en las tasas, posiblemente debido a campañas de concienciación y mejor acceso a recursos de salud mental.
# 2. **Promedio de Tasas por País:**
#    - Los países del este de Europa, como **Lituania** y **Letonia**, tienen promedios significativamente más altos, lo que marca áreas clave para intervenciones dirigidas.
#    - Por otro lado, **Chipre** y **Malta** tienen las tasas más bajas, lo que podría servir de modelo para estrategias exitosas de prevención.
# 3. **Distribución por Género:**
#    - Las tasas de suicidio en **hombres** son mucho más altas que en **mujeres**, una constante que resalta en todos los países analizados.
#    - Estas diferencias subrayan la necesidad de estrategias específicas por género, como abordar el estigma asociado con buscar ayuda en los hombres.
#  **Observaciones**
# - Los datos indican que es crítico implementar políticas focalizadas para los países y grupos demográficos más vulnerables.
# - Las estrategias deben incluir campañas públicas, mejora de los recursos de salud mental y enfoques dirigidos por género.
# - Una revisión a fondo de las tendencias a lo largo del tiempo puede ayudar a identificar periodos críticos de intervención.
# **Notebook 4: KPI (Key Performance Indicators)**
# **Objetivo del Notebook**
# El objetivo principal de este notebook es definir, calcular y analizar los KPI clave basados en el análisis exploratorio de datos (EDA) previo. Estos indicadores servirán como métricas esenciales para medir el estado actual y el progreso en las áreas analizadas.
#  **Estructura del Notebook**
# 1. **Selección de KPIs Clave:**
#    - Identificar los indicadores relevantes para cada dataset.
#    - Justificar la selección de cada KPI en base a su utilidad.
# 2. **Cálculo de KPIs:**
#    - Definir las fórmulas necesarias para calcular cada indicador.
#    - Implementar cálculos basados en los datos procesados.
# 3. **Visualización de KPIs:**
#    - Representar gráficamente los indicadores para facilitar la interpretación.
# 4. **Análisis e Interpretación:**
#    - Extraer conclusiones y recomendaciones basadas en los valores calculados.
#  **KPIs Iniciales Propuestos**
# - **Tasa de suicidios por género:** Diferencia porcentual entre hombres y mujeres.
# - **Promedio anual de suicidios por país:** Identificar tendencias de cada región.
# - **Variación interanual en tasas de suicidio:** Para evaluar cambios significativos.
# - **Tasa de tratamiento por historial familiar:** Relacionar salud mental y antecedentes familiares.
# - **Distribución de suicidios según tamaño de empresa:** Identificar puntos críticos en entornos laborales
# Cargar los datasets
dataframes = {
    "Tasas Suicidio Europa Temporal ICD10": pd.read_csv("data_limpio/Tasas_Suicidio_Europa_Temporal_ICD10_Limpio.csv"),
    "Salud Mental Entorno Laboral": pd.read_csv("data_limpio/Salud_Mental_Entorno_Laboral_Limpio.csv"),
    "Suicidio Europa Datos ICD10": pd.read_csv("data_limpio/Suicidio_Europa_Datos_ICD10_Limpio.csv"),
}
# 1. Tasa de Suicidios por Género
# Este KPI nos permitirá observar la diferencia porcentual entre hombres y mujeres en las tasas de suicidio.
# Cálculo de tasas de suicidio por género usando el dataset Suicidio Europa Datos ICD10
suicidios_genero = dataframes["Suicidio Europa Datos ICD10"].groupby("sex")["OBS_VALUE"].sum()
# Verificar los valores únicos en la columna `sex`
print(suicidios_genero)
# Cálculo de la tasa entre hombres y mujeres
tasa_genero = (suicidios_genero["Males"] - suicidios_genero["Females"]) / suicidios_genero.sum() * 100
print(f"Tasa de suicidios por género (hombres vs mujeres): {tasa_genero:.2f}%")
# 2. Promedio Anual de Suicidios por País
# Con este KPI identificaremos las tendencias promedio de suicidios en cada país a lo largo del tiempo.
# Cálculo del promedio anual por país
promedio_anual_pais = dataframes["Tasas Suicidio Europa Temporal ICD10"].groupby("geo")["OBS_VALUE"].mean()
print(promedio_anual_pais.sort_values(ascending=False))
#3. Variación Interanual en Tasas de Suicidio
# Este indicador nos muestra cómo han cambiado las tasas año tras año.
# Cálculo de la variación interanual sin rellenar valores faltantes
data["variacion"] = data_sorted.groupby("geo")["OBS_VALUE"].pct_change(fill_method=None) * 100

# Análisis descriptivo de las variaciones
descriptive_stats = data.groupby("geo")["variacion"].describe()
print(descriptive_stats)

# **Notebook 4: KPI (Key Performance Indicators)**
#  **Objetivo del Notebook**
# Este notebook tiene como objetivo definir, calcular y analizar KPIs clave basados en el análisis exploratorio de datos (EDA) previo. Estos indicadores son esenciales para medir tendencias, identificar diferencias significativas y establecer prioridades para futuras estrategias de prevención y acción.
# ## **1. Tasa de Suicidios por Género**
# ### **Resultados**
# - **Suicidios en hombres:** 73,865
# - **Suicidios en mujeres:** 22,512
# - **Tasa de suicidios (hombres vs mujeres):** 26.64%
#  **Conclusiones**
# 1. Los hombres representan una proporción significativamente mayor de los suicidios registrados.
# 2. Es crucial priorizar políticas y estrategias dirigidas a abordar factores específicos de género, como el estigma en la búsqueda de ayuda para la salud mental.
#  **2. Promedio Anual de Suicidios por País**
# ### **Resultados**
# - Países con tasas más altas:
#   - **Lituania (LT):** 31.84
#   - **Hungría (HU):** 27.49
#   - **Eslovenia (SI):** 27.47
# - Países con tasas más bajas:
#   - **Chipre (CY):** 2.95
#   - **Turquía (TR):** 4.71
#   - **Grecia (EL):** 5.64
#**Conclusiones**
# 1. Los países del este de Europa presentan tasas significativamente más altas, sugiriendo la necesidad de enfoques preventivos específicos para estas regiones.
# 2. Los países con tasas más bajas pueden proporcionar insights valiosos sobre estrategias efectivas, pero también es importante explorar si existen problemas de subregistro.
# **3. Variación Interanual de Tasas de Suicidio**
# ### **Resultados**
# - Países con mayor variación:
#   - **Serbia (RS):** Máxima variación de 1479.46%.
#   - **Eslovenia (SI):** Máxima variación de 3082.18%.
#   - **Portugal (PT):** Máxima variación de 805.63%.
# - Países más estables:
#   - **Finlandia (FI):** Variación media de 21.20% con baja dispersión.
#   - **Noruega (NO):** Variación media de 30.37% con bajo rango de fluctuación.
# **Conclusiones**
# 1. Serbia y Eslovenia muestran fluctuaciones extremas en las tasas de suicidio, posiblemente relacionadas con eventos significativos o cambios en políticas.
# 2. Países como Finlandia y Noruega destacan por su estabilidad, lo que podría reflejar sistemas de apoyo social más robustos.
# 3. Algunos países presentan valores "infinitos" debido a tasas iguales a cero en ciertos años, lo cual debe manejarse cuidadosamente para evitar sesgos.
#  **Conclusiones Generales**
# 1. Los datos refuerzan la necesidad de enfoques regionales específicos para abordar las altas tasas de suicidio en Europa del Este y las variaciones interanuales extremas en países clave.
# 2. Las diferencias de género subrayan la importancia de estrategias centradas en los hombres, abordando el estigma asociado a la búsqueda de ayuda.
# 3. La variación interanual puede proporcionar insights valiosos sobre la efectividad de las políticas previas y los impactos de eventos externos.
# # **Notebook 5: Análisis de Datos**
# 
# ## **Introducción**
# En este notebook consolidamos los hallazgos obtenidos en los análisis previos y exploramos aspectos adicionales de los datos. Nos enfocamos en tres niveles:
# 1. **Comunidades Autónomas de España:** Analizamos patrones locales.
# 2. **España a Nivel General:** Evaluamos tendencias nacionales y su relación con las comunidades autónomas.
# 3. **Europa:** Realizamos un análisis comparativo entre países europeos.
# 4. **Métodos de Suicidio:** Identificamos los métodos más utilizados y sus tendencias.
# 5. **Salud Mental en el Entorno Laboral:** Exploramos la relación entre estrés laboral y tendencias suicidas.
# Cargar datasets
dataframes = {
    "Tasas Suicidio Europa Temporal": pd.read_csv("data_limpio/Tasas_Suicidio_Europa_Temporal_ICD10_Limpio.csv"),
    "Suicidio Europa Datos ICD10": pd.read_csv("data_limpio/Suicidio_Europa_Datos_ICD10_Limpio.csv"),
    "Datos Comunidades España": pd.read_csv("data_limpio/Demografía_Residencia_Suicidio_Limpio.csv"),
    "Métodos de Suicidio": pd.read_csv("data_limpio/Métodos_Suicidio_Demografía_Limpio.csv"),
    "Salud Mental Entorno Laboral": pd.read_csv("data_limpio/Salud_Mental_Entorno_Laboral_Limpio.csv")
}
# ### 1: Comunidades Autónomas de España

# Filtrar y preparar los datos del dataset de métodos de suicidio
metodos_suicidio = dataframes["Métodos de Suicidio"].copy()

# Filtrar datos para excluir "Extranjero" y considerar solo las comunidades autónomas
comunidades_filtradas = metodos_suicidio[
    (metodos_suicidio["comunidad"] != "Extranjero") &
    (metodos_suicidio["método"] == "Total")
]

# Identificar y eliminar outliers utilizando el método del rango intercuartílico (IQR)
Q1 = comunidades_filtradas["total"].quantile(0.25)
Q3 = comunidades_filtradas["total"].quantile(0.75)
IQR = Q3 - Q1
filtro = (comunidades_filtradas["total"] >= (Q1 - 1.5 * IQR)) & (comunidades_filtradas["total"] <= (Q3 + 1.5 * IQR))
comunidades_sin_outliers = comunidades_filtradas[filtro]

# Crear la gráfica de barras para mostrar el total de suicidios por comunidad autónoma (sin outliers)
plt.figure(figsize=(12, 6))
sns.barplot(
    data=comunidades_sin_outliers, 
    x="comunidad", 
    y="total", 
    palette="viridis"
)
plt.title("Distribución de Suicidios por Comunidad (Sin Outliers)")
plt.xlabel("Comunidad Autónoma")
plt.ylabel("Total de Suicidios")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ### 2: España a Nivel General

# Trabajaremos con el dataset de Tasas Suicidio Europa Temporal
tasas_europa = dataframes["Tasas Suicidio Europa Temporal"].copy()

# Filtrar datos solo para España
tasas_espana = tasas_europa[tasas_europa["geo"] == "ES"]

# Agrupar los datos por año y sumar los valores observados (OBS_VALUE)
suicidios_anuales = tasas_espana.groupby("TIME_PERIOD")["OBS_VALUE"].sum().reset_index()

# Renombrar columnas para claridad
suicidios_anuales.columns = ["Año", "Total Suicidios"]

# Identificar y eliminar outliers utilizando el método del rango intercuartílico (IQR)
Q1 = suicidios_anuales["Total Suicidios"].quantile(0.25)
Q3 = suicidios_anuales["Total Suicidios"].quantile(0.75)
IQR = Q3 - Q1
filtro = (suicidios_anuales["Total Suicidios"] >= (Q1 - 1.5 * IQR)) & (suicidios_anuales["Total Suicidios"] <= (Q3 + 1.5 * IQR))
suicidios_anuales_sin_outliers = suicidios_anuales[filtro]

# Crear la gráfica de evolución de suicidios a nivel nacional (sin outliers)
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=suicidios_anuales_sin_outliers, 
    x="Año", 
    y="Total Suicidios", 
    marker="o", 
    color="blue"
)
plt.title("Evolución Anual de Suicidios en España (Sin Outliers)")
plt.xlabel("Año")
plt.ylabel("Total de Suicidios")
plt.grid(True)
plt.tight_layout()
plt.show()

# ### 3: Europa en su conjunto.

# Trabajaremos con el dataset de Tasas Suicidio Europa Temporal
tasas_europa = dataframes["Tasas Suicidio Europa Temporal"].copy()

# Agrupar los datos por país (geo) y calcular la tasa promedio de suicidio
tasas_por_pais = tasas_europa.groupby("geo")["OBS_VALUE"].mean().reset_index()

# Renombrar columnas para mayor claridad
tasas_por_pais.columns = ["País", "Tasa Promedio Suicidios"]

# Identificar y eliminar outliers utilizando el método del rango intercuartílico (IQR)
Q1 = tasas_por_pais["Tasa Promedio Suicidios"].quantile(0.25)
Q3 = tasas_por_pais["Tasa Promedio Suicidios"].quantile(0.75)
IQR = Q3 - Q1
filtro = (tasas_por_pais["Tasa Promedio Suicidios"] >= (Q1 - 1.5 * IQR)) & (tasas_por_pais["Tasa Promedio Suicidios"] <= (Q3 + 1.5 * IQR))
tasas_por_pais_sin_outliers = tasas_por_pais[filtro]

# Ordenar los países por la tasa promedio
tasas_por_pais_sin_outliers = tasas_por_pais_sin_outliers.sort_values(by="Tasa Promedio Suicidios", ascending=False)

# Crear una gráfica de barras para comparar las tasas promedio de suicidio por país (sin outliers)
plt.figure(figsize=(14, 8))
sns.barplot(
    data=tasas_por_pais_sin_outliers, 
    x="Tasa Promedio Suicidios", 
    y="País", 
    palette="mako"
)
plt.title("Tasa Promedio de Suicidios en Europa por País (Sin Outliers)")
plt.xlabel("Tasa Promedio de Suicidios (por 100,000 habitantes)")
plt.ylabel("País")
plt.tight_layout()
plt.show()


# ### 4: Métodos de Suicidio

# Trabajaremos con el dataset de Métodos de Suicidio
metodos_suicidio = dataframes["Métodos de Suicidio"].copy()

# Filtrar datos para excluir "Total" en la columna de método
metodos_filtrados = metodos_suicidio[
    metodos_suicidio["método"] != "Total"
]

# Agrupar los datos por método y sumar los valores totales
metodos_agrupados = metodos_filtrados.groupby("método")["total"].sum().reset_index()

# Renombrar columnas para mayor claridad
metodos_agrupados.columns = ["Método", "Total Suicidios"]

# Identificar y eliminar outliers utilizando el método del rango intercuartílico (IQR)
Q1 = metodos_agrupados["Total Suicidios"].quantile(0.25)
Q3 = metodos_agrupados["Total Suicidios"].quantile(0.75)
IQR = Q3 - Q1
filtro = (metodos_agrupados["Total Suicidios"] >= (Q1 - 1.5 * IQR)) & (metodos_agrupados["Total Suicidios"] <= (Q3 + 1.5 * IQR))
metodos_sin_outliers = metodos_agrupados[filtro]

# Ordenar los métodos por el total de suicidios
metodos_sin_outliers = metodos_sin_outliers.sort_values(by="Total Suicidios", ascending=False)

# Simplificar etiquetas para mejorar legibilidad
metodos_sin_outliers["Método"] = metodos_sin_outliers["Método"].apply(lambda x: x.split(". ")[-1])

# Crear una gráfica de barras horizontal para destacar los métodos más utilizados
plt.figure(figsize=(12, 8))
sns.barplot(
    data=metodos_sin_outliers, 
    x="Total Suicidios", 
    y="Método", 
    palette="crest"
)
plt.title("Métodos de Suicidio Más Utilizados (Sin Outliers)")
plt.xlabel("Total de Suicidios")
plt.ylabel("Método")
plt.tight_layout()
plt.show()

# ### 5: Impacto de la Salud Mental en el Entorno Laboral.

# Cargar el dataset de Salud Mental Entorno Laboral
salud_laboral = dataframes["Salud Mental Entorno Laboral"].copy()

# Analizar la relación entre la búsqueda de tratamiento y el tamaño de la empresa
tratamiento_por_empresa = salud_laboral.groupby("tamano_empresa")["Sought Treatment"].mean().reset_index()
tratamiento_por_empresa.columns = ["Tamaño de Empresa", "Porcentaje que Busca Tratamiento"]

# Crear gráfica para visualizar el impacto del tamaño de la empresa
plt.figure(figsize=(10, 6))
sns.barplot(
    data=tratamiento_por_empresa, 
    x="Tamaño de Empresa", 
    y="Porcentaje que Busca Tratamiento", 
    palette="coolwarm"
)
plt.title("Impacto del Tamaño de la Empresa en la Búsqueda de Tratamiento")
plt.xlabel("Tamaño de la Empresa")
plt.ylabel("Porcentaje que Busca Tratamiento")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


#  5_2:Analizar el impacto del historial familia

# Analizar la relación entre el historial familiar de enfermedades mentales y la búsqueda de tratamiento
tratamiento_por_historial = salud_laboral.groupby("Family History of Mental Illness")["Sought Treatment"].mean().reset_index()
tratamiento_por_historial.columns = ["Historial Familiar", "Porcentaje que Busca Tratamiento"]

# Crear gráfica para visualizar el impacto del historial familiar
plt.figure(figsize=(8, 5))
sns.barplot(
    data=tratamiento_por_historial, 
    x="Historial Familiar", 
    y="Porcentaje que Busca Tratamiento", 
    palette="pastel"
)
plt.title("Impacto del Historial Familiar en la Búsqueda de Tratamiento")
plt.xlabel("Historial Familiar de Enfermedades Mentales")
plt.ylabel("Porcentaje que Busca Tratamiento")
plt.tight_layout()
plt.show()

# ### 5_3: Diferencias de Género

# Analizar la relación entre el género y la búsqueda de tratamiento
tratamiento_por_genero = salud_laboral.groupby("Gender")["Sought Treatment"].mean().reset_index()
tratamiento_por_genero.columns = ["Género", "Porcentaje que Busca Tratamiento"]

# Crear gráfica para visualizar las diferencias de género
plt.figure(figsize=(8, 5))
sns.barplot(
    data=tratamiento_por_genero, 
    x="Género", 
    y="Porcentaje que Busca Tratamiento", 
    palette="coolwarm"
)
plt.title("Diferencias de Género en la Búsqueda de Tratamiento")
plt.xlabel("Género")
plt.ylabel("Porcentaje que Busca Tratamiento")
plt.tight_layout()
plt.show()

# # **Reporte Final: Análisis de Datos**
# 
# ## **1. Resumen del Proyecto**
# Este notebook aborda el análisis exploratorio de datos (EDA) y KPIs relacionados con suicidios en España y Europa, así como la búsqueda de tratamiento en el entorno laboral. Los objetivos incluyen identificar tendencias clave y ofrecer insights para estrategias de prevención.
# ## **2. Análisis Exploratorio de Datos**
# ### **Suicidios en España**
# - Grupos más afectados: **50-54 años** y **75-79 años**.
# - Métodos predominantes: **ahorcamiento** en hombres y **envenenamiento** en mujeres.
# - Comunidades más afectadas: **Andalucía**, **Cataluña**, y **Comunitat Valenciana**.
#  **Métodos de Suicidio**
# - Principales: **ahorcamiento** y **saltos desde lugares elevados**.
# - Diferencias por género: hombres tienden a métodos más letales; mujeres recurren más al envenenamiento.
# ## **3. KPIs Clave**
# ### **Tasa de Suicidios por Género**
# - Hombres: 73,865 vs. Mujeres: 22,512.
# - Necesidad de estrategias centradas en hombres por el estigma asociado.
# ### **Promedio Anual por País**
# - Países con mayores tasas: **Lituania** y **Hungría**.
# - Países con menores tasas: **Chipre** y **Malta**.
# ### **Variación Interanual**
# - Fluctuaciones extremas en países como Serbia y Eslovenia.
# - Estabilidad en Finlandia y Noruega.
## **4. Salud Mental en el Entorno Laboral**
# ### **Impacto del Historial Familiar**
# - Las personas con antecedentes familiares tienen más probabilidades de buscar ayuda.
#  **Diferencias de Género**
# - Las mujeres muestran mayor predisposición a buscar tratamiento en comparación con los hombres.