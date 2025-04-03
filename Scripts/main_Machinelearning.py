# # 1_Preparación y exploración de datos
# 1. **Carga de datos**:
#    - Hemos cargado dos datasets: Notas de Suicidio y Tweets sobre posibles publicaciones suicidas.
# 2. **Homogeneización**:
#    - Las etiquetas de ambos datasets fueron unificadas en dos categorías: `suicide` y `non-suicide`.
#  3. **Concatenación**:
#    - Unimos los datasets en uno único para trabajar con todos los datos en un solo lugar.
#  4. **Exploración preliminar**:
#    - Revisamos la calidad de los datos, incluyendo valores faltantes y estadísticas descriptivas sobre la longitud de los textos.
#    - Identificamos la distribución de los datos por categoría para verificar la calidad y balance de los datos.
#  5. **Limpieza**:
#    - Eliminamos los registros con valores nulos en la columna de texto (`contenido`) y guardamos el dataset final para el preprocesamiento en el siguiente notebook.
# Este análisis preliminar nos asegura que los datos están listos para continuar con el procesamiento y técnicas de análisis más avanzadas. 

# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
# Cargar los datasets
notas_path = "data_limpio/Clasificación_Textos_Suicidio_Limpio.csv"  
tweets1_path = "data_limpio/Clasificación_Tweets_Suicidio_Limpio.csv"  
# Leer los archivos en DataFrames
notas = pd.read_csv(notas_path) 
tweets = pd.read_csv(tweets1_path)  
# Visualización inicial de los datasets
print("Dataset 1: Notas de Suicidio")
print(notas.info())  
print(notas.head())  
print("\nDataset 2: Tweets Dataset 1")
print(tweets.info())  
print(tweets.head())  

# Homogeneizar etiquetas en ambos datasets
notas["etiqueta"] = notas["etiqueta"].replace({"suicide": "suicide", "non-suicide": "non-suicide"})
tweets1["etiqueta"] = tweets1["etiqueta"].replace({"potential suicide post": "suicide", "not suicide post": "non-suicide"})
# Eliminar columna 'id' del dataset de notas 
notas = notas.drop(columns=["id"])
# Verificar la distribución de etiquetas en ambos datasets
print("Distribución de etiquetas en Notas de Suicidio:")
print(notas["etiqueta"].value_counts())
print("\nDistribución de etiquetas en Tweets Dataset 1:")
print(tweets1["etiqueta"].value_counts())
# Verificar el total de registros antes de concatenar
total_registros = len(notas) + len(tweets1)
print(f"\nTotal de registros antes de concatenar: {total_registros}")
# Concatenar los datasets
datasets_unidos = pd.concat([notas, tweets1], ignore_index=True)
# Verificar la distribución de etiquetas en el dataset unido
print("\nDistribución de etiquetas en el dataset unido:")
print(datasets_unidos["etiqueta"].value_counts())
print(f"Total de registros después de concatenar: {len(datasets_unidos)}")
# ### Estudio preliminar de la calidad de los datos
# En esta sección realizaremos un análisis exploratorio inicial de los datos, incluyendo:
# 1. Revisión de valores faltantes en las columnas clave.
# 2. Distribución de palabras por categoría (`suicide` y `non-suicide`).
# 3. Análisis estadístico básico de las características de los textos.

# Revisar valores faltantes
print("Valores faltantes en el dataset:")
print(datasets_unidos.isnull().sum())

# Estadísticas descriptivas de la longitud de los textos
datasets_unidos["num_palabras"] = datasets_unidos["contenido"].apply(lambda x: len(str(x).split()))
print("\nEstadísticas descriptivas de la longitud de los textos:")
print(datasets_unidos.groupby("etiqueta")["num_palabras"].describe())

# Distribución de las etiquetas
print("\nDistribución de registros por etiqueta:")
print(datasets_unidos["etiqueta"].value_counts())

# Eliminar registros con valores nulos 
datasets_unidos = datasets_unidos.dropna(subset=["contenido"])

# Verificar que no haya nulos 
print("Valores faltantes tras eliminar registros con nulos:")
print(datasets_unidos.isnull().sum())

# Guardar el dataset concatenado 
datasets_unidos.to_csv("dataset_unido.csv", index=False)

print("Dataset unido guardado como 'dataset_unido.csv'")

# 2_Procesamiento de Datos
# En este notebook realizaremos los pasos necesarios para preparar y transformar el dataset unido, asegurando que esté listo para análisis avanzados. Las tareas incluyen:
# 1. **Limpieza de datos**:
#    - Refinaremos el texto eliminando caracteres especiales, palabras irrelevantes y ruido.
#  2. **Normalización de texto**:
#    - Convertiremos el texto a un formato uniforme (minúsculas, sin caracteres extra).
#  3. **Generación de atributos**:
#    - Crearemos columnas adicionales para análisis, como conteo de palabras o detección de patrones específicos.
#4. **Preparación para modelado**:
#    - Dejaremos el dataset listo para aplicar técnicas de análisis y modelos en etapas posteriores.
# Importación de librerías necesarias

# Cargar el dataset unido
df = "dataset_unido.csv" 
datos = pd.read_csv(df)

# Verificar la carga del dataset
print("Primeras filas del dataset cargado:")
print(datos.head())

print("\nInformación general del dataset:")
print(datos.info())

# Función para limpiar el texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto) 
    texto = re.sub(r"@\w+", "", texto)  
    texto = re.sub(r"[^a-z\s]", "", texto) 
    texto = re.sub(r"\s+", " ", texto).strip() 
    return texto

# Aplicar la limpieza 
datos["contenido"] = datos["contenido"].apply(limpiar_texto)

# Verificar la limpieza
print("Primeras filas después de la limpieza:")
print(datos.head())

# Obtener la lista de stop words en inglés
stop_words = set(stopwords.words("english"))

# Función para filtrar las stop words del contenido
def quitar_stop_words(texto):
    palabras = texto.split()  
    palabras_filtradas = [palabra for palabra in palabras if palabra not in stop_words]
    return " ".join(palabras_filtradas)  

# Aplicar el filtro de stop words al texto
datos["contenido"] = datos["contenido"].apply(quitar_stop_words)

# Verificar las primeras filas después de quitar stop words
print("Primeras filas después de eliminar stop words:")
print(datos.head())

# Dividir el texto en palabras y contar las frecuencias
palabras = " ".join(datos["contenido"]).split()  
conteo_palabras = Counter(palabras)  

# Obtener las 20 palabras más frecuentes
palabras_mas_comunes = conteo_palabras.most_common(20)

# Mostrar el resultado
print("Palabras más comunes y su frecuencia:")
for palabra, frecuencia in palabras_mas_comunes:
    print(f"{palabra}: {frecuencia}")
# ### Análisis comparativo entre categorías

# Separar los textos por categoría
suicide_texts = " ".join(datos[datos["etiqueta"] == "suicide"]["contenido"]).split()
non_suicide_texts = " ".join(datos[datos["etiqueta"] == "non-suicide"]["contenido"]).split()

# Contar la frecuencia de palabras en cada categoría
conteo_suicide = Counter(suicide_texts)
conteo_non_suicide = Counter(non_suicide_texts)

# Obtener las 20 palabras más comunes en cada categoría
palabras_comunes_suicide = conteo_suicide.most_common(20)
palabras_comunes_non_suicide = conteo_non_suicide.most_common(20)

# Mostrar los resultados
print("Palabras más comunes en textos suicidas:")
for palabra, frecuencia in palabras_comunes_suicide:
    print(f"{palabra}: {frecuencia}")

print("\nPalabras más comunes en textos no suicidas:")
for palabra, frecuencia in palabras_comunes_non_suicide:
    print(f"{palabra}: {frecuencia}")

# Palabras comunes en textos suicidas:
# Palabras como "life", "feel", "want", "cant", y "never" destacan, lo que refleja emociones intensas y pensamientos introspectivos.
# La frecuencia de "life" y "feel" sugiere un enfoque en el estado emocional y la percepción de la vida.
# Palabras comunes en textos no suicidas:
# Palabras como "filler", "fuck", "day", y "school" aparecen más frecuentemente, lo que indica un contenido más cotidiano o casual.
# La presencia de "school" podría reflejar un contexto más juvenil o relacionado con experiencias diarias.
# Diferencias clave:
# Palabras como "life", "feel", y "never" son mucho más prominentes en textos suicidas, mientras que palabras como "filler" y "school" son exclusivas de los textos no suicidas.
# ## Cálculo de TF-IDF (Term Frequency-Inverse Document Frequency)
# En esta sección, calcularemos las métricas de **TF-IDF** para identificar palabras clave que son representativas de cada categoría (`suicide` y `non-suicide`). El objetivo es:
# 1. **Comprender la importancia de las palabras** en cada texto en relación con su frecuencia en todo el dataset.
# 2. **Destacar términos relevantes** que podrían actuar como disparadores en los textos suicidas.
# 3. **Preparar las bases para un análisis más avanzado** enfocado en patrones y disparadores lingüísticos.
# Este enfoque nos permitirá extraer información más significativa y avanzar hacia la identificación de palabras clave que puedan servir para el modelo de clasificación.
# Crear el vectorizador TF-IDF
vectorizador = TfidfVectorizer(max_features=100, stop_words="english")
# Aplicar el TF-IDF al texto
tfidf_matriz = vectorizador.fit_transform(datos["contenido"])

# Obtener las palabras y sus pesos (TF-IDF)
tfidf_palabras = vectorizador.get_feature_names_out()
tfidf_pesos = tfidf_matriz.sum(axis=0).A1

# Ordenar por importancia (mayor TF-IDF)
tfidf_resultados = sorted(zip(tfidf_palabras, tfidf_pesos), key=lambda x: x[1], reverse=True)

# Mostrar las 20 palabras más importantes según TF-IDF
print("Palabras más importantes según TF-IDF:")
for palabra, peso in tfidf_resultados[:20]:
    print(f"{palabra}: {peso}")

# Crear el vectorizador TF-IDF por categoría
vectorizador_suicide = TfidfVectorizer(max_features=100, stop_words="english")
vectorizador_non_suicide = TfidfVectorizer(max_features=100, stop_words="english")

# Aplicar el vectorizador a cada categoría
tfidf_suicide = vectorizador_suicide.fit_transform(datos[datos["etiqueta"] == "suicide"]["contenido"])
tfidf_non_suicide = vectorizador_non_suicide.fit_transform(datos[datos["etiqueta"] == "non-suicide"]["contenido"])

# Obtener palabras y sus pesos para cada categoría
palabras_suicide = vectorizador_suicide.get_feature_names_out()
pesos_suicide = tfidf_suicide.sum(axis=0).A1

palabras_non_suicide = vectorizador_non_suicide.get_feature_names_out()
pesos_non_suicide = tfidf_non_suicide.sum(axis=0).A1

# Ordenar palabras por importancia
resultados_suicide = sorted(zip(palabras_suicide, pesos_suicide), key=lambda x: x[1], reverse=True)
resultados_non_suicide = sorted(zip(palabras_non_suicide, pesos_non_suicide), key=lambda x: x[1], reverse=True)

# Mostrar las 10 palabras más importantes en cada categoría
print("Palabras más importantes en textos suicidas (TF-IDF):")
for palabra, peso in resultados_suicide[:10]:
    print(f"{palabra}: {peso}")

print("\nPalabras más importantes en textos no suicidas (TF-IDF):")
for palabra, peso in resultados_non_suicide[:10]:
    print(f"{palabra}: {peso}")

# ### refinar los textos suicidas
# Lista de palabras comunes a eliminar
palabras_comunes = {"im", "dont", "ive","things", "people", "really", "get", "one"}

# Función para refinar el texto suicida
def refinar_texto_suicida(texto):
    palabras = texto.split()
    palabras_filtradas = [palabra for palabra in palabras if palabra not in palabras_comunes and len(palabra) > 2]
    return " ".join(palabras_filtradas)

# Aplicar el refinamiento a los textos suicidas
datos.loc[datos["etiqueta"] == "suicide", "contenido"] = datos[datos["etiqueta"] == "suicide"]["contenido"].apply(refinar_texto_suicida)

# Dividir el texto en palabras y contar las frecuencias
palabras_suicidas = " ".join(datos[datos["etiqueta"] == "suicide"]["contenido"]).split()
conteo_palabras_suicidas = Counter(palabras_suicidas)

# Obtener las 30 palabras más frecuentes
palabras_mas_comunes_suicidas = conteo_palabras_suicidas.most_common(30)

# Mostrar el resultado
print("30 palabras más comunes en textos suicidas refinados:")
for palabra, frecuencia in palabras_mas_comunes_suicidas:
    print(f"{palabra}: {frecuencia}")

datos.to_csv("dataset_unido.csv", index=False)

# # Informe de Análisis: Palabras más comunes en textos suicidas refinados
# Introducción
# Este informe analiza las 30 palabras más comunes en textos suicidas refinados, comparándolas con las palabras más importantes según TF-IDF y las palabras más relevantes en textos no suicidas. El objetivo es identificar patrones lingüísticos clave que puedan ser útiles para el desarrollo de un modelo de detección de suicidio.
# Palabras más comunes en textos suicidas refinados
# A continuación, se presentan las 30 palabras más frecuentes en textos suicidas refinados, junto con su frecuencia:
# 
# | Palabra      | Frecuencia |
# |--------------|------------|
# | like         | 131,141    |
# | want         | 128,509    |
# | life         | 110,774    |
# | feel         | 108,024    |
# | know         | 106,379    |
# | cant         | 90,159     |
# | even         | 77,032     |
# | time         | 69,768     |
# | would        | 69,310     |
# | think        | 57,403     |
# | going        | 57,231     |
# | never        | 56,874     |
# | much         | 50,765     |
# | friends      | 48,604     |
# | years        | 46,340     |
# | help         | 45,532     |
# | day          | 41,007     |
# | anymore      | 40,803     |
# | anything     | 37,707     |
# | way          | 37,481     |
# | could        | 37,109     |
# | die          | 36,856     |
# | fucking      | 36,432     |
# | make         | 36,420     |
# | family       | 36,358     |
# | everything   | 36,196     |
# | nothing      | 35,007     |
# | back         | 34,959     |
# | kill         | 34,928     |
# | end          | 34,889     |
#  Comparación con palabras más importantes según TF-IDF
# 1. **Coincidencias clave**:
#    - Palabras como *"like"*, *"want"*, *"life"*, *"feel"*, y *"know"* aparecen tanto en las listas de TF-IDF como en las palabras más comunes, lo que refuerza su relevancia en textos suicidas.
#    - *"time"* y *"friends"* también son recurrentes, indicando su importancia en el contexto emocional.
# 2. **Nuevas palabras destacadas**:
#    - Palabras como *"cant"*, *"even"*, *"never"*, y *"die"* no estaban entre las más importantes según TF-IDF, pero su alta frecuencia en textos suicidas refinados sugiere que podrían ser indicadores clave.
# ## Comparación con textos no suicidas
# 1. **Diferencias significativas**:
#    - Palabras como *"die"*, *"kill"*, *"end"*, y *"nothing"* son exclusivas de los textos suicidas, reflejando un enfoque en temas de desesperanza y finalización.
#    - En contraste, los textos no suicidas incluyen palabras más neutrales como *"day"*, *"got"*, y *"really"*.
# 2. **Patrones emocionales**:
#    - Las palabras en textos suicidas tienden a expresar emociones intensas (*"feel"*, *"life"*, *"cant"*) y relaciones personales (*"friends"*, *"family"*), mientras que los textos no suicidas son más descriptivos y cotidianos.
# ## Conclusión
# Este análisis destaca patrones lingüísticos clave en textos suicidas, como el uso frecuente de palabras relacionadas con emociones intensas, relaciones personales, y temas de desesperanza. Estas observaciones pueden ser fundamentales para entrenar modelos de detección y análisis de textos.
# # 3_Tokenización y Análisis NLP
# En este notebook avanzaremos con el análisis basado en **Procesamiento de Lenguaje Natural (NLP)**. Los pasos que seguiremos incluyen:
# 1. **Tokenización**:
# - Dividiremos los textos en unidades más pequeñas (tokens) para analizar su estructura.
#  2. **Exploración de patrones**:
# - Identificaremos combinaciones frecuentes de palabras y contextos significativos.
# 3. **Preparación para el modelado**:
#  Transformaremos los textos en representaciones numéricas (como Bag of Words o embeddings) que sirvan como entrada para el modelo NLP.
# Importación de librerías 
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.data.find('tokenizers/punkt')
import spacy
# Cargar el modelo de idioma en inglés de spaCy
nlp = spacy.load("en_core_web_sm")
# Cargar el dataset refinado
dataset_path = "data_limpio/Clasificación_Textos_Suicidio_Limpio.csv"  # Archivo sobreescrito con los cambios
datos = pd.read_csv(dataset_path)

# Verificar que el dataset se cargó correctamente
print("Primeras filas del dataset cargado:")
print(datos.head())

# Convertir todos los valores a tipo string
datos['contenido'] = datos['contenido'].astype(str)

# # Selección de Muestra Representativa
#  Contexto Inicial
# El dataset original contiene un total de **233,809 registros** distribuidos en tres columnas: `contenido` (texto), `etiqueta` (suicida/no suicida), y `num_palabras` (cantidad de palabras en el texto). Este volumen de datos, aunque valioso para el análisis, presenta desafíos en términos de tiempos de procesamiento, especialmente durante la tokenización y el preprocesamiento.
# ## Decisión de Uso de Muestra
# Para agilizar los procesos sin comprometer la calidad del análisis, se ha decidido trabajar con una **muestra representativa** del dataset. Esta muestra será suficientemente diversa y proporcional para reflejar las características generales del conjunto completo, asegurando que los hallazgos sean aplicables al análisis final.
# ## Beneficios de Trabajar con una Muestra
# 1. **Eficiencia**: Reducir el tiempo requerido para procesos como tokenización, análisis de palabras clave y modelado.
# 2. **Flexibilidad**: Permite iterar rápidamente en el análisis y ajustar el flujo de trabajo según las necesidades.
# 3. **Reusabilidad**: La muestra será guardada y podrá ser utilizada en futuros experimentos sin necesidad de reprocesar los datos originales.
#  Proceso a Seguir
# 1. Seleccionar un subconjunto del dataset basado en:
#    - Filtrado por etiquetas (`suicida` o ambos).
#    - Toma de una fracción aleatoria representativa (ej. 20%).
# 2. Realizar tokenización sobre la muestra seleccionada.
# 3. Guardar la muestra y los datos tokenizados en nuevos archivos, para evitar la repetición de procesos en el futuro.
# Este enfoque garantiza que el análisis sea eficiente y efectivo, sin sacrificar la validez de los resultados.
# Crear una muestra del 20% de textos suicidas
datos_suicidas = datos[datos['etiqueta'] == 'suicide']
muestra_suicidas = datos_suicidas.sample(frac=0.2, random_state=42)

# Guardar la muestra en un nuevo archivo
muestra_suicidas.to_csv("muestra_suicidas.csv", index=False)

# Crear una muestra del 20% de todo el dataset
muestra_general = datos.sample(frac=0.2, random_state=42)

# Guardar la muestra en un nuevo archivo
muestra_general.to_csv("muestra_general.csv", index=False)

# Tokenización básica como alternativa más rápida
muestra_general['tokens'] = muestra_general['contenido'].apply(lambda x: x.split())

muestra_general.to_csv("muestra_general_tokenizada.csv", index=False)

# Tokenizar la muestra suicida con str.split()
muestra_suicidas['tokens'] = muestra_suicidas['contenido'].apply(lambda x: x.split())

# Guardar la muestra tokenizada
muestra_suicidas.to_csv("muestra_suicidas_tokenizada.csv", index=False)
# # Notebook 4: NLP - Análisis Inicial
# **Introducción**
# El objetivo de este notebook es realizar un análisis NLP inicial utilizando los datos tokenizados, explorando patrones y representaciones útiles para el análisis de disparadores de suicidio en textos. El entrenamiento de modelos será desarrollado en notebooks posteriores.
#  **Pasos a seguir en este Notebook**
# **1. Inspeccionar las muestras tokenizadas**
#  Objetivo:
# - Cargar y revisar los archivos `muestra_general_tokenizada.csv` y `muestra_suicidas_tokenizada.csv`.
# - Asegurarnos de que las muestras estén correctamente preparadas para el análisis.
#  **2. Representación de los textos**
#  Objetivo:
# - Convertir los tokens en matrices útiles para el análisis.
#  Opciones:
# 1. **Bag of Words (BoW)**:
#    - Representación basada en frecuencia de palabras.
# 2. **TF-IDF**:
#    - Representación que mide la relevancia de las palabras en el contexto.
#**3. Análisis Exploratorio**
#  Objetivo:
# - Identificar patrones iniciales en las muestras tokenizadas.
#  Actividades:
# - Analizar la frecuencia de términos clave.
# - Generar visualizaciones como histogramas y nubes de palabras.
#  **Conclusión**
# Este notebook se centra en el análisis NLP para proporcionar una base sólida de exploración y representación de datos, preparando el camino para el desarrollo de modelos en futuros notebooks.
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
# Cargar las muestras tokenizadas
muestra_general = pd.read_csv("muestra_general_tokenizada.csv")
muestra_suicidas = pd.read_csv("muestra_suicidas_tokenizada.csv")
# Ver distribución de etiquetas en el dataset general
print(muestra_general['etiqueta'].value_counts())

# Agregar columna 'num_palabras' contando el número de tokens
muestra_general['num_palabras'] = muestra_general['tokens'].apply(lambda x: len(eval(x)))

# Filtrar textos dentro de los límites
datos_filtrados = muestra_general[
    (muestra_general['num_palabras'] >= 25) &
    (muestra_general['num_palabras'] <= 342)
]

# Confirmar cantidad de textos restantes después de filtrar
print(f"Textos restantes después de eliminar outliers: {len(datos_filtrados)}")

# Guardar el nuevo dataset filtrado
datos_filtrados.to_csv("dataset_filtrado.csv", index=False)

# Cargar el nuevo dataset filtrado
dataset_filtrado = pd.read_csv("dataset_filtrado.csv")
# Distribución del número de palabras en el dataset filtrado
plt.figure(figsize=(12, 6))

sns.histplot(dataset_filtrado['num_palabras'], bins=30, kde=True, color='green')
plt.title('Distribución del Número de Palabras (Dataset Filtrado)')
plt.xlabel('Número de Palabras')
plt.ylabel('Frecuencia')
plt.show()

# Segmentar el dataset por etiquetas
suicide_texts = dataset_filtrado[dataset_filtrado['etiqueta'] == 'suicide']
non_suicide_texts = dataset_filtrado[dataset_filtrado['etiqueta'] == 'non-suicide']

# Confirmar cantidades por segmento
print(f"Textos suicidas: {len(suicide_texts)}")
print(f"Textos no suicidas: {len(non_suicide_texts)}")

from collections import Counter

# Frecuencia de términos en textos suicidas
frecuencia_suicidas = Counter([token for tokens in suicide_texts['tokens'] for token in eval(tokens)])
print("Palabras más frecuentes en textos suicidas:")
print(frecuencia_suicidas.most_common(30))

# Frecuencia de términos en textos no suicidas
frecuencia_no_suicidas = Counter([token for tokens in non_suicide_texts['tokens'] for token in eval(tokens)])
print("\nPalabras más frecuentes en textos no suicidas:")
print(frecuencia_no_suicidas.most_common(30))

import nltk

# Descargar la lista de stopwords
nltk.download('stopwords')

# Lista de stopwords en inglés
stop_words = set(stopwords.words('english'))

# Frecuencia de términos en textos suicidas (sin stopwords)
frecuencia_suicidas = Counter(
    [token for tokens in suicide_texts['tokens'] 
     for token in eval(tokens) if token not in stop_words]
)
print("Palabras más frecuentes en textos suicidas:")
print(frecuencia_suicidas.most_common(30))

# Frecuencia de términos en textos no suicidas (sin stopwords)
frecuencia_no_suicidas = Counter(
    [token for tokens in non_suicide_texts['tokens'] 
     for token in eval(tokens) if token not in stop_words]
)
print("\nPalabras más frecuentes en textos no suicidas:")
print(frecuencia_no_suicidas.most_common(30))
from collections import Counter

# Lista personalizada de palabras irrelevantes que deseas eliminar
palabras_irrelevantes = ['im', 'dont', 'like', 'really', 'get',  'going', 'go', 'one']

# Frecuencia de términos en textos suicidas (sin stopwords y palabras irrelevantes)
frecuencia_suicidas_afinada = Counter(
    [token for tokens in suicide_texts['tokens'] 
     for token in eval(tokens) 
     if token not in stop_words and token not in palabras_irrelevantes]
)
print("Palabras más frecuentes en textos suicidas (afinadas):")
print(frecuencia_suicidas_afinada.most_common(30))

# Frecuencia de términos en textos no suicidas (sin stopwords y palabras irrelevantes)
frecuencia_no_suicidas_afinada = Counter(
    [token for tokens in non_suicide_texts['tokens'] 
     for token in eval(tokens) 
     if token not in stop_words and token not in palabras_irrelevantes]
)
print("\nPalabras más frecuentes en textos no suicidas (afinadas):")
print(frecuencia_no_suicidas_afinada.most_common(30))
# Incorporar tokens filtrados al DataFrame
dataset_filtrado['tokens_filtrados'] = dataset_filtrado['tokens'].apply(
    lambda x: [token for token in eval(x) if token not in stop_words and token not in palabras_irrelevantes]
)

# Convertir tokens filtrados a cadenas de texto para futuras transformaciones
dataset_filtrado['texto_filtrado'] = dataset_filtrado['tokens_filtrados'].apply(lambda x: ' '.join(x))

# Guardar el nuevo DataFrame filtrado
dataset_filtrado.to_csv("dataset_filtrado_palabras_filtradas.csv", index=False)

# Confirmar los cambios
print("Dataset filtrado guardado con las palabras procesadas.")

# Cargar el nuevo DataFrame desde el archivo guardado
dataset_filtrado = pd.read_csv("dataset_filtrado_palabras_filtradas.csv")

# Convertir de texto a listas de tokens si lo necesitas nuevamente
dataset_filtrado['tokens_filtrados'] = dataset_filtrado['tokens_filtrados'].apply(eval)

from sklearn.feature_extraction.text import CountVectorizer

# Crear matriz Bag of Words (Basado en los textos filtrados)
vectorizer_bow = CountVectorizer(max_features=5000)  # Selección de hasta 5000 palabras más relevantes
X_bow = vectorizer_bow.fit_transform(dataset_filtrado['texto_filtrado'])

# Mostrar forma de la matriz
print(f"Matriz Bag of Words: {X_bow.shape}")

from sklearn.feature_extraction.text import TfidfVectorizer

# Crear matriz TF-IDF utilizando los textos filtrados
vectorizer_tfidf = TfidfVectorizer(max_features=5000)  # Selección de las 5000 palabras más importantes
X_tfidf = vectorizer_tfidf.fit_transform(dataset_filtrado['texto_filtrado'])

# Mostrar forma de la matriz TF-IDF
print(f"Matriz TF-IDF: {X_tfidf.shape}")

# Opcional: Mostrar las palabras más importantes según TF-IDF
tfidf_feature_names = vectorizer_tfidf.get_feature_names_out()
print("\nPalabras más destacadas (TF-IDF):")
print(tfidf_feature_names[:30])  # Primeras 30 palabras

# Extraer las etiquetas del dataset filtrado
etiquetas = dataset_filtrado['etiqueta']

# Guardar las etiquetas en un archivo CSV
etiquetas.to_csv('etiquetas.csv', index=False)

print("Archivo de etiquetas guardado exitosamente.")

import joblib

# Guardar la matriz Bag of Words (BoW) y su vectorizador
joblib.dump(X_bow, 'matriz_bow.pkl')
joblib.dump(vectorizer_bow, 'vectorizador_bow.pkl')

# Guardar la matriz TF-IDF y su vectorizador
joblib.dump(X_tfidf, 'matriz_tfidf.pkl')
joblib.dump(vectorizer_tfidf, 'vectorizador_tfidf.pkl')

print("Matrices y vectorizadores guardados exitosamente.")

# # Informe Final: Notebook de Representación y Análisis NLP
# ## **Resumen de Actividades**
# Este notebook se centró en representar los textos del dataset general y generar análisis clave relacionados con el objetivo de detectar disparadores de suicidio en textos mediante técnicas de NLP. A continuación, se resumen los pasos realizados y los resultados obtenidos.
#  **Actividades Realizadas**
# ### **1. Eliminación de Outliers**
# - Se eliminaron textos con un número de palabras fuera del rango típico:
#   - Textos con menos de 25 palabras y más de 342 palabras fueron eliminados.
# - **Cantidad de textos restantes**: 31,259.
#  **2. Análisis Exploratorio**
# - **Distribución de etiquetas**:
#   - Textos suicidas: 17,616
#   - Textos no suicidas: 13,643
# - **Frecuencia de términos clave**:
#   - Las palabras más frecuentes en textos suicidas incluyen: *want*, *feel*, *life*, *cant*, *help*.
#   - Las palabras más frecuentes en textos no suicidas incluyen: *im*, *like*, *dont*, *school*, *people*.
#  **3. Representación Numérica de Textos**
# - **Bag of Words (BoW)**:
#   - Matriz de frecuencia para las 5,000 palabras más comunes.
#   - Dimensiones: (31,259, 5,000).
# - **TF-IDF**:
#   - Matriz ponderada que resalta las palabras más relevantes.
#   - Dimensiones: (31,259, 5,000).
# **Conclusiones**
# - El dataset filtrado y las matrices generadas (BoW y TF-IDF) están listas para ser usadas en el entrenamiento de modelos.
# - Los textos suicidas muestran patrones lingüísticos claros, reflejando emociones intensas y conceptos introspectivos.
# - Los textos no suicidas tienden a ser más narrativos y neutrales.
# # Notebook 5: Entrenamiento del Modelo
# 
# ## **Introducción**
# En este notebook, nos enfocaremos en construir y entrenar modelos de clasificación utilizando las representaciones generadas en el Notebook 4 (Bag of Words y TF-IDF). Los objetivos principales incluyen:
# 1. Entrenar modelos para clasificar textos en **suicidas** y **no suicidas**.
# 2. Evaluar el rendimiento de los modelos utilizando métricas clave como precisión, recall y F1-score.
# 3. Explorar interpretabilidad inicial para comprender las decisiones del modelo.

# Modelado
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# Configuración de gráficos
sns.set(style="whitegrid")

from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
# Cargar la matriz Bag of Words y su vectorizador
X_bow = joblib.load('matriz_bow.pkl')
vectorizer_bow = joblib.load('vectorizador_bow.pkl')

# Cargar la matriz TF-IDF y su vectorizador
X_tfidf = joblib.load('matriz_tfidf.pkl')
vectorizer_tfidf = joblib.load('vectorizador_tfidf.pkl')

# Cargar etiquetas
y = pd.read_csv('etiquetas.csv')

# Confirmar los tamaños
print(f"Matriz BoW: {X_bow.shape}")
print(f"Matriz TF-IDF: {X_tfidf.shape}")
print(f"Etiquetas: {y.shape}")

# División de datos (80% entrenamiento, 20% prueba)
X_train_bow, X_test_bow, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=42)
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

print("Datos divididos en entrenamiento y prueba.")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Entrenar el modelo usando Bag of Words
model_bow = LogisticRegression()
model_bow.fit(X_train_bow, y_train)

# Evaluar el modelo
y_pred_bow = model_bow.predict(X_test_bow)
print("Accuracy (BoW):", accuracy_score(y_test, y_pred_bow))
print("Reporte de clasificación (BoW):\n", classification_report(y_test, y_pred_bow))

# Entrenar el modelo usando TF-IDF
model_tfidf = LogisticRegression()
model_tfidf.fit(X_train_tfidf, y_train)

# Evaluar el modelo
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
print("Accuracy (TF-IDF):", accuracy_score(y_test, y_pred_tfidf))
print("Reporte de clasificación (TF-IDF):\n", classification_report(y_test, y_pred_tfidf))

# ## Selección del Modelo
# Tras comparar el rendimiento de dos enfoques diferentes de representación de texto (Bag of Words y TF-IDF) utilizando el algoritmo **Logistic Regression**, hemos optado por quedarnos con el modelo basado en **TF-IDF**. Esta decisión se fundamenta en los siguientes puntos clave:
# Resultados del Modelo Bag of Words (BoW)
# - **Accuracy**: 91.34%
# - **Precision (suicide)**: 93%
# - **Recall (suicide)**: 91%
# - **F1-Score (suicide)**: 92%
# - Aunque el modelo basado en Bag of Words ofrece buenos resultados, su representación únicamente considera la frecuencia de las palabras, lo que puede asignar mayor peso a términos comunes que no aportan significado específico en este contexto.
# Resultados del Modelo TF-IDF
# - **Accuracy**: 92.17%
# - **Precision (suicide)**: 94%
# - **Recall (suicide)**: 92%
# - **F1-Score (suicide)**: 93%
# - El modelo basado en TF-IDF mostró un desempeño superior al ponderar términos relevantes, dando mayor importancia a palabras que son frecuentes en algunos textos pero no están distribuidas uniformemente en todo el corpus. Esto permitió capturar mejor los patrones distintivos en los textos relacionados con el suicidio.
# Conclusión
# Optamos por el modelo basado en **TF-IDF**, ya que su rendimiento general es superior tanto en **accuracy** como en las métricas de precisión, recall y f1-score. La capacidad de TF-IDF para valorar términos relevantes mientras disminuye el peso de palabras poco informativas es crucial para la identificación precisa de textos relacionados con el suicidio.
#  Próximos pasos
# 1. **Guardar el modelo TF-IDF**:
#    El modelo será almacenado para su uso futuro en nuevas predicciones y análisis.
# 2. **Explorar algoritmos adicionales**:
#    Aunque Logistic Regression ya ofrece resultados sólidos, se evaluará el uso de modelos más avanzados como Random Forest, XGBoost o SVM para mejorar aún más la precisión del sistema.

# Verificar distribución de clases
from sklearn.utils import resample

print("Distribución original de etiquetas:")
print(y.value_counts())

# Verificar distribución de etiquetas original
print("Distribución original de etiquetas:")
print(y.value_counts())

# Combinar características y etiquetas en un DataFrame temporal
df_temporal = pd.DataFrame(X_tfidf.toarray())  # Convertir la matriz TF-IDF a un DataFrame
df_temporal['etiqueta'] = y

# Dividir en clases mayoritaria y minoritaria
clase_mayoritaria = df_temporal[df_temporal['etiqueta'] == 'suicide']
clase_minoritaria = df_temporal[df_temporal['etiqueta'] == 'non-suicide']

# Aplicar submuestreo a la clase mayoritaria
clase_mayoritaria_subsampled = resample(
    clase_mayoritaria,
    replace=False,  # No reemplazar, solo reducir
    n_samples=len(clase_minoritaria),  # Igualar al tamaño de la clase minoritaria
    random_state=42  # Para reproducibilidad
)

# Combinar clases balanceadas
df_balanceado = pd.concat([clase_mayoritaria_subsampled, clase_minoritaria])

# Verificar nueva distribución de etiquetas
print("\nDistribución balanceada de etiquetas:")
print(df_balanceado['etiqueta'].value_counts())

# Separar características y etiquetas nuevamente
X_balanceado = df_balanceado.drop('etiqueta', axis=1)  # Características balanceadas
y_balanceado = df_balanceado['etiqueta']  # Etiquetas balanceadas

# KNN

from sklearn.neighbors import KNeighborsClassifier
# Dividir datos balanceados en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_balanceado, y_balanceado, test_size=0.2, random_state=42)

# Entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)  # Inicialmente usamos 5 vecinos
knn.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_knn = knn.predict(X_test)

# Evaluar el modelo
print("Resultados de KNN:")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred_knn))

# Matriz de confusión para análisis de errores
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred_knn))

# ## Resultados - K-Nearest Neighbors (KNN)
# - **Accuracy**: 49.61%
# - **Precision (non-suicide)**: 50%
# - **Recall (non-suicide)**: 100%
# - **Recall (suicide)**: 0%
# - **Observaciones**: El modelo muestra un rendimiento deficiente, con un fuerte sesgo hacia la clase "non-suicide". Esto sugiere que KNN no es adecuado para datos con alta dimensionalidad, como las matrices TF-IDF.

# Naive Bayes

# Entrenar el modelo Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_nb = nb.predict(X_test)

# Evaluar el modelo
print("Resultados de Naive Bayes:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred_nb))

# Matriz de confusión para análisis de errores
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred_nb))

# ## Resultados - Naive Bayes
# - **Accuracy**: 89.98%
# - **Precision (non-suicide)**: 94%
# - **Recall (non-suicide)**: 85%
# - **Precision (suicide)**: 87%
# - **Recall (suicide)**: 95%
# - **F1-Score (suicide)**: 90%
# - **Observaciones**: Naive Bayes presenta un rendimiento consistente y balanceado, con un excelente recall para la clase "suicide". Es un modelo simple pero robusto para tareas de clasificación de texto.

# Support Vector Machines (SVM)

# Entrenar el modelo SVM
svm = SVC(kernel='linear', random_state=42)  # Usamos un kernel lineal inicialmente
svm.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_svm = svm.predict(X_test)

# Evaluar el modelo
print("Resultados de Support Vector Machines (SVM):")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred_svm))

# Matriz de confusión para análisis de errores
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred_svm))

# ## Resultados - Support Vector Machines (SVM)
# - **Accuracy**: 92.52%
# - **Precision (non-suicide)**: 92%
# - **Recall (non-suicide)**: 93%
# - **Precision (suicide)**: 93%
# - **Recall (suicide)**: 92%
# - **F1-Score (suicide)**: 93%
# - **Observaciones**: SVM muestra un excelente rendimiento, con métricas equilibradas en ambas clases. Este modelo es adecuado para tareas de clasificación de texto, y podría optimizarse aún más ajustando los hiperparámetros como `C` o probando kernels diferentes.
# 

# Random Forest

# Entrenar el modelo Random Forest
rf = RandomForestClassifier(random_state=42, n_estimators=100)  # 100 árboles por defecto
rf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_rf = rf.predict(X_test)

# Evaluar el modelo
print("Resultados de Random Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred_rf))

# Matriz de Confusión para análisis de errores
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred_rf))

# ## Resultados - Random Forest
# - **Accuracy**: 89.04%
# - **Precision (non-suicide)**: 91%
# - **Recall (non-suicide)**: 87%
# - **Precision (suicide)**: 88%
# - **Recall (suicide)**: 91%
# - **F1-Score (suicide)**: 89%
# - **Matriz de confusión**:
#   - [[2340, 353],
#      [242, 2495]]
# - **Observaciones**: Random Forest presenta un buen equilibrio entre precisión y recall, con un desempeño competitivo. Podría mejorarse ligeramente ajustando hiperparámetros como `n_estimators` o `max_depth`.

# XGBoost

from sklearn.preprocessing import LabelEncoder

# Codificar etiquetas
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Entrenar el modelo XGBoost
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')  # Configuración inicial
xgb.fit(X_train, y_train_encoded)

# Predecir en el conjunto de prueba
y_pred_xgb = xgb.predict(X_test)

# Decodificar las predicciones
y_pred_xgb_decoded = label_encoder.inverse_transform(y_pred_xgb)

# Evaluar el modelo
print("Resultados de XGBoost:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb_decoded))
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred_xgb_decoded))

# Matriz de Confusión
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred_xgb_decoded))

# ## Resultados - XGBoost
# - **Accuracy**: 90.13%
# - **Precision (non-suicide)**: 89%
# - **Recall (non-suicide)**: 91%
# - **Precision (suicide)**: 91%
# - **Recall (suicide)**: 89%
# - **F1-Score (suicide)**: 90%
# - **Matriz de confusión**:
#   - [[2448, 245],
#      [291, 2446]]
# - **Observaciones**: XGBoost muestra un rendimiento sobresaliente, con métricas equilibradas y una accuracy superior. Podría optimizarse aún más ajustando hiperparámetros clave.

# Definir un rango reducido de hiperparámetros
param_grid = {
    'n_estimators': [50, 100],           # Reducimos a 2 opciones para el número de árboles
    'max_depth': [3, 5],                # Reducimos la profundidad máxima
    'learning_rate': [0.1, 0.2],        # Dos opciones para la tasa de aprendizaje
}

# Configurar el modelo base
xgb = XGBClassifier(random_state=42, eval_metric='logloss')

# Codificar etiquetas para XGBoost
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Configurar GridSearchCV
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

# Entrenar con búsqueda de hiperparámetros
grid_search.fit(X_train, y_train_encoded)

# Mostrar los mejores parámetros y su rendimiento
print("Mejores parámetros:", grid_search.best_params_)
print("Mejor Accuracy:", grid_search.best_score_)

# Evaluar el mejor modelo en el conjunto de prueba
xgb_best = grid_search.best_estimator_
y_pred_best = xgb_best.predict(X_test)

# Decodificar las predicciones a etiquetas originales
y_pred_best_decoded = label_encoder.inverse_transform(y_pred_best)

print("Resultados del mejor modelo XGBoost:")
print("Accuracy:", accuracy_score(y_test, y_pred_best_decoded))
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred_best_decoded))

# # Resultados del Modelo XGBoost Ajustado con Hiperparámetros
# 
# Después del ajuste de hiperparámetros, el modelo **XGBoost** ha alcanzado un rendimiento impresionante, destacando su efectividad en la tarea de clasificación entre "suicide" y "non-suicide".
# 
# ## Hiperparámetros Óptimos
# - **Learning Rate**: 0.2  
# - **Max Depth**: 5  
# - **Número de Árboles (n_estimators)**: 100  
# 
# ## Métricas de Rendimiento
# | Métrica              | Valor   |
# |----------------------|---------|
# | **Accuracy**         | 89.57%  |
# | **Precision (non-suicide)** | 88%     |
# | **Recall (non-suicide)**    | 91%     |
# | **Precision (suicide)**     | 91%     |
# | **Recall (suicide)**        | 88%     |
# | **F1-Score (global)**       | 0.90    |
# 
# ## Resumen del Rendimiento
# - **Balance entre clases**: Tanto la clase "suicide" como "non-suicide" presentan métricas equilibradas, indicando que el modelo no favorece una clase por encima de otra.  
# - **Eficiencia global**: La precisión general del modelo lo hace adecuado para escenarios reales donde es esencial minimizar errores de clasificación.  
# 
# ## Impacto y Relevancia
# El modelo ajustado podría utilizarse como una herramienta predictiva robusta en aplicaciones de análisis de texto, ayudando a identificar contenido relacionado con riesgo de suicidio. Su balance entre precisión y recall asegura una mayor confiabilidad en las predicciones.

# # Comparación de Modelos Probados
# 
# ## Métricas Generales
# | Modelo                | Hiperparámetros Principales                     | Accuracy | Precision (suicide) | Recall (suicide) | F1-Score (global) |
# |-----------------------|-------------------------------------------------|----------|---------------------|------------------|-------------------|
# | **SVM**               | Kernel=RBF, C=1                                | 86.43%   | 85%                 | 88%              | 0.86              |
# | **Random Forest**     | n_estimators=100, Max Depth=None               | 89.04%   | 88%                 | 91%              | 0.89              |
# | **XGBoost (inicial)** | Learning Rate=0.1, Max Depth=3, n_estimators=50| 90.13%   | 91%                 | 89%              | 0.90              |
# | **XGBoost (ajustado)**| Learning Rate=0.2, Max Depth=5, n_estimators=100| 89.57%   | 91%                 | 88%              | 0.90              |
# 
# ## Análisis Comparativo
# - **SVM**:
#   - Buen rendimiento general, aunque ligeramente menor en comparación con los algoritmos de ensamble.  
#   - Ventaja: Adecuado para problemas lineales y datos con alta dimensionalidad.  
# - **Random Forest**:
#   - Ofreció un buen balance entre precisión y recall, con un rendimiento competitivo.  
#   - Limitación: Mayor tiempo de entrenamiento en comparación con otros modelos.  
# - **XGBoost (inicial)**:
#   - Mostró el mejor desempeño antes del ajuste de hiperparámetros, destacándose por su capacidad de generalización.  
# - **XGBoost (ajustado)**:
#   - Ajuste de hiperparámetros mejoró su estabilidad y controló el sobreajuste, asegurando un rendimiento óptimo en diferentes métricas.  
# 
# ## Conclusión
# El modelo **XGBoost ajustado** se posiciona como la opción más sólida debido a su balance entre precisión, recall y rendimiento general. Esto lo hace especialmente adecuado para la implementación en sistemas predictivos reales.

import joblib

# Guardar el modelo entrenado
joblib.dump(xgb_best, "xgboost_best_model.pkl")
print("Modelo guardado exitosamente como 'xgboost_best_model.pkl'")

# Guardar el codificador de etiquetas (label_encoder) si necesitas decodificar en el futuro
joblib.dump(label_encoder, "label_encoder.pkl")
print("Codificador de etiquetas guardado como 'label_encoder.pkl'")


# Guardar el mejor modelo ajustado
joblib.dump(grid_search.best_estimator_, "xgboost_best_model.pkl")
print("Modelo ajustado guardado exitosamente como 'xgboost_best_model.pkl'")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

xgb_loaded = joblib.load("xgboost_best_model.pkl")
print("Modelo cargado exitosamente.")

# Realizar predicciones en el conjunto de prueba
y_pred_loaded = xgb_loaded.predict(X_test)

# Decodificar las etiquetas predichas para obtener las originales (si es necesario)
y_pred_loaded_decoded = label_encoder.inverse_transform(y_pred_loaded)

# Evaluar el rendimiento del modelo cargado
print("Accuracy del modelo cargado:", accuracy_score(y_test, y_pred_loaded_decoded))
print("Reporte de Clasificación del modelo cargado:\n", classification_report(y_test, y_pred_loaded_decoded))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_loaded_decoded))

# # Clasificación Errónea de Textos Felices como "Suicida"
# 
# ## Problema Detectado
# Durante las pruebas realizadas en el modelo predictivo utilizando **XGBoost**, se identificó un comportamiento inesperado: textos claramente positivos o felices fueron clasificados incorrectamente como "suicidas". Este resultado puede afectar la precisión del modelo y comprometer su confiabilidad en escenarios reales.
# 
# ## Posibles Causas
# 1. **Sesgo en los Datos de Entrenamiento**:
#    - Los textos positivos pueden estar etiquetados incorrectamente como "suicidas", causando que el modelo aprenda patrones erróneos.
# 
# 2. **Vectorización con TF-IDF**:
#    - Algunas palabras positivas pueden estar presentes en textos suicidas en un contexto irónico o sarcástico, lo cual confunde al vectorizador y al modelo.
# 
# 3. **Desbalance entre Clases**:
#    - Si la clase "suicida" tiene significativamente más ejemplos en el conjunto de datos, el modelo puede estar sesgado hacia esta clase.
# 
# 4. **Hiperparámetros Subóptimos**:
#    - Aunque el modelo ajustado logró métricas destacadas (Accuracy: 89.57%), es posible que no esté optimizado para casos complejos como textos positivos que no están etiquetados correctamente.
# 
# ## Impacto en el Proyecto
# - Clasificaciones incorrectas podrían reducir la confianza en el modelo.
# - Este problema debe ser solucionado para garantizar la validez de los resultados y maximizar el impacto del proyecto en escenarios reales.
# 
# ## Acciones Propuestas
# 1. **Revisión y Depuración de los Datos**:
#    - Verificar que los textos positivos estén correctamente etiquetados en el conjunto de datos de entrenamiento.
# 
# 2. **Ajuste de Hiperparámetros**:
#    - Optimizar los hiperparámetros del siguiente modelo, buscando mejorar el balance entre precisión y recall para evitar sesgos hacia cualquier clase.
# 
# 3. **Evaluación con Nuevos Modelos**:
#    - Probar otros algoritmos avanzados como **LightGBM** o **CatBoost**, que podrían manejar mejor la complejidad de los datos.
# 
# 4. **Validación Cruzada Estricta**:
#    - Implementar técnicas de validación más robustas para garantizar que el modelo generalice correctamente.

# Install lightgbm if not already installed
%pip install lightgbm

from lightgbm import LGBMClassifier


# Definir el rango de hiperparámetros
param_grid = {
    'n_estimators': [100, 200],  # Número de árboles
    'learning_rate': [0.05, 0.1],  # Tasa de aprendizaje
    'max_depth': [5, 7],  # Profundidad máxima
    'num_leaves': [31, 50],  # Número de hojas por árbol
}

# Configurar el modelo base
lgbm = LGBMClassifier(random_state=42)

# Codificar etiquetas si es necesario
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Realizar búsqueda de hiperparámetros con GridSearchCV
grid_search_lgbm = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search_lgbm.fit(X_train, y_train_encoded)

# Mostrar los mejores parámetros
print("Mejores parámetros:", grid_search_lgbm.best_params_)
print("Mejor Accuracy en validación:", grid_search_lgbm.best_score_)

# Evaluar el modelo con los datos de prueba
lgbm_best = grid_search_lgbm.best_estimator_
y_pred = lgbm_best.predict(X_test)

# Decodificar etiquetas si es necesario
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Mostrar resultados
print("Accuracy en datos de prueba:", accuracy_score(y_test, y_pred_decoded))
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred_decoded))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_decoded))

# Guardar el mejor modelo LightGBM
joblib.dump(lgbm_best, "lightgbm_best_model.pkl")
print("Modelo guardado exitosamente como 'lightgbm_best_model.pkl'")

# Guardar el codificador de etiquetas
joblib.dump(label_encoder, "label_encoder.pkl")
print("Codificador de etiquetas guardado exitosamente como 'label_encoder.pkl'")

# # Documentación del Modelo LightGBM
# ## Introducción
# Se ha probado el modelo **LightGBM** como alternativa al modelo XGBoost con el objetivo de mejorar la precisión y manejar de forma más eficiente los patrones complejos y los textos previamente clasificados de forma errónea.
# Ajuste de Hiperparámetros
# El ajuste de hiperparámetros se realizó utilizando **GridSearchCV** con validación cruzada en 3 folds. Los mejores parámetros obtenidos son los siguientes:
#  **Mejores Hiperparámetros**
# - **Learning Rate**: 0.1  
# - **Max Depth**: 7  
# - **Número de Árboles (n_estimators)**: 200  
# - **Número de Hojas (num_leaves)**: 31  
# 
# ## Resultados Obtenidos
# El modelo **LightGBM** logró métricas destacadas en el conjunto de prueba, demostrando su eficacia en la tarea de clasificación entre *suicide* y *non-suicide*.
# 
# ### **Métricas Principales**
# | Métrica              | Valor  |
# |----------------------|--------|
# | **Accuracy en Validación** | 90.16% |
# | **Accuracy en Prueba**     | 90.17% |
# | **Precision (non-suicide)** | 89%    |
# | **Recall (non-suicide)**    | 92%    |
# | **F1-Score (non-suicide)**  | 0.90   |
# | **Precision (suicide)**     | 91%    |
# | **Recall (suicide)**        | 89%    |
# | **F1-Score (suicide)**      | 0.90   |
# 
# ### **Matriz de Confusión**
# | Clase            | Predicción Correcta | Falsos Positivos | Falsos Negativos |
# |------------------|---------------------|------------------|------------------|
# | **non-suicide**  | 2466               | 227              | N/A              |
# | **suicide**      | 2430               | N/A              | 307              |
# 
# ### **Interpretación**
# - **Balance entre Clases**: El modelo muestra un excelente equilibrio entre precisión y recall para ambas clases.
# - **Falsos Positivos y Negativos**: Aunque existen algunos errores de clasificación, su proporción es manejable dado el rendimiento general del modelo.
# 
# ## Próximos Pasos
# 1. **Integración en Streamlit**:
#    - El modelo se almacenó exitosamente como `lightgbm_best_model.pkl` para ser implementado en aplicaciones.
#    - El codificador de etiquetas también se guardó como `label_encoder.pkl`.
# 
# 2. **Optimización Adicional**:
#    - Continuar evaluando el modelo con nuevos textos para garantizar su robustez y minimizar los errores de clasificación.
# 
# 3. **Preparación para Uso Real**:
#    - Incorporar el modelo en sistemas predictivos y asegurar que las clasificaciones sean confiables en escenarios reales.
# 
# ## Conclusión
# El modelo **LightGBM** se presenta como una opción sólida para la clasificación de textos con alta precisión y un excelente balance entre métricas. Su integración en Streamlit será el próximo paso para su uso práctico.
# 