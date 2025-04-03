# Funcion de visualización para explorar dataset.
def explorar_dataset(data, titulo):
    """
    Muestra las principales características de un dataset.

    Parámetros:
    - data (DataFrame): Dataset que se quiere explorar.
    - titulo (str): Título descriptivo para el dataset.

    Retorna:
    Un resumen con la información principal del dataset.
    """
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

# Función para validar datasets
def validar_datos(data, nombre):
    print(f"===== Validando {nombre} =====")
    print(f"Dimensiones: {data.shape}")
    print(f"Columnas: {data.columns.tolist()}")
    print(f"Primeras filas:\n{data.head()}")
    print(f"Valores nulos:\n{data.isnull().sum()}")
    print("=" * 50)


# Función para analizar valores faltantes
def analizar_valores_faltantes(data, titulo):
    print(f"===== Análisis de Valores Faltantes: {titulo} =====")


# Limpiar texto
def limpiar_texto(texto):
    texto = texto.lower()  # Convertir a minúsculas
    texto = re.sub(r"[^a-zA-Z0-9\s]", "", texto)  # Eliminar caracteres especiales
    texto = texto.strip()  # Eliminar espacios extra
    return texto

# Función para separar edades
def separar_edades(edad):
    try:
        if "a" in edad: 
            partes = edad.replace("De ", "").replace(" años", "").split(" a ")
            return int(partes[0]), int(partes[1])
        elif "Menores" in edad:  
            return 0, 15
        elif "Todas las edades" in edad:  
            return None, None
        else:
            return int(edad), int(edad)  
    except ValueError:
        return None, None  


# Función para limpiar outliers
def eliminar_outliers(data, columnas):
    data_filtrado = data.copy()
    
    for columna in columnas:
        Q1 = data[columna].quantile(0.25)
        Q3 = data[columna].quantile(0.75) 
        IQR = Q3 - Q1
        
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        data_filtrado = data_filtrado[(data_filtrado[columna] >= limite_inferior) & 
                                       (data_filtrado[columna] <= limite_superior)]
    
    return data_filtrado

# Función para filtrar las stop words del contenido
def quitar_stop_words(texto):
    palabras = texto.split()  
    palabras_filtradas = [palabra for palabra in palabras if palabra not in stop_words]
    return " ".join(palabras_filtradas)  