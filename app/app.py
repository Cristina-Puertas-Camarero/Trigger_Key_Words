import streamlit as st
import sys
import subprocess

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image

# Configurar la página 
st.set_page_config(
    page_title="Trigger Key Words App",
    page_icon="💡",
    layout="wide",  # Mantener pantalla amplia
)

# Función para cambiar el estilo de la aplicación
def agregar_estilo():
    st.markdown(
        """
        <style>
        /* Fondo de la aplicación */
        .stApp {
            background-color: #b3f0ff;  /* Fondo general */
        }

        /* Fondo completo de la barra lateral */
        div[data-testid="stSidebar"] {
            background-color: #9dc2c9;  /* Fondo azul */
        }

        /* Forzar fondo de la barra lateral completa */
        section[data-testid="stSidebar"] {
            background-color: #9dc2c9 !important;  /* Fondo azul específico */
            padding: 20px;  /* Espaciado */
        }

        /* Botones interactivos (color azul oscuro) */
        button {
            background-color: #003366 !important; /* Fondo azul oscuro */
            color: white !important; /* Texto en blanco */
            border-radius: 10px; /* Bordes redondeados */
            border: none !important;
        }

        /* Botones cuando se pasa el cursor por encima */
        button:hover {
            background-color: #0055a4 !important; /* Azul un poco más claro */
            color: white !important;
        }

        /* Links y otros elementos interactivos */
        a, a:visited {
            color: #003366 !important; /* Azul oscuro */
        }

        /* Links cuando se pasa el cursor por encima */
        a:hover {
            color: #0055a4 !important; /* Azul claro */
        }

        /* Ajustar los textos destacados en azul oscuro */
        .stMarkdown span {
            color: #003366 !important;  /* Texto resaltado */
            font-weight: bold;
        }

        /* Ajustar los textos en la barra lateral */
        section[data-testid="stSidebar"] .css-1lcbmhc {
            background-color: #9dc2c9 !important; /* Azul para contenedores internos */
            color: #404040 !important; /* Ajuste de texto */
        }

        /* Ajustar color de texto en general */
        section[data-testid="stSidebar"] .css-1d391kg {
            color: #404040 !important;  /* Texto oscuro */
            font-size: 18px;
            font-weight: bold;
        }

        /* Ajustar el color de la barra superior (botones predeterminados de Streamlit) */
        header {
            background-color: #b3f0ff !important;  /* Consistente con el fondo de la página */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Aplicar el estilo personalizado
agregar_estilo()

# Configuración global de gráficos
sns.set_palette("Set2")  # Usar la paleta Set2 en todos los gráficos


# Configurar las páginas en el menú lateral con iconos
pagina = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Presentation",
        "📊 Data Analysis",
        "🛠️ Machine Learning ",
        "🤖 Predictive Model",
        "📜 Conclusions and Reflections"
    ],
    index=0  # Seleccionar la primera página por defecto
)

    
    
    
# Pagina 1 Presentación
if pagina == "🏠 Presentation":
    st.title("Trigger Key Words")
    st.subheader("Analyzing patterns and proposing innovative strategies to tackle one of the most pressing challenges in public health.")

    # Imagen principal
    st.image("clipping_30Jebg_5288.png", caption="Autor Javirroyo", use_container_width=True)
    
    # Descripción principal con texto resaltado en azul oscuro
    st.markdown("""
    <span style='color:#003366; font-weight:bold;'>This project aims to address the issue of suicide from an academic perspective, analyzing relevant data to identify patterns and design effective prevention strategies.</span>
    <span style='color:#003366; font-weight:bold;'>This work does not constitute a formal study nor is it officially endorsed; it should be viewed solely as an educational initiative with informative purposes</span>

    Suicide poses a critical challenge in the social and public health domains, affecting millions of people worldwide. Statistics in Spain and Europe reveal an alarming reality that calls for heightened awareness and the development of more effective prevention strategies
    """, unsafe_allow_html=True)
    
    # Fuentes de datos como expanders
    with st.expander("Sources of Data Used"):
        st.markdown("""
        - **Suicide Notes:** Dataset obtained from GitHub  
        - **Related Tweets:** Public datasets extracted from Kaggle  
        - **European Data:** Eurostat data on suicide rates by country  
        - **National Data:** Information provided by the National Statistics Institute (INE)
        """)

    # Metodología ampliada: procesos de análisis y Machine Learning
    with st.expander("Methodology: Analysis and Machine Learning Processes"):
        st.markdown("""
        - **Data Preparation and Cleaning**:
            - Removal of null and duplicate values.
            - Column normalization and format standardization.
        - **Exploratory Data Analysis (EDA)**:
            - Visualization of distributions by age, gender, and region.
            - Identification of relevant patterns within the data.
        - **Predictive Machine Learning Modeling**:
            - Utilization of models such as SVM, Random Forest, and XGBoost.
            - Evaluation of key metrics including accuracy, precision, recall, and F1-score.
        - **Optimization**:
            - Hyperparameter tuning to enhance model performance.
        - **Results Visualization**:
            - Tables, charts, and confusion matrices for result interpretation.
        """)

    # Métricas actualizadas
    st.metric(label="Daily suicides in Spain", value="11 deaths per day")
    st.metric(label="Daily suicides in Europe", value="80 deaths per day")

    # Lista de teléfonos de ayuda
    with st.expander("Suicide Prevention Hotlines in Europe"):
        st.markdown("""
        - **Spain**: 024 (24/7 helpline).
        - **Germany**: 0800-111-0-111 or 0800-111-0-222.
        - **France**: 01 45 39 40 00.
        - **Italy**: 800 86 00 22.
        - **United Kingdom**: 0800 689 5652.
        - **Other European countries**: Please refer to the official prevention page in each region.
        """)

    # Imagen secundaria sólo al final
    st.image("643945903_221240743_1706x1676.png", caption="Not talking about suicide is a form of suicide", use_container_width=True)

# Página 2: Análisis de Datos
elif pagina == "📊 Data Analysis":
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Título de la página
    st.title("Interactive Exploration of Suicide Data")
    st.subheader("Visualizations, Metrics, and Conclusions")

    # Descripción de la sección
    st.markdown("""
    Welcome to the **Data Analysis section**. Here, you can explore interactive charts.
    """)

    # Intentar cargar los datos
    try:
        demografia_residencia = pd.read_csv("Demografía_Residencia_Suicidio_Limpio.csv")
        metodos_suicidio = pd.read_csv("Métodos_Suicidio_Demografía_Limpio.csv")
        tasas_europa = pd.read_csv("Tasas_Suicidio_Europa_Temporal_ICD10_Limpio.csv")
    except FileNotFoundError as e:
        st.error(f"Error: No se encontraron los archivos de datos necesarios. Detalles: {e}")
        st.stop()

    # Selección interactiva de análisis
    seleccion = st.radio(
        "Select the analysis you wish to explore:",
        ["Distribution in Spain", "Distribution by Age and Gender", "Suicide Methods", "European Trends"]
    )

    # Distribución por Comunidad Autónoma
    if seleccion == "Distribution in Spain":
        st.subheader("Suicide distribution in Spain by autonomous community")

        # Filtrar datos relevantes
        comunidades_sin_total = metodos_suicidio[metodos_suicidio["comunidad"] != "Total"]

        # Gráfico de barras por comunidad
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(
            data=comunidades_sin_total.groupby("comunidad")["total"].sum().reset_index(),
            x="comunidad", 
            y="total", 
            palette="Set2", 
            ax=ax
        )
        ax.set_title("Distribution of suicides by autonomous community", fontsize=16, weight="bold")
        ax.set_xlabel("Autonomous Community", fontsize=14)
        ax.set_ylabel("Total suicides", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

    # Distribución por Edad y Género
    elif seleccion == "Distribution by Age and Gender":
        st.subheader("Distribution by Age and Gender in Spain")

        # Filtrar datos
        edades_suicidios = demografia_residencia[demografia_residencia["comunidad"] != "Total"]

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(
            data=edades_suicidios,
            x="edad",
            y="total",
            hue="sexo",
            palette="Set2",
            ax=ax
        )
        ax.set_title("Distribution by Age and Gender in Spain", fontsize=16, weight="bold")
        ax.set_xlabel("Rango de Edad", fontsize=14)
        ax.set_ylabel("Total de Suicidios", fontsize=14)
        ax.legend(title="Género", fontsize=12, title_fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

    # Métodos de Suicidio
    elif seleccion == "Suicide methods":
        st.subheader("Suicide Methods by Community and Gender")

        # Filtrar datos para excluir "Total"
        metodos_filtrados = metodos_suicidio[metodos_suicidio["método"] != "Total"]

        # Selección interactiva
        comunidad_seleccionada = st.selectbox(
            "Select an autonomous community:",
            metodos_filtrados["comunidad"].unique()
        )
        datos_comunidad = metodos_filtrados[metodos_filtrados["comunidad"] == comunidad_seleccionada]

        # Gráfico de barras por método y género
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(
            data=datos_comunidad,
            x="método",
            y="total",
            hue="sexo",
            palette="Set2",
            ax=ax
        )
        ax.set_title(f"Métodos de Suicidio en {comunidad_seleccionada}", fontsize=16, weight="bold")
        ax.set_xlabel("Método", fontsize=14)
        ax.set_ylabel("Total de Suicidios", fontsize=14)
        ax.legend(title="Género", fontsize=12, title_fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

    # Tendencias Europeas
    elif seleccion == "European Trends":
        st.subheader("Suicide Trends in Europe")

        tasas_por_pais = tasas_europa.groupby("geo")["OBS_VALUE"].mean().reset_index()
        tasas_por_pais.columns = ["País", "Tasa Promedio"]
        tasas_por_pais = tasas_por_pais.sort_values(by="Tasa Promedio", ascending=False)

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(
            data=tasas_por_pais, 
            x="Tasa Promedio", 
            y="País", 
            palette="mako", 
            ax=ax
        )
        ax.set_title("Tasa Promedio de Suicidios en Europa", fontsize=16, weight="bold")
        ax.set_xlabel("Tasa Promedio (por 100,000 habitantes)", fontsize=14)
        ax.set_ylabel("País", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
    # Segunda parte: KPIs, análisis y conclusiones
    # Tabla de KPIs sin índices numéricos
    st.markdown("### 🌟 Key Indicators (KPIs)")
    kpis_data = [
        ["Suicide Rate in Men", "77.6%"],
        ["Suicide Rate in Women", "22.4%"],
        ["Highest Annual Rate (Lithuania)", "31.84%"],
        ["Lowest Annual Rate (Cyprus)", "2.95%"],
        ["Most Affected Autonomous Communities (Spain)", "Andalucía, Cataluña, Comunitat Valenciana"],
        ["Most Frequent Methods", "Hanging, Poisoning, Jumping from high places"],
        ["Most Affected Age Groups", "50-54 age y 75-79 age"]
    ]
    df_kpis = pd.DataFrame(kpis_data, columns=["Indicador", "Valor"])
    st.table(df_kpis)

    # Resumen del análisis
    st.markdown("### 📊 Analysis Summary ")
    st.markdown("""
    The analysis of suicide data in Spain and Europe has allowed us to identify key trends:

    - **Gender**: The proportion of suicides is significantly higher in men (77.6%) than in women (22.4%).
    - **Age**: The most affected age ranges are 50-54 years and 75-79 years.
    - **Region**: Andalucía, Cataluña, and Comunitat Valenciana are the autonomous communities with the highest number of suicides in Spain.
    - **Methods**: Hanging is the most used method, followed by poisoning and jumping from high places.
    - **Europe**: Lithuania has the highest average suicide rate in Europe (31.84%), while Cyprus has the lowest (2.95%).

    These findings are essential for understanding the factors contributing to suicide and guiding tailored prevention strategies.
    """)
    st.markdown("### 📊 Dashboard Power BI ")
    st.markdown("""The image below is a screenshot of the dashboard developed in Power BI. This dashboard presents key visualizations for data analysis.
    """)
    st.image("captura_powerbi.png", caption="You can access the file in Power BI from my Github repository", use_container_width=True)


    # Reflexión final y enlace con Machine Learning
    st.markdown("### 🌐 Final Reflection and Next Steps")
    st.markdown("""
    **A Global Issue: Suicide Prevention**

    Suicide is a **multifactorial global issue** that affects millions of people each year. The analyzed data has revealed critical patterns, allowing us to better understand its causes and characteristics. However, suicide prevention requires **coordinated efforts** by public and private entities, combining educational, social, and technological approaches.

    ---

    ### **Key Development: A Machine Learning Model**

    A groundbreaking tool we have developed is a **Machine Learning model** designed to analyze text for warning signs. This model could become a valuable resource for identifying high-risk cases and enabling early intervention.

    ---

    ### **What’s Next?**

    In the following section, we will explore:
    1. **Creation**
    2. **Optimization**
    3. **Validation**

    of this predictive model to showcase how it contributes to tailored prevention strategies.
    """)


# Página 3: Proceso de Machine Learning
elif pagina == "🛠️ Machine Learning ":
    
    # Título y descripción de la página
    st.title("Machine Learning Process")
    st.header("Model Preparation, Training and Comparison")

    st.markdown("""
    ### Machine Learning Process for Text Classification

    In this section, we detail the Machine Learning process for classifying texts as **suicidal** or **non-suicidal**. 
    This includes:
    1. **Data Preparation**: Organizing and preprocessing text data for analysis.
    2. **Model Training**: Using algorithms to create predictive models.
    3. **Benchmark Evaluation**: Comparing model performance and selecting the final model.

    These steps ensure a robust and accurate classification process.
    """)
    
    st.image("images.png", caption="talking about suicide is not suicide", width=200)
    
    
    # Data Preparation
    st.subheader("1. Data Preparation")
    st.markdown("""
    - **Data Loading**: Datasets of classified notes and tweets were combined into the categories `suicide` and `non-suicide`.
    - **Text Cleaning**:
        - Removal of special characters and null values.
        - Conversion to lowercase and elimination of textual noise.
    - **Numerical Representation**:
        - Transformation of textual data using techniques such as **TF-IDF**.
    """)
     # Divide the page into two columns
    col1, col2 = st.columns(2)
    
    # Column 1: Content
    with col1:
        st.markdown("### Stop Words Processing")
        
        st.markdown("""
        
        Before tokenization, we identify and remove **stop words**, 
        which are common words that do not provide meaningful insight for analysis. 

        This process helps reduce noise in the data and focuses on identifying significant patterns, 
        ultimately improving the model's effectiveness and precision.
        """)

    # In the second column, insert the image with specific dimensions
    with col2:
        st.markdown("### Data Processing")  # Optional title in the column
        # Load the image
        image = Image.open("output1.png")
        # Resize the image to ID card dimensions (approximately)
        image = image.resize((380, 220))  # Adjust dimensions as needed
        st.image(image, caption="Most common words in suicidal texts", use_container_width=False)
     # Data for suicidal and non-suicidal words
    suicidal_words = [
        ('want', 15457), ('feel', 12285), ('life', 12150), ('know', 11670), ('cant', 10923),
        ('ive', 10703), ('even', 7438), ('people', 7283), ('time', 6522), ('would', 6510),
        ('think', 6271), ('never', 5476), ('help', 5413), ('much', 5313), ('anymore', 5071),
        ('friends', 5000), ('die', 4789), ('years', 4594), ('kill', 4420), ('suicide', 4367),
        ('fucking', 4323), ('end', 4227), ('day', 4185), ('way', 4040), ('live', 4035),
        ('anything', 3968)
    ]

    non_suicidal_words = [
        ('know', 3623), ('people', 3340), ('want', 2798), ('time', 2307),
        ('feel', 2214), ('would', 2207), ('even', 2192), ('think', 2189),
        ('got', 2085), ('ive', 2031), ('day', 1983), ('friends', 1909), ('cant', 1857),
        ('help', 1607)
        
        
    ]

    # Convert data to dataframes
    suicidal_df = pd.DataFrame(suicidal_words, columns=['Word', 'Frequency'])
    non_suicidal_df = pd.DataFrame(non_suicidal_words, columns=['Word', 'Frequency'])

    # Merge data into one dataframe for comparison
    comparison_df = pd.merge(
        suicidal_df.rename(columns={'Frequency': 'Suicidal'}),
        non_suicidal_df.rename(columns={'Frequency': 'Non-Suicidal'}),
        on='Word',
        how='outer'
    ).fillna(0)

    # Plot comparative bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(comparison_df['Word'], comparison_df['Suicidal'], label='Suicidal', color='#66c2a5', alpha=0.8)  # Turquoise-like (Set2)
    ax.bar(comparison_df['Word'], comparison_df['Non-Suicidal'], label='Non-Suicidal', color='#fc8d62', alpha=0.8)  # Salmon-like (Set2)
    ax.set_xlabel("Words")
    ax.set_ylabel("Frequency")
    ax.set_title("Word Frequency Comparison: Suicidal vs Non-Suicidal Texts")
    ax.legend()
    plt.xticks(rotation=45)

    # Display results
    st.pyplot(fig)
    # Evaluated Models
    st.subheader("2. Evaluated Models")
    st.markdown("""
    The following models were evaluated for this problem:
    - **K-Nearest Neighbors (KNN)**: Low performance due to class imbalance and high dimensionality.
    - **Random Forest**: Excellent balance between precision and recall.
    - **Naive Bayes**: Reasonable performance, but inferior to Random Forest and other advanced models.
    - **XGBoost**: Very good performance, with high accuracy and generalization.
    - **LightGBM**: Selected as the final model for its superior overall performance in terms of precision, recall, and efficiency.
    """)

    # Model Metrics Comparison
    st.subheader("3. Model Metrics Comparison")
    metrics_df = pd.DataFrame({
    "Model": ["KNN", "Random Forest", "Naive Bayes", "XGBoost", "LightGBM"],
    "Accuracy (%)": [44, 89.04, 86.43, 89.57, 90.17],
    "Macro F1-Score (%)": [31, 89.00, 84.50, 89.50, 90.00],
    "Weighted Avg (%)": [28, 88.00, 85.00, 89.00, 90.00]
    })

    st.markdown("### Performance Comparison Table")
    st.dataframe(metrics_df.style.format({
        "Accuracy (%)": "{:.2f}",
        "Macro F1-Score (%)": "{:.2f}",
        "Weighted Avg (%)": "{:.2f}"
    }))

    # Comparative Models: Bar Chart
    st.markdown("### Model Comparison: Bar Chart")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Percentage"), 
        x="Model",
        y="Percentage", 
        hue="Metric",
        palette="Set2"
    )
    ax.set_title("Model Comparison by Metrics", fontsize=16, weight="bold")
    ax.set_ylabel("Percentage", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    plt.xticks(rotation=0, fontsize=12)
    st.pyplot(fig)
    # Legend block for the graph
    st.markdown("""
    **Legend**:  
    - **Accuracy**: Represents the overall correctness of the model's predictions. .  
    - **Macro F1-Score**: Averages the F1-scores of all classes, treating them equally regardless of imbalance.   
    - **Weighted Average**: The F1-score calculated with consideration for class weights.   
    """)
    # Prediction Distribution for Selected Model (LightGBM)
    st.subheader("4. Prediction Distribution for LightGBM Model")
    predictions = pd.DataFrame({
        "Class": ["Suicidal", "Non-Suicidal"],
        "Count": [2430, 2466]  # Data from confusion matrix
    })

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    sns.barplot(data=predictions, x="Class", y="Count", palette="viridis", ax=ax2)
    ax2.set_title("Prediction Distribution of LightGBM Model", fontsize=16)
    ax2.set_xlabel("Class", fontsize=12)
    ax2.set_ylabel("Case Count", fontsize=12)
    plt.xticks(rotation=0, fontsize=12)
    st.pyplot(fig2)

    # Predicted Class Proportions
    st.markdown("### Predicted Class Proportions by LightGBM Model")
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.pie(predictions["Count"], labels=predictions["Class"], autopct="%1.1f%%", startangle=90, colors=["#FF9999", "#66B2FF"])
    ax3.set_title("Class Proportion Predictions", fontsize=16)
    st.pyplot(fig3)

    # Conclusions
    st.subheader("5. Process Conclusions")
    st.markdown("""
    - **Selected Model:** LightGBM, chosen for its superior performance with **90.17% accuracy**, and excellent balance between precision and recall.
    - Robust data cleaning and numerical representation steps were implemented, enabling effective analysis of complex data.
    - This model shows great potential for use in predictive systems for suicide prevention.
    """)

# Página 4: Predictive Model
elif pagina == "🤖 Predictive Model":
    import os
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    import streamlit as st

    # Título y descripción de la página
    st.title("Predictive Risk Model")
    st.header("Text Classification: Suicidal vs. Non-Suicidal")

    st.markdown("""
    In this section, you can analyze input texts, and the predictive model will classify the content as **suicidal** or **non-suicidal** based on prior training.
    """)

    # Cargar el modelo y el vectorizador TF-IDF
    try:
        modelo_path = "lightgbm_best_model.pkl"  # Nuevo modelo entrenado LightGBM
        vectorizador_path = "vectorizador_tfidf.pkl"

        # Verificar existencia de archivos
        if not os.path.exists(modelo_path) or not os.path.exists(vectorizador_path):
            raise FileNotFoundError("No se encontraron los archivos 'lightgbm_best_model.pkl' o 'vectorizador_tfidf.pkl'. Por favor, verifica las rutas.")

        # Cargar modelo ajustado y vectorizador
        modelo_cargado = joblib.load(modelo_path)
        vectorizador = joblib.load(vectorizador_path)

        st.success("Model and vectorizer loaded successfully.")

    except Exception as e:
        st.error(f"Error al cargar el modelo o vectorizador: {e}")
        st.stop()

    # Área para ingresar texto
    texto_usuario = st.text_area("Write the text you want to analyze:", height=150)

    # Botón para realizar predicción
    if st.button("Analyze text"):
        if texto_usuario.strip():  # Verificar que el texto no esté vacío
            try:
                # Preprocesar el texto ingresado
                texto_procesado = texto_usuario.lower().strip()  # Convertir a minúsculas y eliminar espacios extra
                
                # Vectorizar el texto
                texto_vectorizado = vectorizador.transform([texto_procesado])

                # Realizar predicción con el modelo cargado
                prediccion = modelo_cargado.predict(texto_vectorizado)

                # Interpretar resultado de la predicción
                etiqueta_predicha = "Suicide" if prediccion[0] == 1 else "No Suicide"

                # Mostrar resultado
                if etiqueta_predicha == "Suicide":
                    st.error("⚠️ **The entered text is classified as SUICIDE.**")
                    st.markdown("""
                    ### Please seek help immediately.
                    - **Helpline:** 024 (Spain)
                    - **Toll-free number:** Available 24/7.
                    - **Additional information:** Contact mental health professionals or emergency services in your area.
                    """)
                else:
                    st.success("The entered text is classified as: **Non-Suicidal**.")
            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")
        else:
            st.warning("Por favor, introduce un texto válido para analizar.")

    # Mostrar imagen
    st.image("obsession.jpg", caption="Imagen de Javirroyo", use_container_width=True)




# Página 5: Reporte Final
elif pagina == "📜 Conclusions and Reflections":

    st.title("📜 Conclusions and Reflections")
    st.header("Opportunities for Improvement and Appreciation")
    
    # Reflection on the Project
    st.markdown("### 🌟 Reflections and Future Improvements")
    st.markdown("""
    This project represents a significant step in leveraging technology for suicide prevention, yet there are areas for improvement:
    - **Dataset Expansion**: Incorporate global data and multilingual texts to improve representativeness.
    - **Model Optimization**: Experiment with new deep learning algorithms to capture more complex patterns.
    - **Real-Time Implementation**: Develop a practical application to analyze social media posts and texts to alert possible risks.
    - **Interdisciplinary Collaboration**: Partner with mental health and ethics professionals to ensure the appropriate use of technology.
    """)

    st.image("images (2).png", caption="Prevention is everyone’s responsibility", width=200)
    
    # Subtitle
    st.subheader("📲 A Step Towards Suicide Prevention")

    # Description with Visual Formatting
    st.markdown("""
    **The ultimate goal of this project** is to implement our algorithm in real-time across social media platforms such as:  

    - 📱 **WhatsApp**  
    - 📸 **Instagram**  

    This would enable effective detection of warning signs related to suicide. Our approach aims to:  
    - **Identify alerts**, not only for the affected individuals themselves but also for their **friends and family**.  
    - **Enable early intervention** in critical situations.  
    - **Help prevent potential tragedies** before they unfold.  

    With the power of **data analysis** and **machine learning**, we strongly believe in technology's potential to make a positive impact and save lives. 🌍💡
    """)

    # Academic Disclaimer
    st.markdown("### 📚 Academic Disclaimer")
    st.markdown("""
    This project was developed as part of an academic endeavor and **does not have medical backing**. 
    The data and conclusions should not be considered an official diagnosis or treatment. Its purpose is to contribute to research and learning in the field of data analysis and Machine Learning.
    """)

    # Closing and Acknowledgments
    st.markdown("### 🙏 Acknowledgments")
    st.markdown("""
    This project is my **final project at Ironhack**, the result of months of learning and dedication. I want to express my gratitude to my instructors, colleagues, and the Ironhack team for all the support and shared knowledge. This achievement is also thanks to the collaborative and enriching environment of the program.
    """)

    # Final Contact
    st.markdown("### Connect with Me:")
    st.markdown("""
    **Cristina Puertas Camarero**  
    **cris.puertascamarero@gmail.com**  
    - **GitHub:** [Cristina-Puertas-Camarero](https://github.com/Cristina-Puertas-Camarero)  
    - **LinkedIn:** [Cristina Puertas Camarero](https://www.linkedin.com/in/cristina-puertas-camarero-8955a6349/)  
    """)
    
    # Display an image with a size similar to a two-euro coin
    st.image("IMG_2685.jpg", caption="Cristina", width=100)


# Highlighted Help Phone Line
st.sidebar.markdown("### Suicide Prevention Hotline in Spain: **024 (free and available 24/7)**")
st.sidebar.markdown("Check support lines in other countries if you need help.")






