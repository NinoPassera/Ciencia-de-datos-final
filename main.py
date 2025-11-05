"""
Página principal de la aplicación
Muestra información sobre la empresa, el sistema de bicicletas compartidas y cómo funciona
"""

import streamlit as st

def main_page():
    st.title("🚴 Sistema de Predicción de Destinos en Bicicleta")
    st.markdown("---")
    
    st.markdown("""
    ## 📋 Acerca del Sistema de Bicicletas Compartidas
    
    [Aquí va la información sobre la empresa y el sistema de bicicletas compartidas]
    
    ### 🏢 Sobre la Empresa
    
    [Descripción de la empresa, su misión, valores, etc.]
    
    ### 🚲 Cómo Funciona el Sistema
    
    [Explicación de cómo funciona el sistema de bicicletas compartidas:
    - Cómo los usuarios toman prestadas las bicicletas
    - Cómo funcionan las estaciones
    - Proceso de devolución
    - Tarifas y membresías
    - Ubicación de las estaciones
    - etc.]
    
    ### 📍 Cobertura y Ubicación
    
    [Información sobre la cobertura geográfica del sistema, número de estaciones, etc.]
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 🎯 Objetivo del Proyecto
    
    Este proyecto utiliza técnicas de machine learning para predecir el destino de viajes en bicicleta,
    ayudando a optimizar la distribución de bicicletas y mejorar la experiencia del usuario.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 📊 Navegación
    
    - **Inicio**: Esta página con información sobre la empresa y el sistema de bicicletas
    - **Explicación del Modelo**: Información técnica detallada sobre el modelo, características, pesos y resultados
    - **Visualizaciones**: Gráficos interactivos mostrando análisis y hallazgos de los datos
    - **Modelo**: Interfaz para probar el modelo con datos nuevos y realizar predicciones
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 👥 Autores
    
    Proyecto de ciencia de datos para predicción de destinos en sistemas de bicicletas compartidas.
    """)

