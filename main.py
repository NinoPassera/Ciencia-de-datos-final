"""
Página principal de la aplicación
Muestra información del proyecto y descripción general
"""

import streamlit as st

def main_page():
    st.title("🚴 Sistema de Predicción de Destinos en Bicicleta")
    st.markdown("---")
    
    st.markdown("""
    ## 📋 Descripción del Proyecto
    
    Esta aplicación implementa un sistema avanzado de machine learning para predecir destinos de viajes 
    en bicicleta basado en datos históricos de usuarios, características temporales, geográficas y patrones 
    de comportamiento.
    
    El modelo utiliza **Random Forest** con **29 características** que incluyen:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🗺️ Características Geográficas
        - Coordenadas de origen (lat/lon)
        - Coordenadas de destino favorito (lat/lon)
        - Zona geográfica
        - Capacidad de estación
        - Estaciones cercanas
        """)
    
    with col2:
        st.markdown("""
        ### ⏰ Características Temporales
        - Hora del día
        - Día de la semana
        - Mes del año
        - Período del día
        - Fin de semana / Hora pico
        """)
    
    with col3:
        st.markdown("""
        ### 👤 Características de Usuario
        - Historial de viajes
        - Frecuencia semanal
        - Duración promedio
        - Distancia promedio
        - Consistencia horaria
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 🎯 Resultados del Modelo (versión actual)
    
    - **Accuracy**: 60.64%
    - **OOB score**: 60.84%
    - **Destinos únicos (tras filtrado)**: 94
    - **Registros de entrenamiento**: 120,677 (test: 30,170)
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 🔥 Características Más Importantes (estimación)
    
    1. **Longitud Destino Favorito** (~29.27%)
    2. **Latitud Destino Favorito** (~28.22%)
    3. **Distancia promedio del usuario** (~6.28%)
    4. **Latitud de origen** (~4.98%)
    5. **Longitud de origen** (~4.84%)
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 📊 Navegación
    
    - **Inicio**: Esta página con información general del proyecto
    - **Visualizaciones**: Gráficos interactivos con Altair mostrando análisis y hallazgos
    - **Modelo**: Interfaz para probar el modelo con datos nuevos
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 👥 Autores
    
    Proyecto de ciencia de datos para predicción de destinos en sistemas de bicicletas compartidas.
    """)

