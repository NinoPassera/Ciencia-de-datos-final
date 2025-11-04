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
    
    El modelo utiliza **Random Forest** con **27 características** que incluyen:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🗺️ Características Geográficas
        - Coordenadas de origen (lat/lon)
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
    ## 🎯 Resultados del Modelo
    
    - **Accuracy**: 53.66% (mejora de +6.65% vs modelo original)
    - **Validación cruzada**: 47.02% (+/- 2.58%)
    - **Destinos únicos**: 89 estaciones
    - **Registros de entrenamiento**: 150,064
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 🔥 Características Más Importantes
    
    1. **Distancia promedio del usuario** (10.40%) - ⭐ La más predictiva!
    2. **Mes del año** (6.62%)
    3. **Hora de salida** (6.50%)
    4. **Longitud de origen** (5.82%)
    5. **Duración promedio** (5.82%)
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

