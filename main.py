"""
Página principal de la aplicación
Muestra información sobre la empresa, el sistema de bicicletas compartidas y cómo funciona
"""

import streamlit as st

def main_page():
    st.title("🚴 Sistema de Predicción de Destinos en Bicicleta")
    st.markdown("---")
    
    st.markdown("""
    ## 📋 Acerca del Sistema biciTRAN
    
    **biciTRAN** es el sistema automatizado de alquiler de bicicletas públicas de la ciudad de Mendoza, 
    Argentina. Es una opción de transporte público accesible y saludable que permite a los ciudadanos 
    movilizarse por la ciudad de manera sostenible.
    
    ### 🏢 Sobre el Sistema
    
    biciTRAN es un sistema público de bicicletas compartidas que forma parte de la red de transporte 
    urbano de Mendoza. El sistema está diseñado para facilitar la movilidad dentro de la ciudad, 
    ofreciendo una alternativa ecológica y práctica al transporte tradicional.
    
    ### 🚲 Cómo Funciona el Sistema
    
    El sistema opera a través de estaciones automatizadas distribuidas estratégicamente por la ciudad:
    
    1. **Estaciones y Bicicletas**: El sistema cuenta con múltiples estaciones ubicadas en puntos 
       estratégicos de la ciudad (plazas, instituciones públicas, centros de transporte, etc.). 
       Cada estación tiene capacidad para 10 bicicletas.
    
    2. **Proceso de Uso**: Los usuarios pueden:
       - Descargar la aplicación móvil "biciTRAN" o escanear el código QR en las estaciones
       - Escanear el código de la bicicleta usando Bluetooth o ingresar manualmente la matrícula
       - Retirar la bicicleta de la estación
       - Realizar su viaje y devolver la bicicleta en cualquier estación del sistema
    
    3. **Características Técnicas**: Las bicicletas cuentan con:
       - Sistema de traba U para seguridad
       - Candado Bluetooth con rastreo en tiempo real
       - Panel solar integrado para alimentar el sistema electrónico
       - Diseño robusto y funcional para uso urbano
    
    ### 📍 Cobertura y Ubicación
    
    El sistema cuenta con estaciones distribuidas en diferentes zonas de la ciudad, incluyendo:
    - Plazas principales (Plaza 25 de Mayo, Plaza del Soldado, Plaza Constituyentes, etc.)
    - Instituciones públicas (Municipalidad, Legislatura, Hospital Cullen, etc.)
    - Centros de transporte (Estación Mitre, Estación Belgrano)
    - Puntos de interés cultural y turístico (Teatro Municipal, Escuela Nacional, etc.)
    
    Cada estación tiene capacidad para 10 bicicletas, permitiendo un flujo continuo de usuarios 
    en diferentes horarios del día.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 🎯 Objetivo del Proyecto
    
    Este proyecto utiliza técnicas de machine learning para analizar y predecir patrones de comportamiento 
    en el uso del sistema biciTRAN. Los datos analizados provienen directamente del sistema operativo 
    de bicicletas compartidas, incluyendo información sobre:
    
    - **Viajes realizados**: origen, destino, fecha y hora de cada viaje
    - **Comportamiento de usuarios**: patrones de uso, frecuencia, preferencias horarias
    - **Patrones temporales**: distribución de viajes por hora, día de la semana y mes
    - **Patrones geográficos**: flujos de movimiento entre estaciones, destinos más frecuentes
    
    ### 📊 Análisis de Datos
    
    El objetivo principal es entender cómo se comportan los usuarios del sistema para:
    - Identificar patrones de movilidad urbana
    - Analizar la demanda en diferentes estaciones y horarios
    - Predecir destinos probables basados en características del viaje y del usuario
    - Optimizar la distribución de bicicletas entre estaciones
    - Mejorar la planificación y gestión del sistema
    
    Los datos utilizados en este análisis son reales y provienen del sistema operativo de biciTRAN, 
    proporcionando insights valiosos sobre el comportamiento de los usuarios y los patrones de uso 
    del sistema de bicicletas compartidas.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 📊 Navegación
    
    - **Inicio**: Esta página con información sobre biciTRAN y el sistema de bicicletas compartidas
    - **Explicación del Modelo**: Información técnica detallada sobre el modelo de machine learning, 
      características utilizadas, pesos y resultados
    - **Visualizaciones**: Gráficos interactivos mostrando análisis y hallazgos de los datos reales del sistema
    - **Modelo**: Interfaz para probar el modelo con datos nuevos y realizar predicciones de destinos
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## ℹ️ Información del Sistema
    
    Para más información sobre biciTRAN, puedes visitar el sitio oficial: 
    [https://bicitran.stmendoza.com](https://bicitran.stmendoza.com)
    
    ---
    
    **Proyecto de ciencia de datos** para análisis y predicción de patrones en sistemas de bicicletas compartidas.
    """)

