"""
Página del modelo - Inferencia y visualización
Permite al usuario ingresar datos nuevos y probar el modelo entrenado
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from lib import load_model, load_preprocessor, process_input

def model_page():
    st.title("🤖 Modelo de Predicción")
    st.markdown("---")
    
    # Cargar modelo
    modelo = load_model()
    if modelo is None:
        st.error("No se pudo cargar el modelo. Por favor, asegúrate de que el modelo esté en la carpeta static/")
        return
    
    # Cargar preprocessor
    preprocessor = load_preprocessor()
    if preprocessor is None:
        st.error("No se pudo cargar el preprocessor.")
        return
    
    # Mostrar información del modelo
    st.subheader("📊 Información del Modelo")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if hasattr(modelo, 'n_estimators'):
            st.metric("Número de Árboles", modelo.n_estimators)
        else:
            st.metric("Número de Árboles", "N/A")
    
    with col2:
        if hasattr(modelo, 'max_depth'):
            st.metric("Profundidad Máxima", modelo.max_depth if modelo.max_depth else "Sin límite")
        else:
            st.metric("Profundidad Máxima", "N/A")
    
    with col3:
        if hasattr(modelo, 'classes_'):
            st.metric("Destinos Únicos", len(modelo.classes_))
        else:
            st.metric("Destinos Únicos", "N/A")
    
    st.markdown("---")
    
    # Visualización de importancia de características
    st.subheader("🔍 Importancia de Características")
    
    if hasattr(modelo, 'feature_importances_'):
        importance = modelo.feature_importances_
        feature_names = modelo.feature_names_in_ if hasattr(modelo, 'feature_names_in_') else [f'feature_{i}' for i in range(len(importance))]
        
        imp_df = pd.DataFrame({
            'caracteristica': feature_names,
            'importancia': importance
        }).sort_values('importancia', ascending=False).head(15)
        
        chart = (
            alt.Chart(imp_df)
            .mark_bar()
            .encode(
                x=alt.X('importancia:Q', 
                       title='Importancia (Gini)', 
                       axis=alt.Axis(format='.4f')),
                y=alt.Y('caracteristica:N', 
                       sort='-x', 
                       title='Característica'),
                tooltip=[
                    alt.Tooltip('caracteristica:N', title='Característica'),
                    alt.Tooltip('importancia:Q', title='Importancia', format='.4f')
                ],
                color=alt.Color('importancia:Q', 
                               scale=alt.Scale(scheme='blues'), 
                               legend=None)
            )
            .properties(
                width=700,
                height=400,
                title='Top 15 Características Más Importantes'
            )
        )
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("El modelo no tiene información de importancia de características.")
    
    st.markdown("---")
    
    # Interfaz de inferencia
    st.subheader("🔮 Probar el Modelo con Datos Nuevos")
    st.markdown("""
    Ingresa los datos de un viaje para predecir el destino más probable.
    Si no tienes datos del historial del usuario, se usarán valores por defecto.
    """)
    
    # Formulario de entrada
    with st.form("form_prediccion"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📍 Datos Geográficos")
            origen_lat = st.number_input(
                "Latitud de Origen",
                value=-32.89,
                min_value=-90.0,
                max_value=90.0,
                step=0.00001,
                format="%.5f",
                help="Latitud de la estación de origen (ej: -32.89 para Mendoza)"
            )
            origen_lon = st.number_input(
                "Longitud de Origen",
                value=-68.84,
                min_value=-180.0,
                max_value=180.0,
                step=0.00001,
                format="%.5f",
                help="Longitud de la estación de origen (ej: -68.84 para Mendoza)"
            )
        
        with col2:
            st.markdown("### ⏰ Datos Temporales")
            hora_salida = st.slider(
                "Hora de Salida",
                min_value=0,
                max_value=23,
                value=8,
                help="Hora del día cuando inicia el viaje (0-23)"
            )
            dia_semana = st.selectbox(
                "Día de la Semana",
                options=[0, 1, 2, 3, 4, 5, 6],
                format_func=lambda x: ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'][x],
                help="Día de la semana (0=Lunes, 6=Domingo)"
            )
            mes = st.slider(
                "Mes",
                min_value=1,
                max_value=12,
                value=3,
                help="Mes del año (1-12)"
            )
        
        st.markdown("---")
        st.markdown("### 👤 Datos del Usuario (Opcionales)")
        st.markdown("*Si no conoces estos datos, déjalos en los valores por defecto*")
        
        col3, col4 = st.columns(2)
        
        with col3:
            viajes_totales = st.number_input(
                "Viajes Totales del Usuario",
                min_value=0,
                value=25,
                help="Número total de viajes que ha hecho el usuario"
            )
            semanas_activas = st.number_input(
                "Semanas Activas",
                min_value=1,
                value=10,
                help="Número de semanas diferentes en que el usuario ha usado el servicio"
            )
            duracion_promedio_min = st.number_input(
                "Duración Promedio (minutos)",
                min_value=0.0,
                value=20.0,
                step=0.1,
                help="Duración promedio de viajes del usuario en minutos"
            )
            distancia_promedio_usuario = st.number_input(
                "Distancia Promedio del Usuario",
                min_value=0.0,
                value=0.025,
                step=0.001,
                format="%.5f",
                help="Distancia promedio que recorre el usuario en sus viajes"
            )
        
        with col4:
            variedad_destinos = st.number_input(
                "Variedad de Destinos",
                min_value=1,
                value=8,
                help="Número de destinos únicos que visita el usuario"
            )
            variedad_origenes = st.number_input(
                "Variedad de Orígenes",
                min_value=1,
                value=5,
                help="Número de orígenes únicos que usa el usuario"
            )
            consistencia_horaria = st.number_input(
                "Consistencia Horaria",
                min_value=0.0,
                value=3.0,
                step=0.1,
                help="Desviación estándar de horas de viaje (menor = más consistente)"
            )
            dia_favorito = st.selectbox(
                "Día Favorito",
                options=[0, 1, 2, 3, 4, 5, 6],
                format_func=lambda x: ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'][x],
                help="Día de la semana favorito del usuario"
            )
        
        # Frecuencias semanales
        st.markdown("#### Frecuencias Semanales (Opcional)")
        col5, col6, col7 = st.columns(3)
        
        with col5:
            frecuencia_lunes = st.number_input("Viajes Lunes", min_value=0, value=5)
            frecuencia_martes = st.number_input("Viajes Martes", min_value=0, value=4)
            frecuencia_miercoles = st.number_input("Viajes Miércoles", min_value=0, value=4)
        
        with col6:
            frecuencia_jueves = st.number_input("Viajes Jueves", min_value=0, value=4)
            frecuencia_viernes = st.number_input("Viajes Viernes", min_value=0, value=5)
            frecuencia_sabado = st.number_input("Viajes Sábado", min_value=0, value=3)
        
        with col7:
            frecuencia_domingo = st.number_input("Viajes Domingo", min_value=0, value=2)
        
        # Botón de predicción
        submitted = st.form_submit_button("🔮 Predecir Destino", use_container_width=True)
    
    if submitted:
        st.markdown("---")
        st.subheader("📊 Resultados de la Predicción")
        
        # Preparar datos de entrada
        input_data = {
            'origen_lat': origen_lat,
            'origen_lon': origen_lon,
            'hora_salida': hora_salida,
            'dia_semana': dia_semana,
            'mes': mes,
            'viajes_totales': viajes_totales,
            'semanas_activas': semanas_activas,
            'viajes_por_semana': viajes_totales / semanas_activas if semanas_activas > 0 else 0,
            'duracion_promedio_min': duracion_promedio_min,
            'variedad_destinos': variedad_destinos,
            'variedad_origenes': variedad_origenes,
            'consistencia_horaria': consistencia_horaria,
            'distancia_promedio_usuario': distancia_promedio_usuario,
            'dia_favorito': dia_favorito,
            'frecuencia_lunes': frecuencia_lunes,
            'frecuencia_martes': frecuencia_martes,
            'frecuencia_miercoles': frecuencia_miercoles,
            'frecuencia_jueves': frecuencia_jueves,
            'frecuencia_viernes': frecuencia_viernes,
            'frecuencia_sabado': frecuencia_sabado,
            'frecuencia_domingo': frecuencia_domingo
        }
        
        try:
            # Procesar input
            X_processed = process_input(input_data, preprocessor)
            
            # Hacer predicción
            prediccion = modelo.predict(X_processed)[0]
            probabilidades = modelo.predict_proba(X_processed)[0]
            
            # Mostrar resultado principal
            st.success(f"🎯 **Destino Predicho**: {prediccion}")
            
            # Mostrar top 5 predicciones
            top_indices = np.argsort(probabilidades)[-5:][::-1]
            top_probs = probabilidades[top_indices]
            top_classes = modelo.classes_[top_indices]
            
            st.markdown("### Top 5 Destinos Más Probables")
            
            # Crear DataFrame para visualización
            pred_df = pd.DataFrame({
                'destino': top_classes,
                'probabilidad': top_probs
            })
            
            # Visualización con Altair
            chart = (
                alt.Chart(pred_df)
                .mark_bar()
                .encode(
                    x=alt.X('probabilidad:Q', 
                           title='Probabilidad', 
                           axis=alt.Axis(format='.2%'),
                           scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y('destino:N', 
                           sort='-x', 
                           title='Destino Predicho'),
                    tooltip=[
                        alt.Tooltip('destino:N', title='Destino'),
                        alt.Tooltip('probabilidad:Q', title='Probabilidad', format='.2%')
                    ],
                    color=alt.Color('probabilidad:Q', 
                                   scale=alt.Scale(scheme='greens'), 
                                   legend=None)
                )
                .properties(
                    width=700,
                    height=300,
                    title='Top 5 Destinos Más Probables'
                )
            )
            
            st.altair_chart(chart, width='stretch')
            
            # Tabla de resultados
            st.markdown("### Tabla de Resultados")
            pred_df['probabilidad'] = (pred_df['probabilidad'] * 100).round(2)
            pred_df.columns = ['Destino', 'Probabilidad (%)']
            st.dataframe(pred_df, width='stretch')
            
        except Exception as e:
            st.error(f"Error al procesar la predicción: {e}")
            st.exception(e)

