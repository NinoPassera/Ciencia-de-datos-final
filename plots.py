"""
Página de visualizaciones interactivas con Altair
Implementa 2-3 visualizaciones aplicando principios de gramática de gráficos
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
# from lib import load_model  # No se usa directamente

def plots_page():
    st.title("📊 Visualizaciones Interactivas")
    st.markdown("---")
    
    # Intentar cargar modelo para obtener importancia (opcional)
    try:
        from lib import load_model
        modelo = load_model()
    except:
        modelo = None
        st.warning("No se pudo cargar el modelo para mostrar importancia. Las visualizaciones de datos siguen disponibles.")
    
    # Cargar datos del dataset
    try:
        # Intentar diferentes rutas posibles
        dataset_paths = [
            "dataset_modelo_final.csv",
            "../prediccion/dataset_modelo_final.csv",
            "../../prediccion/dataset_modelo_final.csv"
        ]
        df = None
        for path in dataset_paths:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    st.success(f"Dataset cargado: {len(df):,} registros desde {path}")
                    break
            except FileNotFoundError:
                continue
        
        if df is None:
            st.warning("⚠️ No se encontró el dataset. Las visualizaciones de datos no estarán disponibles.")
            st.info("💡 Puedes copiar el dataset desde la carpeta prediccion/ a esta carpeta o ajustar la ruta.")
            # Continuar sin dataset - mostrar solo importancia del modelo si está disponible
            if modelo is not None:
                st.markdown("---")
                st.markdown("## 1. Importancia de Características del Modelo")
                if hasattr(modelo, 'feature_importances_'):
                    importance = modelo.feature_importances_
                    feature_names = modelo.feature_names_in_ if hasattr(modelo, 'feature_names_in_') else [f'feature_{i}' for i in range(len(importance))]
                    
                    imp_df = pd.DataFrame({
                        'caracteristica': feature_names,
                        'importancia': importance
                    }).sort_values('importancia', ascending=False).head(15)
                    
                    chart1 = (
                        alt.Chart(imp_df)
                        .mark_bar()
                        .encode(
                            x=alt.X('importancia:Q', 
                                   title='Importancia (Gini)', 
                                   axis=alt.Axis(format='.4f')),
                            y=alt.Y('caracteristica:N', 
                                   sort='-x', 
                                   title='Característica',
                                   axis=alt.Axis(labelLimit=1000)),
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
                            height=500,
                            title='Top 15 Características Más Importantes del Modelo'
                        )
                    )
                    st.altair_chart(chart1, width='stretch')
            return
    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")
        return
    
    # Visualización 1: Importancia de Características
    st.markdown("## 1. Importancia de Características del Modelo")
    st.markdown("""
    Este gráfico muestra las características más importantes para el modelo Random Forest.
    La importancia se calcula como la reducción promedio de impureza que aporta cada característica.
    """)
    
    if hasattr(modelo, 'feature_importances_'):
        importance = modelo.feature_importances_
        feature_names = modelo.feature_names_in_ if hasattr(modelo, 'feature_names_in_') else [f'feature_{i}' for i in range(len(importance))]
        
        imp_df = pd.DataFrame({
            'caracteristica': feature_names,
            'importancia': importance
        }).sort_values('importancia', ascending=False).head(15)
        
        chart1 = (
            alt.Chart(imp_df)
            .mark_bar()
            .encode(
                x=alt.X('importancia:Q', 
                       title='Importancia (Gini)', 
                       axis=alt.Axis(format='.4f'),
                       scale=alt.Scale(domain=[0, imp_df['importancia'].max() * 1.1])),
                y=alt.Y('caracteristica:N', 
                       sort='-x', 
                       title='Característica',
                       axis=alt.Axis(labelLimit=1000)),
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
                height=500,
                title='Top 15 Características Más Importantes del Modelo'
            )
        )
        
        st.altair_chart(chart1, width='stretch')
    else:
        st.info("El modelo no tiene información de importancia de características.")
    
    st.markdown("---")
    
    # Visualización 2: Distribución Temporal de Viajes
    st.markdown("## 2. Distribución Temporal de Viajes")
    st.markdown("""
    Análisis de patrones temporales en los viajes. Muestra la distribución de viajes por hora del día
    y día de la semana, revelando patrones de comportamiento de los usuarios.
    """)
    
    # Crear visualización de distribución por hora
    hora_counts = df['hora_salida'].value_counts().sort_index().reset_index()
    hora_counts.columns = ['hora', 'cantidad_viajes']
    
    chart2a = (
        alt.Chart(hora_counts)
        .mark_area(opacity=0.7, interpolate='monotone')
        .encode(
            x=alt.X('hora:Q', 
                   title='Hora del Día (0-23)', 
                   axis=alt.Axis(format='d'),
                   scale=alt.Scale(domain=[0, 23])),
            y=alt.Y('cantidad_viajes:Q', 
                   title='Cantidad de Viajes',
                   axis=alt.Axis(format=',')),
            tooltip=[
                alt.Tooltip('hora:Q', title='Hora', format='d'),
                alt.Tooltip('cantidad_viajes:Q', title='Viajes', format=',')
            ],
            color=alt.value('#4A90E2')
        )
        .properties(
            width=700,
            height=300,
            title='Distribución de Viajes por Hora del Día'
        )
    )
    
    st.altair_chart(chart2a, width='stretch')
    
    # Crear visualización de distribución por día de semana
    dias_nombres = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    dia_counts = df['dia_semana'].value_counts().sort_index().reset_index()
    dia_counts.columns = ['dia_semana', 'cantidad_viajes']
    dia_counts['dia_nombre'] = dia_counts['dia_semana'].map(lambda x: dias_nombres[x] if x < 7 else 'Otro')
    
    chart2b = (
        alt.Chart(dia_counts)
        .mark_bar()
        .encode(
            x=alt.X('dia_nombre:N', 
                   title='Día de la Semana',
                   sort=dias_nombres),
            y=alt.Y('cantidad_viajes:Q', 
                   title='Cantidad de Viajes',
                   axis=alt.Axis(format=',')),
            tooltip=[
                alt.Tooltip('dia_nombre:N', title='Día'),
                alt.Tooltip('cantidad_viajes:Q', title='Viajes', format=',')
            ],
            color=alt.Color('cantidad_viajes:Q', 
                          scale=alt.Scale(scheme='viridis'), 
                          legend=None)
        )
        .properties(
            width=700,
            height=300,
            title='Distribución de Viajes por Día de la Semana'
        )
    )
    
    st.altair_chart(chart2b, width='stretch')
    
    st.markdown("---")
    
    # Visualización 3: Análisis Geográfico - Top Destinos
    st.markdown("## 3. Top Destinos Más Frecuentes")
    st.markdown("""
    Análisis de los destinos más populares en el sistema. Muestra las estaciones destino más frecuentes,
    lo que ayuda a entender los patrones de movilidad y demanda en diferentes zonas.
    """)
    
    # Top 15 destinos
    top_destinos = df['destino'].value_counts().head(15).reset_index()
    top_destinos.columns = ['destino', 'cantidad_viajes']
    top_destinos['porcentaje'] = (top_destinos['cantidad_viajes'] / len(df) * 100).round(2)
    
    chart3 = (
        alt.Chart(top_destinos)
        .mark_bar()
        .encode(
            x=alt.X('cantidad_viajes:Q', 
                   title='Cantidad de Viajes',
                   axis=alt.Axis(format=',')),
            y=alt.Y('destino:N', 
                   sort='-x', 
                   title='Estación Destino'),
            tooltip=[
                alt.Tooltip('destino:N', title='Destino'),
                alt.Tooltip('cantidad_viajes:Q', title='Viajes', format=','),
                alt.Tooltip('porcentaje:Q', title='Porcentaje', format='.2f')
            ],
            color=alt.Color('cantidad_viajes:Q', 
                          scale=alt.Scale(scheme='reds'), 
                          legend=None)
        )
        .properties(
            width=700,
            height=500,
            title='Top 15 Estaciones Destino Más Frecuentes'
        )
    )
    
    st.altair_chart(chart3, width='stretch')
    
    # Estadísticas adicionales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Viajes", f"{len(df):,}")
    with col2:
        st.metric("Destinos Únicos", f"{df['destino'].nunique()}")
    with col3:
        st.metric("Usuarios Únicos", f"{df.get('Usuario_key', pd.Series()).nunique() if 'Usuario_key' in df.columns else 'N/A'}")
    with col4:
        st.metric("Destino Más Frecuente", f"{top_destinos.iloc[0]['destino'][:20]}...")

