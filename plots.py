"""
Página de visualizaciones interactivas con Altair
Implementa 2-3 visualizaciones aplicando principios de gramática de gráficos
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import seaborn as sns
import matplotlib.pyplot as plt
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
                    
                    # Mapeo de nombres de features a nombres descriptivos en español
                    nombres_descriptivos = {
                        'lat_destino_favorito': 'Latitud Destino Favorito',
                        'lon_destino_favorito': 'Longitud Destino Favorito',
                        'destino_favorito_encoded': 'Destino Favorito (Codificado)',
                        'origen_lat': 'Latitud Origen',
                        'origen_lon': 'Longitud Origen',
                        'hora_salida': 'Hora de Salida',
                        'dia_semana': 'Día de la Semana',
                        'mes': 'Mes',
                        'viajes_totales': 'Viajes Totales',
                        'semanas_activas': 'Semanas Activas',
                        'viajes_por_semana': 'Viajes por Semana',
                        'duracion_promedio_min': 'Duración Promedio (min)',
                        'periodo_dia_numerico': 'Período del Día',
                        'es_fin_semana': 'Es Fin de Semana',
                        'es_hora_pico': 'Es Hora Pico',
                        'zona_origen': 'Zona Origen',
                        'capacidad_origen': 'Capacidad Estación Origen',
                        'estaciones_cercanas_origen': 'Estaciones Cercanas Origen',
                        'variedad_destinos': 'Variedad Destinos',
                        'variedad_origenes': 'Variedad Orígenes',
                        'consistencia_horaria': 'Consistencia Horaria',
                        'distancia_promedio_usuario': 'Distancia Promedio Usuario',
                        'dia_favorito': 'Día Favorito',
                        'frecuencia_lunes': 'Frecuencia Lunes',
                        'frecuencia_martes': 'Frecuencia Martes',
                        'frecuencia_miercoles': 'Frecuencia Miércoles',
                        'frecuencia_jueves': 'Frecuencia Jueves',
                        'frecuencia_viernes': 'Frecuencia Viernes',
                        'frecuencia_sabado': 'Frecuencia Sábado',
                        'frecuencia_domingo': 'Frecuencia Domingo'
                    }
                    
                    # Aplicar nombres descriptivos
                    feature_names_descriptivos = [nombres_descriptivos.get(name, name) for name in feature_names]
                    
                    imp_df = pd.DataFrame({
                        'caracteristica': feature_names_descriptivos,
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
        
        # Mapeo de nombres de features a nombres descriptivos en español
        nombres_descriptivos = {
            'lat_destino_favorito': 'Latitud Destino Favorito',
            'lon_destino_favorito': 'Longitud Destino Favorito',
            'destino_favorito_encoded': 'Destino Favorito (Codificado)',
            'origen_lat': 'Latitud Origen',
            'origen_lon': 'Longitud Origen',
            'hora_salida': 'Hora de Salida',
            'dia_semana': 'Día de la Semana',
            'mes': 'Mes',
            'viajes_totales': 'Viajes Totales',
            'semanas_activas': 'Semanas Activas',
            'viajes_por_semana': 'Viajes por Semana',
            'duracion_promedio_min': 'Duración Promedio (min)',
            'periodo_dia_numerico': 'Período del Día',
            'es_fin_semana': 'Es Fin de Semana',
            'es_hora_pico': 'Es Hora Pico',
            'zona_origen': 'Zona Origen',
            'capacidad_origen': 'Capacidad Estación Origen',
            'estaciones_cercanas_origen': 'Estaciones Cercanas Origen',
            'variedad_destinos': 'Variedad Destinos',
            'variedad_origenes': 'Variedad Orígenes',
            'consistencia_horaria': 'Consistencia Horaria',
            'distancia_promedio_usuario': 'Distancia Promedio Usuario',
            'dia_favorito': 'Día Favorito',
            'frecuencia_lunes': 'Frecuencia Lunes',
            'frecuencia_martes': 'Frecuencia Martes',
            'frecuencia_miercoles': 'Frecuencia Miércoles',
            'frecuencia_jueves': 'Frecuencia Jueves',
            'frecuencia_viernes': 'Frecuencia Viernes',
            'frecuencia_sabado': 'Frecuencia Sábado',
            'frecuencia_domingo': 'Frecuencia Domingo'
        }
        
        # Aplicar nombres descriptivos
        feature_names_descriptivos = [nombres_descriptivos.get(name, name) for name in feature_names]
        
        imp_df = pd.DataFrame({
            'caracteristica': feature_names_descriptivos,
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
    
    st.markdown("---")
    
    # Visualización 4: Matriz Origen-Destino (Heatmap)
    st.markdown("## 4. Matriz de Probabilidad Origen-Destino")
    st.markdown("""
    Este heatmap muestra la probabilidad de que un viaje desde una estación origen termine en una estación destino.
    Los valores representan el porcentaje de viajes desde cada origen hacia cada destino.
    Solo se muestran las estaciones más frecuentes (top 15) para mejor legibilidad.
    """)
    
    # Verificar que existen las columnas origen y destino
    if 'origen' not in df.columns or 'destino' not in df.columns:
        st.warning("⚠️ El dataset no contiene las columnas 'origen' y 'destino' necesarias para este gráfico.")
        st.info("💡 Ejecuta crear_dataset_final.py para generar el dataset con estas columnas.")
    else:
        # Crear DataFrame con origen y destino (similar a df_viajes)
        df_viajes = df[['origen', 'destino']].copy()
        
        # Top 15 por origen y destino
        top_origen = df_viajes["origen"].value_counts().head(15).index
        top_destino = df_viajes["destino"].value_counts().head(15).index
        
        # Matriz Origen x Destino, filtrada a top-15
        matriz_top = pd.crosstab(df_viajes["origen"], df_viajes["destino"])
        matriz_top = matriz_top.loc[matriz_top.index.intersection(top_origen), 
                                    matriz_top.columns.intersection(top_destino)]
        
        # Orden por totales (ayuda a ver estructura)
        filas = matriz_top.sum(axis=1).sort_values(ascending=False).index
        cols = matriz_top.sum(axis=0).sort_values(ascending=False).index
        matriz_top = matriz_top.loc[filas, cols]
        
        # Usar el mismo orden en ambos ejes
        # Orden común: respetá el orden de columnas (cols) y quedate con las que también están en filas
        orden_comun = cols.intersection(filas, sort=False)
        
        # Si no hay intersección, usar las que hay
        if len(orden_comun) == 0:
            orden_comun = filas.intersection(cols, sort=False)
        
        # Reindexar filas y columnas con el mismo orden (cuadrada y sincronizada)
        if len(orden_comun) > 0:
            matriz_sync = matriz_top.reindex(index=orden_comun, columns=orden_comun, fill_value=0)
        else:
            # Si no hay intersección, usar todas las que hay pero ordenadas
            matriz_sync = matriz_top.copy()
        
        # Normalización por fila (probabilidad de destino dado origen)
        matriz_norm = matriz_sync.div(matriz_sync.sum(axis=1), axis=0).fillna(0)
        
        # Crear el heatmap con matplotlib/seaborn
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Convertir a porcentajes y reemplazar 0 con NaN para mejor visualización
        matriz_plot = matriz_norm.replace(0, np.nan) * 100
        
        sns.heatmap(
            matriz_plot,
            annot=True, 
            fmt=".1f",  # mostrar valores con un decimal
            cmap="Purples",
            cbar=True,
            linewidths=0.6, 
            linecolor="#DDDDDD",
            square=True,
            ax=ax,
            cbar_kws={'label': 'Probabilidad (%)'}
        )
        
        ax.set_title("Probabilidad de destino (%) dado origen", fontsize=16, pad=20)
        ax.set_xlabel("Destino", fontsize=12)
        ax.set_ylabel("Origen", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Mostrar en Streamlit
        st.pyplot(fig)
        plt.close(fig)
        
        # Estadísticas de la matriz
        st.markdown("**Información de la Matriz:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Estaciones Origen", len(matriz_sync))
        with col2:
            st.metric("Estaciones Destino", len(matriz_sync.columns))
        with col3:
            # Calcular el porcentaje de viajes cubiertos por estos top
            total_viajes = len(df_viajes)
            viajes_en_matriz = matriz_sync.sum().sum()
            porcentaje = (viajes_en_matriz / total_viajes * 100) if total_viajes > 0 else 0
            st.metric("Cobertura", f"{porcentaje:.1f}%")

