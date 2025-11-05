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
            return
    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")
        return
    
    # Visualización 1: Distribución Temporal de Viajes
    st.markdown("## 1. Distribución Temporal de Viajes")
    st.markdown("""
    Análisis de patrones temporales en los viajes. Muestra la distribución de viajes por hora del día
    y día de la semana, revelando patrones de comportamiento de los usuarios.
    """)
    
    # Selectores de filtros temporales
    col_filtro1, col_filtro2 = st.columns(2)
    
    with col_filtro1:
        # Mapeo de meses a nombres
        meses_nombres = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
            5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
            9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }
        
        meses_disponibles = sorted(df['mes'].unique())
        opciones_meses = ['Todos los meses'] + [meses_nombres[m] for m in meses_disponibles]
        
        mes_seleccionado = st.selectbox(
            "📅 Filtrar por Mes",
            options=opciones_meses,
            index=0,
            help="Selecciona un mes específico o 'Todos los meses' para ver todos los datos"
        )
    
    with col_filtro2:
        # Mapeo de temporadas (hemisferio sur)
        temporadas = {
            'Todas las temporadas': None,
            'Verano (Dic-Ene-Feb)': [12, 1, 2],
            'Otoño (Mar-Abr-May)': [3, 4, 5],
            'Invierno (Jun-Jul-Ago)': [6, 7, 8],
            'Primavera (Sep-Oct-Nov)': [9, 10, 11]
        }
        
        temporada_seleccionada = st.selectbox(
            "🌤️ Filtrar por Temporada",
            options=list(temporadas.keys()),
            index=0,
            help="Selecciona una temporada del año para filtrar los datos"
        )
    
    # Aplicar filtros
    df_filtrado = df.copy()
    
    # Filtro por mes
    filtro_mes_aplicado = False
    if mes_seleccionado != 'Todos los meses':
        mes_numero = [k for k, v in meses_nombres.items() if v == mes_seleccionado][0]
        df_filtrado = df_filtrado[df_filtrado['mes'] == mes_numero]
        filtro_mes_aplicado = True
    
    # Filtro por temporada
    filtro_temporada_aplicado = False
    if temporada_seleccionada != 'Todas las temporadas':
        meses_temporada = temporadas[temporada_seleccionada]
        if filtro_mes_aplicado:
            # Si ya hay un mes seleccionado, verificar que esté en la temporada
            if mes_numero in meses_temporada:
                # El mes ya está filtrado, no necesitamos filtrar más
                filtro_temporada_aplicado = True
            else:
                # El mes seleccionado no está en la temporada, no hay datos
                df_filtrado = df_filtrado[df_filtrado['mes'].isin([])]  # DataFrame vacío
                filtro_temporada_aplicado = True
        else:
            # Solo filtrar por temporada
            df_filtrado = df_filtrado[df_filtrado['mes'].isin(meses_temporada)]
            filtro_temporada_aplicado = True
    
    # Mostrar resumen de filtros aplicados
    if len(df_filtrado) < len(df):
        st.info(f"📊 Mostrando {len(df_filtrado):,} viajes de {len(df):,} totales (filtros aplicados)")
    
    # Validar que hay datos después de filtrar
    if len(df_filtrado) == 0:
        st.warning("⚠️ No hay datos disponibles para los filtros seleccionados. Por favor, ajusta los filtros.")
        st.markdown("---")
        return
    
    st.markdown("---")
    
    # Crear visualización de distribución por hora
    hora_counts = df_filtrado['hora_salida'].value_counts().sort_index().reset_index()
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
    dia_counts = df_filtrado['dia_semana'].value_counts().sort_index().reset_index()
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
    
    # Visualización 2: Análisis Geográfico - Top Destinos
    st.markdown("## 2. Top Destinos Más Frecuentes")
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
    
    # Visualización 3: Matriz Origen-Destino (Heatmap)
    st.markdown("## 3. Matriz de Probabilidad Origen-Destino")
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

