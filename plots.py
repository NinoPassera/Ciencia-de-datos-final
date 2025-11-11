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
import folium
from streamlit_folium import st_folium
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
    Análisis de patrones temporales en los viajes. Muestra la distribución de viajes por hora del día,
    revelando patrones de comportamiento de los usuarios.
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
                   axis=alt.Axis(format=',d')),
            tooltip=[
                alt.Tooltip('hora:Q', title='Hora', format='d'),
                alt.Tooltip('cantidad_viajes:Q', title='Viajes', format=',d')
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
    
    st.markdown("---")
    
    # Visualización 2: Análisis Geográfico - Top Destinos
    st.markdown("## 2. Top Destinos Más Frecuentes")
    st.markdown("""
    Análisis de los destinos más populares en el sistema. Muestra las estaciones destino más frecuentes,
    lo que ayuda a entender los patrones de movilidad y demanda en diferentes zonas.
    """)
    
    # Filtros temporales (mes y temporada)
    col_filtro_temp1, col_filtro_temp2 = st.columns(2)
    
    with col_filtro_temp1:
        # Mapeo de meses a nombres
        meses_nombres_dest = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
            5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
            9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }
        
        meses_disponibles_dest = sorted(df['mes'].unique())
        opciones_meses_dest = ['Todos los meses'] + [meses_nombres_dest[m] for m in meses_disponibles_dest]
        
        mes_seleccionado_dest = st.selectbox(
            "📅 Filtrar por Mes",
            options=opciones_meses_dest,
            index=0,
            help="Selecciona un mes específico o 'Todos los meses' para ver todos los datos",
            key="mes_selector_destinos"
        )
    
    with col_filtro_temp2:
        # Mapeo de temporadas (hemisferio sur)
        temporadas_dest = {
            'Todas las temporadas': None,
            'Verano (Dic-Ene-Feb)': [12, 1, 2],
            'Otoño (Mar-Abr-May)': [3, 4, 5],
            'Invierno (Jun-Jul-Ago)': [6, 7, 8],
            'Primavera (Sep-Oct-Nov)': [9, 10, 11]
        }
        
        temporada_seleccionada_dest = st.selectbox(
            "🌤️ Filtrar por Temporada",
            options=list(temporadas_dest.keys()),
            index=0,
            help="Selecciona una temporada del año para filtrar los datos",
            key="temporada_selector_destinos"
        )
    
    # Aplicar filtros temporales
    df_filtrado_dest = df.copy()
    
    # Filtro por mes
    filtro_mes_aplicado_dest = False
    if mes_seleccionado_dest != 'Todos los meses':
        mes_numero_dest = [k for k, v in meses_nombres_dest.items() if v == mes_seleccionado_dest][0]
        df_filtrado_dest = df_filtrado_dest[df_filtrado_dest['mes'] == mes_numero_dest]
        filtro_mes_aplicado_dest = True
    
    # Filtro por temporada
    filtro_temporada_aplicado_dest = False
    if temporada_seleccionada_dest != 'Todas las temporadas':
        meses_temporada_dest = temporadas_dest[temporada_seleccionada_dest]
        if filtro_mes_aplicado_dest:
            if mes_numero_dest in meses_temporada_dest:
                filtro_temporada_aplicado_dest = True
            else:
                df_filtrado_dest = df_filtrado_dest[df_filtrado_dest['mes'].isin([])]
                filtro_temporada_aplicado_dest = True
        else:
            df_filtrado_dest = df_filtrado_dest[df_filtrado_dest['mes'].isin(meses_temporada_dest)]
            filtro_temporada_aplicado_dest = True
    
    # Mostrar resumen de filtros aplicados
    if len(df_filtrado_dest) < len(df):
        st.info(f"📊 Mostrando {len(df_filtrado_dest):,} viajes de {len(df):,} totales (filtros aplicados)")
    
    # Validar que hay datos después de filtrar
    if len(df_filtrado_dest) == 0:
        st.warning("⚠️ No hay datos disponibles para los filtros seleccionados. Por favor, ajusta los filtros.")
        st.markdown("---")
    else:
        # Top destinos (mostrar top 15)
        top_destinos = df_filtrado_dest['destino'].value_counts().head(15).reset_index()
        
        top_destinos.columns = ['destino', 'cantidad_viajes']
        top_destinos['porcentaje'] = (top_destinos['cantidad_viajes'] / len(df_filtrado_dest) * 100).round(2)
        
        titulo_grafico = 'Top 15 Estaciones Destino Más Frecuentes'
        
        chart3 = (
            alt.Chart(top_destinos)
            .mark_bar()
            .encode(
                x=alt.X('cantidad_viajes:Q', 
                       title='Cantidad de Viajes',
                       axis=alt.Axis(format=',d')),
                y=alt.Y('destino:N', 
                       sort='-x', 
                       title='Estación Destino'),
                tooltip=[
                    alt.Tooltip('destino:N', title='Destino'),
                    alt.Tooltip('cantidad_viajes:Q', title='Viajes', format=',d'),
                    alt.Tooltip('porcentaje:Q', title='Porcentaje', format='.2f')
                ],
                color=alt.Color('cantidad_viajes:Q', 
                              scale=alt.Scale(scheme='reds'), 
                              legend=None)
            )
            .properties(
                width=700,
                height=500,
                title=titulo_grafico
            )
        )
        
        st.altair_chart(chart3, width='stretch')
        
        # Estadísticas adicionales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Viajes", f"{len(df_filtrado_dest):,}")
        with col2:
            st.metric("Destinos Únicos", f"{df_filtrado_dest['destino'].nunique()}")
        with col3:
            st.metric("Usuarios Únicos", f"{df_filtrado_dest.get('Usuario_key', pd.Series()).nunique() if 'Usuario_key' in df_filtrado_dest.columns else 'N/A'}")
        with col4:
            if len(top_destinos) > 0:
                st.metric("Destino Más Frecuente", f"{top_destinos.iloc[0]['destino'][:20]}...")
            else:
                st.metric("Destino Más Frecuente", "N/A")
    
    st.markdown("---")
    
    # Visualización 3: Mapa de Calor de Estaciones
    st.markdown("## 3. Mapa de Calor de Estaciones")
    st.markdown("""
    Este mapa muestra la frecuencia de uso de cada estación. Las estaciones más grandes 
    indican mayor cantidad de viajes (tanto como origen como destino).
    """)
    
    if True:  # Mapa de calor siempre visible
        # Calcular frecuencia de uso por estación (origen + destino)
        if 'origen' in df.columns and 'destino' in df.columns:
            # Contar apariciones como origen
            frecuencia_origen = df['origen'].value_counts().reset_index()
            frecuencia_origen.columns = ['estacion', 'frecuencia_origen']
            
            # Contar apariciones como destino
            frecuencia_destino = df['destino'].value_counts().reset_index()
            frecuencia_destino.columns = ['estacion', 'frecuencia_destino']
            
            # Combinar y sumar
            frecuencia_total = frecuencia_origen.merge(
                frecuencia_destino, 
                on='estacion', 
                how='outer'
            ).fillna(0)
            frecuencia_total['frecuencia_total'] = (
                frecuencia_total['frecuencia_origen'] + 
                frecuencia_total['frecuencia_destino']
            )
            
            # Obtener coordenadas de las estaciones (si están en el CSV)
            if 'origen_lat' in df.columns and 'origen_lon' in df.columns:
                # Obtener coordenadas de origen
                coords_origen = df.groupby('origen').agg({
                    'origen_lat': 'first',
                    'origen_lon': 'first'
                }).reset_index()
                coords_origen.columns = ['estacion', 'lat', 'lon']
                
                # Obtener coordenadas de destino desde estaciones.json
                from lib import load_stations
                estaciones_dict = load_stations()
                
                coords_destino_list = []
                for estacion in frecuencia_total['estacion'].unique():
                    if estacion in estaciones_dict:
                        coords_destino_list.append({
                            'estacion': estacion,
                            'lat': estaciones_dict[estacion]['lat'],
                            'lon': estaciones_dict[estacion]['lon']
                        })
                
                if coords_destino_list:
                    coords_destino = pd.DataFrame(coords_destino_list)
                    # Combinar coordenadas
                    coords = pd.concat([coords_origen, coords_destino]).drop_duplicates('estacion')
                else:
                    coords = coords_origen
                
                # Merge con frecuencia
                frecuencia_con_coords = frecuencia_total.merge(
                    coords, 
                    on='estacion', 
                    how='left'
                ).dropna(subset=['lat', 'lon'])
                
                # Crear mapa base con Folium
                # Centro de Mendoza: -32.89, -68.84
                mapa = folium.Map(
                    location=[-32.89, -68.84],
                    zoom_start=13,
                    tiles='OpenStreetMap'
                )
                
                # Normalizar frecuencia para el tamaño de los círculos y el color
                max_frecuencia = frecuencia_con_coords['frecuencia_total'].max()
                min_frecuencia = frecuencia_con_coords['frecuencia_total'].min()
                
                # Agregar marcadores circulares para cada estación
                for _, row in frecuencia_con_coords.iterrows():
                    # Calcular tamaño del círculo (entre 5 y 30 metros de radio)
                    radio = 5 + (row['frecuencia_total'] / max_frecuencia) * 25
                    
                    # Calcular color (rojo más intenso = mayor frecuencia)
                    intensidad = int((row['frecuencia_total'] / max_frecuencia) * 255)
                    color_hex = f'#{intensidad:02x}0000'
                    
                    # Crear círculo
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=radio,
                        popup=folium.Popup(
                            f"""
                            <b>{row['estacion']}</b><br>
                            Frecuencia Total: {row['frecuencia_total']:,.0f}<br>
                            Como Origen: {row['frecuencia_origen']:,.0f}<br>
                            Como Destino: {row['frecuencia_destino']:,.0f}<br>
                            Lat: {row['lat']:.5f}, Lon: {row['lon']:.5f}
                            """,
                            max_width=300
                        ),
                        tooltip=f"{row['estacion']}: {row['frecuencia_total']:,.0f} viajes",
                        color='darkred',
                        fill=True,
                        fillColor=color_hex,
                        fillOpacity=0.6,
                        weight=2
                    ).add_to(mapa)
                
                # Mostrar mapa
                st_folium(mapa, width=700, height=500, returned_objects=[])
            else:
                st.warning("⚠️ No se encontraron coordenadas en el dataset. Usando gráfico de barras alternativo.")
                # Gráfico alternativo de barras
                top_estaciones = frecuencia_total.sort_values('frecuencia_total', ascending=False).head(20)
                chart_barras = (
                    alt.Chart(top_estaciones)
                    .mark_bar()
                    .encode(
                        x=alt.X('frecuencia_total:Q', title='Frecuencia Total', axis=alt.Axis(format=',d')),
                        y=alt.Y('estacion:N', sort='-x', title='Estación'),
                        color=alt.Color('frecuencia_total:Q', scale=alt.Scale(scheme='reds'), legend=None),
                        tooltip=[
                            alt.Tooltip('estacion:N', title='Estación'),
                            alt.Tooltip('frecuencia_total:Q', title='Frecuencia Total', format=',d')
                        ]
                    )
                    .properties(
                        width=700,
                        height=600,
                        title='Top 20 Estaciones por Frecuencia de Uso'
                    )
                )
                st.altair_chart(chart_barras, use_container_width=True)
        else:
            st.warning("⚠️ No se encontraron las columnas 'origen' y 'destino' necesarias para este gráfico.")
    
    st.markdown("---")
    
    # Visualización 4: Evolución Mensual - Línea de Tiempo de Tendencias
    st.markdown("## 4. Evolución Mensual: Línea de Tiempo de Tendencias")
    st.markdown("""
    Este gráfico muestra la evolución temporal de los viajes a lo largo del tiempo. 
    Puedes ver las tendencias mensuales y filtrar por un mes específico para analizar patrones detallados.
    """)
    
    # Verificar que existe la columna 'mes'
    if 'mes' not in df.columns:
        st.warning("⚠️ El dataset no contiene la columna 'mes' necesaria para este gráfico.")
    else:
        # Mapeo de meses a nombres
        meses_nombres = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
            5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
            9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }
        
        # Crear datos para la evolución temporal
        # Agrupar por mes y contar viajes
        evolucion_mensual = df.groupby('mes').size().reset_index(name='cantidad_viajes')
        evolucion_mensual['mes_nombre'] = evolucion_mensual['mes'].map(meses_nombres)
        
        # Si hay columna 'semana', también podemos hacer evolución semanal
        evolucion_semanal = None
        if 'semana' in df.columns:
            # Agrupar por semana y contar viajes
            evolucion_semanal = df.groupby('semana').size().reset_index(name='cantidad_viajes')
            # Ordenar por semana (asumiendo formato YYYY-WW)
            evolucion_semanal = evolucion_semanal.sort_values('semana')
        
        # Crear gráfico principal de evolución mensual
        chart_evolucion = (
            alt.Chart(evolucion_mensual)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X('mes:O', 
                       title='Mes',
                       axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('cantidad_viajes:Q', 
                       title='Cantidad de Viajes',
                       axis=alt.Axis(format=',d')),
                tooltip=[
                    alt.Tooltip('mes_nombre:N', title='Mes'),
                    alt.Tooltip('cantidad_viajes:Q', title='Viajes', format=',d')
                ],
                color=alt.value('#1f77b4')
            )
            .properties(
                width=700,
                height=400,
                title='Evolución Mensual de Viajes'
            )
        )
        
        # Agregar área debajo de la línea
        chart_area = (
            alt.Chart(evolucion_mensual)
            .mark_area(opacity=0.3)
            .encode(
                x=alt.X('mes:O', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('cantidad_viajes:Q', axis=alt.Axis(format=',d')),
                color=alt.value('#1f77b4')
            )
        )
        
        chart_final = chart_area + chart_evolucion
        
        st.altair_chart(chart_final, use_container_width=True)
        
        # Selector de mes (opcional, para filtrar el detalle)
        meses_disponibles = sorted(df['mes'].unique())
        opciones_meses = ['Todos los meses'] + [meses_nombres[m] for m in meses_disponibles]
        
        mes_seleccionado_evo = st.selectbox(
            "📅 Filtrar por Mes (Opcional)",
            options=opciones_meses,
            index=0,
            help="Selecciona un mes específico para ver su evolución detallada, o 'Todos los meses' para ver la evolución completa",
            key="mes_evolucion_selector"
        )
        
        # Aplicar filtro de mes si se seleccionó uno
        df_filtrado_evo = df.copy()
        mostrar_detalle = False
        if mes_seleccionado_evo != 'Todos los meses':
            mes_numero = [k for k, v in meses_nombres.items() if v == mes_seleccionado_evo][0]
            df_filtrado_evo = df_filtrado_evo[df_filtrado_evo['mes'] == mes_numero]
            mostrar_detalle = True
            
            # Si hay filtro de mes, mostrar evolución por día de la semana o por semana del mes
            if 'dia_semana' in df_filtrado_evo.columns:
                evolucion_dia = df_filtrado_evo.groupby('dia_semana').size().reset_index(name='cantidad_viajes')
                dias_nombres = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 
                               4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
                evolucion_dia['dia_nombre'] = evolucion_dia['dia_semana'].map(dias_nombres)
        
        # Si hay filtro de mes, mostrar gráfico detallado
        if mostrar_detalle:
            st.markdown(f"### 📊 Detalle del Mes Seleccionado: {mes_seleccionado_evo}")
            
            if 'dia_semana' in df_filtrado_evo.columns:
                # Gráfico de evolución por día de la semana
                chart_dia = (
                    alt.Chart(evolucion_dia)
                    .mark_bar()
                    .encode(
                        x=alt.X('dia_nombre:N', 
                               title='Día de la Semana',
                               sort=['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']),
                        y=alt.Y('cantidad_viajes:Q', 
                               title='Cantidad de Viajes',
                               axis=alt.Axis(format=',d')),
                        color=alt.Color('cantidad_viajes:Q', 
                                       scale=alt.Scale(scheme='blues'),
                                       legend=None),
                        tooltip=[
                            alt.Tooltip('dia_nombre:N', title='Día'),
                            alt.Tooltip('cantidad_viajes:Q', title='Viajes', format=',d')
                        ]
                    )
                    .properties(
                        width=700,
                        height=300,
                        title=f'Distribución de Viajes por Día de la Semana - {mes_seleccionado_evo}'
                    )
                )
                st.altair_chart(chart_dia, use_container_width=True)
            
            # Estadísticas del mes seleccionado
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Viajes", f"{len(df_filtrado_evo):,}")
            with col2:
                if 'origen' in df_filtrado_evo.columns:
                    origenes_unicos = df_filtrado_evo['origen'].nunique()
                    st.metric("Estaciones Origen Únicas", origenes_unicos)
            with col3:
                if 'destino' in df_filtrado_evo.columns:
                    destinos_unicos = df_filtrado_evo['destino'].nunique()
                    st.metric("Estaciones Destino Únicas", destinos_unicos)
        
        # Mostrar evolución semanal si está disponible y no hay filtro de mes
        if evolucion_semanal is not None and not mostrar_detalle:
            st.markdown("### 📈 Evolución Semanal")
            
            chart_semanal = (
                alt.Chart(evolucion_semanal)
                .mark_line(point=True, strokeWidth=2)
                .encode(
                    x=alt.X('semana:N', 
                           title='Semana',
                           axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('cantidad_viajes:Q', 
                           title='Cantidad de Viajes',
                           axis=alt.Axis(format=',d')),
                    tooltip=[
                        alt.Tooltip('semana:N', title='Semana'),
                        alt.Tooltip('cantidad_viajes:Q', title='Viajes', format=',d')
                    ],
                    color=alt.value('#ff7f0e')
                )
                .properties(
                    width=700,
                    height=300,
                    title='Evolución Semanal de Viajes'
                )
            )
            st.altair_chart(chart_semanal, use_container_width=True)
        
        # Estadísticas generales
        st.markdown("**📊 Estadísticas de Evolución:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            max_mes = evolucion_mensual.loc[evolucion_mensual['cantidad_viajes'].idxmax()]
            st.metric("Mes con Más Viajes", 
                     f"{max_mes['mes_nombre']}: {max_mes['cantidad_viajes']:,}")
        with col2:
            min_mes = evolucion_mensual.loc[evolucion_mensual['cantidad_viajes'].idxmin()]
            st.metric("Mes con Menos Viajes", 
                     f"{min_mes['mes_nombre']}: {min_mes['cantidad_viajes']:,}")
        with col3:
            promedio = evolucion_mensual['cantidad_viajes'].mean()
            st.metric("Promedio Mensual", f"{promedio:,.0f} viajes")
    
    st.markdown("---")
    
    # Visualización 5: Matriz Origen-Destino (Heatmap)
    st.markdown("## 5. Matriz de Probabilidad Origen-Destino")
    st.markdown("""
    Este heatmap muestra la probabilidad de que un viaje desde una estación origen termine en una estación destino.
    Los valores representan el porcentaje de viajes desde cada origen hacia cada destino.
    Puedes seleccionar qué estaciones quieres visualizar en el heatmap.
    """)
    
    # Verificar que existen las columnas origen y destino
    if 'origen' not in df.columns or 'destino' not in df.columns:
        st.warning("⚠️ El dataset no contiene las columnas 'origen' y 'destino' necesarias para este gráfico.")
        st.info("💡 Ejecuta crear_dataset_final.py para generar el dataset con estas columnas.")
    else:
        # Crear DataFrame con origen y destino (similar a df_viajes)
        df_viajes = df[['origen', 'destino']].copy()
        
        # Obtener todas las estaciones únicas (origen y destino)
        todas_estaciones_origen = sorted(df_viajes["origen"].unique())
        todas_estaciones_destino = sorted(df_viajes["destino"].unique())
        
        # Top 15 por origen y destino (para valores por defecto)
        top_origen = df_viajes["origen"].value_counts().head(15).index.tolist()
        top_destino = df_viajes["destino"].value_counts().head(15).index.tolist()
        
        # Selector de estaciones (opcional, si no selecciona nada usa top 15)
        col_selector1, col_selector2 = st.columns(2)
        
        with col_selector1:
            estaciones_origen_seleccionadas = st.multiselect(
                "📍 Estaciones Origen (Opcional)",
                options=todas_estaciones_origen,
                default=[],
                help="Selecciona estaciones específicas de origen. Si no seleccionas ninguna, se mostrarán las top 15 por defecto."
            )
        
        with col_selector2:
            estaciones_destino_seleccionadas = st.multiselect(
                "🎯 Estaciones Destino (Opcional)",
                options=todas_estaciones_destino,
                default=[],
                help="Selecciona estaciones específicas de destino. Si no seleccionas ninguna, se mostrarán las top 15 por defecto."
            )
        
        # Usar selección del usuario o top 15 por defecto
        if len(estaciones_origen_seleccionadas) == 0:
            estaciones_origen_finales = top_origen
        else:
            estaciones_origen_finales = estaciones_origen_seleccionadas
        
        if len(estaciones_destino_seleccionadas) == 0:
            estaciones_destino_finales = top_destino
        else:
            estaciones_destino_finales = estaciones_destino_seleccionadas
        
        # Validar que hay estaciones para mostrar
        if len(estaciones_origen_finales) == 0 or len(estaciones_destino_finales) == 0:
            st.warning("⚠️ No hay estaciones disponibles para mostrar.")
        else:
            # Matriz Origen x Destino, filtrada según selección
            matriz_top = pd.crosstab(df_viajes["origen"], df_viajes["destino"])
            matriz_top = matriz_top.loc[matriz_top.index.intersection(estaciones_origen_finales), 
                                        matriz_top.columns.intersection(estaciones_destino_finales)]
            
            # Validar que hay datos después de filtrar
            if matriz_top.empty:
                st.warning("⚠️ No hay datos disponibles para las estaciones seleccionadas.")
            else:
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
                    # Calcular el porcentaje de viajes cubiertos por estas estaciones
                    total_viajes = len(df_viajes)
                    viajes_en_matriz = matriz_sync.sum().sum()
                    porcentaje = (viajes_en_matriz / total_viajes * 100) if total_viajes > 0 else 0
                    st.metric("Cobertura", f"{porcentaje:.1f}%")

