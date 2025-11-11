"""
Página de explicación del modelo
Muestra información técnica detallada sobre el modelo, características, pesos, y resultados
"""

import streamlit as st
import pandas as pd
import altair as alt
from lib import load_model, render_feature_importance

def explicacion_modelo_page():
    st.title("📚 Explicación del Modelo")
    st.markdown("---")
    
    # Intentar cargar el modelo para mostrar información detallada
    try:
        modelo = load_model()
        modelo_cargado = True
    except Exception as e:
        modelo = None
        modelo_cargado = False
        st.warning("⚠️ No se pudo cargar el modelo. Se mostrará información general.")
    
    st.markdown("""
    ## 📋 Descripción del Modelo
    
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
    
    # Información técnica del modelo
    st.markdown("## 🎯 Resultados del Modelo")
    
    if modelo_cargado:
        # Mostrar información del modelo
        col1, col2, col3, col4 = st.columns(4)
        
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
        
        with col4:
            if hasattr(modelo, 'oob_score_'):
                st.metric("OOB Score", f"{modelo.oob_score_:.2%}")
            else:
                st.metric("OOB Score", "N/A")
        
        st.markdown("""
        - **Accuracy**: 60.64%
        - **OOB score**: 60.84%
        - **Destinos únicos (tras filtrado)**: 94
        - **Registros de entrenamiento**: 120,677 (test: 30,170)
        """)
    else:
        st.markdown("""
        - **Accuracy**: 60.64%
        - **OOB score**: 60.84%
        - **Destinos únicos (tras filtrado)**: 94
        - **Registros de entrenamiento**: 120,677 (test: 30,170)
        """)
    
    st.markdown("---")
    
    # Importancia de características
    st.markdown("## 🔥 Importancia de Características (Pesos del Modelo)")
    st.markdown("""
    La importancia de cada característica se calcula como la reducción promedio de impureza (Gini) 
    que aporta cada característica al modelo Random Forest. Valores más altos indican mayor 
    capacidad predictiva.
    """)
    
    if modelo_cargado and hasattr(modelo, 'feature_importances_'):
        # Crear gráfico con colores por categoría
        try:
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
            
            # Mapeo de características a categorías
            categorias = {
                # Geográficas
                'origen_lat': '🗺️ Geográficas',
                'origen_lon': '🗺️ Geográficas',
                'lat_destino_favorito': '🗺️ Geográficas',
                'lon_destino_favorito': '🗺️ Geográficas',
                'zona_origen': '🗺️ Geográficas',
                'capacidad_origen': '🗺️ Geográficas',
                'estaciones_cercanas_origen': '🗺️ Geográficas',
                
                # Temporales
                'hora_salida': '⏰ Temporales',
                'dia_semana': '⏰ Temporales',
                'mes': '⏰ Temporales',
                'periodo_dia_numerico': '⏰ Temporales',
                'es_fin_semana': '⏰ Temporales',
                'es_hora_pico': '⏰ Temporales',
                
                # Usuario
                'viajes_totales': '👤 Usuario',
                'semanas_activas': '👤 Usuario',
                'viajes_por_semana': '👤 Usuario',
                'duracion_promedio_min': '👤 Usuario',
                'variedad_destinos': '👤 Usuario',
                'variedad_origenes': '👤 Usuario',
                'consistencia_horaria': '👤 Usuario',
                'distancia_promedio_usuario': '👤 Usuario',
                'dia_favorito': '👤 Usuario',
                'frecuencia_lunes': '👤 Usuario',
                'frecuencia_martes': '👤 Usuario',
                'frecuencia_miercoles': '👤 Usuario',
                'frecuencia_jueves': '👤 Usuario',
                'frecuencia_viernes': '👤 Usuario',
                'frecuencia_sabado': '👤 Usuario',
                'frecuencia_domingo': '👤 Usuario',
                'destino_favorito_encoded': '🗺️ Geográficas'  # Fallback
            }
            
            # Aplicar nombres descriptivos y categorías
            feature_names_descriptivos = [nombres_descriptivos.get(name, name) for name in feature_names]
            feature_categorias = [categorias.get(name, 'Otros') for name in feature_names]
            
            # Crear DataFrame
            imp_df = pd.DataFrame({
                'feature': feature_names_descriptivos,
                'importance': importance,
                'categoria': feature_categorias
            }).sort_values('importance', ascending=False).head(29)
            
            # Crear gráfico con colores por categoría
            chart = (
                alt.Chart(imp_df)
                .mark_bar()
                .encode(
                    x=alt.X('importance:Q', 
                           title='Importancia (Gini)', 
                           axis=alt.Axis(format='.4f')),
                    y=alt.Y('feature:N', 
                           sort='-x', 
                           title='Característica',
                           axis=alt.Axis(labelLimit=1000)),
                    tooltip=[
                        alt.Tooltip('feature:N', title='Característica'),
                        alt.Tooltip('categoria:N', title='Categoría'),
                        alt.Tooltip('importance:Q', title='Importancia', format='.4f')
                    ],
                    color=alt.Color('categoria:N',
                                   scale=alt.Scale(
                                       domain=['🗺️ Geográficas', '⏰ Temporales', '👤 Usuario'],
                                       range=['#1f77b4', '#ff7f0e', '#2ca02c']  # Azul, Naranja, Verde
                                   ),
                                   legend=alt.Legend(
                                       title='Categoría',
                                       orient='bottom',
                                       titleFontSize=12,
                                       labelFontSize=11
                                   ))
                )
                .properties(
                    width=700,
                    height=800,
                    title='Importancia de las 29 Características del Modelo'
                )
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            # Leyenda explicativa
            st.markdown("""
            **📊 Leyenda de Colores:**
            
            - 🔵 **Azul (🗺️ Geográficas)**: Características relacionadas con ubicación geográfica
              - Coordenadas de origen (lat/lon)
              - Coordenadas de destino favorito (lat/lon)
              - Zona geográfica
              - Capacidad de estación
              - Estaciones cercanas
            
            - 🟠 **Naranja (⏰ Temporales)**: Características relacionadas con tiempo y momento
              - Hora del día
              - Día de la semana
              - Mes del año
              - Período del día
              - Fin de semana / Hora pico
            
            - 🟢 **Verde (👤 Usuario)**: Características relacionadas con el comportamiento del usuario
              - Historial de viajes
              - Frecuencia semanal
              - Duración promedio
              - Distancia promedio
              - Consistencia horaria
              - Frecuencias por día de la semana
            """)
            
        except Exception as e:
            st.error(f"Error al generar gráfico de importancia: {e}")
            # Fallback: mostrar tabla
            importance = modelo.feature_importances_
            feature_names = modelo.feature_names_in_ if hasattr(modelo, 'feature_names_in_') else [f'feature_{i}' for i in range(len(importance))]
            
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
            
            feature_names_descriptivos = [nombres_descriptivos.get(name, name) for name in feature_names]
            
            imp_df = pd.DataFrame({
                'Característica': feature_names_descriptivos,
                'Importancia': importance,
                'Porcentaje': importance * 100
            }).sort_values('Importancia', ascending=False)
            
            st.dataframe(
                imp_df.style.format({'Importancia': '{:.6f}', 'Porcentaje': '{:.2f}%'}),
                use_container_width=True,
                height=600
            )
    else:
        st.info("💡 Carga el modelo para ver la importancia detallada de cada característica.")
        st.markdown("""
        ### Top 5 Características Más Importantes (estimación):
        
        1. **Longitud Destino Favorito** (~29.27%)
        2. **Latitud Destino Favorito** (~28.22%)
        3. **Distancia promedio del usuario** (~6.28%)
        4. **Latitud de origen** (~4.98%)
        5. **Longitud de origen** (~4.84%)
        """)
    
    st.markdown("---")
    
    # Hiperparámetros del modelo
    st.markdown("## ⚙️ Hiperparámetros del Modelo")
    
    if modelo_cargado:
        st.markdown("""
        Los hiperparámetros utilizados para entrenar el modelo Random Forest son:
        """)
        
        hiperparametros = {
            'Parámetro': [
                'n_estimators',
                'max_depth',
                'min_samples_split',
                'min_samples_leaf',
                'max_features',
                'bootstrap',
                'oob_score',
                'random_state'
            ],
            'Valor': [
                getattr(modelo, 'n_estimators', 'N/A'),
                getattr(modelo, 'max_depth', 'N/A'),
                getattr(modelo, 'min_samples_split', 'N/A'),
                getattr(modelo, 'min_samples_leaf', 'N/A'),
                getattr(modelo, 'max_features', 'N/A'),
                getattr(modelo, 'bootstrap', 'N/A'),
                getattr(modelo, 'oob_score', 'N/A'),
                getattr(modelo, 'random_state', 'N/A')
            ]
        }
        
        df_hiper = pd.DataFrame(hiperparametros)
        st.dataframe(df_hiper, use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        - **n_estimators**: 95 (número de árboles en el bosque)
        - **max_depth**: 15 (profundidad máxima de cada árbol)
        - **min_samples_split**: 15 (mínimo de muestras para dividir un nodo)
        - **min_samples_leaf**: 5 (mínimo de muestras en una hoja)
        - **max_features**: 0.5 (fracción de features a considerar por split)
        - **bootstrap**: True (muestreo con reemplazo)
        - **oob_score**: True (puntaje out-of-bag)
        - **random_state**: 42 (semilla para reproducibilidad)
        """)
    
    st.markdown("---")
    
    # Detalles de las características
    st.markdown("## 📊 Detalle de Características")
    
    st.markdown("""
    ### Características Geográficas
    
    - **origen_lat, origen_lon**: Coordenadas geográficas de la estación de origen del viaje
    - **lat_destino_favorito, lon_destino_favorito**: Coordenadas del destino favorito del usuario (estación más visitada históricamente)
    - **zona_origen**: Zona geográfica clasificada (1: Centro, 2: Cerca, 3: Periferia, 4: Lejos)
    - **capacidad_origen**: Capacidad total de bicicletas de la estación de origen
    - **estaciones_cercanas_origen**: Número de estaciones cercanas (dentro de 0.01 grados) a la estación de origen
    
    ### Características Temporales
    
    - **hora_salida**: Hora del día en que inicia el viaje (0-23)
    - **dia_semana**: Día de la semana (0: Lunes, 6: Domingo)
    - **mes**: Mes del año (1-12)
    - **periodo_dia_numerico**: Período del día (0: Madrugada, 1: Mañana, 2: Tarde, 3: Noche)
    - **es_fin_semana**: Indicador binario (1: Sábado/Domingo, 0: Día laboral)
    - **es_hora_pico**: Indicador binario (1: Horas pico 7-9 y 17-19, 0: Otras horas)
    
    ### Características de Usuario
    
    - **viajes_totales**: Total de viajes realizados por el usuario
    - **semanas_activas**: Número de semanas en las que el usuario ha realizado al menos un viaje
    - **viajes_por_semana**: Promedio de viajes por semana
    - **duracion_promedio_min**: Duración promedio de los viajes del usuario (en minutos)
    - **variedad_destinos**: Número de destinos únicos visitados por el usuario
    - **variedad_origenes**: Número de orígenes únicos utilizados por el usuario
    - **consistencia_horaria**: Desviación estándar de las horas de inicio de viajes (menor valor = más consistente)
    - **distancia_promedio_usuario**: Distancia promedio de los viajes del usuario (en grados)
    - **dia_favorito**: Día de la semana más frecuente para viajes (0-6)
    - **frecuencia_lunes a frecuencia_domingo**: Contador de viajes por día de la semana
    """)
    
    st.markdown("---")
    
    # Metodología
    st.markdown("## 🔬 Metodología")
    
    st.markdown("""
    ### Proceso de Entrenamiento
    
    1. **Preprocesamiento de datos**: Limpieza y normalización de datos históricos de viajes
    2. **Feature Engineering**: Creación de características derivadas de los datos originales
    3. **División de datos**: 80% entrenamiento, 20% test
    4. **Entrenamiento**: Random Forest Classifier con los hiperparámetros optimizados
    5. **Validación**: Usando Out-of-Bag (OOB) score y validación cruzada
    
    ### Métricas de Evaluación
    
    - **Accuracy**: Porcentaje de predicciones correctas sobre el total
    - **OOB Score**: Puntaje out-of-bag, estimación de la precisión sin necesidad de un conjunto de validación separado
    - **Top-K Accuracy**: Porcentaje de casos donde el destino real está en las K predicciones más probables
    """)

