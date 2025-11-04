"""
Funciones auxiliares para la aplicación Streamlit
- Carga de modelo
- Procesamiento de datos de entrada
- Visualizaciones con Altair
- Transformers personalizados para cálculo de features
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import altair as alt

# Centro de Mendoza para cálculo de zona geográfica
CENTRO_LAT = -32.89
CENTRO_LON = -68.84

# ============================================================================
# TRANSFORMERS PERSONALIZADOS
# ============================================================================

class FeatureEngineeringTemporal(BaseEstimator, TransformerMixin):
    """Calcula features temporales derivadas"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Clasificar período del día
        def clasificar_periodo(hora):
            if 6 <= hora < 12: return 1    # mañana
            elif 12 <= hora < 18: return 2 # tarde
            elif 18 <= hora < 24: return 3 # noche
            else: return 0                # madrugada
        
        X['periodo_dia_numerico'] = X['hora_salida'].apply(clasificar_periodo)
        X['es_fin_semana'] = X['dia_semana'].isin([5, 6]).astype(int)
        X['es_hora_pico'] = X['hora_salida'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        return X


class FeatureEngineeringGeografica(BaseEstimator, TransformerMixin):
    """Calcula features geográficas"""
    
    def __init__(self, estaciones_data=None):
        self.estaciones_data = estaciones_data
        self.estaciones_dict = None
        self.estaciones_cercanas_dict = None
    
    def fit(self, X, y=None):
        if self.estaciones_data is not None:
            # Crear diccionario de estaciones por coordenadas
            self.estaciones_dict = {}
            for _, row in self.estaciones_data.iterrows():
                lat = row.get('station_lat', row.get('lat', None))
                lon = row.get('station_lon', row.get('lon', None))
                if pd.notna(lat) and pd.notna(lon):
                    key = (round(lat, 5), round(lon, 5))
                    self.estaciones_dict[key] = {
                        'capacidad': row.get('station_capacity', row.get('capacity', 0)),
                        'zona': self._clasificar_zona(lat, lon)
                    }
            
            # Calcular estaciones cercanas
            if len(self.estaciones_data) > 0:
                coords = self.estaciones_data[['station_lat', 'station_lon']].values
                if len(coords) > 0:
                    nbrs = NearestNeighbors(n_neighbors=min(10, len(coords)), algorithm='ball_tree')
                    nbrs.fit(coords)
                    self.estaciones_cercanas_dict = {}
                    for idx, row in self.estaciones_data.iterrows():
                        lat = row.get('station_lat', row.get('lat', None))
                        lon = row.get('station_lon', row.get('lon', None))
                        if pd.notna(lat) and pd.notna(lon):
                            coords_point = [[lat, lon]]
                            distances, indices = nbrs.kneighbors(coords_point)
                            cercanas = (distances[0] <= 0.01).sum() - 1
                            key = (round(lat, 5), round(lon, 5))
                            self.estaciones_cercanas_dict[key] = max(0, cercanas)
        
        return self
    
    def _clasificar_zona(self, lat, lon):
        """Clasifica zona geográfica"""
        if pd.isna(lat) or pd.isna(lon):
            return 0
        dist_lat = abs(lat - CENTRO_LAT)
        dist_lon = abs(lon - CENTRO_LON)
        if dist_lat < 0.02 and dist_lon < 0.02:
            return 1    # Centro
        elif dist_lat < 0.05 and dist_lon < 0.05:
            return 2    # Cerca del centro
        elif dist_lat < 0.1 and dist_lon < 0.1:
            return 3    # Periferia
        else:
            return 4    # Lejos
    
    def transform(self, X):
        X = X.copy()
        
        # Calcular zona_origen
        X['zona_origen'] = X.apply(
            lambda row: self._clasificar_zona(row.get('origen_lat', 0), row.get('origen_lon', 0)),
            axis=1
        )
        
        # Buscar capacidad y estaciones cercanas si tenemos datos
        if self.estaciones_dict is not None:
            X['capacidad_origen'] = X.apply(
                lambda row: self._get_capacidad(row.get('origen_lat', 0), row.get('origen_lon', 0)),
                axis=1
            )
            X['estaciones_cercanas_origen'] = X.apply(
                lambda row: self._get_estaciones_cercanas(row.get('origen_lat', 0), row.get('origen_lon', 0)),
                axis=1
            )
        else:
            # Valores por defecto si no hay datos de estaciones
            X['capacidad_origen'] = 15  # Valor promedio
            X['estaciones_cercanas_origen'] = 5  # Valor promedio
        
        return X
    
    def _get_capacidad(self, lat, lon):
        """Obtiene capacidad de estación más cercana"""
        if self.estaciones_dict is None:
            return 15
        key = (round(lat, 5), round(lon, 5))
        return self.estaciones_dict.get(key, {}).get('capacidad', 15)
    
    def _get_estaciones_cercanas(self, lat, lon):
        """Obtiene número de estaciones cercanas"""
        if self.estaciones_cercanas_dict is None:
            return 5
        key = (round(lat, 5), round(lon, 5))
        return self.estaciones_cercanas_dict.get(key, 5)


class FeatureEngineeringUsuario(BaseEstimator, TransformerMixin):
    """Calcula features de usuario (usa valores por defecto si no hay historial)"""
    
    def fit(self, X, y=None):
        # Calcular valores promedio del dataset de entrenamiento
        self.valores_default = {
            'viajes_totales': 25,
            'semanas_activas': 10,
            'viajes_por_semana': 2.5,
            'duracion_promedio_min': 20.0,
            'variedad_destinos': 8,
            'variedad_origenes': 5,
            'consistencia_horaria': 3.0,
            'distancia_promedio_usuario': 0.025,
            'dia_favorito': 0,
            'frecuencia_lunes': 5,
            'frecuencia_martes': 4,
            'frecuencia_miercoles': 4,
            'frecuencia_jueves': 4,
            'frecuencia_viernes': 5,
            'frecuencia_sabado': 3,
            'frecuencia_domingo': 2
        }
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Agregar features de usuario si no existen
        for feature, default in self.valores_default.items():
            if feature not in X.columns:
                X[feature] = default
        
        # Asegurar que semanas_activas no sea 0 para evitar división por cero
        X['semanas_activas'] = X['semanas_activas'].replace(0, 1)
        
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Selecciona las features finales en el orden correcto"""
    
    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Asegurar que todas las features existan
        missing_features = [f for f in self.features if f not in X.columns]
        if missing_features:
            # Agregar features faltantes con valores por defecto
            for f in missing_features:
                X[f] = 0
        
        # Seleccionar features en el orden correcto
        return X[self.features]


# ============================================================================
# FUNCIONES DE CARGA Y PROCESAMIENTO
# ============================================================================

def load_model():
    """Carga el modelo Random Forest entrenado"""
    # Intentar diferentes rutas posibles
    model_paths = [
        "static/modelo_random_forest_final_tunado.pkl",
        "modelo_random_forest_final_tunado.pkl",
        "../prediccion/modelo_random_forest_final_tunado.pkl"
    ]
    
    for model_path in model_paths:
        try:
            if os.path.exists(model_path):
                modelo = joblib.load(model_path)
                return modelo
        except Exception as e:
            continue
    
    # Si no se encuentra, mostrar advertencia pero no error
    try:
        import streamlit as st
        st.warning(f"⚠️ No se encontró el modelo. Algunas funcionalidades no estarán disponibles. Por favor, asegúrate de que el modelo esté en static/")
    except:
        pass
    return None


def load_preprocessor():
    """Carga el preprocessor guardado"""
    # Intentar diferentes rutas posibles
    preprocessor_paths = [
        "static/preprocessor.pkl",
        "preprocessor.pkl",
        "../prediccion/preprocessor.pkl"
    ]
    
    for preprocessor_path in preprocessor_paths:
        try:
            if os.path.exists(preprocessor_path):
                preprocessor = joblib.load(preprocessor_path)
                return preprocessor
        except Exception as e:
            continue
    
    # Si no existe, crear uno nuevo
    try:
        return create_preprocessor()
    except Exception as e:
        try:
            import streamlit as st
            st.warning(f"No se pudo crear el preprocessor: {e}")
        except:
            pass
        return None


def create_preprocessor():
    """Crea un preprocessor con todos los transformers"""
    from sklearn.pipeline import Pipeline
    
    # Features finales en el orden correcto
    features_finales = [
        'origen_lat', 'origen_lon',
        'hora_salida', 'dia_semana', 'mes',
        'viajes_totales', 'semanas_activas', 'viajes_por_semana', 'duracion_promedio_min',
        'periodo_dia_numerico', 'es_fin_semana', 'es_hora_pico', 'zona_origen',
        'capacidad_origen', 'estaciones_cercanas_origen', 'variedad_destinos', 'variedad_origenes',
        'consistencia_horaria', 'distancia_promedio_usuario', 'dia_favorito',
        'frecuencia_lunes', 'frecuencia_martes', 'frecuencia_miercoles',
        'frecuencia_jueves', 'frecuencia_viernes', 'frecuencia_sabado', 'frecuencia_domingo'
    ]
    
    # Intentar cargar datos de estaciones si existen
    estaciones_data = None
    try:
        estaciones_path = "../prediccion/station_data_enriched (1).csv"
        estaciones_data = pd.read_csv(estaciones_path)
        # Normalizar nombres de columnas
        if 'station_lat' not in estaciones_data.columns:
            if 'lat' in estaciones_data.columns:
                estaciones_data['station_lat'] = estaciones_data['lat']
        if 'station_lon' not in estaciones_data.columns:
            if 'lon' in estaciones_data.columns:
                estaciones_data['station_lon'] = estaciones_data['lon']
    except:
        pass
    
    preprocessor = Pipeline([
        ('temporal', FeatureEngineeringTemporal()),
        ('geografica', FeatureEngineeringGeografica(estaciones_data=estaciones_data)),
        ('usuario', FeatureEngineeringUsuario()),
        ('selector', FeatureSelector(features_finales))
    ])
    
    # Fit con datos dummy para inicializar
    dummy_data = pd.DataFrame({
        'origen_lat': [-32.89],
        'origen_lon': [-68.84],
        'hora_salida': [8],
        'dia_semana': [0],
        'mes': [3]
    })
    preprocessor.fit(dummy_data)
    
    return preprocessor


def process_input(input_data: dict, preprocessor):
    """Procesa datos de entrada del usuario"""
    # Convertir a DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Aplicar preprocessor
    X_processed = preprocessor.transform(df_input)
    
    return X_processed


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def render_feature_importance(modelo, top_n=15):
    """Visualiza importancia de características con Altair"""
    if not hasattr(modelo, 'feature_importances_'):
        st.info("El modelo no tiene información de importancia de características.")
        return
    
    importance = modelo.feature_importances_
    feature_names = modelo.feature_names_in_ if hasattr(modelo, 'feature_names_in_') else [f'feature_{i}' for i in range(len(importance))]
    
    # Crear DataFrame
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Crear visualización con Altair
    chart = (
        alt.Chart(imp_df)
        .mark_bar()
        .encode(
            x=alt.X('importance:Q', title='Importancia', axis=alt.Axis(format='.4f')),
            y=alt.Y('feature:N', sort='-x', title='Característica'),
            tooltip=['feature', alt.Tooltip('importance:Q', format='.4f')],
            color=alt.Color('importance:Q', scale=alt.Scale(scheme='blues'), legend=None)
        )
        .properties(
            width=700,
            height=400,
            title='Top 15 Características Más Importantes'
        )
    )
    
    st.altair_chart(chart, width='stretch')



