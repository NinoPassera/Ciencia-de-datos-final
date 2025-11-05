#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para entrenar modelo Random Forest con feature de destino favorito
Incluye:
- Cálculo de destino_favorito por usuario
- Codificación con LabelEncoder
- Modelo con 28 features (27 originales + destino_favorito_encoded)
"""

import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

def main():
    print("=" * 70)
    print("ENTRENAMIENTO DE MODELO CON DESTINO FAVORITO")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    RANDOM_SEED = 42
    
    # Verificar que existe el dataset
    dataset_path = "prediccion/dataset_modelo_final.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "../prediccion/dataset_modelo_final.csv"
        if not os.path.exists(dataset_path):
            print("[ERROR] No se encontró dataset_modelo_final.csv")
            return
    
    # Cargar datos
    print("\nCargando dataset final...")
    df = pd.read_csv(dataset_path)
    print(f"[OK] Dataset cargado: {len(df):,} registros")
    
    # Verificar si existe Usuario_key, si no, calcularlo
    if 'Usuario_key' not in df.columns:
        print("[ADVERTENCIA] No se encontró Usuario_key. Calculando desde datos originales...")
        # Calcular Usuario_key sintético basado en características únicas
        df['Usuario_key'] = (
            df['origen_lat'].round(4).astype(str) + '_' +
            df['origen_lon'].round(4).astype(str) + '_' +
            df['viajes_totales'].astype(str) + '_' +
            df['semanas_activas'].astype(str)
        )
    
    # Usar coordenadas de destino favorito del CSV (si existen)
    if 'lat_destino_favorito' in df.columns and 'lon_destino_favorito' in df.columns:
        print("\nUsando coordenadas de destino favorito del CSV...")
        
        # Cargar estaciones para buscar el nombre del destino favorito por coordenadas
        import json
        estaciones_dict = {}
        
        # Intentar cargar estaciones desde JSON
        estaciones_dict_coords = {}  # Diccionario: (lat, lon) -> nombre
        estaciones_paths = [
            "static/estaciones.json",
            "../prediccion/estaciones.json",
            "estaciones.json"
        ]
        
        for path in estaciones_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        estaciones_json = json.load(f)
                    # Convertir formato JSON a diccionario de coordenadas
                    for nombre, datos in estaciones_json.items():
                        if isinstance(datos, dict) and 'lat' in datos and 'lon' in datos:
                            lat = round(float(datos['lat']), 5)
                            lon = round(float(datos['lon']), 5)
                            estaciones_dict_coords[(lat, lon)] = nombre
                    if estaciones_dict_coords:
                        break
                except:
                    continue
        
        # Si no hay JSON, intentar cargar desde CSV de estaciones
        if not estaciones_dict_coords:
            estaciones_csv_paths = [
                "../prediccion/station_data_enriched (1).csv",
                "prediccion/station_data_enriched (1).csv",
                "station_data_enriched (1).csv"
            ]
            for path in estaciones_csv_paths:
                if os.path.exists(path):
                    try:
                        df_estaciones = pd.read_csv(path)
                        for _, row in df_estaciones.iterrows():
                            lat = row.get('station_lat', row.get('lat', None))
                            lon = row.get('station_lon', row.get('lon', None))
                            nombre = row.get('station_name', row.get('name', ''))
                            if pd.notna(lat) and pd.notna(lon) and nombre:
                                key = (round(float(lat), 5), round(float(lon), 5))
                                estaciones_dict_coords[key] = str(nombre).strip()
                        if estaciones_dict_coords:
                            break
                    except:
                        continue
        
        # Buscar estación más cercana a las coordenadas de destino favorito
        def buscar_estacion_por_coordenadas(lat, lon):
            if pd.isna(lat) or pd.isna(lon):
                return None
            # Buscar estación exacta o más cercana
            key = (round(float(lat), 5), round(float(lon), 5))
            if key in estaciones_dict_coords:
                return estaciones_dict_coords[key]
            # Si no hay exacta, buscar la más cercana
            min_dist = float('inf')
            estacion_cercana = None
            for (est_lat, est_lon), nombre in estaciones_dict_coords.items():
                dist = ((lat - est_lat)**2 + (lon - est_lon)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    estacion_cercana = nombre
            return estacion_cercana if min_dist < 0.01 else None  # Tolerancia de ~1km
        
        # Aplicar búsqueda de estación por coordenadas
        mask_coords = df['lat_destino_favorito'].notna() & df['lon_destino_favorito'].notna()
        df.loc[mask_coords, 'destino_favorito'] = df.loc[mask_coords].apply(
            lambda row: buscar_estacion_por_coordenadas(
                row['lat_destino_favorito'], 
                row['lon_destino_favorito']
            ), axis=1
        )
        
        print(f"[OK] Destino favorito encontrado para {df['destino_favorito'].notna().sum():,} registros")
    else:
        # Si no hay coordenadas, calcular destino favorito por usuario (fallback)
        print("\n[ADVERTENCIA] No se encontraron coordenadas de destino favorito. Calculando desde destinos...")
        destino_favorito_por_usuario = df.groupby('Usuario_key')['destino'].agg(
            lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else None
        ).to_dict()
        df['destino_favorito'] = df['Usuario_key'].map(destino_favorito_por_usuario)
    
    # Codificar destino_favorito con LabelEncoder
    print("Codificando destino favorito...")
    label_encoder = LabelEncoder()
    
    # Asegurar que todos los destinos favoritos sean strings y no nulos
    destinos_favoritos_validos = df['destino_favorito'].dropna().astype(str)
    if len(destinos_favoritos_validos) > 0:
        label_encoder.fit(destinos_favoritos_validos.unique())
        
        # Codificar
        mask_notna = df['destino_favorito'].notna()
        df.loc[mask_notna, 'destino_favorito_encoded'] = label_encoder.transform(
            df.loc[mask_notna, 'destino_favorito'].astype(str)
        )
        df.loc[~mask_notna, 'destino_favorito_encoded'] = 0
        
        print(f"[OK] LabelEncoder creado con {len(label_encoder.classes_)} clases")
    else:
        print("[ADVERTENCIA] No se encontraron destinos favoritos válidos")
        df['destino_favorito_encoded'] = 0
    
    # Features base (27 características originales)
    features_originales = [
        'origen_lat','origen_lon',
        'hora_salida','dia_semana','mes',
        'viajes_totales','semanas_activas','viajes_por_semana','duracion_promedio_min'
    ]
    features_mejoradas = [
        'periodo_dia_numerico','es_fin_semana','es_hora_pico','zona_origen',
        'capacidad_origen','estaciones_cercanas_origen','variedad_destinos','variedad_origenes',
        'consistencia_horaria','distancia_promedio_usuario','dia_favorito',
        'frecuencia_lunes','frecuencia_martes','frecuencia_miercoles',
        'frecuencia_jueves','frecuencia_viernes','frecuencia_sabado','frecuencia_domingo'
    ]
    features_base = features_originales + features_mejoradas
    
    # Agregar destino_favorito_encoded
    features_finales = features_base + ['destino_favorito_encoded']
    
    X = df[features_finales].fillna(0)
    y = df['destino'].astype(str)
    
    print(f"\nFeatures base: {len(features_base)}")
    print(f"Features finales (con destino_favorito_encoded): {len(features_finales)}")
    print(f"Destinos únicos: {y.nunique()}")
    
    # Filtrar destinos con muy pocos registros (necesarios para estratificación y reducir tamaño)
    print("\nFiltrando destinos con pocos registros...")
    destino_counts = y.value_counts()
    # Filtrar destinos con al menos 50 registros (más agresivo para reducir clases y tamaño <100MB)
    destinos_validos = destino_counts[destino_counts >= 50].index
    mask_validos = y.isin(destinos_validos)
    X = X[mask_validos].reset_index(drop=True)
    y = y[mask_validos].reset_index(drop=True)
    
    print(f"Registros después de filtrar: {len(X):,}")
    print(f"Destinos únicos después de filtrar: {y.nunique()}")
    
    # Split
    print("\nDividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Entrenamiento: {len(X_train):,}")
    print(f"Prueba: {len(X_test):,}")
    
    # Modelo Random Forest (optimizado para tamaño <100MB)
    print("\n" + "=" * 70)
    print("ENTRENANDO MODELO RANDOM FOREST")
    print("=" * 70)
    print("Hiperparámetros (optimizados para <100MB):")
    print("  - n_estimators: 100 (balance tamaño/performance)")
    print("  - max_depth: 15 (balance)")
    print("  - min_samples_split: 15 (balance)")
    print("  - min_samples_leaf: 5 (balance)")
    print("  - max_features: 0.5 (igual)")
    print("  - Filtro: destinos con >= 50 registros (reduce clases y tamaño <100MB)")
    print("=" * 70)
    
    modelo = RandomForestClassifier(
        n_estimators=100,         # Volver a 100 (equilibrio tamaño/performance)
        max_depth=15,             # Volver a 15 (equilibrio)
        min_samples_split=15,    # Volver a 15 (equilibrio)
        min_samples_leaf=5,      # Volver a 5 (equilibrio)
        max_features=0.5,         # Mantener igual
        bootstrap=True,
        oob_score=True,
        class_weight=None,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    # Entrenamiento
    print("\nEntrenando modelo...")
    t0 = time.time()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - t0
    print(f"[OK] Entrenamiento completado en {tiempo_entrenamiento:.2f} segundos")
    print(f"OOB score: {modelo.oob_score_*100:.2f}%")
    
    # Evaluación
    print("\nEvaluando modelo...")
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    
    print(f"\n[RESULTADOS EN TEST]")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1 macro: {f1m*100:.2f}%")
    
    # Top-k
    if hasattr(modelo, "predict_proba"):
        proba_test = modelo.predict_proba(X_test)
        top3 = top_k_accuracy_score(y_test, proba_test, k=3, labels=modelo.classes_)
        top5 = top_k_accuracy_score(y_test, proba_test, k=5, labels=modelo.classes_)
        print(f"Top-3 accuracy: {top3*100:.2f}%")
        print(f"Top-5 accuracy: {top5*100:.2f}%")
    
    # Importancias
    importances = pd.Series(modelo.feature_importances_, index=features_finales).sort_values(ascending=False)
    print(f"\n[TOP 10 FEATURES IMPORTANTES]")
    for i, (feature, importance) in enumerate(importances.head(10).items(), 1):
        tag = "[NUEVA]" if feature == 'destino_favorito_encoded' else "[ORIGINAL]"
        print(f"{i:2d}. {tag} {feature}: {importance:.6f}")
    
    # Guardar modelo y LabelEncoder
    print("\n" + "=" * 70)
    print("GUARDANDO MODELO Y LABEL ENCODER")
    print("=" * 70)
    
    # Crear carpeta static si no existe (para app-streamlit)
    if not os.path.exists("static"):
        os.makedirs("static")
        print("[OK] Carpeta static creada")
    
    # Crear carpeta modelos si no existe (backup)
    if not os.path.exists("modelos"):
        os.makedirs("modelos")
        print("[OK] Carpeta modelos creada")
    
    # Guardar en static/ (para uso en app-streamlit)
    model_file = "static/modelo_con_destino_favorito.pkl"
    le_file = "static/label_encoder_destino_favorito.pkl"
    features_file = "modelos/features_con_destino_favorito.txt"
    
    # Guardar modelo con compresión (reduce tamaño significativamente)
    print(f"\nGuardando modelo en: {model_file}")
    print("Usando compresión de joblib para reducir tamaño...")
    joblib.dump(modelo, model_file, compress=3)
    
    # Guardar LabelEncoder
    print(f"Guardando LabelEncoder en: {le_file}")
    joblib.dump(label_encoder, le_file)
    
    # Guardar lista de features
    print(f"Guardando lista de features en: {features_file}")
    with open(features_file, 'w', encoding='utf-8') as f:
        f.write("Features del modelo con destino favorito:\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total features: {len(features_finales)}\n\n")
        f.write("Features originales:\n")
        for feat in features_originales:
            f.write(f"  - {feat}\n")
        f.write("\nFeatures mejoradas:\n")
        for feat in features_mejoradas:
            f.write(f"  - {feat}\n")
        f.write("\nFeatures nuevas:\n")
        f.write(f"  - destino_favorito_encoded\n")
    
    # Verificar tamaño
    file_size = os.path.getsize(model_file) / (1024 * 1024)
    print(f"\n[OK] Modelo guardado ({file_size:.2f} MB)")
    print(f"[OK] LabelEncoder guardado")
    print(f"[OK] Lista de features guardada")
    
    print("\n" + "=" * 70)
    print("[OK] PROCESO COMPLETADO")
    print("=" * 70)
    print(f"\nResumen:")
    print(f"  - Accuracy: {acc*100:.2f}%")
    print(f"  - OOB score: {modelo.oob_score_*100:.2f}%")
    print(f"  - Features: {len(features_finales)}")
    print(f"  - Importancia destino_favorito_encoded: {importances['destino_favorito_encoded']:.6f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
