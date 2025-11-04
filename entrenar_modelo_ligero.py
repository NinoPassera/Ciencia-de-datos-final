#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para entrenar una versión más ligera del modelo Random Forest
Optimizado para reducir el tamaño del archivo manteniendo buen rendimiento
- Hiperparámetros reducidos: n_estimators=200, max_depth=20
- Compresión de joblib para reducir tamaño del archivo
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
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

def main():
    print("=" * 70)
    print("ENTRENAMIENTO DE MODELO LIGERO PARA STREAMLIT")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    RANDOM_SEED = 42
    
    # Verificar que existe el dataset
    dataset_path = "dataset_modelo_final.csv"
    if not os.path.exists(dataset_path):
        # Intentar desde el directorio padre
        dataset_path = "../prediccion/dataset_modelo_final.csv"
        if not os.path.exists(dataset_path):
            print("[ERROR] No se encontró dataset_modelo_final.csv")
            print("Por favor, asegúrate de que el dataset esté disponible")
            return
    
    # Cargar datos
    print("\nCargando dataset final...")
    df = pd.read_csv(dataset_path)
    print(f"[OK] Dataset cargado: {len(df):,} registros")
    
    # Features (27 características en el orden correcto)
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
    features = features_originales + features_mejoradas
    
    X = df[features].fillna(0)
    y = df['destino'].astype(str)
    
    print(f"\nFeatures: {len(features)}")
    print(f"Destinos únicos: {y.nunique()}")
    
    # Split
    print("\nDividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Entrenamiento: {len(X_train):,}")
    print(f"Prueba: {len(X_test):,}")
    
    # Modelo RF ULTRA LIGERO (hiperparámetros reducidos para <100MB)
    print("\n" + "=" * 70)
    print("CONFIGURACIÓN DEL MODELO ULTRA LIGERO")
    print("=" * 70)
    print("Hiperparámetros (reducidos para <100MB y Git normal):")
    print("  - n_estimators: 100 (vs 600 original)")
    print("  - max_depth: 15 (vs 32 original)")
    print("  - min_samples_split: 15 (vs 10 original)")
    print("  - min_samples_leaf: 5 (vs 1 original)")
    print("  - max_features: 0.5 (igual)")
    print("=" * 70)
    
    modelo = RandomForestClassifier(
        n_estimators=100,        # Reducido de 200 a 100 (reduce tamaño ~2x)
        max_depth=15,            # Reducido de 20 a 15 (reduce profundidad)
        min_samples_split=15,    # Aumentado de 10 a 15 (árboles más pequeños)
        min_samples_leaf=5,      # Aumentado de 3 a 5 (árboles más compactos)
        max_features=0.5,        # Mantener igual
        bootstrap=True,
        oob_score=True,
        class_weight=None,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    # Entrenamiento
    print("\nEntrenando modelo ligero...")
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
    importances = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=False)
    print(f"\n[TOP 10 FEATURES IMPORTANTES]")
    for i, (feature, importance) in enumerate(importances.head(10).items(), 1):
        tag = "[NUEVA]" if feature in features_mejoradas else "[ORIGINAL]"
        print(f"{i:2d}. {tag} {feature}: {importance:.4f}")
    
    # Guardar modelo CON COMPRESIÓN
    print("\n" + "=" * 70)
    print("GUARDANDO MODELO CON COMPRESIÓN")
    print("=" * 70)
    
    # Crear carpeta static si no existe
    if not os.path.exists("static"):
        os.makedirs("static")
        print("[OK] Carpeta static creada")
    
    model_file = "static/modelo_random_forest_final_tunado.pkl"
    
    # Guardar con compresión (reduce tamaño significativamente)
    print(f"\nGuardando modelo en: {model_file}")
    print("Usando compresión de joblib para reducir tamaño...")
    
    t0 = time.time()
    joblib.dump(modelo, model_file, compress=3)  # compress=3 es un buen balance
    tiempo_guardado = time.time() - t0
    
    # Verificar tamaño del archivo
    file_size = os.path.getsize(model_file)
    file_size_mb = file_size / (1024 * 1024)
    file_size_gb = file_size / (1024 * 1024 * 1024)
    
    print(f"[OK] Modelo guardado en {tiempo_guardado:.2f} segundos")
    if file_size_gb >= 1:
        print(f"[INFO] Tamaño del archivo: {file_size_gb:.2f} GB")
    else:
        print(f"[INFO] Tamaño del archivo: {file_size_mb:.2f} MB")
    
    # Comparación con modelo original (si existe)
    original_model_path = "../prediccion/modelo_random_forest_final_tunado.pkl"
    if os.path.exists(original_model_path):
        original_size = os.path.getsize(original_model_path)
        original_size_gb = original_size / (1024 * 1024 * 1024)
        reduccion = (1 - file_size / original_size) * 100
        print(f"\n[COMPARACIÓN]")
        print(f"Modelo original: {original_size_gb:.2f} GB")
        if file_size_gb >= 1:
            print(f"Modelo ligero: {file_size_gb:.2f} GB")
        else:
            print(f"Modelo ligero: {file_size_mb:.2f} MB")
        print(f"Reducción: {reduccion:.1f}%")
    
    print("\n" + "=" * 70)
    print("[OK] PROCESO COMPLETADO")
    print("=" * 70)
    print(f"\nEl modelo está listo para usar en Streamlit!")
    print(f"Archivo: {model_file}")
    
    # Resumen final
    print(f"\n[RESUMEN]")
    print(f"  - Accuracy: {acc*100:.2f}%")
    print(f"  - OOB score: {modelo.oob_score_*100:.2f}%")
    if file_size_gb >= 1:
        print(f"  - Tamaño: {file_size_gb:.2f} GB")
    else:
        print(f"  - Tamaño: {file_size_mb:.2f} MB")
    print(f"  - Tiempo entrenamiento: {tiempo_entrenamiento:.2f} segundos")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

