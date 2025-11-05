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
    
    # Calcular destino_favorito por usuario (destino más frecuente de cada usuario)
    print("\nCalculando destino favorito por usuario...")
    destino_favorito_por_usuario = df.groupby('Usuario_key')['destino'].agg(
        lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else None
    ).to_dict()
    
    df['destino_favorito'] = df['Usuario_key'].map(destino_favorito_por_usuario)
    
    # Codificar destino_favorito con LabelEncoder
    print("Codificando destino favorito...")
    label_encoder = LabelEncoder()
    
    # Asegurar que todos los destinos favoritos sean strings y no nulos
    destinos_favoritos_validos = df['destino_favorito'].dropna().astype(str)
    label_encoder.fit(destinos_favoritos_validos.unique())
    
    # Codificar
    mask_notna = df['destino_favorito'].notna()
    df.loc[mask_notna, 'destino_favorito_encoded'] = label_encoder.transform(
        df.loc[mask_notna, 'destino_favorito'].astype(str)
    )
    df.loc[~mask_notna, 'destino_favorito_encoded'] = 0
    
    print(f"[OK] LabelEncoder creado con {len(label_encoder.classes_)} clases")
    
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
    print("  - n_estimators: 100 (vs 200 original)")
    print("  - max_depth: 15 (vs 20 original)")
    print("  - min_samples_split: 15 (vs 10 original)")
    print("  - min_samples_leaf: 5 (vs 3 original)")
    print("  - max_features: 0.5 (igual)")
    print("=" * 70)
    
    modelo = RandomForestClassifier(
        n_estimators=100,        # Reducido de 200 a 100 (reduce tamaño ~2x)
        max_depth=15,             # Reducido de 20 a 15 (reduce profundidad)
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
