#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para procesar usuarios del dataset y guardarlos en static/
- Extrae usuarios únicos con sus métricas
- Guarda en formato JSON para fácil acceso
"""

import pandas as pd
import json
import os

def main():
    print("=" * 70)
    print("PROCESAMIENTO DE USUARIOS")
    print("=" * 70)
    
    # Rutas posibles del archivo CSV (priorizar el de prediccion que tiene las nuevas columnas)
    csv_paths = [
        "../prediccion/dataset_modelo_final.csv",
        "prediccion/dataset_modelo_final.csv",
        "dataset_modelo_final.csv"
    ]
    
    df = None
    csv_path_usado = None
    
    for csv_path in csv_paths:
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                csv_path_usado = csv_path
                print(f"[OK] CSV cargado desde: {csv_path}")
                print(f"     Total de registros: {len(df):,}")
                break
        except Exception as e:
            continue
    
    if df is None:
        print("[ERROR] No se encontró dataset_modelo_final.csv")
        print("Rutas probadas:")
        for path in csv_paths:
            print(f"  - {path}")
        return
    
    # Verificar que tiene Usuario_key
    if 'Usuario_key' not in df.columns:
        print("[ADVERTENCIA] No se encontró Usuario_key. Calculando...")
        df['Usuario_key'] = (
            df['origen_lat'].round(4).astype(str) + '_' +
            df['origen_lon'].round(4).astype(str) + '_' +
            df['viajes_totales'].astype(str) + '_' +
            df['semanas_activas'].astype(str)
        )
    
    # Agrupar por usuario y obtener métricas promedio/únicas
    print("\nProcesando usuarios únicos...")
    
    usuarios_resumen = df.groupby('Usuario_key').agg({
        'viajes_totales': 'first',
        'semanas_activas': 'first',
        'viajes_por_semana': 'first',
        'duracion_promedio_min': 'mean',
        'variedad_destinos': 'first',
        'variedad_origenes': 'first',
        'consistencia_horaria': 'mean',
        'distancia_promedio_usuario': 'mean',
        'dia_favorito': lambda x: x.mode()[0] if len(x.mode()) > 0 else 0,
        'frecuencia_lunes': 'first',
        'frecuencia_martes': 'first',
        'frecuencia_miercoles': 'first',
        'frecuencia_jueves': 'first',
        'frecuencia_viernes': 'first',
        'frecuencia_sabado': 'first',
        'frecuencia_domingo': 'first'
    }).reset_index()
    
    # Agregar coordenadas de destino favorito si existen en el CSV
    if 'lat_destino_favorito' in df.columns and 'lon_destino_favorito' in df.columns:
        usuarios_resumen = usuarios_resumen.merge(
            df.groupby('Usuario_key').agg({
                'lat_destino_favorito': 'first',
                'lon_destino_favorito': 'first'
            }).reset_index(),
            on='Usuario_key',
            how='left'
        )
    else:
        usuarios_resumen['lat_destino_favorito'] = None
        usuarios_resumen['lon_destino_favorito'] = None
    
    # Crear diccionario de usuarios
    usuarios_dict = {}
    
    # Cargar estaciones para obtener nombre del destino favorito
    estaciones_dict = {}
    estaciones_paths = [
        "static/estaciones.json",
        "estaciones.json",
        "../prediccion/estaciones.json"
    ]
    
    for path in estaciones_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    estaciones_json = json.load(f)
                    # Convertir a diccionario: (lat, lon) -> nombre
                    for nombre, datos in estaciones_json.items():
                        if isinstance(datos, dict) and 'lat' in datos and 'lon' in datos:
                            lat = round(float(datos['lat']), 5)
                            lon = round(float(datos['lon']), 5)
                            estaciones_dict[(lat, lon)] = nombre
                break
            except:
                continue
    
    # Función para buscar estación por coordenadas
    def buscar_estacion_por_coordenadas(lat, lon):
        if pd.isna(lat) or pd.isna(lon):
            return None
        key = (round(float(lat), 5), round(float(lon), 5))
        if key in estaciones_dict:
            return estaciones_dict[key]
        # Buscar más cercana
        min_dist = float('inf')
        estacion_cercana = None
        for (est_lat, est_lon), nombre in estaciones_dict.items():
            dist = ((lat - est_lat)**2 + (lon - est_lon)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                estacion_cercana = nombre
        return estacion_cercana if min_dist < 0.01 else None
    
    # Procesar cada usuario
    for _, row in usuarios_resumen.iterrows():
        usuario_key = row['Usuario_key']
        
        # Obtener destino favorito por coordenadas
        destino_favorito = None
        if pd.notna(row['lat_destino_favorito']) and pd.notna(row['lon_destino_favorito']):
            destino_favorito = buscar_estacion_por_coordenadas(
                row['lat_destino_favorito'],
                row['lon_destino_favorito']
            )
        
        # Crear nombre descriptivo para el usuario
        viajes = int(row['viajes_totales']) if pd.notna(row['viajes_totales']) else 0
        semanas = int(row['semanas_activas']) if pd.notna(row['semanas_activas']) else 0
        
        # El usuario_key es el nombre real del usuario
        nombre_real = usuario_key
        
        # Crear nombre descriptivo con nombre real, tipo y viajes
        if viajes < 10:
            tipo = "Ocasional"
        elif viajes < 30:
            tipo = "Regular"
        elif viajes < 50:
            tipo = "Frecuente"
        else:
            tipo = "Activo"
        
        nombre_usuario = f"{nombre_real} - {tipo} ({viajes} viajes)"
        
        # Crear diccionario del usuario
        usuarios_dict[usuario_key] = {
            'nombre': nombre_usuario,
            'viajes_totales': int(row['viajes_totales']) if pd.notna(row['viajes_totales']) else 25,
            'semanas_activas': int(row['semanas_activas']) if pd.notna(row['semanas_activas']) else 10,
            'viajes_por_semana': float(row['viajes_por_semana']) if pd.notna(row['viajes_por_semana']) else 2.5,
            'duracion_promedio_min': float(row['duracion_promedio_min']) if pd.notna(row['duracion_promedio_min']) else 20.0,
            'variedad_destinos': int(row['variedad_destinos']) if pd.notna(row['variedad_destinos']) else 8,
            'variedad_origenes': int(row['variedad_origenes']) if pd.notna(row['variedad_origenes']) else 5,
            'consistencia_horaria': float(row['consistencia_horaria']) if pd.notna(row['consistencia_horaria']) else 3.0,
            'distancia_promedio_usuario': float(row['distancia_promedio_usuario']) if pd.notna(row['distancia_promedio_usuario']) else 0.025,
            'dia_favorito': int(row['dia_favorito']) if pd.notna(row['dia_favorito']) else 0,
            'frecuencia_lunes': int(row['frecuencia_lunes']) if pd.notna(row['frecuencia_lunes']) else 5,
            'frecuencia_martes': int(row['frecuencia_martes']) if pd.notna(row['frecuencia_martes']) else 4,
            'frecuencia_miercoles': int(row['frecuencia_miercoles']) if pd.notna(row['frecuencia_miercoles']) else 4,
            'frecuencia_jueves': int(row['frecuencia_jueves']) if pd.notna(row['frecuencia_jueves']) else 4,
            'frecuencia_viernes': int(row['frecuencia_viernes']) if pd.notna(row['frecuencia_viernes']) else 5,
            'frecuencia_sabado': int(row['frecuencia_sabado']) if pd.notna(row['frecuencia_sabado']) else 3,
            'frecuencia_domingo': int(row['frecuencia_domingo']) if pd.notna(row['frecuencia_domingo']) else 2,
            'destino_favorito': destino_favorito
        }
    
    # Limitar a usuarios más representativos (top 50 por viajes)
    usuarios_ordenados = sorted(
        usuarios_dict.items(),
        key=lambda x: x[1]['viajes_totales'],
        reverse=True
    )[:50]  # Top 50 usuarios
    
    usuarios_dict_final = dict(usuarios_ordenados)
    
    # Guardar en JSON
    output_file = "static/usuarios.json"
    
    # Crear carpeta static si no existe
    if not os.path.exists("static"):
        os.makedirs("static")
        print("[OK] Carpeta static creada")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(usuarios_dict_final, f, indent=2, ensure_ascii=False)
    
    file_size_kb = os.path.getsize(output_file) / 1024
    
    print(f"\n[OK] Usuarios procesados: {len(usuarios_dict_final)}")
    print(f"[OK] Archivo guardado en: {output_file}")
    print(f"     Tamaño: {file_size_kb:.2f} KB")
    
    print("\n" + "=" * 70)
    print("[OK] PROCESO COMPLETADO")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

