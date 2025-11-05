#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para procesar el archivo de estaciones y guardarlo en static/
- Extrae estaciones únicas
- Guarda en formato JSON para fácil acceso
"""

import pandas as pd
import json
import os

def main():
    print("=" * 70)
    print("PROCESAMIENTO DE ESTACIONES")
    print("=" * 70)
    
    # Rutas posibles del archivo CSV
    csv_paths = [
        "../prediccion/station_data_enriched (1).csv",
        "prediccion/station_data_enriched (1).csv",
        "station_data_enriched (1).csv"
    ]
    
    df_estaciones = None
    csv_path_usado = None
    
    for csv_path in csv_paths:
        try:
            if os.path.exists(csv_path):
                df_estaciones = pd.read_csv(csv_path)
                csv_path_usado = csv_path
                print(f"[OK] CSV cargado desde: {csv_path}")
                print(f"     Total de registros: {len(df_estaciones):,}")
                break
        except Exception as e:
            continue
    
    if df_estaciones is None:
        print("[ERROR] No se encontró el archivo CSV de estaciones")
        print("Rutas probadas:")
        for path in csv_paths:
            print(f"  - {path}")
        return
    
    # Normalizar nombres de columnas
    if 'station_name' not in df_estaciones.columns:
        if 'name' in df_estaciones.columns:
            df_estaciones['station_name'] = df_estaciones['name']
    
    if 'station_lat' not in df_estaciones.columns:
        if 'lat' in df_estaciones.columns:
            df_estaciones['station_lat'] = df_estaciones['lat']
    
    if 'station_lon' not in df_estaciones.columns:
        if 'lon' in df_estaciones.columns:
            df_estaciones['station_lon'] = df_estaciones['lon']
    
    # Verificar que tenemos las columnas necesarias
    if 'station_name' not in df_estaciones.columns:
        print("[ERROR] No se encontró la columna 'station_name' o 'name'")
        return
    
    if 'station_lat' not in df_estaciones.columns:
        print("[ERROR] No se encontró la columna 'station_lat' o 'lat'")
        return
    
    if 'station_lon' not in df_estaciones.columns:
        print("[ERROR] No se encontró la columna 'station_lon' o 'lon'")
        return
    
    # Estaciones a excluir
    estaciones_excluidas = ["Hub-prueba", "TALLER BICITRAN"]
    
    # Extraer estaciones únicas por nombre
    print("\nProcesando estaciones únicas...")
    
    estaciones_dict = {}
    estaciones_por_nombre = {}
    
    for _, row in df_estaciones.iterrows():
        nombre = row.get('station_name', None)
        lat = row.get('station_lat', None)
        lon = row.get('station_lon', None)
        capacidad = row.get('station_capacity', row.get('capacity', 15))
        
        if pd.notna(nombre) and pd.notna(lat) and pd.notna(lon):
            # Limpiar el nombre
            nombre_limpio = str(nombre).strip()
            
            # Excluir estaciones no deseadas
            if nombre_limpio in estaciones_excluidas:
                continue
            
            # Si ya existe este nombre, verificar si son las mismas coordenadas
            if nombre_limpio in estaciones_por_nombre:
                # Verificar si las coordenadas son similares (dentro de 0.001 grados)
                lat_existente = estaciones_por_nombre[nombre_limpio]['lat']
                lon_existente = estaciones_por_nombre[nombre_limpio]['lon']
                
                if abs(float(lat) - lat_existente) < 0.001 and abs(float(lon) - lon_existente) < 0.001:
                    # Es la misma estación, mantener la que tenga mayor capacidad
                    if capacidad > estaciones_por_nombre[nombre_limpio]['capacidad']:
                        estaciones_por_nombre[nombre_limpio] = {
                            'lat': float(lat),
                            'lon': float(lon),
                            'capacidad': int(capacidad) if pd.notna(capacidad) else 15
                        }
                    continue
                else:
                    # Mismo nombre pero diferentes coordenadas - agregar sufijo
                    nombre_limpio = f"{nombre_limpio} ({lat:.5f}, {lon:.5f})"
            
            # Agregar estación única
            estaciones_por_nombre[nombre_limpio] = {
                'lat': float(lat),
                'lon': float(lon),
                'capacidad': int(capacidad) if pd.notna(capacidad) else 15
            }
    
    print(f"[OK] Estaciones únicas encontradas: {len(estaciones_por_nombre)}")
    print(f"[OK] Estaciones excluidas: {estaciones_excluidas}")
    
    # Crear carpeta static si no existe
    if not os.path.exists("static"):
        os.makedirs("static")
        print("[OK] Carpeta static creada")
    
    # Guardar en JSON
    json_path = "static/estaciones.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(estaciones_por_nombre, f, ensure_ascii=False, indent=2)
    
    # Verificar tamaño del archivo
    file_size = os.path.getsize(json_path)
    file_size_kb = file_size / 1024
    
    print(f"\n[OK] Estaciones guardadas en: {json_path}")
    print(f"     Tamaño: {file_size_kb:.2f} KB")
    print(f"     Total de estaciones: {len(estaciones_por_nombre)}")
    
    # Mostrar algunas estaciones de ejemplo
    print("\n[EJEMPLO] Primeras 5 estaciones:")
    for i, (nombre, datos) in enumerate(list(estaciones_por_nombre.items())[:5], 1):
        print(f"  {i}. {nombre}")
        print(f"     Lat: {datos['lat']:.5f}, Lon: {datos['lon']:.5f}, Capacidad: {datos['capacidad']}")
    
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

