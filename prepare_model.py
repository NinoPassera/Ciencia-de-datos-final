"""
Script para preparar el modelo y preprocessor para la app Streamlit
Copia el modelo desde la carpeta prediccion y crea el preprocessor
"""

import os
import shutil
import joblib
import pandas as pd
from lib import create_preprocessor

def main():
    print("=" * 70)
    print("PREPARACIÓN DE MODELO Y PREPROCESSOR PARA STREAMLIT")
    print("=" * 70)
    
    # Verificar que existe la carpeta static
    if not os.path.exists("static"):
        os.makedirs("static")
        print("✓ Carpeta static creada")
    
    # Copiar modelo desde prediccion
    modelo_sources = [
        "../prediccion/modelo_random_forest_final_tunado.pkl",
        "../../prediccion/modelo_random_forest_final_tunado.pkl",
        "prediccion/modelo_random_forest_final_tunado.pkl"
    ]
    modelo_dest = "static/modelo_random_forest_final_tunado.pkl"
    
    modelo_copiado = False
    for modelo_source in modelo_sources:
        if os.path.exists(modelo_source):
            print(f"\nCopiando modelo desde {modelo_source}...")
            try:
                shutil.copy2(modelo_source, modelo_dest)
                print(f"✓ Modelo copiado a {modelo_dest}")
                modelo_copiado = True
                break
            except Exception as e:
                print(f"✗ Error al copiar modelo: {e}")
                continue
    
    if not modelo_copiado:
        print(f"⚠ No se encontró el modelo en ninguna de las rutas probadas:")
        for path in modelo_sources:
            print(f"  - {path}")
        print("Por favor, asegúrate de que el modelo esté en la carpeta prediccion/")
        print("O copia manualmente el modelo a static/modelo_random_forest_final_tunado.pkl")
    
    # Crear y guardar preprocessor
    print("\nCreando preprocessor...")
    try:
        preprocessor = create_preprocessor()
        preprocessor_path = "static/preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_path)
        print(f"✓ Preprocessor guardado en {preprocessor_path}")
    except Exception as e:
        print(f"✗ Error al crear preprocessor: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("✓ Preparación completada")
    print("=" * 70)
    print("\nPuedes ejecutar la app con: streamlit run app.py")

if __name__ == "__main__":
    main()

