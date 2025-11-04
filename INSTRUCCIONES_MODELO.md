# Instrucciones para Agregar el Modelo

El modelo final tunado (`modelo_random_forest_final_tunado.pkl`) es muy grande (~10.6 GB) y necesita ser agregado con Git LFS.

## Pasos para agregar el modelo:

1. **Asegúrate de tener espacio en disco** (al menos 11 GB libres)

2. **Copia el modelo a la carpeta static/**:
   ```bash
   cd app-streamlit
   Copy-Item ..\prediccion\modelo_random_forest_final_tunado.pkl static\modelo_random_forest_final_tunado.pkl
   ```

3. **Agrega el modelo con Git LFS**:
   ```bash
   git add static/modelo_random_forest_final_tunado.pkl
   git commit -m "Add final tuned model with Git LFS"
   git push origin main
   ```

4. **Verifica que Git LFS está configurado**:
   ```bash
   git lfs track "static/*.pkl"
   ```

## Nota importante:

- El modelo es muy grande, así que el push puede tardar varios minutos
- Necesitas tener Git LFS instalado: `git lfs install`
- GitHub tiene límites para Git LFS (1 GB gratis, luego pago)

## Alternativa (si no puedes subir el modelo):

La app funciona sin el modelo, mostrando advertencias. Las visualizaciones del dataset funcionan correctamente.

