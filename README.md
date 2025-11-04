# 🚴 Sistema de Predicción de Destinos en Bicicleta - App Streamlit

Aplicación web interactiva para explorar y probar el modelo de predicción de destinos en bicicleta.

## 📋 Descripción

Esta aplicación Streamlit permite:
- **Visualizar** los resultados del análisis y modelado con gráficos interactivos de Altair
- **Explorar** los datos y patrones encontrados
- **Probar** el modelo entrenado ingresando datos nuevos

## 🚀 Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Preparar el modelo y preprocessor:
```bash
python prepare_model.py
```

Este script copiará el modelo desde `../prediccion/` y creará el preprocessor necesario.

## ▶️ Ejecutar la Aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá en tu navegador en `http://localhost:8501`

## 📁 Estructura del Proyecto

```
app-streamlit/
├── app.py              # Configuración de navegación
├── main.py             # Página principal
├── plots.py            # Visualizaciones interactivas
├── model.py            # Interfaz de inferencia
├── lib.py              # Funciones auxiliares y pipelines
├── requirements.txt    # Dependencias
├── prepare_model.py    # Script para preparar modelo
├── static/             # Modelos y recursos
│   ├── modelo_random_forest_final_tunado.pkl
│   └── preprocessor.pkl
└── README.md
```

## 🎨 Características

### Visualizaciones Interactivas (Altair)

1. **Importancia de Características**: Top 15 características más importantes del modelo
2. **Distribución Temporal**: Análisis de patrones por hora del día y día de la semana
3. **Top Destinos**: Estaciones destino más frecuentes

### Interfaz de Inferencia

- Formulario interactivo para ingresar datos de un viaje
- Predicción del destino más probable
- Top 5 destinos con sus probabilidades
- Visualización interactiva de resultados

## 🔧 Pipeline de Preprocesamiento

La aplicación utiliza pipelines de sklearn para:
- Calcular features temporales derivadas
- Calcular features geográficas
- Procesar datos del usuario
- Seleccionar features en el orden correcto

## 📊 Modelo

- **Algoritmo**: Random Forest Classifier
- **Features**: 27 características
- **Accuracy**: 53.66%
- **Destinos**: 89 estaciones únicas

## 📝 Notas

- Si no tienes datos del historial del usuario, se usarán valores por defecto basados en promedios del dataset
- El modelo requiere acceso a `dataset_modelo_final.csv` para las visualizaciones
- Los datos de estaciones (`station_data_enriched.csv`) son opcionales pero mejoran la precisión de las features geográficas

## 🌐 Despliegue en Streamlit Cloud

1. Sube el proyecto a GitHub
2. Conecta tu repositorio en [Streamlit Cloud](https://streamlit.io/cloud)
3. Asegúrate de que el modelo esté en la carpeta `static/`
4. La aplicación se desplegará automáticamente

