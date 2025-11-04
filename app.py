import streamlit as st
from main import main_page
from plots import plots_page
from model import model_page

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Predicción de Destinos en Bicicleta",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Navegación multi-página
main_page_obj = st.Page(main_page, title="Inicio", icon="🏠", default=True)
plots_page_obj = st.Page(plots_page, title="Visualizaciones", icon="📊")
model_page_obj = st.Page(model_page, title="Modelo", icon="🤖")

pg = st.navigation([main_page_obj, plots_page_obj, model_page_obj])
pg.run()

