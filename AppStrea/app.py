import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Configuración inicial de Streamlit
st.set_page_config(page_title="Análisis Económico", layout="wide")

# Cargar datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("/Users/anayelicocoletzi/Documents/MCD/1_ArqDatos/test_streamlit/data_w3.csv")  # Asegúrate de que el nombre del archivo sea correcto

df = cargar_datos()
df['date'] = pd.to_datetime(df['date'])  # Convertir la columna 'date' a tipo datetime

# Título de la app
st.title("📊 Análisis Económico: Regresiones Lineales")

# Sección 1: Vista de Datos
st.header("🔍 Exploración de Datos")

# Mostrar DataFrame
st.subheader("📌 Datos Económicos")
st.dataframe(df)

# Mostrar estadísticas descriptivas
st.subheader("📊 Resumen Estadístico")
st.write(df.describe())

# Gráfico de líneas para ver evolución temporal
st.subheader("📈 Evolución Temporal de las Variables")
variable_seleccionada = st.selectbox("Selecciona una variable para visualizar:", df.columns[1:])
fig, ax = plt.subplots(figsize=(10, 5))  # Tamaño ajustado del gráfico
sns.lineplot(data=df, x="date", y=variable_seleccionada, marker="o", ax=ax)
ax.set_title(f"Evolución de {variable_seleccionada} en el tiempo")
ax.set_xlabel("Fecha")
ax.set_ylabel(variable_seleccionada)

# Configurar el eje x para mostrar un valor cada año o cada 2 años
ax.xaxis.set_major_locator(mdates.YearLocator(base=2))  # Mostrar una marca por año (o base=2 para cada 2 años)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Formatear como año
plt.xticks(rotation=45)  # Rotar las etiquetas para mejor legibilidad
plt.tight_layout()  # Ajustar el layout para evitar superposiciones
st.pyplot(fig)

# Mostrar pairplot para ver correlaciones
st.subheader("📉 Matriz de Correlaciones (Pairplot)")
fig = sns.pairplot(df, diag_kind="kde", height=2)  # Tamaño reducido del pairplot
st.pyplot(fig)

# Mostrar histogramas para ver distribuciones
st.subheader("📊 Distribución de Variables")
variable_hist = st.selectbox("Selecciona una variable para ver su distribución:", df.columns[1:])
fig, ax = plt.subplots(figsize=(6, 4))  # Tamaño reducido del histograma
sns.histplot(df[variable_hist], bins=30, kde=True, ax=ax)
ax.set_title(f"Distribución de {variable_hist}")
ax.set_xlabel(variable_hist)
st.pyplot(fig)

# Sección 2: Regresiones Lineales
st.header("📈 Regresiones Lineales")

# Selección de variables para regresión
st.subheader("⚙️ Configuración de Regresión")
x_var = st.selectbox("Selecciona la variable independiente (X):", df.columns[1:])
y_var = st.selectbox("Selecciona la variable dependiente (Y):", df.columns[1:])

# Función para realizar regresión
def regresion_lineal(df, x_var, y_var):
    X = df[[x_var]]
    X = sm.add_constant(X)  # Agregar término de intercepto
    y = df[y_var]
    
    modelo = sm.OLS(y, X).fit()  # Ajustar modelo
    return modelo

# Ejecutar regresión si las variables son diferentes
if x_var != y_var:
    modelo = regresion_lineal(df, x_var, y_var)
    
    st.subheader("📈 Resumen de la Regresión")
    st.text(modelo.summary())

    # Graficar scatter plot con regresión
    st.subheader("📉 Gráfico de Dispersión con Línea de Regresión")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_var, y=y_var, alpha=0.7, ax=ax)

    # Línea de regresión
    X = sm.add_constant(df[x_var])
    df["predicted"] = modelo.predict(X)
    sns.lineplot(data=df, x=x_var, y="predicted", color="red", label="Regresión", ax=ax)

    ax.set_title(f"Regresión: {y_var} vs {x_var}")
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.legend()
    
    st.pyplot(fig)
else:
    st.warning("⚠️ Selecciona dos variables diferentes para realizar la regresión.")