
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Título
st.title("Predicción de Aprobación de Estudiantes con Árbol de Decisión")
st.markdown("Este modelo usa notas de parciales, proyecto y examen final para predecir si un estudiante aprobará la materia.")

# Cargar datos desde archivo CSV
@st.cache_data
def cargar_datos():
    return pd.read_csv("estudiantes_notas_finales.csv")

df = cargar_datos()

# Mostrar los primeros datos
st.subheader("Datos cargados")
st.write(df.head())

# Gráficos simples
st.subheader("Distribución de Notas")
st.bar_chart(df[["Primer_Parcial", "Segundo_Parcial", "Proyecto", "Examen_Final", "Nota_Final"]].mean())

# Preparar datos
X = df[["Primer_Parcial", "Segundo_Parcial", "Proyecto", "Examen_Final"]]
y = df["Aprobado"]

# Separar datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo árbol de decisión
modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
modelo.fit(X_train, y_train)

# Evaluación
y_pred = modelo.predict(X_test)
st.subheader("Evaluación del modelo")
st.text(f"Precisión: {accuracy_score(y_test, y_pred):.2f}")
st.text("Matriz de Confusión:")
st.write(confusion_matrix(y_test, y_pred))
st.text("Reporte de Clasificación:")
st.text(classification_report(y_test, y_pred))

# Visualizar árbol
st.subheader("Visualización del Árbol de Decisión")
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(modelo, feature_names=X.columns, class_names=["No", "Sí"], filled=True, rounded=True, fontsize=10)
st.pyplot(fig)

# Formulario interactivo
st.subheader("¿Aprobaría este estudiante?")
with st.form("formulario_prediccion"):
    p1 = st.number_input("Primer Parcial", 0.0, 100.0, 50.0)
    p2 = st.number_input("Segundo Parcial", 0.0, 100.0, 50.0)
    proy = st.number_input("Proyecto", 0.0, 100.0, 50.0)
    exf = st.number_input("Examen Final", 0.0, 100.0, 50.0)
    submitted = st.form_submit_button("Predecir")

    if submitted:
        datos_nuevos = pd.DataFrame([[p1, p2, proy, exf]], columns=X.columns)
        prediccion = modelo.predict(datos_nuevos)[0]
        st.success(f"Resultado: {'✅ Aprobado' if prediccion == 'Sí' else '❌ Reprobado'}")
