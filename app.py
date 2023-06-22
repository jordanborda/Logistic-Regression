import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go

st.set_option('deprecation.showPyplotGlobalUse', False)
# Crear un marco para el autor
st.subheader("Autor: Jordan Piero Borda Colque")
st.markdown(
    """
    **Tecnicas Estadisticas Multivariadas**
    """
)
# Cargar dato
def load_data():
    return pd.read_csv("mary.csv")

data = load_data()

# Título de la aplicación
st.title("Regresión Logística para Tratamiento de Tuberculosis")

# Mostrar datos
if st.checkbox("Mostrar datos"):
    st.write(data)

# Selección de variables
features = data.drop('abandono del tratamiento', axis=1)
target = data['abandono del tratamiento']

# Opciones de regresión logística
solver = st.sidebar.selectbox("Seleccione un solver", ("liblinear", "newton-cg", "lbfgs", "sag", "saga"))
penalty = st.sidebar.selectbox("Seleccione el tipo de regularización", ("l1", "l2"))
C = st.sidebar.slider("Parámetro de regularización (C)", 0.01, 10.0, 1.0)
random_state = st.sidebar.slider("Semilla aleatoria para la división de datos", 1, 100, 42)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=random_state)

# Crear modelo de regresión logística
model = LogisticRegression(solver=solver, penalty=penalty, C=C, max_iter=1000)

# Entrenar modelo
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)
y_score = model.decision_function(X_test)

# Mostrar resultados
if st.checkbox("Mostrar resultados"):
    st.subheader("Resultados")
    report = classification_report(y_test, y_pred, output_dict=True)

    # Crear una tabla para mostrar las métricas de clasificación
    metrics = pd.DataFrame(report).transpose()
    metrics = metrics.round(2)
    st.dataframe(metrics)

    # Mostrar una barra de progreso para visualizar las métricas importantes
    accuracy = metrics.loc["accuracy", "f1-score"]
    
    if "1" in metrics.index:
        precision = metrics.loc["1", "precision"]
        recall = metrics.loc["1", "recall"]
        
        col1, col2, col3 = st.beta_columns(3)

        with col1:
            st.progress(accuracy)
            st.caption("Exactitud")

        with col2:
            st.progress(precision)
            st.caption("Precisión")

        with col3:
            st.progress(recall)
            st.caption("Recall")
    else:
        st.warning("La clase '1' no está presente en el conjunto de datos de prueba.")


# Mostrar matriz de confusión
if st.checkbox("Mostrar matriz de confusión"):
    st.subheader("Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, color_continuous_scale='viridis', labels=dict(x="Predicho", y="Verdadero", color="Conteo"))
    st.plotly_chart(fig)

# Mostrar coeficientes
if st.checkbox("Mostrar coeficientes"):
    st.subheader("Coeficientes")
    coef = pd.DataFrame(model.coef_[0], index=features.columns, columns=["Coeficiente"])
    coef.sort_values("Coeficiente", inplace=True)
    fig = px.bar(coef, x=coef.index, y='Coeficiente', title="Coeficientes del Modelo", labels={"Coeficiente": "Valor", "index": "Característica"})
    st.plotly_chart(fig)

# Mostrar Curva ROC
if st.checkbox("Mostrar Curva ROC"):
    st.subheader("Curva ROC")
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score
    from scipy import interp
    from itertools import cycle
    
    # Binarizar las etiquetas
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]

    # Calcular la puntuación de la clase para cada clase
    y_score = model.decision_function(X_test)

    # Calcular las tasas de verdaderos y falsos positivos para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calcular micro-promedio de la curva ROC y el área
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Gráfico de la curva ROC para cada clase
    fig = go.Figure()
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], mode='lines',
                                 name=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})',
                                 line=dict(color=color)))

    # Línea de clasificación aleatoria
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guessing', line=dict(color='navy', dash='dash')))

    # Estilizar gráfico
    fig.update_layout(autosize=False, xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', title='Receiver Operating Characteristic')

    # Mostrar gráfico
    st.plotly_chart(fig)




# Mostrar correlaciones entre características
if st.checkbox("Mostrar Correlaciones entre Características"):
    st.subheader("Correlaciones entre Características")
    corr = features.corr()
    fig = px.imshow(corr, color_continuous_scale=[(0, 'blue'), (0.5, 'white'), (1, 'red')])
    st.plotly_chart(fig)

# Agregar gráfico de regresión logística
# Mostrar Gráfico de Regresión Logística
if st.checkbox("Mostrar Gráfico de Regresión Logística"):
    st.subheader("Gráfico de Regresión Logística")
    x_values = model.decision_function(X_test)[:, 0]  # Seleccionar la columna correspondiente a la clase 0
    
    # Crear el gráfico de regresión logística con Plotly
    fig = px.scatter(x=x_values, y=y_test, trendline="ols", trendline_color_override="red")
    fig.update_layout(title="Gráfico de Regresión Logística", xaxis_title="Valores de Decisión", yaxis_title="Valores Reales")
    
    st.plotly_chart(fig)


    # Calcular estadísticas de los coeficientes del modelo
    coefs = pd.DataFrame(model.coef_[0], index=features.columns, columns=["Coeficiente"])
    coefs["Abs(Coeficiente)"] = np.abs(coefs["Coeficiente"])
    coefs.sort_values("Abs(Coeficiente)", ascending=False, inplace=True)
    st.subheader("Estadísticas de los Coeficientes")
    st.write(coefs)
    # Calcular la proporción de clases en los datos de prueba
    class_counts = y_test.value_counts(normalize=True)
    st.subheader("Proporción de Clases en los Datos de Prueba")
    st.write(class_counts)

    # Calcular precisión del modelo
    accuracy = model.score(X_test, y_test)
    st.subheader("Precisión del Modelo")
    st.write(f"La precisión del modelo es: {accuracy:.4f}")

    # Calcular sensibilidad y especificidad del modelo
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    st.subheader("Sensibilidad y Especificidad del Modelo")
    st.write(f"Sensibilidad: {sensitivity:.4f}")
    st.write(f"Especificidad: {specificity:.4f}")



# Calcular la matriz de correlación
corr_matrix = features.corr()

# Mostrar el heatmap de correlación
if st.checkbox("Mostrar Heatmap de Correlación"):
    st.subheader("Heatmap de Correlación")
    
    # Crear el heatmap de correlación con Plotly
    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
                                   colorscale='Viridis', colorbar=dict(title="Correlación"), zmin=-1, zmax=1))
    fig.update_layout(title="Heatmap de Correlación")
    
    st.plotly_chart(fig)



# Calcular la frecuencia de cada clase
class_counts = data['abandono del tratamiento'].value_counts()

# Definir colores personalizados
colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)']

# Mostrar el gráfico de barras con Plotly
st.subheader("Distribución de la Variable Objetivo")
fig = go.Figure(data=[go.Bar(x=class_counts.index,
                            y=class_counts.values,
                            marker=dict(color=colors))])
fig.update_layout(title='Distribución de la Variable Objetivo',
                  xaxis_title='Clase',
                  yaxis_title='Frecuencia')
st.plotly_chart(fig)

from sklearn.model_selection import learning_curve

# Calcular la curva de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(model, features, target, cv=5)

# Calcular las medias y desviaciones estándar de los puntajes
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Definir colores personalizados
colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)']

# Mostrar la curva de aprendizaje con Plotly
st.subheader("Curva de Aprendizaje")
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_sizes,
                         y=train_scores_mean,
                         mode='lines',
                         name='Train',
                         line=dict(color=colors[0])))
fig.add_trace(go.Scatter(x=train_sizes,
                         y=test_scores_mean,
                         mode='lines',
                         name='Validation',
                         line=dict(color=colors[1])))
fig.add_trace(go.Scatter(x=train_sizes,
                         y=train_scores_mean - train_scores_std,
                         mode='lines',
                         name='Train (std)',
                         line=dict(color=colors[0], dash='dash')))
fig.add_trace(go.Scatter(x=train_sizes,
                         y=train_scores_mean + train_scores_std,
                         mode='lines',
                         name='Train (std)',
                         line=dict(color=colors[0], dash='dash')))
fig.add_trace(go.Scatter(x=train_sizes,
                         y=test_scores_mean - test_scores_std,
                         mode='lines',
                         name='Validation (std)',
                         line=dict(color=colors[1], dash='dash')))
fig.add_trace(go.Scatter(x=train_sizes,
                         y=test_scores_mean + test_scores_std,
                         mode='lines',
                         name='Validation (std)',
                         line=dict(color=colors[1], dash='dash')))
fig.update_layout(title='Curva de Aprendizaje',
                  xaxis_title='Número de muestras de entrenamiento',
                  yaxis_title='Precisión')
st.plotly_chart(fig)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import io
import pydotplus

# Crear un modelo de árbol de decisión
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Convertir las etiquetas de clase a cadenas de texto
class_names = model.classes_.astype(str)

# Crear el gráfico de árbol de decisión con Graphviz
dot_data = tree.export_graphviz(dt_model, out_file=None, feature_names=features.columns, class_names=class_names, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)

# Convertir el gráfico a una imagen
image = graph.create_png()
stream = io.BytesIO(image)

# Mostrar la imagen en la aplicación
st.subheader("Gráfico de Árbol de Decisión")
st.image(stream, use_column_width=True)




