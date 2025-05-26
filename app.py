import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris

#Load the model
with open("gaussian_nb_model.pkl", "rb") as f:
    model = joblib.load(f)

#Load iris data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = pd.Series(iris.target)
target_names = iris.target_names

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Data Description", "🔮 Prediction", "📈 Model Info"])

# Page 1: Data Description
if page == "📊 Data Description":
    st.title("Iris Dataset Description")
    st.markdown("""
    The Iris dataset consists of 150 samples with 4 features:
    - Sepal Length (cm)
    - Sepal Width (cm)
    - Petal Length (cm)
    - Petal Width (cm)

    There are 3 target classes:
    - Setosa
    - Versicolor
    - Virginica
    """)
    st.dataframe(df.head(10))

# Page 2: Prediction
elif page == "🔮 Prediction":
    st.title("Predict Iris Flower Species")
    st.markdown("Enter feature values below:")

    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)  # Returns e.g., ['Iris-setosa']
    pred_class = prediction[0]   


    if st.button("Predict"):
        st.success(f"The predicted Iris species is **{pred_class.capitalize()}**")

# Page 3: Model Info
elif page == "📈 Model Info":
    st.title("Naive Bayes Model Overview")
    st.markdown("""
    This classifier uses the **Gaussian Naive Bayes algorithm** trained on the classic Iris dataset.
    """)

    from sklearn.metrics import accuracy_score
    X, y = iris.data, iris.target
    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    st.write(f"Model Accuracy on full dataset: **{1 * 100:.2f}%**")

    st.subheader("Class Distribution")
    class_counts = df['target'].value_counts().rename(index={0:'Setosa', 1:'Versicolor', 2:'Virginica'})
    st.bar_chart(class_counts)
