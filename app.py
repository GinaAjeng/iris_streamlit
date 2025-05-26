import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
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
page = st.sidebar.radio("Go to", ["📊 Data Description", "🔮 Prediction", "📈 Model Training Results"])

# Page 1: Data Description + EDA
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
    st.subheader("📌 First 10 Rows of the Dataset")
    st.dataframe(df.head(10))

    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("📌 Class Distribution")
    class_counts = df['target'].value_counts().rename(index={0:'Setosa', 1:'Versicolor', 2:'Virginica'})
    st.bar_chart(class_counts)

    st.subheader("📈 Pairplot of Features")
    # Combine target with names
    df['species'] = df['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    fig1 = sns.pairplot(df, hue="species", height=2.5)
    st.pyplot(fig1)

    st.subheader("📉 Correlation Heatmap")
    corr = df.iloc[:, :4].corr()
    fig2, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig2)

# Page 2: Prediction
elif page == "🔮 Prediction":
    st.title("Predict Iris Flower Species")
    st.markdown("Enter feature values below:")

    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    pred_name = str(prediction[0])  # Langsung ambil nama prediksi (string)

    if st.button("Predict"):
        st.success(f"The predicted Iris species is **{pred_name}**")


# Page 3: Model Training Results
elif page == "📈 Model Training Results":
    st.title("Train and Evaluate Naive Bayes Model")

    st.markdown("""
    You can train a **Gaussian Naive Bayes** model using a different train/test split.
    Adjust the slider below to change the test set proportion.
    """)

    # Slider for test size
    test_size = st.slider("Test Set Size (%)", 10, 50, 30, step=5) / 100.0

    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Split the dataset
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train model
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.subheader("🔍 Evaluation Results")
    st.write(f"Test Size: **{int(test_size * 100)}%**")
    st.write(f"Accuracy: **{acc * 100:.2f}%**")

    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)
