import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Streamlit app
st.title("Iris Species Classifier")

# Sidebar for user input
st.sidebar.header("Input Features")


def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', float(df['sepal length (cm)'].min()),
                                     float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
    sepal_width = st.sidebar.slider('Sepal width (cm)', float(df['sepal width (cm)'].min()),
                                    float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
    petal_length = st.sidebar.slider('Petal length (cm)', float(df['petal length (cm)'].min()),
                                     float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
    petal_width = st.sidebar.slider('Petal width (cm)', float(df['petal width (cm)'].min()),
                                    float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))

    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

# Display user input
st.subheader("User Input Features")
st.write(input_df)

# Model Training
X = df.drop(columns='species')
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Prediction
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

# Display Prediction
st.subheader("Prediction")
predicted_species = prediction[0]  # Prediction already contains the species name
st.write(f"Predicted Species: {predicted_species}")

# Display Prediction Probability
st.subheader("Prediction Probability")
st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))
