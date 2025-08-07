import streamlit as st
import pickle
import numpy as np

# Load saved models
with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("kmean.pkl", "rb") as f:
    kmeans = pickle.load(f)

# OPTIONAL: Load a vectorizer if your model needs it (e.g., TF-IDF)
try:
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    vectorizer = None

# App UI
st.title("📰 News Cluster Predictor")

option = st.radio("Choose input type:", ["News Heading", "News Body"])

user_input = st.text_area("Enter your text here:")

if st.button("Predict Cluster"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocessing: Convert text to vector
        if vectorizer:
            vector = vectorizer.transform([user_input])
        else:
            st.error("Vectorizer not found. Please ensure vectorizer.pkl is available.")
            st.stop()

        # PCA transformation
        reduced_vector = pca.transform(vector.toarray())

        # Predict cluster
        prediction = kmeans.predict(reduced_vector)

        st.success(f"🔍 Predicted Cluster: **{prediction[0]}**")
