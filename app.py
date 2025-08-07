# import streamlit as st
# import joblib
# import numpy as np

# # --- Load Models ---
# @st.cache_resource
# def load_models():
#     try:
#         vectorizer = joblib.load("vectorizer.pkl")
#         pca = joblib.load("pca.pkl")
#         kmeans = joblib.load("kmean.pkl")
#         return vectorizer, pca, kmeans
#     except Exception as e:
#         st.error(f"Model loading failed: {e}")
#         return None, None, None

# vectorizer, pca, kmeans = load_models()

# # --- Streamlit UI ---
# st.title("📰 News Cluster Predictor")
# st.write("Enter a **news heading** or **news body**, and the app will predict which cluster it belongs to.")

# input_type = st.radio("Select Input Type:", ["News Heading", "News Body"])
# text_input = st.text_area(f"Enter your {input_type.lower()} below:")

# if st.button("Predict Cluster"):
#     if not text_input.strip():
#         st.warning("Please enter some text.")
#     elif None in (vectorizer, pca, kmeans):
#         st.error("Models could not be loaded. Check your files.")
#     else:
#         try:
#             # Step 1: Vectorize text
#             vector = vectorizer.transform([text_input])

#             # Step 2: PCA transformation
#             reduced_vector = pca.transform(vector.toarray())

#             # Step 3: Predict cluster
#             cluster = kmeans.predict(reduced_vector)[0]

#             st.success(f"🔎 Predicted Cluster: **{cluster}**")

#         except Exception as e:
#             st.error(f"Prediction failed: {e}")









import streamlit as st
import joblib

# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load("vectorizer.pkl")
        kmeans = joblib.load("kmean.pkl")
        return vectorizer, kmeans
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

vectorizer, kmeans = load_models()

# --- Streamlit UI ---
st.title("📰 News Cluster Predictor")
st.write("Enter a **news heading** or **news body**, and the app will predict which cluster it belongs to.")

input_type = st.radio("Select Input Type:", ["News Heading", "News Body"])
text_input = st.text_area(f"Enter your {input_type.lower()} below:")

if st.button("Predict Cluster"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    elif None in (vectorizer, kmeans):
        st.error("Models could not be loaded. Check your files.")
    else:
        try:
            # Vectorize the input
            vector = vectorizer.transform([text_input])

            # Predict cluster (no PCA)
            cluster = kmeans.predict(vector)[0]

            st.success(f"🔎 Predicted Cluster: **{cluster}**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
