

# import streamlit as st
# import joblib

# # --- Load Models ---
# @st.cache_resource
# def load_models():
#     try:
#         vectorizer = joblib.load("vectorizer.pkl")
#         kmeans = joblib.load("kmean.pkl")
#         return vectorizer, kmeans
#     except Exception as e:
#         st.error(f"Model loading failed: {e}")
#         return None, None

# vectorizer, kmeans = load_models()

# # --- Streamlit UI ---
# st.title("📰 News Cluster Predictor")
# st.write("Enter a **news heading** or **news body**, and the app will predict which cluster it belongs to.")

# input_type = st.radio("Select Input Type:", ["News Heading", "News Body"])
# text_input = st.text_area(f"Enter your {input_type.lower()} below:")

# if st.button("Predict Cluster"):
#     if not text_input.strip():
#         st.warning("Please enter some text.")
#     elif None in (vectorizer, kmeans):
#         st.error("Models could not be loaded. Check your files.")
#     else:
#         try:
#             # Vectorize the input
#             vector = vectorizer.transform([text_input])

#             # Predict cluster (no PCA)
#             cluster = kmeans.predict(vector)[0]

#             st.success(f"🔎 Predicted Cluster: **{cluster}**")

#         except Exception as e:
#             st.error(f"Prediction failed: {e}")













import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np

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

            # Predict cluster
            cluster = kmeans.predict(vector)[0]
            st.success(f"🔎 Predicted Cluster: **{cluster}**")

            # Cluster info
            st.subheader("📊 Cluster Size Overview")

            # Generate dummy cluster distribution (equal sizes as placeholder)
            n_clusters = kmeans.n_clusters
            placeholder_sizes = np.ones(n_clusters)

            # Highlight predicted cluster
            fig, ax = plt.subplots()
            bars = ax.bar(range(n_clusters), placeholder_sizes)

            for i, bar in enumerate(bars):
                if i == cluster:
                    bar.set_color('orange')

            ax.set_xlabel("Cluster")
            ax.set_ylabel("Sample Count (mock)")
            ax.set_title("Mock Cluster Size Distribution")
            st.pyplot(fig)

            st.caption("Note: This chart uses placeholder values since training data is not available.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
