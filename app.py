

# # import streamlit as st
# # import joblib

# # # --- Load Models ---
# # @st.cache_resource
# # def load_models():
# #     try:
# #         vectorizer = joblib.load("vectorizer.pkl")
# #         kmeans = joblib.load("kmean.pkl")
# #         return vectorizer, kmeans
# #     except Exception as e:
# #         st.error(f"Model loading failed: {e}")
# #         return None, None

# # vectorizer, kmeans = load_models()

# # # --- Streamlit UI ---
# # st.title("📰 News Cluster Predictor")
# # st.write("Enter a **news heading** or **news body**, and the app will predict which cluster it belongs to.")

# # input_type = st.radio("Select Input Type:", ["News Heading", "News Body"])
# # text_input = st.text_area(f"Enter your {input_type.lower()} below:")

# # if st.button("Predict Cluster"):
# #     if not text_input.strip():
# #         st.warning("Please enter some text.")
# #     elif None in (vectorizer, kmeans):
# #         st.error("Models could not be loaded. Check your files.")
# #     else:
# #         try:
# #             # Vectorize the input
# #             vector = vectorizer.transform([text_input])

# #             # Predict cluster (no PCA)
# #             cluster = kmeans.predict(vector)[0]

# #             st.success(f"🔎 Predicted Cluster: **{cluster}**")

# #         except Exception as e:
# #             st.error(f"Prediction failed: {e}")













# import streamlit as st
# import joblib
# import matplotlib.pyplot as plt
# import numpy as np

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

#             # Predict cluster
#             cluster = kmeans.predict(vector)[0]
#             st.success(f"🔎 Predicted Cluster: **{cluster}**")

#             # Cluster info
#             st.subheader("📊 Cluster Size Overview")

#             # Generate dummy cluster distribution (equal sizes as placeholder)
#             n_clusters = kmeans.n_clusters
#             placeholder_sizes = np.ones(n_clusters)

#             # Highlight predicted cluster
#             fig, ax = plt.subplots()
#             bars = ax.bar(range(n_clusters), placeholder_sizes)

#             for i, bar in enumerate(bars):
#                 if i == cluster:
#                     bar.set_color('orange')

#             ax.set_xlabel("Cluster")
#             ax.set_ylabel("Sample Count (mock)")
#             ax.set_title("Mock Cluster Size Distribution")
#             st.pyplot(fig)

#             st.caption("Note: This chart uses placeholder values since training data is not available.")

#         except Exception as e:
#             st.error(f"Prediction failed: {e}")






import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------
# Load saved models
# --------------------
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

# --------------------
# Cluster → Category mapping
# (Adjust this according to your dataset)
# --------------------
cluster_labels = {
    0: "Politics",
    1: "Sports",
    2: "Business",
    3: "Entertainment",
    4: "Technology",
}

# --------------------
# Load dataset for charts
# --------------------
try:
    df_heading = pd.read_csv("news heading.csv")
    df_body = pd.read_csv("news body.csv")

    # For charting — vectorize & predict cluster for each news
    if vectorizer:
        vectors = vectorizer.transform(df_heading["text"])
        reduced_vectors = pca.transform(vectors.toarray())
        preds = kmeans.predict(reduced_vectors)
        df_heading["Cluster"] = preds
        df_heading["Category"] = df_heading["Cluster"].map(cluster_labels)
    else:
        st.warning("⚠ Cannot show category distribution — vectorizer not found.")
        df_heading = None

except FileNotFoundError:
    df_heading = None

# --------------------
# Streamlit App UI
# --------------------
st.set_page_config(page_title="📰 News Cluster Predictor", layout="wide")
st.title("📰 News Cluster Predictor")

option = st.radio("Choose input type:", ["News Heading", "News Body"])
user_input = st.text_area("✏ Enter your news text here:")

if st.button("🔍 Predict Cluster"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        if vectorizer:
            vector = vectorizer.transform([user_input])
            reduced_vector = pca.transform(vector.toarray())
            prediction = kmeans.predict(reduced_vector)[0]
            category = cluster_labels.get(prediction, f"Cluster {prediction}")

            st.success(f"✅ Predicted Category: **{category}** (Cluster {prediction})")
        else:
            st.error("Vectorizer not found. Please ensure vectorizer.pkl is available.")

# --------------------
# Show category distribution chart
# --------------------
if df_heading is not None:
    st.subheader("📊 Category Distribution in Dataset")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df_heading, x="Category", palette="viridis", order=df_heading["Category"].value_counts().index)
    plt.xticks(rotation=45)
    plt.title("Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    st.pyplot(fig)
