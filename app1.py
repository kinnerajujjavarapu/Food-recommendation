import streamlit as st
import sys
st.write("Streamlit version:", st.__version__)
st.write("Python version:", sys.version)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Function to load and preprocess data
@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_excel(path, sheet_name='English_version', header=None)
    df.columns = df.iloc[1]
    df = df.drop(index=0).reset_index(drop=True)
    df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]

    df['ghg_total'] = pd.to_numeric(df['ghg_total'], errors='coerce')
    df['cliped_count'] = pd.to_numeric(df['cliped_count'], errors='coerce').fillna(0)

    nutrient_cols = ['protein_(g)', 'total_fiber_(g)', 'fat_(g)', 'salt_equivalent_(g)']
    for col in nutrient_cols:
        if col not in df.columns:
            df[col] = 0
    df[nutrient_cols] = df[nutrient_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    scaler = MinMaxScaler()
    df[['protein_score', 'fiber_score']] = scaler.fit_transform(df[['protein_(g)', 'total_fiber_(g)']])
    df[['fat_score', 'salt_score']] = 1 - scaler.fit_transform(df[['fat_(g)', 'salt_equivalent_(g)']])
    df['nutrient_score'] = (
        0.4 * df['protein_score'] +
        0.3 * df['fiber_score'] +
        0.2 * df['fat_score'] +
        0.1 * df['salt_score']
    )

    df['combined_text'] = (
        df['keywords'].fillna('') + ' ' +
        df['recipe_description'].fillna('')
    ).str.lower()

    if 'price' not in df.columns:
        raise ValueError("The dataset must contain a 'price' column.")

    df['price'] = df['price'].astype(str).str.extract(r'([\d\.]+)')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['price'] = df['price'].fillna(df['price'].mean() if pd.api.types.is_numeric_dtype(df['price']) else 0)

    df = df.dropna(subset=['recipe_name', 'ghg_total']).reset_index(drop=True)
    return df

# Function to compute embeddings
@st.cache_resource(show_spinner=False)
def load_model_and_embeddings(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['combined_text'].tolist(), convert_to_tensor=True, show_progress_bar=True)
    return model, embeddings

# Recommendation logic
def hybrid_recommend(df, embeddings, model, user_query, top_n=5, price_limit=None, alpha=0.7):
    user_embedding = model.encode([user_query], convert_to_tensor=True)
    content_sim = cosine_similarity(user_embedding.cpu().numpy(), embeddings.cpu().numpy())[0]

    sim_norm = (content_sim - np.min(content_sim)) / (np.max(content_sim) - np.min(content_sim))
    nutrient_norm = df['nutrient_score'].values
    price_norm = 1 - (df['price'] - df['price'].min()) / (df['price'].max() - df['price'].min())
    ghg_norm = 1 - (df['ghg_total'] - df['ghg_total'].min()) / (df['ghg_total'].max() - df['ghg_total'].min())
    df['ghg_score'] = ghg_norm.round(3)

    final_scores = (
        alpha * sim_norm +
        0.15 * nutrient_norm +
        0.1 * price_norm +
        0.05 * ghg_norm
    )
    df['score'] = final_scores.round(3)

    candidates = df.sort_values(by='score', ascending=False)
    if price_limit is not None and price_limit > 0:
        candidates = candidates[candidates['price'] <= price_limit]

    return candidates[['recipe_name', 'nutrient_score', 'ghg_total', 'ghg_score', 'price', 'score']].head(top_n)

# Streamlit UI
st.title("🥗 Sustainable Recipe Recommender (Hybrid: Nutrients + Price + GHG + Content)")

# Load data and model
df = load_data('recipe_data_with_Eng_name.xlsx', sheet_name='English_version', header=None, engine='openpyxl')  # Ensure the file is in the same directory or update path
model, embeddings = load_model_and_embeddings(df)

# User inputs
user_input = st.text_input("Enter recipe keywords or requirements (e.g. 'high protein low fat curry'):")

top_n = st.slider("📋 Number of recommendations to show", 1, 20, 5)
price_limit = st.number_input("💰 Max Price Limit (0 means no limit)", min_value=0.0, value=0.0, step=50.0, format="%.1f")

if st.button("Get Recommendations"):
    if not user_input.strip():
        st.warning("Please enter recipe requirements.")
    else:
        limit = None if price_limit == 0 else price_limit
        results = hybrid_recommend(df, embeddings, model, user_input, top_n=top_n, price_limit=limit)
        if results.empty:
            st.info("No recipes found matching your criteria.")
        else:
            st.dataframe(results.reset_index(drop=True))
