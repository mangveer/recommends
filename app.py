import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load the Netflix dataset
netflix_dataset = pd.read_csv("netflix.csv")

# Load pre-trained models
filledna = pickle.load(open("movies_list.pkl", "rb"))
cosine_sim2 = pickle.load(open("similarity.pkl", "rb"))

# Create indices
indices = pd.Series(filledna.index, index=filledna["title"])


# Function to get recommendations
def get_recommendations_new(title, cosine_sim=cosine_sim2):
    title = title.replace(" ", "").lower()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return netflix_dataset["title"].iloc[movie_indices]


# Streamlit app
st.title("Netflix Recommendation System")

option = st.selectbox("Select a title:", filledna["title"])

if st.button("Get Recommendations"):
    recommendations = get_recommendations_new(option)
    st.write(recommendations)
