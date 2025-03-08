import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load precomputed document embeddings and documents
embeddings = np.load("document_embeddings.npy")
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

def retrieve_top_k(query_embedding, embeddings, k=5):
    """Retrieve top-k most similar documents using cosine similarity."""
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]

# Streamlit UI
st.title("Information Retrieval System")
st.write("Search Reuters News Articles Using Word Embeddings")

# Input query
query = st.text_input("Enter your search query:")

# Placeholder for query embedding generation
def get_query_embedding(query):
    return np.random.rand(embeddings.shape[1])  # Replace with actual embedding function

if st.button("Search"):
    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, embeddings)
    
    # Display results
    st.write("### Top 5 Relevant Documents:")
    for doc, score in results:
        st.write(f"- **{doc.strip()}** (Score: {score:.4f})")
