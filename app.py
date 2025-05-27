import os
import json
import logging

import numpy as np
import pandas as pd
import faiss
import streamlit as st
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Pydantic model ────────────────────────────────────────────────────────────
class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

# ─── Document store with precompute support ───────────────────────────────────
class DocumentStore:
    def __init__(self, csv_path: str = "Articles.csv"):
        self.csv_path = csv_path
        self.dimension = 384
        self.index = None
        self.documents = []
        self.model = None
        self._load_index_or_csv()

    def _load_index_or_csv(self):
        # Try loading precomputed index & metadata
        if os.path.exists("index.faiss") and os.path.exists("metadata.json"):
            logger.info("Loading precomputed FAISS index and metadata")
            self.index = faiss.read_index("index.faiss")
            with open("metadata.json", "r") as f:
                self.documents = json.load(f)
            logger.info(f"Loaded {len(self.documents)} documents from metadata.json")
            return
        # Fallback: embed from CSV
        self._load_csv_and_embed()

    def _load_csv_and_embed(self):
        logger.info(f"Looking for the CSV at: {self.csv_path}")
        if not os.path.exists(self.csv_path):
            logger.error(f"CSV file not found at {self.csv_path}")
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")

        logger.info(f"Loading the CSV from {self.csv_path}")
        df = pd.read_csv(self.csv_path, encoding="latin1")
        required_columns = ["Article", "Date", "Heading", "NewsType"]
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns: {df.columns.tolist()}")
            raise ValueError("CSV must contain Article, Date, Heading, NewsType columns")

        articles = df["Article"].astype(str).tolist()
        logger.info(f"Loaded {len(articles)} articles from CSV")

        # load model and compute embeddings
        logger.info("Loading SentenceTransformer model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Computing embeddings for all articles...")
        embeddings = self.model.encode(articles, show_progress_bar=True)

        # build FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        logger.info(f"Added {len(articles)} embeddings to FAISS index")

        # store document metadata
        for idx, row in df.iterrows():
            self.documents.append({
                "id": idx,
                "article": str(row["Article"]),
                "date": str(row["Date"]),
                "heading": str(row["Heading"]),
                "news_type": str(row["NewsType"])
            })
        logger.info(f"Saved {len(self.documents)} documents for search at runtime")

    def search(self, query: str, top_k: int = 5):
        if self.index is None:
            raise RuntimeError("Index not loaded")
        # embed the query (load model if needed)
        if self.model is None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        query_emb = self.model.encode([query])[0]

        distances, indices = self.index.search(np.array([query_emb]), top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents) and idx != -1:
                doc = self.documents[idx]
                similarity = 1 - (dist / 2)
                results.append({**doc, "similarity": float(similarity)})
        return results

# ─── Streamlit caching for performance ────────────────────────────────────────
@st.cache_resource
def get_document_store(csv_path: str = "Articles.csv") -> DocumentStore:
    return DocumentStore(csv_path)

# ─── Streamlit App UI ─────────────────────────────────────────────────────────
st.set_page_config(page_title="News Article Similarity Search", layout="wide")
st.title("News Article Similarity Search")

# Sidebar: inputs
st.sidebar.header("Search Parameters")
query = st.sidebar.text_input("Enter search query:")
top_k = st.sidebar.slider("Number of results (top_k):", 1, 10, 5)
search_button = st.sidebar.button("Search")

# Load document store
doc_store = get_document_store()

if search_button and query:
    with st.spinner("Searching articles..."):
        results = doc_store.search(query, top_k)

    if results:
        st.subheader(f"Top {len(results)} results for \"{query}\"")
        for r in results:
            st.markdown(f"**{r['heading']}** ({r['date']}) - *{r['news_type']}*\n> Similarity: {r['similarity']:.2f}\n\n{r['article']}")
            st.write("---")
    else:
        st.warning("No results found. Try a different query.")
else:
    st.info("Enter a query in the sidebar and click 'Search' to find similar news articles.")
