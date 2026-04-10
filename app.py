import streamlit as st
import nltk
import re
import os
import requests
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import torch

# --- Load Environment Variables ---
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# --- NLTK Data ---
@st.cache_resource
def download_nltk_data():
    for res in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
        nltk.download(res, quiet=True)

download_nltk_data()

# --- Page Config ---
st.set_page_config(
    page_title="Web Plagiarism Shield",
    page_icon="🛡️",
    layout="centered",
)

# --- Custom CSS for Deep Dark Theme ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    .stApp {
        background-color: #000000 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #0b0b0b !important;
        border-right: 1px solid #222222;
    }

    /* Fix for standard streamlit text elements */
    .stMarkdown, .stText, p, span, h1, h2, h3, h4, label {
        color: #ffffff !important;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }

    /* Selectbox/Slider labels */
    .stSelectbox label, .stSlider label {
        color: #ffffff !important;
    }
    
    /* Center the uploader and main button */
    .stFileUploader {
        padding: 20px;
        background: #111111;
        border-radius: 15px;
        border: 1px dashed #333333;
        margin-bottom: 2rem;
        color: #ffffff !important;
    }

    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: #ffffff !important;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .stButton>button:hover {
        background: #2a5298;
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #aaaaaa !important;
        font-weight: 600;
    }

    .stMetric {
        background: #111111;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #222222;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    
    /* Highlight Cards */
    .highlight-card {
        padding: 28px;
        border-radius: 15px;
        background: #111111;
        border: 1px solid #222222;
        border-left: 8px solid #1e3c72;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }

    .segment-label {
        color: #2a5298;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }

    .segment-content {
        color: #ffffff !important;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 20px;
        background: #1a1a1a;
        padding: 15px;
        border-radius: 8px;
    }

    .source-label {
        color: #ff4b4b;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .source-content {
        color: #cccccc !important;
        font-size: 1rem;
        line-height: 1.5;
        font-style: italic;
    }

    .confidence-badge {
        display: inline-block;
        padding: 4px 12px;
        background: #1e3c72;
        color: #ffffff;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-top: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Utility Functions ---

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def preprocess_text(text, advanced=True):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    if advanced:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)
    return text

@st.cache_resource
def load_model(model_name):
    return SentenceTransformer(model_name)

def google_search(query):
    if not SERPAPI_KEY:
        st.error("SerpAPI Key not found in .env file!")
        return []
    
    try:
        url = "https://serpapi.com/search"
        params = {
            "engine": "google",
            "q": query[:200], # Google query limit approx
            "api_key": SERPAPI_KEY,
            "num": 3
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return []
        
        data = response.json()
        results = []
        if "organic_results" in data:
            for r in data["organic_results"]:
                if "snippet" in r:
                    results.append(r["snippet"])
        return results
    except Exception as e:
        return []

def get_similarity_score(s1, s2, model, alpha=0.3):
    # Process for TF-IDF
    proc1 = preprocess_text(s1)
    proc2 = preprocess_text(s2)
    
    if not proc1.strip() or not proc2.strip():
        return 0.0
        
    # TF-IDF
    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([proc1, proc2])
        tfidf_score = cosine_similarity(vectors[0], vectors[1])[0][0]
    except:
        tfidf_score = 0.0
        
    # BERT
    emb = model.encode([s1, s2])
    bert_score = cosine_similarity([emb[0]], [emb[1]])[0][0]
    
    return (alpha * tfidf_score) + ((1 - alpha) * bert_score)

# --- Main UI ---

st.title("🛡️ Web Plagiarism Shield")
st.markdown("Deep-scan your documents against the live web using AI.")
st.markdown("---")

# Sidebar
st.sidebar.header("Scan Settings")
threshold = st.sidebar.slider("Detection Sensitivity", 0.0, 1.0, 0.45)
alpha = st.sidebar.slider("Keyword vs Semantic Weight", 0.0, 1.0, 0.3)
st.sidebar.info("High Alpha = Focus on exact keywords\n\nLow Alpha = Focus on meaning/context")

# Main Area
st.subheader("📤 Upload Document")
suspicious_file = st.file_uploader("Upload PDF or Text file to scan", type=["pdf", "txt"])

if st.button("🚀 Start Web Deep Scan"):
    if not suspicious_file:
        st.error("Please upload a document first.")
    elif not SERPAPI_KEY:
        st.error("SerpAPI key is missing from .env file.")
    else:
        # Load model early to cache it
        with st.spinner("Loading AI Models..."):
            model = load_model('all-MiniLM-L6-v2')
            
        # 1. Extraction
        if suspicious_file.type == "application/pdf":
            raw_text = extract_text_from_pdf(suspicious_file)
        else:
            raw_text = suspicious_file.read().decode("utf-8")
            
        sentences = sent_tokenize(raw_text)
        
        if not sentences:
            st.error("The document appears to be empty.")
        else:
            st.info(f"Found {len(sentences)} segments. Initiating web search...")
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            plagiarized_count = 0
            
            for i, sent in enumerate(sentences):
                status_text.text(f"Scanning segment {i+1} of {len(sentences)}...")
                
                # Search web
                web_snippets = google_search(sent)
                
                max_score = 0
                best_match = ""
                
                if web_snippets:
                    for snippet in web_snippets:
                        score = get_similarity_score(sent, snippet, model, alpha)
                        if score > max_score:
                            max_score = score
                            best_match = snippet
                
                if max_score > threshold:
                    results.append({
                        "Original": sent,
                        "Web Match": best_match,
                        "Confidence": max_score
                    })
                    plagiarized_count += 1
                
                progress_bar.progress((i + 1) / len(sentences))
                
            status_text.empty()
            progress_bar.empty()
            
            # --- Results Presentation ---
            st.subheader("🔍 Scan Results")
            
            p_percent = (plagiarized_count / len(sentences))
            
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Overall Match Rate", f"{p_percent:.1%}")
            res_col2.metric("Flagged Segments", f"{plagiarized_count} / {len(sentences)}")
            
            if plagiarized_count > 0:
                st.warning("Potential plagiarism detected. Review the matches below.")
                
                for res in results:
                    with st.container():
                        st.markdown(f"""
                        <div class="highlight-card">
                            <div class="segment-label">Original Segment</div>
                            <div class="segment-content">"{res['Original']}"</div>
                            <div class="source-label">Potential Web Source</div>
                            <div class="source-content">"{res['Web Match']}"</div>
                            <div class="confidence-badge">
                                AI Confidence: {res['Confidence']:.2f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Export option
                df = pd.DataFrame(results)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Match Report (CSV)",
                    data=csv,
                    file_name='plagiarism_report.csv',
                    mime='text/csv',
                )
            else:
                st.success("No significant plagiarism detected. Your document appears original based on live web results.")

else:
    st.info("💡 Note: A full deep-scan takes time as it checks each sentence against live search results.")
