# 🛡️ Web Plagiarism Shield

A premium, deep-learning powered plagiarism detection system that scans your documents against the live web in real-time. Built with **Streamlit** and **Sentence Transformers (BERT)**.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-BERT%20%2F%20TF--IDF-blue?style=for-the-badge)

## ✨ Features

- **Live Web Deep-Scan**: Queries the Google search engine (via SerpAPI) to find potential matches across the entire internet.
- **Hybrid AI Engine**: Combines **TF-IDF Keyword Matching** with **BERT Semantic Analysis** to catch both direct copying and paraphrasing.
- **Deep Dark Theme**: A high-contrast, premium UI designed for focus and readability.
- **Detailed Match Reporting**: View original segments side-by-side with web matches, complete with AI confidence scores.
- **Export Capabilities**: Download your full scan results as a CSV report.
- **PDF & Text Support**: Seamlessly extract text from PDF documents and raw text files.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A [SerpAPI Key](https://serpapi.com/) (for live web searching)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ishanrt119/NLP-FA.git
   cd NLP-FA
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**:
   Create a `.env` file in the root directory and add your SerpAPI key:
   ```env
   SERPAPI_KEY=your_serp_api_key_here
   ```

### Running the App

```bash
streamlit run app.py
```

## 🛠️ How it Works

1. **Extraction**: Text is extracted from the uploaded PDF or TXT file using `PyPDF2`.
2. **Segmentation**: The document is tokenized into individual sentences using `NLTK`.
3. **Web Search**: For every sentence, the app queries Google to retrieve the most relevant snippets.
4. **Scoring**:
   - **TF-IDF**: Calculates lexical similarity.
   - **BERT**: Calculates semantic similarity using the `all-MiniLM-L6-v2` transformer model.
5. **Report**: Matches exceeding the user-defined threshold are flagged and displayed in the dashboard.

## 🔐 Security & Privacy

- **API Keys**: Keys are managed via `.env` and are never displayed in the UI.
- **Local Data**: Your documents are processed in-memory and are not stored on any server (except for the snippets sent to SerpAPI for search).

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Created with ❤️ for NLP Final Assignment.*