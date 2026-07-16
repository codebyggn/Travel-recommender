# 🌍 Smart Travel Recommendation System

A hybrid **Information Retrieval (IR)** and **Machine Learning (ML)** based travel recommendation system developed using Python and Streamlit.

The system helps users discover travel destinations across India by combining:

- 🔍 TF-IDF Search
- 📚 BM25 Ranking
- ⚡ Hybrid Retrieval
- ⭐ Content-Based Recommendation
- 📊 Interactive Analytics Dashboard

---

## 📸 Application Preview

### 🏠 Home

![Home](assets/home.png)

### 🔍 Search

![Search](assets/search.png)

### ⭐ Recommendation

![Recommendation](assets/recommendation.png)

### 📊 Analytics

![Analytics](assets/analytics.png)

---

# ✨ Features

- 🔍 Hybrid search using **TF-IDF** and **BM25**
- ⭐ Content-based travel destination recommendation
- 📊 Interactive analytics dashboard
- 🌏 Search destinations by activities, season, or travel preference
- ⚡ Fast retrieval using Information Retrieval techniques
- 🎨 Minimal and responsive Streamlit interface

---

# 🧠 Algorithms Used

## Information Retrieval

- TF-IDF Vectorization
- BM25 Ranking
- Hybrid Score Combination

## Machine Learning

- Content-Based Recommendation
- Cosine Similarity

---

# 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Framework | Streamlit |
| Data Processing | Pandas |
| Machine Learning | Scikit-learn |
| Information Retrieval | Rank-BM25 |
| Visualization | Streamlit Charts |
| Dataset | Indian Tourism Dataset (JSON) |

---

# 📂 Project Structure

```text
travel-recommender/
│
├── assets/
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── notebooks/
│
├── src/
│   ├── preprocess.py
│   ├── ir_engine.py
│   ├── tfidf_engine.py
│   ├── bm25_engine.py
│   ├── ranking.py
│   ├── ml_engine.py
│   ├── recommender.py
│   ├── evaluation.py
│   └── utils.py
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```
---

# ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/codebyggn/travel-recommender.git
```

### 2. Navigate to the project

```bash
cd travel-recommender
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Launch the Streamlit application using:

```bash
streamlit run streamlit_app.py
```

The application will start at:

```
http://localhost:8501
```

---

# 🔄 System Workflow

```text
                  User Query
                      │
                      ▼
             Text Preprocessing
                      │
                      ▼
          TF-IDF Search + BM25 Search
                      │
                      ▼
              Hybrid Score Ranking
                      │
                      ▼
          Top Relevant Destinations
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
 Analytics Dashboard      Similar Destination
                             Recommendation
```

---

# 📈 Future Enhancements

- Semantic Search using Sentence Transformers
- Real-time travel information APIs
- Interactive map visualization
- Personalized user profiles
- Image-based destination search
- Voice-enabled destination search
- User authentication and saved trips

---

# 🎯 Learning Outcomes

This project demonstrates practical implementation of:

- Information Retrieval (IR)
- Machine Learning (ML)
- Natural Language Processing (NLP)
- Hybrid Search Systems
- Content-Based Recommendation Systems
- Data Preprocessing
- Interactive Web Application Development with Streamlit

---

# 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

Feel free to fork this repository and submit a Pull Request.

---

# 📜 License

This project is intended for educational and academic purposes.

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

Made with ❤️ using Python, Streamlit, Information Retrieval and Machine Learning.

</div>