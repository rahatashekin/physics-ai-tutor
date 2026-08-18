# ⚛️ Physics Tutor Agent

> **বাংলাদেশ Class 9-10 পদার্থবিজ্ঞান — Adaptive RAG Chatbot**

একটি intelligent AI chatbot যা NCTB Class 9-10 Physics বই থেকে শিক্ষার্থীদের পদার্থবিজ্ঞান শেখায়।

---

## ✨ Features

| Feature | বিবরণ |
|---|---|
| 📖 **RAG-based Q&A** | বইয়ের content থেকে accurate উত্তর |
| 🧠 **Adaptive Learning** | শিক্ষার্থীর বোঝার স্তর অনুযায়ী ব্যাখ্যা adjust |
| 📷 **Vision Input** | সমস্যার ছবি পাঠিয়ে সাহায্য নাও |
| 📊 **Physics Diagrams** | v-t graph, force diagram, circuit, wave, projectile |
| 📐 **LaTeX Math** | সূত্র সুন্দরভাবে render |
| 🎯 **Quiz Mode** | Adaptive quiz ও hint system |
| 🌐 **Trilingual** | বাংলা · English · Banglish |

---

## 🚀 Quick Start

### 1. Clone করো
```bash
git clone https://github.com/yourusername/physics-tutor-agent.git
cd physics-tutor-agent
```

### 2. Environment setup করো
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### 3. API Key set করো
```bash
cp .env.example .env
# .env ফাইলে তোমার OpenAI API key লেখো
```

### 4. Physics বইয়ের PDF রাখো
```
data/raw/physics_9_10.pdf
```

### 5. Data pipeline চালাও (একবারই দরকার)
```bash
python scripts/1_extract.py      # PDF → structured text (১০-১৫ মিনিট)
python scripts/2_chunk.py        # Chunking (৫-১০ মিনিট)
python scripts/3_embed_store.py  # Embedding → LanceDB (৫ মিনিট)
```

### 6. App চালাও
```bash
streamlit run app.py
```

Browser এ যাও: `http://localhost:8501` 🎉

---

## 📁 Project Structure

```
physics-tutor-agent/
├── .env.example          ← API key template
├── .gitignore
├── requirements.txt
├── app.py                ← Streamlit UI (main entry point)
│
├── scripts/
│   ├── 1_extract.py      ← Docling PDF extraction
│   ├── 2_chunk.py        ← HybridChunker
│   └── 3_embed_store.py  ← LanceDB embedding
│
├── agent/
│   ├── tools.py          ← Search, vision, diagram tools
│   └── prompts.py        ← System prompts (adaptive)
│
├── memory/
│   └── user_profile.py   ← Adaptive learning engine
│
├── utils/
│   ├── tokenizer.py      ← OpenAI tokenizer wrapper
│   └── latex_formatter.py
│
├── data/
│   ├── raw/              ← PDF রাখার জায়গা (gitignored)
│   └── processed/        ← LanceDB + chunks (gitignored)
│
└── profiles/             ← User session profiles (gitignored)
```

---

## 🛠️ Tech Stack

- **LLM**: OpenAI GPT-4o-mini (vision support সহ)
- **Document Processing**: [Docling](https://github.com/DS4SD/docling)
- **Vector Store**: [LanceDB](https://lancedb.github.io/lancedb/)
- **Embeddings**: OpenAI text-embedding-3-small
- **UI**: [Streamlit](https://streamlit.io)
- **Diagrams**: Matplotlib + Schemdraw

---

## 🌐 Deployment (Streamlit Cloud)

1. GitHub এ push করো (`.env` ও `data/` gitignored তাই safe)
2. [share.streamlit.io](https://share.streamlit.io) এ যাও
3. Repository connect করো
4. Secrets এ `OPENAI_API_KEY` set করো
5. Deploy! 🚀

---

## 📄 License

MIT License — শিক্ষামূলক উদ্দেশ্যে ব্যবহারযোগ্য।

---

*Built with ❤️ for Bangladesh students*
