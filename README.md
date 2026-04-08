
🚀 HalluciCheck

Evidence-Based Hallucination Detection for LLM Outputs

HalluciCheck is a full-stack AI evaluation platform that detects hallucinations in Large Language Model (LLM) responses using retrieval, semantic similarity, NLI reasoning, and evaluation frameworks (RAGAS & DeepEval).

It allows users to upload documents, ask questions, and verify whether an LLM-generated answer is factually grounded in real evidence.

⸻

🧠 Why this project matters

LLMs are powerful—but they hallucinate.

HalluciCheck solves this by:
	•	grounding answers in real documents
	•	measuring faithfulness and consistency
	•	providing transparent explanations + metrics
	•	enabling multi-model comparison

👉 This is not just a demo — it’s a research-grade evaluation pipeline.

⸻

✨ Core Features

🔍 1. Document-Based Verification
	•	Upload one or multiple PDFs
	•	Extract text and chunk documents
	•	Retrieve relevant evidence using semantic search

⸻

🤖 2. Two Modes of Operation

📝 Mode 1: Manual Verification
	•	Paste an LLM answer
	•	Check if it’s supported by the document

⚡ Mode 2: Chat + Verify
	•	Select multiple models:
	•	OpenAI (GPT)
	•	Anthropic (Claude)
	•	Google (Gemini)
	•	Generate answers automatically
	•	Compare results side-by-side

⸻

📊 3. Advanced Evaluation Metrics

Each answer is evaluated using multiple signals:

🧩 Retrieval & Grounding
	•	Similarity score (embeddings)
	•	Evidence chunk ranking

⚖️ Natural Language Inference (NLI)
	•	Entailment
	•	Neutral
	•	Contradiction

📚 External Evaluation Frameworks
	•	RAGAS
	•	Faithfulness
	•	Answer Relevancy
	•	DeepEval
	•	Faithfulness
	•	Relevancy

⸻

🧮 4. Trust Score System

A custom scoring system combines all signals into:
	•	✅ Trust Score (%)
	•	⚠️ Risk Percentage
	•	📌 Final Verdict
	•	Supported
	•	Partially Supported
	•	Unsupported

⸻

📈 5. Visual Analytics Dashboard
	•	Radar charts for metric distribution
	•	Comparison view across models
	•	Evidence highlighting (key sentences)
	•	Full metrics table

⸻

🧪 6. Model Disagreement Detection

Automatically detects when models:
	•	produce conflicting answers
	•	disagree on factual grounding

⸻

🏗️ Architecture Overview

User Input
   ↓
PDF Upload → Text Extraction
   ↓
Chunking (LangChain splitter)
   ↓
Embeddings (SentenceTransformers)
   ↓
FAISS Vector Search
   ↓
Top-K Evidence Retrieval
   ↓
NLI Model (DeBERTa)
   ↓
Evaluation (RAGAS + DeepEval)
   ↓
Custom Metrics + Trust Score
   ↓
Django UI (Charts + Evidence)


⸻

🧰 Tech Stack

🖥️ Backend
	•	Django 5
	•	PostgreSQL (Render)
	•	Gunicorn

🧠 AI / ML
	•	SentenceTransformers
	•	HuggingFace Transformers
	•	FAISS (vector search)
	•	RAGAS
	•	DeepEval

🌐 LLM Providers
	•	OpenAI
	•	Anthropic
	•	Google Gemini

🎨 Frontend
	•	Django Templates
	•	Chart.js (Radar Charts)
	•	Custom CSS UI

☁️ Deployment
	•	Render (Web Service + PostgreSQL)
	•	WhiteNoise (static files)

⸻

⚙️ Installation

1. Clone repo

git clone https://github.com/yourusername/hallucicheck.git
cd hallucicheck

2. Create virtual environment

python -m venv venv
source venv/bin/activate  # Mac/Linux

3. Install dependencies

pip install -r requirements.txt

4. Set environment variables

Create .env:

SECRET_KEY=your_secret_key
DEBUG=True

OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key

DATABASE_URL=sqlite:///db.sqlite3


⸻

5. Run migrations

python manage.py migrate

6. Run server

python manage.py runserver


⸻

🚀 Deployment (Render)
	1.	Create Web Service
	2.	Add PostgreSQL database
	3.	Set environment variables:
	•	DATABASE_URL
	•	API keys
	•	SECRET_KEY
	4.	Start command:

gunicorn config.wsgi:application --timeout 360


⸻

⚠️ Important Notes

🧠 Memory Constraints
	•	Transformer + NLI models are heavy
	•	Use Standard plan or higher on Render

⸻

⏱️ Timeouts
	•	Long AI pipelines require:

--timeout 360


⸻

📄 PDF Limitations
	•	Scanned PDFs may fail
	•	Use text-based PDFs

⸻

🔮 Future Improvements
	•	Async processing (Celery / Redis)
	•	Streaming results
	•	Caching embeddings
	•	Lighter models for production
	•	Real-time dashboards
	•	API version (SaaS)

⸻

🤝 Use Cases
	•	AI product evaluation
	•	LLM benchmarking
	•	Academic research
	•	Enterprise AI safety
	•	Fact-checking pipelines

⸻

🧑‍💻 Author

Anastasios Tsevas
🌐 hallucicheck.onrender.com

⸻

⭐ Final Note

HalluciCheck is not just a tool —
it’s a step toward trustworthy AI systems.

⸻

