# DEX – AI Structural Estimation Engine

DEX is an AI-powered backend system that analyzes structural engineering drawings and automatically generates a **Bill of Materials (BOM)** from PDF plans.

The system combines:

* Vision Models (symbol detection)
* LLM reasoning
* Graph database retrieval
* LangGraph workflow orchestration

to convert engineering drawings into structured material estimates.

---

# 🚀 Features

* 📄 Upload structural drawing PDFs
* 🧠 Automatic page classification (plan / section / text)
* 🔍 Symbol detection using **GroundingDINO**
* 🧾 Detail extraction using **LLM vision models**
* 🧠 GraphRAG retrieval of schedules and rules
* ⚙️ Multi-agent workflow powered by **LangGraph**
* 📊 Final Bill of Materials with pricing estimation
* 🔄 Real-time workflow streaming via **Server-Sent Events**

---

# 🏗️ System Architecture

```
Frontend (React / UI)
        │
        ▼
FastAPI Backend
        │
        ▼
Service Layer
        │
        ▼
LangGraph Workflow
        │
 ┌──────┴─────────────┐
 │                    │
 ▼                    ▼
Vision Pipeline      Graph Database
(GroundingDINO)      (Schedule + Detail Knowledge)
 │
 ▼
LLM Extraction (Gemini / Groq)
 │
 ▼
Final Bill of Materials
```

---

# 📁 Project Structure

```
DEX
│
├── assets/                # Uploaded PDFs
├── output_temp/           # Intermediate processing outputs
│
├── src
│   │
│   ├── api                # FastAPI routes
│   │   ├── routes
│   │   │   ├── upload.py
│   │   │   ├── jobs.py
│   │   │   └── projects.py
│   │   └── main.py
│   │
│   ├── db                 # SQLite database helpers
│   │
│   ├── infrastructure     # Vision + graph integrations
│   │
│   ├── workflow           # LangGraph agent workflows
│   │   ├── common
│   │   ├── tools
│   │   └── workflows
│   │
│   ├── logger.py          # Central logging setup
│   └── service.py         # Workflow execution layer
│
├── requirements.txt
├── checkpoints.sqlite     # LangGraph checkpoint database
└── README.md
```

---

# ⚙️ Requirements

* Python **3.10+**
* CUDA GPU (optional but recommended)
* Neo4j database
* API keys for LLM providers

---

# 🔑 Environment Variables

Create a `.env` file in the root directory.

Example:

```
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

---

# 📦 Installation

Clone the repository:

```
git clone https://github.com/yourusername/dex-ai-estimator.git
cd dex-ai-estimator
```

Create virtual environment:

```
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Running the Backend

Start the FastAPI server:

```
uvicorn src.api.main:app --reload
```

Server will start at:

```
http://localhost:8000
```

API docs available at:

```
http://localhost:8000/docs
```

---

# 📡 API Endpoints

### Upload Structural Drawing

```
POST /api/v1/upload
```

Form Data:

```
file        -> PDF file
start_page  -> first structural page
end_page    -> last structural page
```

Response:

```
{
  "job_id": "uuid",
  "file_path": "assets/job_structural.pdf"
}
```

---

### Stream Workflow Progress

```
GET /api/v1/jobs/stream
```

Query Parameters:

```
job_id
file_path
```

Returns **Server-Sent Events** showing workflow progress.

---

### Get Job Result

```
GET /api/v1/jobs/{job_id}/result
```

Response:

```
{
  "job_id": "...",
  "bom": [...]
}
```

---

### List Projects

```
GET /api/v1/projects
```

Returns list of uploaded estimation jobs.

---

# 🤖 AI Workflow

The system runs a multi-agent workflow:

### Agent 0 – Page Classifier

Classifies pages into:

* text
* floor plan
* section detail

---

### Agent 1 – Text Rule Extractor

Extracts:

* schedules
* notes
* structural rules

Stores results in **graph database**.

---

### Agent 2 – Plan Processor

Processes floor plans:

* table detection
* schedule ingestion
* plan crop extraction

---

### Agent 3 – Detail Extraction

Extracts structural details:

* materials
* fabrication metrics
* detail relationships

---

### Agent 4 – Vision Estimator

Combines:

* detected symbols
* graph knowledge
* LLM reasoning

to generate final **Bill of Materials**.

---

# 🧠 Technologies Used

* **FastAPI**
* **LangGraph**
* **Google Gemini**
* **Groq Llama Vision**
* **GroundingDINO**
* **Neo4j**
* **SQLite**
* **pdfplumber / pypdf**
* **Pandas**

---

# 📊 Output Example

```
Beam W24X62
Total Length: 240 ft
Quantity: 10
Total Weight: 14,880 lbs
Total Cost: $22,320
```

---

# 🛠 Future Improvements

* Material cost optimization
* Multi-floor aggregation
* CAD integration
* Cloud deployment
* Parallel agent execution

---

# 📜 License

MIT License

---

# 👨‍💻 Author

Built for structural estimation automation using AI.
