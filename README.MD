# DAX — AI Structural Estimation System

DAX is an AI-powered system designed to analyze structural engineering drawings and automatically generate material estimates from PDF plans.

The platform processes engineering drawings, extracts structural information, and produces a **Bill of Materials (BOM)** that can be used for project estimation and planning.

The goal of DAX is to reduce the manual effort required to interpret construction drawings and convert them into structured material estimates.

---

# What the System Does

DAX enables users to:

• Upload structural drawing PDFs
• Automatically analyze drawing content
• Extract structural details and schedules
• Process relationships between plans and section details
• Generate a structured Bill of Materials (BOM)
• View and track estimation results through a dashboard

The system combines document analysis, visual understanding, and knowledge retrieval to convert complex engineering drawings into structured project data.

---

# High Level Workflow

```
User Uploads Structural Drawing
            │
            ▼
Document Processing Pipeline
            │
            ▼
Drawing Understanding & Detail Extraction
            │
            ▼
Knowledge Retrieval from Graph Database
            │
            ▼
Bill of Materials Generation
            │
            ▼
Estimation Dashboard
```

---

# Project Structure

```
DAX
│
├── backend
│   ├── assets/          # Uploaded drawing files
│   ├── output_temp/     # Intermediate processing outputs
│   ├── src/             # Backend source code
│   └── README.md        # Backend documentation
│
├── frontend
│   ├── src/             # Frontend source code
│   └── README.md        # Frontend documentation
│
└── README.md            # Project overview (this file)
```

---

# Repository Components

### Backend

The backend is responsible for processing uploaded drawings, running the estimation workflow, and generating the final Bill of Materials.

It manages document analysis, workflow execution, and result generation.

Detailed backend documentation is available here:

```
backend/README.md
```

---

### Frontend

The frontend provides a dashboard interface where users can upload drawings, monitor processing progress, and view generated estimation results.

Detailed frontend documentation is available here:

```
frontend/README.md
```

---

# System Overview

```
Frontend Dashboard
        │
        ▼
Backend Processing Engine
        │
        ▼
AI Workflow Pipeline
        │
        ▼
Graph Knowledge Retrieval
        │
        ▼
Final Bill of Materials
```

---

# Use Cases

DAX can assist in workflows such as:

• Structural material estimation
• Construction planning
• Engineering drawing interpretation
• Automated document analysis
• Project cost estimation

---

## Authors

- [**Sahil Kharche**](https://github.com/sahilkharche999)  
- [**Chetan Padhen**](https://github.com/Chetax)  