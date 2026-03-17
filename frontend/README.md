# DAX Estimation Dashboard

A modern **React + TypeScript dashboard** for the DAX estimation system.
This frontend provides the interface for uploading drawings, tracking processing status, and viewing generated BOM data.

---

# Overview

The DAX dashboard allows users to:

* Upload manufacturing drawings or PDFs
* Monitor document processing
* View extracted Bill of Materials (BOM)
* Track project estimation details
* Inspect traceability information

The UI is designed for **manufacturing estimation workflows**.

---

# Tech Stack

Frontend technologies used:

* **React**
* **TypeScript**
* **Vite**
* **TailwindCSS**
* **Emotion (Styled Components)**

---

# Project Structure

```
src
 ├── app
 │   └── App.tsx
 │
 ├── assets
 │   └── dax_mfg_logo.jpeg
 │
 ├── components
 │   ├── BOMTable.tsx
 │   ├── DashboardHeader.tsx
 │   ├── EditableCell.tsx
 │   ├── EstimationDashboard.tsx
 │   ├── NotificationProvider.tsx
 │   ├── ProcessingView.tsx
 │   ├── Sidebar.tsx
 │   ├── SummaryTab.tsx
 │   ├── TraceabilityView.tsx
 │   └── UploadState.tsx
 │
 ├── styles
 │   ├── fonts.css
 │   ├── index.css
 │   ├── tailwind.css
 │   └── theme.css
 │
 ├── types
 │   ├── bom.ts
 │   ├── project.ts
 │   └── summary.ts
 │
 ├── main.tsx
 └── vite-env.d.ts
```

---

# Installation

Clone the repository:

```
git clone https://github.com/sahilkharche999/ArchDoc-AI
cd frontend
```

Install dependencies:

```
npm install
```

---

# Run Development Server

Start the development server:

```
npm run dev
```

The app will be available at:

```
http://localhost:5173
```

---

# Build for Production

Create optimized production build:

```
npm run build
```

The output files will be generated in:

```
dist/
```

---

# Environment Variables

Create a `.env` file in the project root.

Example:

```
VITE_API_URL=http://localhost:8000
```

---

# Backend Integration

The frontend communicates with the backend API to:

* Upload drawings
* Retrieve processing status
* Fetch generated BOM data

---
