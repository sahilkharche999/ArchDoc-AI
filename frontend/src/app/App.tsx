import { useState,useEffect } from "react";
import { Sidebar } from "../components/Sidebar";
import { UploadState } from "../components/UploadState";
import { ProcessingView } from "../components/ProcessingView";
import { EstimationDashboard } from "../components/EstimationDashboard";
import {Intro} from '../components/Intro';
import {ProjectGrid} from '../components/ProjectGrid';
import { Project } from "../types/project";


type AppState = "upload" | "processing" | "dashboard";

export default function App() {
  const [appState, setAppState] = useState<AppState>("dashboard");
  const [projects, setProjects] = useState<Project[]>([])
  const [selectedProjectId, setSelectedProjectId] =useState<string | null>(null)
  const [search, setSearch] = useState("")
  const [jobId, setJobId] = useState<string | null>(null);
  const [filePath, setFilePath] = useState<string | null>(null);
  const [bomData, setBomData] = useState<any[]>([]);
  const [isStarting, setIsStarting] = useState(false)

const handleSelectProject = async (jobId: string|null) => {
  if (!jobId) {
    setAppState("upload");
    setSelectedProjectId(null);
    return;
  }
  setSelectedProjectId(jobId);
  try {
    const res = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/projects/${jobId}`);
    const project = await res.json();
    console.log("RAW STATUS:", project.status);

    const status = project.status?.trim().toLowerCase();
    console.log("NORMALIZED:", status);

    // 🔴 PROCESSING
    if (status === "processing") {
      setJobId(jobId);
      setFilePath(project.file_path);
      setBomData([]);
      setAppState("processing");
      return;
    }

    // 🔵 COMPLETED
    if (status === "completed") {
      const resultRes = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/jobs/${jobId}/result`);

      if (!resultRes.ok) {
        console.warn("Result not available");
        setAppState("dashboard");
        setBomData([]);
        return;
      }

      const data = await resultRes.json();
  

      setBomData(data.bom || []);
      setAppState("dashboard");
    }

  } catch (err) {
    console.error("Failed to fetch project", err);
  }
};


 const renameProject = async (jobId: string, newName: string) => {
  try {
    await fetch(`${import.meta.env.VITE_API_URL}/api/v1/projects/${jobId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ new_name: newName })
    });

    // refresh UI
    setProjects(prev =>
      prev.map(p =>
        p.job_id === jobId ? { ...p, name: newName } : p
      )
    );

  } catch (err) {
    console.error("Rename failed", err);
  }
};

const deleteProject = async (jobId: string) => {
  try {
    await fetch(`${import.meta.env.VITE_API_URL}/api/v1/projects/${jobId}`, {
      method: "DELETE"
    });

    setProjects(prev => prev.filter(p => p.job_id !== jobId));

  } catch (err) {
    console.error("Delete failed", err);
  }
};



const handleNewEstimation = () => {
    setAppState("upload");
    setSelectedProjectId(null);
  };

  const handleStartProcessing =async (jobId: string, filePath: string) => {
    setIsStarting(true);
  setSelectedProjectId(jobId);
  setJobId(jobId);
  setFilePath(filePath);
  try {
    await fetch(`${import.meta.env.VITE_API_URL}/api/v1/jobs/start`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    job_id: jobId,
    file_path: filePath
  })
});
 setAppState("processing");
  }
  finally {
  setIsStarting(false);
}
};

  const handleProcessingComplete = (result: any) => {
  setBomData(result.bom);
  setAppState("dashboard");
};
 
  const fetchProjects = async () => {
    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/projects`);
      const data = await res.json();
      console.log("PROJECT API RESPONSE:", data);
      setProjects(data.projects);
    } catch (err) {
      console.error("Failed to fetch projects", err);
    }
  };
useEffect(() => {
  fetchProjects();
}, []);

if (isStarting) {
  return (
    <div className="flex items-center justify-center h-screen w-full bg-background">
      <div className="text-center space-y-4">
        <div className="animate-spin h-10 w-10 border-4 border-primary border-t-transparent rounded-full mx-auto"></div>
        <p className="text-lg font-medium">Starting processing...</p>
        <p className="text-sm text-muted-foreground">
          Setting things up for your project
        </p>
      </div>
    </div>
  );
}

if (appState === "upload") {
  return (
    <UploadState onStartProcessing={handleStartProcessing} />
  )
}


if (projects.length === 0 && appState === "dashboard") {
  return <Intro onStart={handleNewEstimation} />
}



if (!selectedProjectId) {
  return (
    <ProjectGrid
      projects={projects}
      search={search}
      setSearch={setSearch}
      onSelect={handleSelectProject}
      onRename={renameProject}
      onDelete={deleteProject}
    />
  )
}


  return (
    <>
      <div className="flex h-screen bg-background">
        {/* Sidebar */}
      
        <Sidebar
          onNewEstimation={handleNewEstimation}
          selectedProjectId={selectedProjectId}
          onSelectProject={handleSelectProject}
        />

        {/* Main Content */}
  <div className="flex-1 overflow-auto">
          {appState === "processing" && jobId && filePath && (
  <ProcessingView
    jobId={jobId}
    filePath={filePath}
    onComplete={handleProcessingComplete}
  />
)}
         {appState === "dashboard" && (
  <EstimationDashboard
    projectId={selectedProjectId}
    bomData={bomData}
  />
)}
</div>
        </div>
        
    </>
  );
}