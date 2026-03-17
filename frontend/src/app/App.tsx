import { useState } from "react";
import { Sidebar } from "../components/Sidebar";
import { UploadState } from "../components/UploadState";
import { ProcessingView } from "../components/ProcessingView";
import { EstimationDashboard } from "../components/EstimationDashboard";
import { NotificationProvider } from "../components/NotificationProvider";

type AppState = "upload" | "processing" | "dashboard";

export default function App() {
  const [appState, setAppState] = useState<AppState>("dashboard");
  const [selectedProjectId, setSelectedProjectId] = useState<string>();
  const [jobId, setJobId] = useState<string | null>(null);
  const [filePath, setFilePath] = useState<string | null>(null);
  const [bomData, setBomData] = useState<any[]>([]);
  

const handleSelectProject = async (jobId: string) => {
  setSelectedProjectId(jobId);

  try {
    const res = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/projects/${jobId}`);
    console.log("Project status response:", res.status);

    const project = await res.json();
    console.log("Project object:", project);

    if (project.status === "Processing") {
      setJobId(jobId);
      setFilePath(project.file_path);
      setAppState("processing");
      return;
    }

    const resultRes = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/jobs/${jobId}/result`);
    const data = await resultRes.json();

    setBomData(data.bom || []);
    setAppState("dashboard");

  } catch (err) {
    console.error("Failed to fetch project", err);
  }
};


  const handleNewEstimation = () => {
    setAppState("upload");
  };

  const handleStartProcessing = (jobId: string, filePath: string) => {
  setSelectedProjectId(jobId);
  setJobId(jobId);
  setFilePath(filePath);
  setAppState("processing");
};

  const handleProcessingComplete = (result: any) => {
  setBomData(result.bom);
  setAppState("dashboard");
};

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
          {appState === "upload" && (
            <UploadState onStartProcessing={handleStartProcessing} />
          )}
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
      <NotificationProvider />
    </>
  );
}