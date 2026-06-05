import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

import { Sidebar } from "../components/Sidebar";
import { ProcessingView } from "../components/ProcessingView";
import { EstimationDashboard } from "../components/EstimationDashboard";
import { QueuedWatcher } from "../components/QueuedWatcher";
import { useAuth } from "../app/context/AuthContext";

export default function ProjectPage() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [status, setStatus] = useState<
    "loading" | "processing" | "completed" | "failed" |  "queued"
  >("loading");

  const [filePath, setFilePath] = useState<string | null>(null);
  const [bomData, setBomData] = useState<any[]>([]);
  const [unreferencedDetails, setUnreferencedDetails] = useState<any[]>([]);
  const [runningJobName, setRunningJobName] = useState<string | null>(null);
  const [message, setMessage] = useState<string | undefined>(undefined);
  const { token } = useAuth();
  const [failedError, setFailedError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;

    const fetchProject = async () => {
      try {
        const res = await fetch(
          `${import.meta.env.VITE_API_URL}/api/v1/projects/${id}`,
          { headers: { "Authorization": `Bearer ${token}` } }
        );

        if (!res.ok) {
          navigate("/projects");
          return;
        }

        const project = await res.json();
        const s = project.status?.toLowerCase();
        if (s === "pending") {
            setStatus("queued");
        } else if (s === "processing") {
            setFilePath(project.file_path);
            setStatus("processing");
        } else if (s === "failed") {
            setStatus("failed");
        }
        else if (s === "completed") {
          const resultRes = await fetch(
            `${import.meta.env.VITE_API_URL}/api/v1/jobs/${id}/result`,
            { headers: { "Authorization": `Bearer ${token}` } }
          );

          const data = await resultRes.json();
          setBomData(data.bom || []);
          setUnreferencedDetails(data.unreferenced_details || []);
          setMessage(data.message);
          setStatus("completed");
        }
      } catch (err) {
        console.error("Failed to fetch project", err);
      }
    };

    fetchProject();
  }, [id]);

  

  useEffect(() => {
      if (status !== "queued") return;
      fetch(`${import.meta.env.VITE_API_URL}/api/v1/projects/`,{ headers: { "Authorization": `Bearer ${token}` } })
          .then(r => r.json())
          .then(data => {
              const running = (data.projects || []).find((p: any) => p.status === "processing");
              if (running) setRunningJobName(running.name);
          });
  }, [status]);

  // loading state
  if (status === "loading") {
    return <div className="p-6">Loading...</div>;
  }
  const getErrorMessage = (error: string | null) => {
    if (!error) return "Something went wrong while processing this project.";
    if (error.includes("RESOURCE_EXHAUSTED") || error.includes("prepayment credits"))
        return "Gemini API credits are depleted. Please top up your Google AI Studio account and try again.";
    if (error.includes("API key"))
        return "Invalid Gemini API key. Please update your API key in Settings.";
    return "Something went wrong while processing. Please try again.";
}

  return (
    <div className="flex h-screen bg-background">
      <Sidebar
        selectedProjectId={id}
        onSelectProject={(pid) => navigate(`/project/${pid}`)}
        onNewEstimation={() => navigate("/upload")}
      />

      <div className="flex-1 overflow-auto">
        {status === "failed" && (
          <div className="flex items-center justify-center h-full">
            <div className="border border-border rounded-xl p-10 max-w-sm w-full text-center space-y-5">

              <div className="w-14 h-14 rounded-full bg-red-500/10 flex items-center justify-center mx-auto">
                <svg xmlns="http://www.w3.org/2000/svg" className="w-7 h-7 text-red-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
                  <path d="M12 9v4" /><path d="M10.363 3.591l-8.106 13.534a1.914 1.914 0 0 0 1.636 2.871h16.214a1.914 1.914 0 0 0 1.636 -2.871l-8.106 -13.534a1.914 1.914 0 0 0 -3.274 0z" /><path d="M12 16h.01" />
                </svg>
              </div>

              <div className="space-y-1">
                <p className="text-base font-medium text-foreground">Processing failed</p>
                <p className="text-sm text-muted-foreground leading-relaxed">
                    {getErrorMessage(failedError)}
                </p>
              </div>

              <div className="flex gap-2 justify-center">
                <button
                  onClick={async () => {
                    await fetch(`${import.meta.env.VITE_API_URL}/api/v1/projects/${id}`, { method: "DELETE" ,headers: { "Authorization": `Bearer ${token}`}});
                    navigate("/projects", { replace: true });
                  }}
                  className="flex items-center gap-1.5 text-sm px-4 py-2 rounded-lg bg-red-500/10 text-red-500 hover:bg-red-500/20 transition-colors cursor-pointer border border-red-500/20"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M4 7l16 0" /><path d="M10 11l0 6" /><path d="M14 11l0 6" /><path d="M5 7l1 12a2 2 0 0 0 2 2h8a2 2 0 0 0 2 -2l1 -12" /><path d="M9 7v-3a1 1 0 0 1 1 -1h4a1 1 0 0 1 1 1v3" /></svg>
                  Delete project
                </button>
                <button
                  onClick={() => navigate("/projects")}
                  className="flex items-center gap-1.5 text-sm px-4 py-2 rounded-lg bg-muted hover:bg-muted/80 transition-colors cursor-pointer border border-border text-foreground"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M5 12l14 0" /><path d="M5 12l6 6" /><path d="M5 12l6 -6" /></svg>
                  Back to projects
                </button>
              </div>

            </div>
          </div>
        )}
        {status === "processing" && filePath && (
          <ProcessingView
            jobId={id!}
            filePath={filePath}
            onComplete={(result) => {
              setBomData(result.bom);
              setUnreferencedDetails(result.unreferenced_details || []);
              setMessage(result.message);
              setStatus("completed");
            }}
            onFailed={(error) => {
                setFailedError(error || null);
                setStatus("failed");
            }}
          />
        )}

        {status === "completed" && (
          <EstimationDashboard projectId={id} bomData={bomData} unreferencedDetails={unreferencedDetails}  message={message}/>
        )}

        {status === "queued" && (
        <div className="flex items-center justify-center h-full">
            <div className="border border-border rounded-xl p-10 max-w-sm w-full text-center space-y-5">
                <div className="w-14 h-14 rounded-full bg-yellow-500/10 flex items-center justify-center mx-auto">
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-7 h-7 text-yellow-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
                        <path d="M12 6l0 6l3 3" />
                        <path d="M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0 -18 0" />
                    </svg>
                </div>
                <div className="space-y-1">
                    <p className="text-base font-medium text-foreground">In Queue</p>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                        This project is waiting to be processed. Another estimation is currently running — this will start automatically when it completes.
                    </p>
                    {runningJobName && (
                        <p className="text-xs text-muted-foreground pt-1">
                            Currently running: <span className="text-foreground font-medium">{runningJobName}</span>
                        </p>
                    )}
                </div>
                <QueuedWatcher
                    jobId={id!}
                    onStarted={(filePath) => {
                        setFilePath(filePath);
                        setStatus("processing");
                    }}
                />
            </div>
        </div>
          )}
         
      </div>
    </div>
  );
}