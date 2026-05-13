import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

import { Sidebar } from "../components/Sidebar";
import { ProcessingView } from "../components/ProcessingView";
import { EstimationDashboard } from "../components/EstimationDashboard";

export default function ProjectPage() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [status, setStatus] = useState<
    "loading" | "processing" | "completed" | "failed"
  >("loading");

  const [filePath, setFilePath] = useState<string | null>(null);
  const [bomData, setBomData] = useState<any[]>([]);

  useEffect(() => {
    if (!id) return;

    const fetchProject = async () => {
      try {
        const res = await fetch(
          `${import.meta.env.VITE_API_URL}/api/v1/projects/${id}`
        );

        if (!res.ok) {
          navigate("/projects");
          return;
        }

        const project = await res.json();
        const s = project.status?.toLowerCase();

        if (s === "pending" ||s === "processing") {
          setFilePath(project.file_path);
          setStatus("processing");
        }
        if (s === "failed") {
          setStatus("failed");
        }

        if (s === "completed") {
          const resultRes = await fetch(
            `${import.meta.env.VITE_API_URL}/api/v1/jobs/${id}/result`
          );

          const data = await resultRes.json();
          setBomData(data.bom || []);
          setStatus("completed");
        }
      } catch (err) {
        console.error("Failed to fetch project", err);
      }
    };

    fetchProject();
  }, [id]);

  // loading state
  if (status === "loading") {
    return <div className="p-6">Loading...</div>;
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
                  Something went wrong while processing this project. You can delete it and start again, or retry if it was a temporary issue.
                </p>
              </div>

              <div className="flex gap-2 justify-center">
                <button
                  onClick={async () => {
                    await fetch(`${import.meta.env.VITE_API_URL}/api/v1/projects/${id}`, { method: "DELETE" });
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
              setStatus("completed");
            }}
            onFailed={() => setStatus("failed")}
          />
        )}

        {status === "completed" && (
          <EstimationDashboard projectId={id} bomData={bomData} />
        )}
      </div>
    </div>
  );
}