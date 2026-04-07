import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

import { Sidebar } from "../components/Sidebar";
import { ProcessingView } from "../components/ProcessingView";
import { EstimationDashboard } from "../components/EstimationDashboard";

export default function ProjectPage() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [status, setStatus] = useState<
    "loading" | "processing" | "completed"
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

        const project = await res.json();
        const s = project.status?.toLowerCase();

        if (s === "processing") {
          setFilePath(project.file_path);
          setStatus("processing");
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
        {status === "processing" && filePath && (
          <ProcessingView
            jobId={id!}
            filePath={filePath}
            onComplete={(result) => {
              setBomData(result.bom);
              setStatus("completed");
            }}
          />
        )}

        {status === "completed" && (
          <EstimationDashboard projectId={id} bomData={bomData} />
        )}
      </div>
    </div>
  );
}