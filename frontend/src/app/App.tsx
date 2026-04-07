import { useEffect, useState } from "react";
import { Routes, Route, Navigate, useNavigate } from "react-router-dom";

import { Intro } from "../components/Intro";
import { ProjectGrid } from "../components/ProjectGrid";
import { UploadState } from "../components/UploadState";
import ProjectPage from "../pages/ProjectPage"

import { Project } from "../types/project";

export default function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [search, setSearch] = useState("");
  const navigate = useNavigate();
  

  // fetch projects
  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const res = await fetch(
          `${import.meta.env.VITE_API_URL}/api/v1/projects/`
        );
        const data = await res.json();
        setProjects(data.projects || []);
      } catch (err) {
        console.error("Failed to fetch projects", err);
      }
    };

    fetchProjects();
  }, []);

  return (
    <Routes>

      <Route path="/" element={<Navigate to="/projects" />} />

  
      <Route
        path="/projects"
        element={
          projects.length === 0 ? (
            <Intro onStart={() => navigate("/upload")} />
          ) : (
            <ProjectGrid
              projects={projects}
              search={search}
              setSearch={setSearch}
              onSelect={(id) => {
                if (!id) navigate("/upload");
                else navigate(`/project/${id}`);
              }}
              onRename={async (jobId: string, newName: string) => {
                try {
                    await fetch(`${import.meta.env.VITE_API_URL}/api/v1/projects/${jobId}`, {
                    method: "PUT",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ new_name: newName }),
                    });

                    setProjects((prev) =>
                    prev.map((p) =>
                        p.job_id === jobId ? { ...p, name: newName } : p
                    )
                    );
                } catch (err) {
                    console.error("Rename failed", err);
                }
                }}

                onDelete={async (jobId: string) => {
                try {
                    await fetch(`${import.meta.env.VITE_API_URL}/api/v1/projects/${jobId}`, {
                    method: "DELETE",
                    });

                    setProjects((prev) =>
                    prev.filter((p) => p.job_id !== jobId)
                    );

                } catch (err) {
                    console.error("Delete failed", err);
                }
                }}
            />
          )
        }
      />

      {/* upload page */}
      <Route
        path="/upload"
        element={
          <UploadState
            onStartProcessing={async (jobId, filePath) => {
              await fetch(
                `${import.meta.env.VITE_API_URL}/api/v1/jobs/start`,
                {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    job_id: jobId,
                    file_path: filePath,
                  }),
                }
              );

              navigate(`/project/${jobId}`);
            }}
          />
        }
      />

      {/* project page */}
      <Route path="/project/:id" element={<ProjectPage />} />
    </Routes>
  );
}