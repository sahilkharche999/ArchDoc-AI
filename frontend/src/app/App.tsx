import { useEffect, useState } from "react";
import { Routes, Route, Navigate, useNavigate ,useLocation} from "react-router-dom";
import { useAuth } from "./context/AuthContext";
import { Intro } from "../components/Intro";
import { ProjectGrid } from "../components/ProjectGrid";
import { UploadState } from "../components/UploadState";
import { ProtectedRoute } from "../components/ProtectedRoute";
import LoginPage from '../pages/LoginPage'
import RegisterPage from '../pages/RegisterPage' 
import ProjectPage from "../pages/ProjectPage"
import SettingsPage from "../pages/SettingsPage";

import { Project } from "../types/project";

export default function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [search, setSearch] = useState("");
  const navigate = useNavigate();
  const { token } = useAuth();
  

  // fetch projects
  const location = useLocation();
  useEffect(() => {
    const fetchProjects = async () => {
      if (!token) return;
      try {
        const res = await fetch(
          `${import.meta.env.VITE_API_URL}/api/v1/projects/`,  { headers: { "Authorization": `Bearer ${token}` } }
        );
        const data = await res.json();
        setProjects(data.projects || []);
      } catch (err) {
        console.error("Failed to fetch projects", err);
      }
    };
     if (location.pathname === "/projects") {
      fetchProjects();
    }
  }, [location.pathname,token]);

  return (
    <Routes>
     <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />

      <Route path="/" element={<Navigate to="/projects" />} />
      

  
      <Route
        path="/projects"
        element={
              <ProtectedRoute>{
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
                    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${token}` },
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
                    headers: { "Authorization": `Bearer ${token}` }
                    });

                    setProjects((prev) =>
                    prev.filter((p) => p.job_id !== jobId)
                    );

                } catch (err) {
                    console.error("Delete failed", err);
                }
                }}
            />
          )}
           </ProtectedRoute>
        }
      />

      {/* upload page */}
      <Route
        path="/upload"
        element={
          <ProtectedRoute>
          <UploadState
            onStartProcessing={async (jobId, filePath) => {
              await fetch(
                `${import.meta.env.VITE_API_URL}/api/v1/jobs/start`,
                {
                  method: "POST",
                  headers: { "Content-Type": "application/json" , "Authorization": `Bearer ${token}`},
                  body: JSON.stringify({
                    job_id: jobId,
                    file_path: filePath,
                  }),
                }
              );

              navigate(`/project/${jobId}`);
            }}
          />
          </ProtectedRoute>
        }
      />
      

      {/* project page */}
      <Route path="/project/:id" element={<ProtectedRoute><ProjectPage /></ProtectedRoute>} />
      <Route path="/settings" element={
                <ProtectedRoute>
                    <SettingsPage />
                </ProtectedRoute>
            } />
    </Routes>
  );
}