import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { FileText, Plus } from "lucide-react";
import { useEffect, useState } from "react";
import { Project } from "../types/project";

interface SidebarProps {
  onNewEstimation: () => void;
  selectedProjectId?: string;
  onSelectProject?: (id: string) => void;
}

export function Sidebar({ onNewEstimation, selectedProjectId, onSelectProject }: SidebarProps) {
  const [projects, setProjects] = useState<Project[]>([]);
  useEffect(() => {

  const fetchProjects = () => {
    fetch(`${import.meta.env.VITE_API_URL}/api/v1/projects`)
      .then(res => res.json())
      .then(data => {
        setProjects(data.projects)
         if (!selectedProjectId &&data.projects.length > 0 && onSelectProject) {
           onSelectProject(data.projects[0].job_id);
       }

      })
      .catch(err => console.error(err));
  };

  fetchProjects(); 
   
  const interval = setInterval(fetchProjects, 300000);

  return () => clearInterval(interval);

}, [selectedProjectId]);
  return (
    <div className="w-[260px] bg-sidebar text-sidebar-foreground h-screen flex flex-col border-r border-sidebar-border">

      <div className="p-6 border-b border-sidebar-border">
        <div className="flex items-center gap-3">
          <img
            src="/assets/dax_mfg_logo.jpeg"
            alt="DAX Logo"
            className="w-8 h-8 rounded"
          />
          <h1 className="text-lg text-white font-semibold">DAX Steel Estimator</h1>
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        <div className="mb-3">
          <h2 className="text-xs uppercase tracking-wider text-sidebar-foreground/60 mb-3">
            Projects
          </h2>
        </div>
        <div className="space-y-2">
          {projects.map((project) => (
            <ProjectItem
              key={project.job_id}
              project={project}
              isSelected={project.job_id === selectedProjectId}
              onClick={() => onSelectProject?.(project.job_id)}
            />
          ))}
        </div>
      </div>

      {/* New Estimation Button */}
      <div className="p-4 border-t border-sidebar-border">
        <Button
          onClick={onNewEstimation}
          className="w-full bg-accent hover:bg-accent/90 text-accent-foreground"
        >
          <Plus className="w-4 h-4 mr-2" />
          New Estimation
        </Button>
      </div>
    </div>
  );
}

function ProjectItem({
  project,
  isSelected,
  onClick
}: {
  project: Project;
  isSelected?: boolean;
  onClick?: () => void;
}) {
  return (
    <div
      className={`p-3 rounded-lg cursor-pointer transition-colors group ${isSelected ? "bg-sidebar-accent" : "hover:bg-sidebar-accent/50"
        }`}
      onClick={onClick}
    >
      <div className="flex items-start justify-between mb-1">
        <h3 className="text-sm text-white line-clamp-1">{project.name}</h3>
        <Badge
          variant={project.status === "Completed" ? "secondary" : "default"}
          className={
            project.status === "Completed"
              ? "bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20 text-xs"
              : "bg-accent/20 text-accent hover:bg-accent/20 text-xs"
          }
        >
          {project.status}
        </Badge>
      </div>
      <p className="text-xs text-sidebar-foreground/60">{project.date}</p>
    </div>
  );
}