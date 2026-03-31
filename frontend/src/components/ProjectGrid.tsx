import React from "react";
import { Card, CardContent } from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Folder } from "lucide-react";
import { Button } from "../components/ui/button";
import { useState } from "react";
import logo from "../assets/dax_mfg_logo.jpeg";
interface Project {
  job_id: string;
  name: string;
  date: string;
  status: string;
}
function formatDate(dateString: string) {
  const date = new Date(dateString)

  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric"
  })
}

interface ProjectGridProps {
  projects: Project[];
  search: string;
  setSearch: React.Dispatch<React.SetStateAction<string>>;
  onSelect: (jobId: string| null) => void;
  onRename: (id: string, name: string) => void;
  onDelete: (id: string) => void;
};

export function ProjectGrid({ projects, search, setSearch, onSelect ,onRename,onDelete}:ProjectGridProps) {
  const filteredProjects = projects.filter((p: Project) =>
    p.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="p-8 w-full h-screen overflow-auto bg-background">
      
      {/* Header */}
      <div className="border-b mb-8">
  <div className="flex justify-between items-center max-w-6xl mx-auto py-4">

    {/* LEFT: Logo + Title */}
    <div className="flex items-center gap-3">
      <img
        src={logo}
        alt="logo"
        className="w-8 h-8 rounded"
      />
      <h1 className="text-xl font-semibold">
        DAX Steel Estimator
      </h1>
    </div>

    {/* RIGHT: Button */}
    <Button onClick={() => onSelect(null)}>
      + New Estimation
    </Button>

  </div>
</div>
<div className="max-w-6xl mx-auto mb-6">
  <h2 className="text-3xl font-bold">Your Projects</h2>
  <p className="text-muted-foreground">
    Select a project to view estimation details
  </p>
</div>


      {/* Search */}
     <div className="max-w-6xl mx-auto mb-8">
  <Input
    placeholder="Search projects..."
    value={search}
    onChange={(e) => setSearch(e.target.value)}
    className="h-11"
  />
</div>

      {/* Grid */}
      <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredProjects.map((project: Project) => (
  <ProjectCard
    key={project.job_id}
    project={project}
    onSelect={onSelect}
    onRename={onRename}
    onDelete={onDelete}
  />
))}
      </div>

      {/* Empty State */}
      {filteredProjects.length === 0 && (
        <div className="text-center mt-20 text-muted-foreground space-y-2">
  <p className="text-lg font-medium">No projects found</p>
  <p className="text-sm">Try creating a new estimation</p>
</div>
      )}
    </div>
  );
}

function ProjectCard({
  project,
  onSelect,
  onRename,
  onDelete
}: {
  project: Project;
  onSelect: (id: string) => void;
  onRename: (id: string, name: string) => void;
  onDelete: (id: string) => void;
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [name, setName] = useState(project.name);

  const handleRename = () => {
    setIsEditing(false);
    if (name.trim() && name !== project.name) {
      onRename(project.job_id, name);
    }
  };

  return (
    <Card
      onClick={() => onSelect(project.job_id)}
        className="cursor-pointer border hover:border-primary/50 hover:shadow-lg transition-all duration-200 group"    >
      <CardContent className="p-5 space-y-4">

        {/* Name */}
        <div className="flex items-center gap-3">
          <Folder className="w-5 h-5 text-primary" />

          {isEditing ? (
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              onBlur={handleRename}
              onKeyDown={(e) => e.key === "Enter" && handleRename()}
              className="border-b outline-none text-lg w-full"
              autoFocus
            />
          ) : (
            <h3 className="font-semibold text-lg truncate">
              {project.name}
            </h3>
          )}
        </div>
        
        <div className="border-t pt-3">
        {/* Date */}
        <p className="text-sm text-muted-foreground">
          Created: {formatDate(project.date)}
        </p>

        {/* Status */}
       <div>
  <span
    className={`text-xs px-2 py-1 rounded ${
      project.status === "completed"
        ? "bg-emerald-500/20 text-emerald-400"
        : "bg-accent/20 text-accent"
    }`}
  >
    {project.status}
  </span>
</div>
        </div>
        {/* Actions */}
        <div className="flex justify-between mt-3 opacity-100 transition">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setIsEditing(true);
            }}
            className="text-sm px-2 cursor-pointer py-1 rounded bg-blue-500/20 text-blue-400 hover:bg-blue-500/30"
          >
            Edit
          </button>

          <button
            onClick={(e) => {
              e.stopPropagation();
              if (confirm("Delete this project?")) {
                onDelete(project.job_id);
              }
            }}
            className="text-sm px-2 py-1 cursor-pointer rounded bg-red-500/20 text-red-400 hover:bg-red-500/30"
          >
            Delete
          </button>
        </div>

      </CardContent>
    </Card>
  );
}