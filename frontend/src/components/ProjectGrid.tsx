import React from "react";
import { Card, CardContent } from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Folder } from "lucide-react";
import { Button } from "../components/ui/button";

interface Project {
  job_id: string;
  name: string;
  date: string;
  status: string;
}

interface ProjectGridProps {
  projects: Project[];
  search: string;
  setSearch: React.Dispatch<React.SetStateAction<string>>;
  onSelect: (jobId: string| null) => void;
};

export function ProjectGrid({ projects, search, setSearch, onSelect }:ProjectGridProps) {
  const filteredProjects = projects.filter((p: Project) =>
    p.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="p-8 w-full h-screen overflow-auto bg-background">
      
      {/* Header */}
      <div className="flex justify-between items-center max-w-6xl mx-auto mb-8">
        <div>
            <h1 className="text-3xl font-bold">Your Projects</h1>
            <p className="text-muted-foreground">
            Select a project to view estimation details
            </p>
        </div>
        <Button onClick={() => onSelect(null)}>
            + New Estimation
        </Button>
        </div>

      {/* Search */}
      <div className="max-w-6xl mx-auto mb-6">
        <Input
          placeholder="Search projects..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      {/* Grid */}
      <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredProjects.map((project: Project) => (
          <Card
            key={project.job_id}
            onClick={() => onSelect(project.job_id)}
            className="cursor-pointer hover:shadow-lg transition-all hover:scale-[1.02]"
          >
            <CardContent className="p-5 space-y-3">
              
              {/* Icon + Name */}
              <div className="flex items-center gap-3">
                <Folder className="w-5 h-5 text-primary" />
                <h3 className="font-semibold text-lg truncate">
                  {project.name}
                </h3>
              </div>

              {/* Date */}
              <p className="text-sm text-muted-foreground">
                Created: {project.date}
              </p>

              {/* Status */}
              <div className="text-xs text-muted-foreground">
                Status: {project.status}
              </div>

            </CardContent>
          </Card>
        ))}
      </div>

      {/* Empty State */}
      {filteredProjects.length === 0 && (
        <div className="text-center mt-20 text-muted-foreground">
          No projects found 
        </div>
      )}
    </div>
  );
}