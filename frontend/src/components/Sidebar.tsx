import {Badge} from "./ui/badge";
import {Button} from "./ui/button";
import {useEffect, useState} from "react";
import {useNavigate} from "react-router-dom";
import {Project} from "../types/project";
import logo from '../assets/dax_mfg_logo.jpeg'
import {Plus, Settings, LogOut} from "lucide-react";
import {useAuth} from "../app/context/AuthContext";


interface SidebarProps {
    onNewEstimation: () => void;
    selectedProjectId?: string | null;
    onSelectProject?: (id: string) => void;
}

export function Sidebar({onNewEstimation, selectedProjectId, onSelectProject}: SidebarProps) {
    const [projects, setProjects] = useState<Project[]>([]);
    const navigate = useNavigate();
    const { token ,logout} = useAuth();
    useEffect(() => {
        if (!token) return; 
        const fetchProjects = async () => {
            try {
                const res = await fetch(
                    `${import.meta.env.VITE_API_URL}/api/v1/projects/`,
                    { headers: { "Authorization": `Bearer ${token}` } }
                );
                const data = await res.json();
                setProjects(data.projects || []);
            } catch (err) {
                console.error("Failed to fetch projects", err);
            }
        };

        fetchProjects();
    }, []);

    useEffect(() => {

        if (projects.length === 0) return;

        const sources: EventSource[] = [];

        projects.forEach((project) => {
            if (project.status !== "processing" && project.status !== "pending") return;

            const es = new EventSource(
                `${import.meta.env.VITE_API_URL}/api/v1/jobs/${project.job_id}/stream`
            );

            es.onmessage = (event) => {
                const data = JSON.parse(event.data);
                const isFinalComplete = data.step === "agent_4_merger" && data.status === "completed";
                const isFailed = data.status === "failed";

                if (isFinalComplete || isFailed) {
                    //  update that project only
                    setProjects(prev =>
                        prev.map(p =>
                            p.job_id === project.job_id
                                ? {...p, status: isFailed ? "failed" : "completed"}
                                : p
                        )
                    );

                    es.close(); // stop listening
                }
            };

            es.onerror = () => {
                es.close();
            };

            sources.push(es);
        });

        return () => {
            sources.forEach(s => s.close());
        };
    }, [projects]);
    
    useEffect(() => {
        if (!token) return; 
        const refresh = () => {
            fetch(`${import.meta.env.VITE_API_URL}/api/v1/projects/`,{ headers: { "Authorization": `Bearer ${token}` }})
                .then(r => r.json())
                .then(data => setProjects(data.projects || []))
                .catch(() => {},
            );
        };
        window.addEventListener("job-completed", refresh);
        return () => window.removeEventListener("job-completed", refresh);
    }, []);

    const renameProject = async (jobId: string, newName: string) => {
        try {
            await fetch(`${import.meta.env.VITE_API_URL}/api/v1/projects/${jobId}`, {
                method: "PUT",
                headers: {"Content-Type": "application/json","Authorization": `Bearer ${token}`},
                body: JSON.stringify({new_name: newName})
            });

            // refresh UI
            setProjects(prev =>
                prev.map(p =>
                    p.job_id === jobId ? {...p, name: newName} : p
                )
            );

        } catch (err) {
            console.error("Rename failed", err);
        }
    };

    const deleteProject = async (jobId: string) => {
        try {
            await fetch(`${import.meta.env.VITE_API_URL}/api/v1/projects/${jobId}`, {
                method: "DELETE",
                headers: { "Authorization": `Bearer ${token}` }
            });
        } catch (err) {
            console.error("Delete failed", err);
        }
        setProjects(prev => prev.filter(p => p.job_id !== jobId));
        if (jobId === selectedProjectId) {
            navigate("/projects", { replace: true });
        }
    };

    return (
        <div
            className="w-[260px] bg-sidebar text-sidebar-foreground h-screen flex flex-col border-r border-sidebar-border">

            <div className="p-6 border-b border-sidebar-border">
                <div className="flex items-center gap-3">
                    <img
                        src={logo}
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
                            onRename={renameProject}
                            onDelete={deleteProject}
                        />
                    ))}
                </div>
            </div>

            {/* New Estimation Button */}
            {/* Bottom Buttons */}
            <div className="p-4 border-t border-sidebar-border space-y-2">
                <Button
                    onClick={() => onNewEstimation()}
                    className="w-full bg-accent hover:bg-accent/90 text-accent-foreground"
                >
                    <Plus className="w-4 h-4 mr-2"/>
                    New Estimation
                </Button>
               <div className="flex gap-2">
                    <Button
                        onClick={() => navigate("/settings")}
                        className="flex-1 text-sidebar-foreground bg-sidebar-accent/30 border border-sidebar-border hover:bg-sidebar-accent cursor-pointer"
                    >
                        <Settings className="w-4 h-4 mr-2"/>
                        Settings
                    </Button>
                    <Button
                        onClick={() => {
                            logout();
                            navigate("/login");
                        }}
                        className="flex-1 text-red-400 bg-red-500/10 border border-red-400/30 hover:bg-red-500/20 cursor-pointer"
                    >
                        <LogOut className="w-4 h-4 mr-2"/>
                        Logout
                    </Button>
                </div>
            </div>
        </div>
    );
}

function ProjectItem({
                         project,
                         isSelected,
                         onClick,
                         onRename,
                         onDelete
                     }: {
    project: Project;
    isSelected?: boolean;
    onClick?: () => void;
    onRename?: (id: string, name: string) => void;
    onDelete?: (id: string) => void;

}) {
    const status = project.status?.toLowerCase();
    const [isEditing, setIsEditing] = useState(false);
    const [name, setName] = useState(project.name);
    const handleRename = () => {
        setIsEditing(false);
        if (name.trim() && name !== project.name) {
            onRename?.(project.job_id, name);
        }
    };

    const formatDate = (dateStr?: string) => {
        if (!dateStr) return "";

        const date = new Date(dateStr);

        return date.toLocaleDateString("en-GB", {
            day: "2-digit",
            month: "short",
            year: "numeric",
        });
    };

    return (
        <div
            className={`p-3 rounded-lg cursor-pointer transition-colors group ${isSelected ? "bg-sidebar-accent" : "hover:bg-sidebar-accent/50"
            }`}
            onClick={onClick}
        >
            <div className="flex items-start justify-between mb-1">
                {isEditing ? (
                    <input
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        onBlur={handleRename}
                        onKeyDown={(e) => e.key === "Enter" && handleRename()}
                        className="text-sm bg-transparent border-b border-white outline-none text-white w-full"
                        autoFocus
                    />
                ) : (
                    <h3 className="text-sm text-white line-clamp-1">
                        {project.name}
                    </h3>
                )}
                <Badge
                    variant={status === "completed" ? "secondary" : "default"}
                    className={
                        status === "completed"
                            ? "bg-emerald-500/20 text-emerald-400 text-xs"
                            : status === "failed"
                            ? "bg-red-500/20 text-red-400 text-xs"
                            : status === "pending"
                            ? "bg-yellow-500/20 text-yellow-400 text-xs"
                            : "bg-accent/20 text-accent text-xs"
                    }
                >
                    {status}
                </Badge>

            </div>
            <p className="text-xs text-sidebar-foreground/60">{formatDate(project.date)}</p>
            <div className="flex justify-between mt-2 opacity-100 transition">
                <button
                    onClick={(e) => {
                        e.stopPropagation();
                        setIsEditing(true);
                    }}
                    style={{cursor: 'pointer'}}
                    className="text-sm px-2 py-1 rounded bg-blue-500/20 text-blue-400 hover:bg-blue-500/30"
                >
                    Edit
                </button>

                <button
                    onClick={(e) => {
                        e.stopPropagation();
                        if (confirm("Delete this project?")) {
                            onDelete?.(project.job_id);
                        }
                    }}
                    style={{cursor: 'pointer'}}
                    className="text-sm px-2 py-1 rounded bg-red-500/20 text-red-400 hover:bg-red-500/30"
                >
                    Delete
                </button>
            </div>
        </div>
    );
}