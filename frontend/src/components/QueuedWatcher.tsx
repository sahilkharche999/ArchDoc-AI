import { useEffect } from "react";
import { useAuth } from "../app/context/AuthContext";
export function QueuedWatcher({ jobId, onStarted }: { jobId: string; onStarted: (filePath: string) => void }) {
    const { logout, token } = useAuth();
    useEffect(() => {
        const es = new EventSource(`${import.meta.env.VITE_API_URL}/api/v1/jobs/${jobId}/stream`);
        es.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.status === "processing") {
                es.close();
                // re-fetch to get file_path
                fetch(`${import.meta.env.VITE_API_URL}/api/v1/projects/${jobId}`,{ headers: { "Authorization": `Bearer ${token}` } })
                    .then(r => r.json())
                    .then(project => onStarted(project.file_path));
            }
        };
        es.onerror = () => es.close();
        return () => es.close();
    }, [jobId]);
    return null;
}