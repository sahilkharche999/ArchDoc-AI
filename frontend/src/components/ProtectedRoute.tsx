import { Navigate } from "react-router-dom";
import { useAuth } from "../app/context/AuthContext";

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
    const { token, isLoading } = useAuth();

    if (isLoading) {
        return <div className="flex items-center justify-center h-screen">
            <p className="text-muted-foreground">Loading...</p>
        </div>;
    }

    if (!token) {
        return <Navigate to="/login" replace />;
    }

    return <>{children}</>;
}