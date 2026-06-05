import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../app/context/AuthContext";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Button } from "../components/ui/button";
import { Key, LogOut, CheckCircle } from "lucide-react";

export default function SettingsPage() {
    const { token, user, logout } = useAuth();
    const navigate = useNavigate();
    const [apiKey, setApiKey] = useState("");
    const [saved, setSaved] = useState(false);
    const [error, setError] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    async function handleSaveKey() {
        if (!apiKey.trim()) return;
        setIsLoading(true);
        setError("");
        try {
            const res = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/auth/api-key`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${token}`
                },
                body: JSON.stringify({ gemini_api_key: apiKey })
            });
            if (!res.ok) {
                setError("Failed to save API key");
                return;
            }
            setSaved(true);
            setApiKey("");
            setTimeout(() => navigate("/projects"), 1500);
        } catch {
            setError("Something went wrong.");
        } finally {
            setIsLoading(false);
        }
    }

    function handleLogout() {
        logout();
        navigate("/login");
    }

    return (
        <div className="flex items-center justify-center min-h-screen bg-background">
            <Card className="w-full max-w-sm">
                <CardHeader className="space-y-1">
                    <CardTitle className="text-xl">Settings</CardTitle>
                    <p className="text-sm text-muted-foreground">{user?.email}</p>
                </CardHeader>
                <CardContent className="space-y-6">
                    <div className="space-y-3">
                        <div className="flex items-center gap-2">
                            <Key className="w-4 h-4 text-muted-foreground" />
                            <p className="text-sm font-medium">Gemini API Key</p>
                        </div>
                        {user?.has_gemini_key && (
                            <div className="flex items-center gap-2 text-emerald-600 text-sm">
                                <CheckCircle className="w-4 h-4" />
                                <span>API key is configured</span>
                            </div>
                        )}
                        <Input
                            type="password"
                            placeholder={user?.has_gemini_key ? "Enter new key to update" : "Enter Gemini API key"}
                            value={apiKey}
                            onChange={e => setApiKey(e.target.value)}
                        />
                        {saved && <p className="text-sm text-emerald-600">API key saved successfully!</p>}
                        {error && <p className="text-sm text-red-500">{error}</p>}
                        <Button
                            onClick={handleSaveKey}
                            disabled={isLoading || !apiKey.trim()}
                            className="w-full bg-accent hover:bg-accent/90 text-accent-foreground cursor-pointer"
                        >
                            {isLoading ? "Saving..." : "Save API Key"}
                        </Button>
                    </div>

                    <div className="border-t pt-4">
                        <Button
                            onClick={handleLogout}
                            variant="outline"
                            className="w-full cursor-pointer"
                        >
                            <LogOut className="w-4 h-4 mr-2" />
                            Sign Out
                        </Button>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}