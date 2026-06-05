import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../app/context/AuthContext";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Button } from "../components/ui/button";
import logo from "../assets/dax_mfg_logo.jpeg";

export default function LoginPage() {
    const { login } = useAuth();
    const navigate = useNavigate();
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    async function handleLogin() {
        if (!email || !password) return;
        setIsLoading(true);
        setError("");
        try {
            const res = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/auth/login`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password })
            });
            const data = await res.json();
            if (!res.ok) {
                setError(data.detail || "Login failed");
                return;
            }
            // Fetch user info
            const meRes = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/auth/me`, {
                headers: { "Authorization": `Bearer ${data.token}` }
            });
            const userData = await meRes.json();
            login(data.token, userData);
            navigate("/projects");
        } catch {
            setError("Something went wrong. Try again.");
        } finally {
            setIsLoading(false);
        }
    }

    return (
        <div className="flex items-center justify-center min-h-screen bg-background">
            <Card className="w-full max-w-sm">
                <CardHeader className="text-center space-y-3">
                    <img src={logo} alt="DAX" className="w-12 h-12 rounded mx-auto" />
                    <CardTitle className="text-2xl">DAX Steel Estimator</CardTitle>
                    <p className="text-sm text-muted-foreground">Sign in to your account</p>
                </CardHeader>
                <CardContent className="space-y-4">
                    <Input
                        type="email"
                        placeholder="Email"
                        value={email}
                        onChange={e => setEmail(e.target.value)}
                        onKeyDown={e => e.key === "Enter" && handleLogin()}
                    />
                    <Input
                        type="password"
                        placeholder="Password"
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                        onKeyDown={e => e.key === "Enter" && handleLogin()}
                    />
                    {error && <p className="text-sm text-red-500">{error}</p>}
                    <Button
                        onClick={handleLogin}
                        disabled={isLoading || !email || !password}
                        className="w-full bg-accent hover:bg-accent/90 text-accent-foreground cursor-pointer"
                    >
                        {isLoading ? "Signing in..." : "Sign In"}
                    </Button>
                    <p className="text-center text-sm text-muted-foreground">
                        Don't have an account?{" "}
                        <Link to="/register" className="text-accent hover:underline">
                            Register
                        </Link>
                    </p>
                </CardContent>
            </Card>
        </div>
    );
}