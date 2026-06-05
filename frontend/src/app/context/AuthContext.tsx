// src/app/context/AuthContext.tsx
import { createContext, useContext, useState, useEffect, ReactNode } from "react";

interface User {
    user_id: string;
    email: string;
    has_gemini_key: boolean;
}

interface AuthContextType {
    token: string | null;
    user: User | null;
    login: (token: string, user: User) => void;
    logout: () => void;
    isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
    const [token, setToken] = useState<string | null>(null);
    const [user, setUser] = useState<User | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        // On app load, check if token exists in localStorage
        const savedToken = localStorage.getItem("dax_token");
        if (savedToken) {
            // Verify token is still valid
            fetch(`${import.meta.env.VITE_API_URL}/api/v1/auth/me`, {
                headers: { "Authorization": `Bearer ${savedToken}` }
            })
            .then(res => {
                if (res.ok) return res.json();
                throw new Error("Token invalid");
            })
            .then(userData => {
                setToken(savedToken);
                setUser(userData);
            })
            .catch(() => {
                localStorage.removeItem("dax_token");
            })
            .finally(() => setIsLoading(false));
        } else {
            setIsLoading(false);
        }
    }, []);

    function login(token: string, user: User) {
        localStorage.setItem("dax_token", token);
        setToken(token);
        setUser(user);
    }

    function logout() {
        localStorage.removeItem("dax_token");
        setToken(null);
        setUser(null);
    }

    return (
        <AuthContext.Provider value={{ token, user, login, logout, isLoading }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
    return ctx;
}