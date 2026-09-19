import { useCallback, useEffect, useState } from "react";

/**
 * Password gate. The backend protects every route except /api/health and
 * /api/auth, and answers 401 when a session is missing or has expired; any
 * fetch that sees a 401 dispatches "sv-unauthorized", which brings this screen
 * back without a page reload.
 */
export default function AuthGate({ children }) {
  const [state, setState] = useState({ checked: false, required: true, authenticated: false });
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const check = useCallback(async () => {
    try {
      const res = await fetch("/api/auth");
      const data = await res.json();
      setState({ checked: true, required: data.required, authenticated: data.authenticated });
    } catch {
      // backend unreachable: let the app render and show its own offline state
      setState({ checked: true, required: false, authenticated: true });
    }
  }, []);

  useEffect(() => {
    check();
    const onUnauthorized = () => setState((s) => ({ ...s, checked: true, required: true, authenticated: false }));
    window.addEventListener("sv-unauthorized", onUnauthorized);
    return () => window.removeEventListener("sv-unauthorized", onUnauthorized);
  }, [check]);

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const res = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (res.ok) {
        setPassword("");
        await check();
      } else {
        setError((await res.json().catch(() => ({}))).error || "Login failed");
      }
    } catch (err) {
      setError(String(err));
    }
    setBusy(false);
  };

  if (!state.checked) return null;
  if (!state.required || state.authenticated) return children;

  return (
    <div className="login-screen">
      <form className="login-card" onSubmit={submit}>
        <h1>
          <span className="brand-dot" /> SecureVision
        </h1>
        <p>This system holds camera footage and enrolled faces. Sign in to continue.</p>
        <input
          type="password"
          value={password}
          autoFocus
          placeholder="Dashboard password"
          onChange={(e) => setPassword(e.target.value)}
        />
        {error && <p className="msg-error">{error}</p>}
        <button className="btn-primary" type="submit" disabled={busy || !password}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
        <p className="login-hint">
          First run? The password is printed in the backend console and saved in
          <code> backend/.secrets/password</code>.
        </p>
      </form>
    </div>
  );
}
