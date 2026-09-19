"""Password gate for the API, the video streams and the live event stream.

This system stores face embeddings and images of identifiable people, so it is
protected by default rather than on request. Without this, anyone who can reach
the port can watch every camera, enroll a face, or delete an enrolled person.

  DASHBOARD_PASSWORD   set the password; otherwise one is generated on first
                       run, printed once, and kept in backend/.secrets/
  AUTH_DISABLED=1      turn the gate off (prints a warning; for a machine that
                       is not reachable from anywhere else)

The browser gets an HttpOnly cookie signed with a per-install secret, so
sessions survive restarts but cannot be forged.
"""
import hmac
import os
import secrets
import stat

from flask import jsonify, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

SECRETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".secrets")
COOKIE = "sv_session"
MAX_AGE_S = 30 * 86400
# open to everyone: liveness checks and the login endpoint itself
PUBLIC_PATHS = {"/health", "/api/health", "/api/login", "/api/auth"}


def _read_or_create(name, factory):
    """Persist a generated secret in backend/.secrets/<name>, readable only by this user."""
    os.makedirs(SECRETS_DIR, exist_ok=True)
    path = os.path.join(SECRETS_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            value = f.read().strip()
        if value:
            return value, False
    value = factory()
    with open(path, "w") as f:
        f.write(value)
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return value, True


class Auth:
    def __init__(self):
        self.disabled = os.environ.get("AUTH_DISABLED", "").strip().lower() in ("1", "true", "yes", "on")
        self.password = os.environ.get("DASHBOARD_PASSWORD") or ""
        self.generated = False
        if self.disabled:
            print("\n⚠  AUTH_DISABLED=1 — the API, cameras and enrolled faces are open to "
                  "anyone who can reach this port.\n", flush=True)
            self._serializer = None
            return
        if not self.password:
            self.password, self.generated = _read_or_create("password", lambda: secrets.token_urlsafe(9))
        secret, _ = _read_or_create("cookie_key", lambda: secrets.token_urlsafe(32))
        self._serializer = URLSafeTimedSerializer(secret, salt="securevision-session")

    def announce(self):
        if self.disabled:
            return
        where = "DASHBOARD_PASSWORD" if os.environ.get("DASHBOARD_PASSWORD") else "backend/.secrets/password"
        if self.generated:
            print("\n" + "=" * 62, flush=True)
            print(f"  Dashboard password: {self.password}", flush=True)
            print("  Generated on first run and saved in backend/.secrets/password", flush=True)
            print("  Change it with the DASHBOARD_PASSWORD environment variable.", flush=True)
            print("=" * 62 + "\n", flush=True)
        else:
            print(f"✓ Authentication on (password from {where})", flush=True)

    # ── session ──────────────────────────────────────────
    def token(self):
        return self._serializer.dumps("ok")

    def valid(self, token):
        if not token:
            return False
        try:
            self._serializer.loads(token, max_age=MAX_AGE_S)
            return True
        except (BadSignature, SignatureExpired):
            return False

    def check_password(self, given):
        return bool(given) and hmac.compare_digest(str(given), self.password)

    def authenticated(self):
        return self.disabled or self.valid(request.cookies.get(COOKIE))

    # ── Flask wiring ─────────────────────────────────────
    def guard(self):
        """before_request hook: allow public paths, else require a valid session."""
        if self.disabled or request.method == "OPTIONS":
            return None
        path = request.path.rstrip("/") or "/"
        if path in PUBLIC_PATHS:
            return None
        if self.valid(request.cookies.get(COOKIE)):
            return None
        return jsonify({"error": "authentication required", "login": "/api/login"}), 401

    def install(self, app):
        app.before_request(self.guard)

        @app.post("/api/login")
        def login():
            data = request.get_json(silent=True) or {}
            if self.disabled:
                return jsonify({"ok": True, "required": False})
            if not self.check_password(data.get("password")):
                return jsonify({"error": "incorrect password"}), 401
            resp = jsonify({"ok": True})
            resp.set_cookie(COOKIE, self.token(), max_age=MAX_AGE_S, httponly=True,
                            samesite="Lax", secure=request.is_secure)
            return resp

        @app.post("/api/logout")
        def logout():
            resp = jsonify({"ok": True})
            resp.delete_cookie(COOKIE)
            return resp

        @app.get("/api/auth")
        def auth_state():
            return jsonify({"required": not self.disabled, "authenticated": self.authenticated()})

        return self
