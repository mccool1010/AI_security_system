"""Enrolled-identity store with backend-safe matching.

Face embeddings are only comparable when produced by the same model. Every
user document records the `face_backend` that produced its embeddings; users
enrolled with a different backend are kept in the database but excluded from
matching and reported as `needs_reenrollment` so the UI can say so.

Older documents have no `face_backend` field. facenet (InceptionResnetV1)
emits unit-length vectors, so a legacy user whose embeddings all have norm ~1
is treated as facenet; anything else is marked `legacy-unknown`.
"""
import threading
import uuid
from datetime import datetime, timezone

import numpy as np

MATCH_THRESHOLD_DEFAULT = 0.60   # cosine similarity; see eval/eval_faces.py to tune
MAX_EMBEDDINGS_PER_USER = 30


def infer_legacy_backend(doc):
    embs = [doc.get("embedding")] + list(doc.get("embeddings") or [])
    norms = [float(np.linalg.norm(e)) for e in embs if e is not None and len(e)]
    if norms and all(abs(n - 1.0) < 0.02 for n in norms):
        return "facenet-vggface2"
    return "legacy-unknown"


def backend_of(doc):
    return doc.get("face_backend") or infer_legacy_backend(doc)


class IdentityStore:
    def __init__(self, users_collection, backend_id, threshold=MATCH_THRESHOLD_DEFAULT):
        self._col = users_collection
        self.backend_id = backend_id
        self.threshold = threshold
        self._lock = threading.Lock()
        self._users = []          # compatible users: {user_id, name, matrix (N×D, L2-normalised)}
        self._incompatible = []   # [{user_id, name, face_backend}]
        self.load_error = None

    # ── Cache ────────────────────────────────────────────────────
    def reload(self):
        users, incompatible = [], []
        try:
            docs = list(self._col.find({}))
            self.load_error = None
        except Exception as e:
            self.load_error = str(e)
            print(f"❌ Failed to load users: {e}", flush=True)
            return
        for d in docs:
            uid = d.get("user_id") or str(d.get("_id"))
            be = backend_of(d)
            if be != self.backend_id:
                incompatible.append({"user_id": uid, "name": d.get("name"), "face_backend": be})
                continue
            embs = list(d.get("embeddings") or [])
            if not embs and d.get("embedding"):
                embs = [d["embedding"]]
            if not embs:
                continue
            m = np.asarray(embs, dtype=np.float32)
            m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
            users.append({"user_id": uid, "name": d.get("name"), "matrix": m})
        with self._lock:
            self._users, self._incompatible = users, incompatible
        print(f"✓ Identities: {len(users)} matchable {[u['name'] for u in users]}", flush=True)
        if incompatible:
            print(f"⚠ {len(incompatible)} enrolled user(s) were made with a different face model and "
                  f"cannot be matched until re-enrolled: "
                  f"{[(u['name'], u['face_backend']) for u in incompatible]}", flush=True)

    def summary(self):
        with self._lock:
            return {
                "backend": self.backend_id,
                "threshold": self.threshold,
                "matchable": [{"user_id": u["user_id"], "name": u["name"], "samples": len(u["matrix"])}
                              for u in self._users],
                "needs_reenrollment": list(self._incompatible),
            }

    # ── Matching ─────────────────────────────────────────────────
    def match(self, embedding):
        """Best-of-N cosine match. Returns {user_id, name, score}; name='unknown' below threshold."""
        e = np.asarray(embedding, dtype=np.float32).ravel()
        e /= np.linalg.norm(e) + 1e-12
        best = {"user_id": None, "name": "unknown", "score": 0.0}
        with self._lock:
            users = self._users
        for u in users:
            if u["matrix"].shape[1] != e.shape[0]:
                continue
            s = float(np.max(u["matrix"] @ e))
            if s > best["score"]:
                best = {"user_id": u["user_id"], "name": u["name"], "score": s}
        best["score"] = round(best["score"], 4)
        if best["score"] < self.threshold:
            best["user_id"], best["name"] = None, "unknown"
        return best

    # ── Enrollment ───────────────────────────────────────────────
    def enroll(self, name, embeddings):
        """Add embeddings to the user called `name` (same backend) or create that user.

        Returns (user_id, total_samples, created).
        """
        name = name.strip()
        embs = [np.asarray(e, dtype=np.float32).tolist() for e in embeddings]
        now = datetime.now(timezone.utc).isoformat()
        existing = self._col.find_one({"name": name, "face_backend": self.backend_id})
        if existing:
            merged = (list(existing.get("embeddings") or []) + embs)[-MAX_EMBEDDINGS_PER_USER:]
            avg = np.mean(np.asarray(merged, dtype=np.float32), axis=0).tolist()
            self._col.update_one(
                {"_id": existing["_id"]},
                {"$set": {"embeddings": merged, "embedding": avg, "samples": len(merged), "updated_at": now}},
            )
            uid, created, total = existing.get("user_id") or str(existing["_id"]), False, len(merged)
        else:
            uid = str(uuid.uuid4())
            avg = np.mean(np.asarray(embs, dtype=np.float32), axis=0).tolist()
            self._col.insert_one({
                "user_id": uid, "name": name, "face_backend": self.backend_id,
                "embeddings": embs, "embedding": avg, "samples": len(embs), "created_at": now,
            })
            created, total = True, len(embs)
        self.reload()
        return uid, total, created
