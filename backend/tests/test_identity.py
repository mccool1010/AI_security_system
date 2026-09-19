import numpy as np
import pytest

from identity import IdentityStore, infer_legacy_backend


class FakeCollection:
    def __init__(self, docs=()):
        self.docs = [dict(d, _id=i) for i, d in enumerate(docs)]

    def find(self, q=None):
        return [d for d in self.docs if all(d.get(k) == v for k, v in (q or {}).items())]

    def find_one(self, q):
        found = self.find(q)
        return found[0] if found else None

    def insert_one(self, doc):
        self.docs.append(dict(doc, _id=len(self.docs)))

    def update_one(self, q, update):
        for d in self.find(q):
            d.update(update["$set"])


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def rand_unit(rng, dim=512):
    return unit(rng.normal(size=dim))


def test_legacy_backend_inference():
    rng = np.random.default_rng(0)
    assert infer_legacy_backend({"embedding": rand_unit(rng).tolist()}) == "facenet-vggface2"
    assert infer_legacy_backend({"embedding": (rand_unit(rng) * 4.8).tolist()}) == "legacy-unknown"


def test_incompatible_users_are_excluded_and_reported():
    rng = np.random.default_rng(1)
    alice = rand_unit(rng)
    col = FakeCollection([
        {"user_id": "a", "name": "Alice", "embedding": alice.tolist()},                       # legacy facenet
        {"user_id": "b", "name": "Bob", "embedding": (rand_unit(rng) * 4.8).tolist()},       # legacy other model
        {"user_id": "c", "name": "Cara", "face_backend": "deepface-arcface",
         "embeddings": [rand_unit(rng).tolist()]},
    ])
    store = IdentityStore(col, "facenet-vggface2")
    store.reload()
    s = store.summary()
    assert [u["name"] for u in s["matchable"]] == ["Alice"]
    assert {u["name"] for u in s["needs_reenrollment"]} == {"Bob", "Cara"}
    assert store.match(alice)["name"] == "Alice"


def test_match_threshold():
    rng = np.random.default_rng(2)
    a = rand_unit(rng)
    store = IdentityStore(FakeCollection([{"user_id": "a", "name": "A", "face_backend": "x",
                                           "embeddings": [a.tolist()]}]), "x", threshold=0.6)
    store.reload()
    near = unit(a + 0.3 * rand_unit(rng))
    assert store.match(near)["name"] == "A"
    stranger = store.match(rand_unit(rng))
    assert stranger["name"] == "unknown" and stranger["user_id"] is None


def test_enroll_merges_same_name_same_backend():
    rng = np.random.default_rng(3)
    col = FakeCollection()
    store = IdentityStore(col, "x")
    uid1, n1, created1 = store.enroll("Hari", [rand_unit(rng) for _ in range(3)])
    uid2, n2, created2 = store.enroll("Hari ", [rand_unit(rng) for _ in range(2)])
    assert created1 and not created2
    assert uid1 == uid2 and n2 == 5
    assert len(col.docs) == 1 and col.docs[0]["face_backend"] == "x"


def test_enroll_does_not_merge_across_backends():
    rng = np.random.default_rng(4)
    col = FakeCollection([{"user_id": "old", "name": "Hari", "embedding": (rand_unit(rng) * 4).tolist()}])
    store = IdentityStore(col, "facenet-vggface2")
    uid, n, created = store.enroll("Hari", [rand_unit(rng)])
    assert created and uid != "old" and len(col.docs) == 2
    assert [u["name"] for u in store.summary()["matchable"]] == ["Hari"]


def test_mismatched_dimensions_never_match():
    store = IdentityStore(FakeCollection([{"user_id": "a", "name": "A", "face_backend": "x",
                                           "embeddings": [unit(np.ones(128)).tolist()]}]), "x")
    store.reload()
    assert store.match(unit(np.ones(512)))["name"] == "unknown"


@pytest.mark.parametrize("n", [1, 40])
def test_embedding_cap(n):
    rng = np.random.default_rng(5)
    col = FakeCollection()
    store = IdentityStore(col, "x")
    store.enroll("Z", [rand_unit(rng) for _ in range(n)])
    store.enroll("Z", [rand_unit(rng) for _ in range(n)])
    assert len(col.docs[0]["embeddings"]) <= 30
