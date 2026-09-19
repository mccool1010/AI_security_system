"""In-process publish/subscribe used for Server-Sent Events.

Each subscriber gets a bounded queue; a slow client drops its oldest messages
instead of blocking the camera workers.
"""
import json
import queue
import threading


class EventBus:
    def __init__(self, max_queue=200):
        self._subs = set()
        self._lock = threading.Lock()
        self._max = max_queue

    def subscribe(self):
        q = queue.Queue(maxsize=self._max)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subs.discard(q)

    @property
    def subscriber_count(self):
        with self._lock:
            return len(self._subs)

    def publish(self, kind, data):
        msg = f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(msg)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(msg)
                except (queue.Empty, queue.Full):
                    pass
