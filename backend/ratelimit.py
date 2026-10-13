import math
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple


class RateLimiter:
    """Sliding-window limiter with a per-session cap and a server-wide cap.

    check() both tests and records the request in one step, so it is safe to
    call from several Streamlit sessions at once. A limit of 0 disables that cap.
    """

    def __init__(self, per_session: int, global_limit: int, window_seconds: int):
        self.per_session = per_session
        self.global_limit = global_limit
        self.window = window_seconds
        self._lock = threading.Lock()
        self._sessions: Dict[str, Deque[float]] = defaultdict(deque)
        self._global: Deque[float] = deque()

    def _prune(self, hits: Deque[float], now: float) -> None:
        while hits and now - hits[0] >= self.window:
            hits.popleft()

    def check(self, session_id: str) -> Tuple[bool, int, str]:
        """Returns (allowed, seconds_until_retry, scope) where scope is 'session' or 'global'."""
        with self._lock:
            now = time.monotonic()
            mine = self._sessions[session_id]
            self._prune(mine, now)
            self._prune(self._global, now)

            if len(self._sessions) > 500:  # drop sessions with no recent activity
                for sid in [s for s, h in self._sessions.items() if not h and s != session_id]:
                    del self._sessions[sid]

            if self.per_session and len(mine) >= self.per_session:
                return False, math.ceil(self.window - (now - mine[0])), "session"
            if self.global_limit and len(self._global) >= self.global_limit:
                return False, math.ceil(self.window - (now - self._global[0])), "global"

            mine.append(now)
            self._global.append(now)
            return True, 0, ""
