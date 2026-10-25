"""OptChat core: append-only log, summary tree, sawtooth view.

Implements the UniiChat design (optchat.md) as durable per-session state:

  main.jsonl   messages, one per line: {i, kind, text, size, date, uid}
  tree.jsonl   tree nodes, one per line: {l, i, text, size}
  view.json    the view: {"covered": P, "lines": [[l, i], ...]}
  meta.json    counters (T, compactions, watermarks, last event ts)

Key invariants from the spec:
- append + flush only; never edit or delete a line; one writer per chat (flock)
- node(0,i) = message i in <=512 bytes; node(l,i) = its two halves merged
- the view covers the log prefix [0, covered); each new built line appends;
  merges happen in batches (sawtooth 64->128 KB) in Taelin push order:
  due = (T + 1)/2^l - i, most due first, oldest of ties, parent must be built
- the view is saved and loaded, never rebuilt (a rebuilt view kills the cache)
- never scan the tree for work: ready nodes live in queues
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .prompt import (
    NODE_MAX, RULER, TASK_MERGE, TASK_SINGLE, TASK_TOO_LONG,
)

logger = logging.getLogger(__name__)

ECHO_CLIP = 30_000       # chars: tool output kept as head+tail
SPLIT_AT = 30_000        # chars: any other long text logs as several messages
BUILD_WINDOW = 8         # a node starts once <8 lines before it are unbuilt
MAX_PARALLEL = 8         # up to 8 compaction calls at once
HIGH_DEFAULT = 128_000   # view sawtooth high water (bytes)
LOW_DEFAULT = 64_000      # view sawtooth low water (bytes)
COMPACT_TARGET = 24_000  # compaction view average size (bytes)
COMPACT_HIGH = 32_000    # compaction view re-merge trigger (bytes)
MAX_LINE_TRIES = 5       # too-long correction rounds per line

# Backpressure: the request only shrinks when the summary frontier (`covered`) keeps pace
# with new entries. If the background builder falls past BACKLOG_TRIGGER entries, a request's
# recent tail rides verbatim and can grow toward the context limit (the overflow that hit
# session 20261024_015150_225691dc). The request path then drains the frontier synchronously
# (bounded by budget/deadline) so the backlog can never grow unbounded.
BACKLOG_TRIGGER = 40      # entries: catch up once the un-summarized tail exceeds this
CATCHUP_BUDGET = 48       # entries: max the frontier may lag before a turn blocks on catch-up
CATCHUP_DEADLINE_S = 20.0  # seconds: upper bound a single turn waits on catch-up

LLMFn = Callable[[Optional[str], List[dict]], str]

# paths whose flock this process already holds (flock is per-fd, not per-process)
_HELD: set = getattr(sys, "_optchat_held", None) or set()
sys._optchat_held = _HELD

_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL,
                                           thread_name_prefix="optchat")
        return _executor


def _b(s: str) -> int:
    return len(s.encode("utf-8", "replace"))


def _clip(text: str, limit: int) -> str:
    if _b(text) <= limit:
        return text
    half = limit // 2
    raw = text.encode("utf-8", "replace")
    head = raw[:half].decode("utf-8", "ignore")
    tail = raw[-half:].decode("utf-8", "ignore")
    return head + "\n…[clipped]…\n" + tail


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") in ("text", "input_text", "output_text"):
                    parts.append(str(p.get("text", "")))
                elif "text" in p:
                    parts.append(str(p.get("text", "")))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(x for x in parts if x)
    return "" if content is None else str(content)


def _safe_id(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", session_id or "default")[:80] or "default"


class Store:
    """Durable per-chat state + background tree builder. One instance per session."""

    def __init__(self, session_id: str, home: Path, llm_fn: Optional[LLMFn] = None):
        self.session_id = session_id
        self.dir = Path(home) / "optchat" / _safe_id(session_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.llm_fn = llm_fn
        self.lock_fp = None
        self.read_only = False
        self._acquire_lock()

        self._lock = threading.RLock()
        self.log: List[dict] = []           # entries: {i,kind,text,size,date,uid}
        self.uid_index: Dict[str, int] = {}  # uid -> first entry id
        self.nodes: Dict[Tuple[int, int], dict] = {}
        self.view: List[Tuple[int, int]] = []
        self.covered = 0
        self.T = 0
        self.compactions = 0
        self.high, self.low = HIGH_DEFAULT, LOW_DEFAULT
        self.system_prompt: Optional[str] = None
        self.pending: set = set()            # (l, i) in flight
        self.failed: set = set()             # (l, i) to retry at next message
        self.view_version = 0
        self._render_cache: Optional[Tuple[int, str]] = None
        self._cv_cache: Optional[Tuple[Any, str]] = None
        self.last_event_ts = 0.0
        self._load()

    # ---------------------------------------------------------------- lock
    def _acquire_lock(self) -> None:
        if self.read_only:
            self._retry_lock()
            return
        key = str(self.dir.resolve())
        if key in _HELD:
            return  # same process, second handle: the process already owns it
        try:
            import fcntl
            self.lock_fp = open(self.dir / "lock", "a+")
            fcntl.flock(self.lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            _HELD.add(key)
        except (ImportError, OSError):
            self.read_only = True
            logger.warning("optchat[%s]: another process owns the chat; read-only view",
                           self.session_id)

    def _retry_lock(self) -> None:
        """A read-only handle retries the non-blocking flock on every event: after a
        plugin hot-reload the previous instance's fd latches us read-only until GC
        closes it; the lease must not be lost permanently."""
        key = str(self.dir.resolve())
        if self.lock_fp is None:
            return
        try:
            import fcntl
            fcntl.flock(self.lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.read_only = False
            _HELD.add(key)
            logger.info("optchat[%s]: lock acquired; resuming writes", self.session_id)
        except (ImportError, OSError):
            pass

    def close(self) -> None:
        try:
            import fcntl
            if self.lock_fp is not None:
                fcntl.flock(self.lock_fp.fileno(), fcntl.LOCK_UN)
                self.lock_fp.close()
                self.lock_fp = None
        except Exception:
            pass

    # ---------------------------------------------------------------- load/save
    def _load(self) -> None:
        for name in ("main.jsonl", "tree.jsonl"):
            path = self.dir / name
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if name == "main.jsonl":
                        self.log.append(rec)
                        uid = rec.get("uid")
                        if uid and uid not in self.uid_index:
                            self.uid_index[uid] = rec["i"]
                    else:
                        self.nodes[(rec["l"], rec["i"])] = rec
        self.T = len(self.log)
        vp = self.dir / "view.json"
        if vp.exists():
            try:
                data = json.loads(vp.read_text(encoding="utf-8"))
                self.view = [tuple(x) for x in data.get("lines", [])]
                self.covered = int(data.get("covered", 0))
            except Exception:
                logger.exception("optchat[%s]: bad view.json; starting empty", self.session_id)
        mp = self.dir / "meta.json"
        if mp.exists():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
                self.compactions = int(meta.get("compactions", 0))
                self.high = int(meta.get("high", HIGH_DEFAULT))
                self.low = int(meta.get("low", LOW_DEFAULT))
            except Exception:
                pass
        sp = self.dir / "sysprompt.txt"
        if sp.exists():
            self.system_prompt = sp.read_text(encoding="utf-8") or None

    def save_system_prompt(self) -> None:
        self._atomic(self.dir / "sysprompt.txt", self.system_prompt or "")

    def save_view(self) -> None:
        self._atomic(self.dir / "view.json",
                     json.dumps({"covered": self.covered, "lines": [list(x) for x in self.view]}))

    def save_meta(self) -> None:
        self.last_event_ts = time.time()
        self._atomic(self.dir / "meta.json", json.dumps({
            "session": self.session_id, "T": self.T, "covered": self.covered,
            "compactions": self.compactions, "high": self.high, "low": self.low,
            "view_bytes": self.view_bytes(), "view_lines": len(self.view),
            "built_nodes": len(self.nodes), "pending": len(self.pending),
            "read_only": self.read_only, "pid": os.getpid(),
            "last_event_ts": self.last_event_ts,
        }))

    def _atomic(self, path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def _append_file(self, name: str, rec: dict) -> None:
        with (self.dir / name).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    # ---------------------------------------------------------------- ingest
    def ingest(self, messages: List[dict]) -> int:
        """Idempotently log host messages (by message_uid). Returns new entries."""
        if self.read_only:
            self._acquire_lock()  # retry the lease (hot-reload latch clears here)
        if self.read_only or not messages:
            return 0
        added = 0
        with self._lock:
            for m in messages:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                if role == "system":
                    continue
                uid = m.get("message_uid") or m.get("uid")
                if uid and str(uid).startswith("optchat:"):
                    continue  # our own carriers / view messages
                if uid and str(uid) in self.uid_index:
                    continue
                if not uid:
                    body = _text_of(m.get("content"))
                    uid = "h:" + hashlib.sha1(
                        (role + "\0" + body).encode("utf-8", "replace")).hexdigest()[:16]
                    if uid in self.uid_index:
                        continue
                date = m.get("timestamp") or m.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S")
                for kind, text in self._expand(m):
                    self._log_entry(kind, text, str(uid), str(date))
                    added += 1
            if added:
                self.save_meta()
        if added:
            self._pump()
        return added

    @staticmethod
    def _expand(m: dict) -> List[Tuple[str, str]]:
        role = m.get("role")
        out: List[Tuple[str, str]] = []
        if role == "user":
            text = _text_of(m.get("content")).strip()
            if text:
                out.append(("user", text))
        elif role == "assistant":
            text = (_text_of(m.get("content")) or "").strip()
            if text:
                out.append(("unii", text))
            for call in m.get("tool_calls") or []:
                fn = (call or {}).get("function") or {}
                name = fn.get("name", "?")
                args = str(fn.get("arguments", ""))
                out.append(("tool", f"calls {name} {args}"))
        elif role == "tool":
            text = _text_of(m.get("content"))
            out.append(("echo", _clip(text, ECHO_CLIP)))
        return out

    def _log_entry(self, kind: str, text: str, uid: str, date: str) -> None:
        chunks = [text] if len(text) <= SPLIT_AT else \
            [text[k:k + SPLIT_AT] for k in range(0, len(text), SPLIT_AT)]
        for chunk in chunks:
            i = self.T
            self.T += 1
            entry = {"i": i, "kind": kind, "text": chunk, "size": _b(chunk),
                     "date": date, "uid": uid}
            self.log.append(entry)
            self.uid_index.setdefault(uid, i)
            self._append_file("main.jsonl", entry)

    # ---------------------------------------------------------------- tree
    def node(self, l: int, i: int) -> Optional[dict]:
        return self.nodes.get((l, i))

    def built(self, l: int, i: int) -> bool:
        return (l, i) in self.nodes

    def line_name(self, l: int, i: int) -> str:
        return f"{i << l}+{1 << l}"

    def _eligible0(self, i: int) -> bool:
        lo = max(0, i - BUILD_WINDOW)
        return all((0, j) in self.nodes for j in range(lo, i))

    def enqueue(self, l: int, i: int) -> None:
        if self.read_only or self.llm_fn is None:
            return
        key = (l, i)
        if key in self.nodes or key in self.pending:
            return
        self.pending.add(key)
        _pool().submit(self._build, l, i)

    def _pump(self) -> None:
        """After appends: start eligible level-0 nodes, retry failures, batch view."""
        with self._lock:
            i = len(self.log) - 1
            starts = set(range(max(0, i - BUILD_WINDOW), i + 1))
            starts |= set(range(self.covered, min(self.covered + BUILD_WINDOW + 1, self.T)))
            for j in sorted(starts):
                if (0, j) not in self.nodes and self._eligible0(j):
                    self.enqueue(0, j)
            for key in list(self.failed):
                self.failed.discard(key)
                self.enqueue(*key)
            self.batch()

    def _build(self, l: int, i: int) -> None:
        fallback = ""
        try:
            if l == 0:
                entry = self.log[i]
                if entry["size"] <= NODE_MAX:
                    text = entry["text"] or "(empty message)"
                else:
                    fallback = _clip(entry["text"], NODE_MAX)
                    task = TASK_SINGLE.format(id=i, ruler=RULER, kind=entry["kind"],
                                              text=entry["text"])
                    text = self._call_line(None, task)
            else:
                a = self.nodes.get((l - 1, 2 * i))
                b = self.nodes.get((l - 1, 2 * i + 1))
                if a is None or b is None:
                    return
                if a["size"] + b["size"] + 1 <= NODE_MAX:
                    text = a["text"] + "\n" + b["text"]
                else:
                    first, count = i << l, 1 << l
                    cv = self.compaction_view(first)
                    fallback = _clip(a["text"] + "\n" + b["text"], NODE_MAX)
                    task = TASK_MERGE.format(a=self.line_name(l - 1, 2 * i),
                                             b=self.line_name(l - 1, 2 * i + 1),
                                             ruler=RULER, id=first,
                                             end=first + count - 1,
                                             line_a=a["text"], line_b=b["text"])
                    text = self._call_line(cv, task)
            text = text.strip() or fallback
            if not text:
                raise RuntimeError("empty line")
            with self._lock:
                self.pending.discard((l, i))
                if (l, i) in self.nodes:
                    return
                rec = {"l": l, "i": i, "text": text, "size": _b(text)}
                self.nodes[(l, i)] = rec
                self.compactions += 1
                self._append_file("tree.jsonl", rec)
                self._after_build(l, i)
                self.save_meta()
        except Exception as exc:  # noqa: BLE001 — a failed call retries at the next message
            logger.warning("optchat[%s]: build node(%d,%d) failed: %s",
                           self.session_id, l, i, exc)
            with self._lock:
                self.pending.discard((l, i))
                self.failed.add((l, i))

    def _call_line(self, compaction_view: Optional[str], task: str) -> str:
        """One node build: same system prompt as turns, ruler, too-long retries."""
        assert self.llm_fn is not None
        head = []
        if compaction_view is not None:
            head.append({"role": "user",
                         "content": "<chat>\n" + compaction_view + "\n</chat>"})
        messages = head + [{"role": "user", "content": task}]
        best = self.llm_fn(self.system_prompt, messages).strip()
        tries = 1
        while _b(best) > NODE_MAX and tries < MAX_LINE_TRIES:
            cut = best.encode("utf-8", "replace")[:NODE_MAX].decode("utf-8", "ignore")
            corr = TASK_TOO_LONG.format(n=_b(best), cut=cut)
            retry_msgs = messages + [{"role": "assistant", "content": best},
                                     {"role": "user", "content": corr}]
            cand = self.llm_fn(self.system_prompt, retry_msgs).strip()
            if _b(cand) < _b(best):
                best = cand
            tries += 1
        return best

    def _after_build(self, l: int, i: int) -> None:
        if l == 0:
            # view contiguity: append ready level-0 lines
            while (0, self.covered) in self.nodes:
                self.view.append((0, self.covered))
                self.covered += 1
                self.view_version += 1
            self.save_view()
            # this build may have made the next window of nodes eligible
            for j in range(i + 1, min(i + BUILD_WINDOW + 2, self.T)):
                if (0, j) not in self.nodes and self._eligible0(j):
                    self.enqueue(0, j)
        # parent ready once both halves are built
        if (l, i ^ 1) in self.nodes:
            self.enqueue(l + 1, i // 2)
        # a parent may have just become available for a pending view merge
        self.batch()

    # ---------------------------------------------------------------- view
    def due(self, l: int, i: int) -> float:
        return (self.T + 1) / (1 << l) - i

    def view_bytes(self) -> int:
        return sum(self.nodes[x]["size"] for x in self.view if x in self.nodes)

    def batch(self) -> None:
        """Sawtooth: over high water, merge most-due built-parent pairs to low."""
        changed = False
        while self.view_bytes() > self.high:
            best = None  # (-due, first_id, pos)
            for pos in range(len(self.view) - 1):
                (l1, i1), (l2, i2) = self.view[pos], self.view[pos + 1]
                if l1 != l2 or i2 != i1 + 1:
                    continue
                if not (self.built(l1, i1) and self.built(l1, i2)):
                    continue
                if not self.built(l1 + 1, i1 // 2):
                    self.enqueue(l1 + 1, i1 // 2)
                    continue
                d = self.due(l1, i1)
                first = i1 << l1
                key = (-d, first)
                if best is None or key < best[0]:
                    best = (key, pos, l1, i1)
            if best is None:
                break  # builds lagging; merge what we can at each new message
            _, pos, l, i = best
            self.view[pos:pos + 2] = [(l + 1, i // 2)]
            self.view_version += 1
            changed = True
        if changed:
            self.save_view()
            self.save_meta()

    def catch_up(self, budget_nodes: int = 0, deadline: Optional[float] = None,
                 target_backlog: int = 0) -> int:
        """Synchronously build the level-0 frontier from ``covered`` up to ``T`` (and merge
        parents), so the summary view covers the whole transcript. Returns level-0 nodes built.

        This is the drain primitive for both the on-demand catch-up command and the per-turn
        backpressure below. Callers bound it with ``budget_nodes`` / ``deadline`` so a slow or
        temporarily-down aux model can never hang a turn. A failed build stops the drain early
        (the background keeps retrying); the backlog then only closes when the aux is healthy —
        exactly when we want it to. ``target_backlog`` stops early once ``T - covered`` drops to
        it, so a backpressure caller can leave a small margin instead of rebuilding all history.
        """
        if self.read_only or self.llm_fn is None:
            return 0
        with self._lock:
            cap = self.T
        target_covered = max(0, cap - target_backlog) if target_backlog else cap
        built = 0
        while self.covered < target_covered:
            if budget_nodes and built >= budget_nodes:
                break
            if deadline is not None and time.time() > deadline:
                break
            had = self.covered
            self._build(0, self.covered)  # synchronous; advances `covered` on success
            if self.covered == had:
                break  # build failed (aux down); background keeps retrying, don't spin
            built += 1
        self.batch()
        self.save_meta()
        return built

    def backlog(self) -> int:
        """Un-summarized entries: ``T - covered``. The verbatim tail that still rides the request."""
        with self._lock:
            return self.T - self.covered

    def render_view(self) -> str:
        with self._lock:
            if self._render_cache and self._render_cache[0] == self.view_version:
                return self._render_cache[1]
            lines = []
            for (l, i) in self.view:
                node = self.nodes.get((l, i))
                if node is None:
                    break  # the view stops at the first unbuilt line
                lines.append(f"{self.line_name(l, i)}|{node['text'].replace(chr(10), ' ')}")
            text = "\n".join(lines)
            self._render_cache = (self.view_version, text)
            return text

    # ------------------------------------------------------- compaction view
    def compaction_view(self, boundary: int) -> str:
        """The chat's view merged further (<= ~24 KB), ending at `boundary`."""
        with self._lock:
            if self._cv_cache and self._cv_cache[0] == (self.view_version, boundary):
                return self._cv_cache[1]
            spans: List[List[Any]] = []  # [first, count, text]
            total = 0
            for (l, i) in self.view:
                node = self.nodes.get((l, i))
                if node is None:
                    break
                first, count = i << l, 1 << l
                if first >= boundary:
                    break
                if first + count > boundary:
                    continue  # the view ends at the node
                spans.append([first, count, node["text"]])
                total += node["size"]
            # merge down to the target with the same sawtooth order
            while total > COMPACT_HIGH and len(spans) > 1:
                best = None
                for pos in range(len(spans) - 1):
                    f1, c1, _ = spans[pos]
                    f2, c2, _ = spans[pos + 1]
                    if f1 + c1 != f2:
                        continue
                    size = c1 + c2
                    d = (self.T - (f2 + c2 - 1)) / size
                    key = (-d, f1)
                    if best is None or key < best[0]:
                        best = (key, pos)
                if best is None:
                    break
                _, pos = best
                f1, c1, t1 = spans[pos]
                f2, c2, t2 = spans[pos + 1]
                merged = t1 + "\n" + t2
                total += 1 + len(t2.encode("utf-8", "replace"))
                spans[pos:pos + 2] = [[f1, c1 + c2, merged]]
            text = "\n".join(f"{f}+{c}|{t.replace(chr(10), ' ')}" for f, c, t in spans)
            self._cv_cache = ((self.view_version, boundary), text)
            return text

    # ---------------------------------------------------------------- tools
    def zoom(self, id_: int, n: int = 1, page: int = 1) -> dict:
        with self._lock:
            if n == 1:
                if not (0 <= id_ < self.T):
                    return {"error": f"no message {id_} (chat has {self.T})"}
                e = self.log[id_]
                return {"id": id_, "kind": e["kind"], "date": e["date"], "text": e["text"]}
            if n & (n - 1) or id_ % n:
                return {"error": "n must be a power of 2 and id a multiple of n"}
            l = n.bit_length() - 1
            i = id_ // n
            if not self.built(l, i):
                return {"error": f"line {id_}+{n} is not summarized yet",
                        "range": [id_, id_ + n - 1]}
            if l == 1:
                return {"lines": [
                    {"id": 2 * i, "kind": self.log[2 * i]["kind"],
                     "text": self.log[2 * i]["text"]},
                    {"id": 2 * i + 1, "kind": self.log[min(2 * i + 1, self.T - 1)]["kind"],
                     "text": self.log[min(2 * i + 1, self.T - 1)]["text"]}
                ] if 2 * i < self.T else []}
            a, b = self.node(l - 1, 2 * i), self.node(l - 1, 2 * i + 1)
            out = []
            for half, node in ((2 * i, a), (2 * i + 1, b)):
                if node is not None:
                    out.append({"line": self.line_name(l - 1, half), "text": node["text"]})
            return {"lines": out} if out else {"error": f"children of {id_}+{n} not built yet"}

    def date(self, id_: int) -> dict:
        with self._lock:
            if not (0 <= id_ < self.T):
                return {"error": f"no message {id_} (chat has {self.T})"}
            return {"id": id_, "date": self.log[id_]["date"]}

    # ---------------------------------------------------------------- status
    def status(self) -> dict:
        with self._lock:
            hist = {}
            for (l, _) in self.view:
                hist[l] = hist.get(l, 0) + 1
            return {
                "session": self.session_id, "T": self.T, "covered": self.covered,
                "view_lines": len(self.view), "view_bytes": self.view_bytes(),
                "view_levels": hist, "high": self.high, "low": self.low,
                "built_nodes": len(self.nodes), "pending": len(self.pending),
                "failed": len(self.failed), "compactions": self.compactions,
                "read_only": self.read_only, "last_event_ts": self.last_event_ts,
                "pid": os.getpid(),
            }


# Registries are pinned to `sys` so they survive plugin hot-reloads: a purged module
# generation's builder threads keep its Store (and flock fd) alive, and a fresh Store
# for the same chat would latch read-only forever. One Store per session per process.
_STORES: Dict[str, Store] = getattr(sys, "_optchat_stores", None) or {}
sys._optchat_stores = _STORES
_STORES_LOCK = threading.Lock()


def get_store(session_id: str, home: Path, llm_fn: Optional[LLMFn] = None) -> Store:
    with _STORES_LOCK:
        store = _STORES.get(session_id)
        if store is None:
            store = Store(session_id, home, llm_fn)
            _STORES[session_id] = store
        elif llm_fn is not None:
            store.llm_fn = llm_fn  # keep compactions on the newest module's LLM handle
        return store


def drain_backlog(budget_nodes: int = 0, deadline: Optional[float] = None,
                  target_backlog: int = 0) -> Dict[str, int]:
    """Once-off catch-up: synchronously drain the un-summarized frontier (``covered`` → ``T``)
    for every open session, returning ``{session_id: nodes_built}`` (``-1`` = failed).

    Recovers from a missed-compaction backlog (e.g. after the aux model was down and the frontier
    stalled). A stuck frontier breaks ``catch_up`` on its first failed build, so a down aux cannot
    make this spin; ``deadline`` bounds the wall-clock when the aux is healthy. Runs in the gateway
    process (it owns every open store's flock), so a ``/optchat-catchup`` command reaches all chats.
    """
    if deadline is None:
        deadline = time.time() + 180.0
    out: Dict[str, int] = {}
    with _STORES_LOCK:
        items = list(_STORES.items())
    for sid, store in items:
        if store.read_only or store.llm_fn is None:
            continue
        try:
            out[sid] = store.catch_up(budget_nodes=budget_nodes, deadline=deadline,
                                      target_backlog=target_backlog)
        except Exception as exc:  # noqa: BLE001 — a catch-up must never break the caller
            logger.warning("optchat catch-up failed for %s: %s", sid, exc)
            out[sid] = -1
    return out