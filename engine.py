"""OptChatEngine: the Hermes ContextEngine that runs the UniiChat design.

Adaptation notes (Hermes vs. the standalone UniiChat app):
- The host persists the full transcript; we never rewrite it except through
  ``compress()`` (manual /compress or the token-threshold fallback). The
  per-request swap happens in ``select_context``: covered history is replaced
  by the <chat> view, and everything not yet covered by the view (the recent
  tail, the incoming message, ephemeral blocks) rides along verbatim. This
  keeps tool-call pairing valid and the prompt-cache prefix stable.
- The protocol lives in a plugin system-prompt section (rendered only while
  this engine is selected), so compaction calls reuse the exact captured
  system prompt — one system prompt for turns and compactions, as the spec
  requires.
- The compaction model is the plugin aux task ``optchat_compaction``, which
  inherits the user's ``compression`` slot by default.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

from .core import HIGH_DEFAULT, LOW_DEFAULT, Store, get_store
from .prompt import DATE_DESC, ZOOM_DESC

logger = logging.getLogger(__name__)

ENGAGE_MIN = 16          # don't take over a chat until it has this many entries
ENGAGE_REPLACE_MIN = 8   # and only replace when this many messages actually go away
VIEW_UID = "optchat:view"
CARRIER_PREFIX = "optchat:carrier"


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


class OptChatEngine(ContextEngine):
    """One shared instance per plugin load; per-agent clones share the stores."""

    def __init__(self, llm=None, aux_task: str = "optchat_compaction"):
        self._llm = llm
        self._aux_task = aux_task
        self.session_id = "default"
        # The view owns the boundary; the host's head/tail protection would
        # fight it. (Only the built-in compressor's preflight reads these.)
        self.protect_first_n = 0
        self.protect_last_n = 0
        # Routine background maintenance stays silent; warnings still surface.
        self.emit_automatic_compaction_status = False
        # Attributes the core reads raw off the compressor but only the
        # built-in ContextCompressor defines (turn_context_compaction.py
        # reads last_real_prompt_tokens / summary_target_ratio without
        # getattr) — a custom engine missing them crashes the turn.
        self.last_real_prompt_tokens = 0
        self.summary_target_ratio = 0.20

    @property
    def name(self) -> str:  # noqa: D401
        return "optchat"

    # ---------------------------------------------------------------- store
    def _llm_fn(self, system: Optional[str], messages: List[dict]) -> str:
        if self._llm is None:
            raise RuntimeError("optchat: no plugin LLM handle")
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        # Compaction prompts run 6-18k tokens against a local model that also
        # serves the main chat; the 30s aux default times them out routinely.
        result = self._llm.complete(task=self._aux_task, messages=msgs,
                                    purpose="optchat-compaction")
        return getattr(result, "text", "") or ""

    def _store(self) -> Store:
        return get_store(self.session_id, _hermes_home(), self._llm_fn)

    # ---------------------------------------------------------------- lifecycle
    def on_session_start(self, session_id: str, **kwargs) -> None:
        self.session_id = session_id or "default"
        store = self._store()
        store.high = min(store.high, HIGH_DEFAULT)
        logger.info("optchat: session %s — T=%d covered=%d view=%dB%s",
                    self.session_id, store.T, store.covered, store.view_bytes(),
                    " (read-only)" if store.read_only else "")

    def on_session_end(self, session_id: str, messages: List[dict]) -> None:
        try:
            store = self._store()
            store.ingest(messages or [])
            store.save_meta()
        except Exception:
            logger.debug("optchat: session end ingest failed", exc_info=True)

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
        self.last_real_prompt_tokens = self.last_prompt_tokens
        self.last_completion_tokens = int(usage.get("completion_tokens") or 0)
        self.last_total_tokens = int(usage.get("total_tokens") or 0)

    def update_model(self, model: str, context_length: int, base_url: str = "",
                     api_key: str = "", provider: str = "", api_mode: str = "") -> None:
        super().update_model(model, context_length, base_url=base_url, api_key=api_key,
                             provider=provider, api_mode=api_mode)
        if context_length:
            # Keep the view under ~35% of the window (bytes at ~3.5 chars/token).
            high = max(16_000, min(HIGH_DEFAULT, int(context_length * 1.2)))
            store = self._store()
            store.high, store.low = high, high // 2
            store.save_meta()

    # ---------------------------------------------------------------- decisions
    def should_compress(self, prompt_tokens: Optional[int] = None) -> bool:
        pt = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if self.threshold_tokens and pt > self.threshold_tokens:
            return True
        store = self._store()
        return store.view_bytes() > store.high  # batch synchronously on demand

    def should_compress_preflight(self, messages: List[dict]) -> bool:
        return False  # rough estimates count the full history; select_context bounds the real request

    def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
        return True

    def has_content_to_compress(self, messages: List[dict]) -> bool:
        return self._store().T >= ENGAGE_MIN

    # ---------------------------------------------------------------- request swap
    def select_context(self, request_messages, *, conversation_messages=None,
                       incoming_message=None, budget_tokens: int = 0):
        store = self._store()
        logger.debug("optchat: select_context (T=%d covered=%d ro=%s req=%d)",
                     store.T, store.covered, store.read_only, len(request_messages or ()))
        if self._llm is None:
            return None
        try:
            store.ingest(conversation_messages or [])  # also retries a lost lease
            if store.read_only:
                return None
            if store.system_prompt is None and request_messages \
                    and request_messages[0].get("role") == "system":
                sp = request_messages[0].get("content")
                if isinstance(sp, str) and sp:
                    store.system_prompt = sp
                    store.save_system_prompt()
            if store.T < ENGAGE_MIN or store.covered < 8:
                return None
            # request_messages arrive STRIPPED of message_uid (persistence-only
            # field), so the boundary must be found on conversation_messages
            # (which keep uids) and mapped onto the request positionally.
            conv = list(conversation_messages or [])
            if conv:
                _, split_c, replaced = self._boundary(conv, store)
                lead = 0
                while lead < len(request_messages) \
                        and (request_messages[lead] or {}).get("role") == "system":
                    lead += 1
                # The request is the stripped copy of the history, possibly
                # head-compacted: assume head-drop, then VERIFY by content so a
                # misaligned map can never swap uncovered messages. Fail-open.
                drop = max(0, len(conv) - len(request_messages))
                # request_messages = build_api_messages(conv) prepends the system prompt
                # (and any prefill), so it is conv shifted forward by exactly the front
                # offset: len(request) - len(conv). Map the store boundary (an index into
                # conv) onto the request by that offset; the old `split = split_c - drop`
                # only handled request SHORTER than conv (head-drop) and clamped the
                # leading system prompt away, so a system-prompted request missed every
                # swap (drop=0 -> split=split_c -> compared conv[k] vs request[k]=conv[k-1]).
                split = split_c + (len(request_messages) - len(conv))
                if split_c >= len(conv) and split >= len(request_messages):
                    split = len(request_messages)
                elif split_c >= len(conv) or not (lead < split < len(request_messages)) \
                        or (conv[split_c] or {}).get("role") != (request_messages[split] or {}).get("role") \
                        or str((conv[split_c] or {}).get("content") or "")[:80] \
                           != str((request_messages[split] or {}).get("content") or "")[:80]:
                    logger.info("optchat: no-swap align-miss (split_c=%s/%d split=%s/%d drop=%d "
                                "lead=%d replaced=%d)", split_c, len(conv), split,
                                len(request_messages), drop, lead, replaced)
                    return None
            else:
                lead, split, replaced = self._boundary(list(request_messages), store)
            if split is None or replaced < ENGAGE_REPLACE_MIN:
                if split is not None:
                    logger.info("optchat: no-swap replaced=%d (split=%s)", replaced, split)
                return None
            view_text = store.render_view()
            if not view_text:
                return None
            view_msg = {"role": "user", "content": "<chat>\n" + view_text + "\n</chat>",
                        "message_uid": f"{VIEW_UID}:{store.session_id}"}
            tail = [m for m in request_messages[split:]
                    if not str((m or {}).get("message_uid") or "").startswith("optchat:")]
            logger.info("optchat: view swap — %d request msgs -> %d (view %dB, %d lines, covered %d/%d)",
                        len(request_messages), 1 + lead + len(tail), store.view_bytes(),
                        len(store.view), store.covered, store.T)
            return list(request_messages[:lead]) + [view_msg] + tail
        except Exception:
            logger.exception("optchat: select_context failed; leaving request unchanged")
            return None

    @staticmethod
    def _boundary(request_messages: List[dict], store: Store):
        """(lead_system, split_index, covered_messages_replaced)."""
        lead = 0
        while lead < len(request_messages) and request_messages[lead].get("role") == "system":
            lead += 1
        replaced = 0
        for idx in range(lead, len(request_messages)):
            uid = str((request_messages[idx] or {}).get("message_uid") or "")
            if uid.startswith("optchat:"):
                continue
            first = store.uid_index.get(uid)
            if first is not None and first < store.covered:
                replaced += 1
                continue
            return lead, idx, replaced
        return lead, len(request_messages), replaced

    # ---------------------------------------------------------------- compress
    def compress(self, messages, current_tokens=None, focus_topic=None,
                 force=False, memory_context="") -> List[dict]:
        store = self._store()
        if store.read_only:
            return list(messages)
        try:
            store.ingest(messages or [])
            store.batch()
            lead, split, replaced = self._boundary(list(messages), store)
            if split is None or replaced < ENGAGE_REPLACE_MIN:
                return list(messages)
            view_text = store.render_view()
            carrier = {"role": "user",
                       "content": "<optchat-view>\n" + view_text + "\n</optchat-view>",
                       "message_uid": f"{CARRIER_PREFIX}:{store.session_id}:{int(time.time())}"}
            tail = [m for m in messages[split:]
                    if not str((m or {}).get("message_uid") or "").startswith("optchat:")]
            self.compression_count += 1
            out = list(messages[:lead]) + [carrier] + tail
            logger.info("optchat: /compress boundary — %d messages -> view(%dB)+%d tail",
                        len(messages), store.view_bytes(), len(tail))
            return out
        except Exception:
            logger.exception("optchat: compress failed; leaving messages unchanged")
            return list(messages)

    def prune_tool_results_only(self, messages, current_tokens=None):
        return messages, 0  # the view already drops old tool payloads from the request

    # ---------------------------------------------------------------- tools
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {"name": "zoom", "description": ZOOM_DESC,
             "parameters": {"type": "object", "properties": {
                 "id": {"type": "integer", "description": "First message id of the view line"},
                 "n": {"type": "integer", "description": "Messages the line covers (power of 2); 1 = the message whole"},
                 "page": {"type": "integer", "description": "Page for long messages (1-based)"}},
                 "required": ["id"]}},
            {"name": "date", "description": DATE_DESC,
             "parameters": {"type": "object", "properties": {
                 "id": {"type": "integer", "description": "Message id"}},
                 "required": ["id"]}},
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        store = self._store()
        try:
            if name == "zoom":
                return json.dumps(store.zoom(int(args.get("id", 0)),
                                             int(args.get("n") or 1),
                                             int(args.get("page") or 1)),
                                  ensure_ascii=False)
            if name == "date":
                return json.dumps(store.date(int(args.get("id", 0))), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    # ---------------------------------------------------------------- misc
    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        try:
            status["optchat"] = self._store().status()
        except Exception:
            status["optchat"] = {"error": "store unavailable"}
        return status

    def clone_for_agent(self) -> "OptChatEngine":
        # Stores are process-global per session; clones only need their own counters.
        clone = OptChatEngine(self._llm, self._aux_task)
        clone.session_id = self.session_id
        return clone