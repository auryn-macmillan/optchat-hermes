"""OptChat / UniiChat — one chat that never ends, as a Hermes context engine.

Enable: Settings -> Plugins (Desktop) or ``hermes plugins enable optchat``,
then select the engine with ``hermes config set context.engine optchat``.
The engine engages per chat: until a chat has ~16 logged messages the request
passes through untouched; after that the covered history is replaced by the
<chat> view and a cheap model keeps the summary tree built in the background.

Design source: https://gist.github.com/VictorTaelin/91837951a5ce5b38f341ec1ba1df6449
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

AUX_TASK = "optchat_compaction"
SECTION_ID = "optchat-memory"


def _engine_selected() -> bool:
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        return str(((cfg.get("context") or {}).get("engine")) or "").strip() == "optchat"
    except Exception:
        return False


def _cmd_optchat_catchup(raw_args: str = "", session_id: str = "") -> str:
    """Slash handler for ``/optchat-catchup``: drain the un-summarized frontier for every open
    chat. Runs in the gateway process, which owns every store's flock, so it reaches chats other
    than the one it's invoked in. Bounded by a wall-clock deadline; a stuck frontier breaks early."""
    from .core import drain_backlog
    built = drain_backlog(budget_nodes=0, deadline=time.time() + 180.0)
    if not built:
        return "optchat: no open session stores to catch up."
    total = sum(n for n in built.values() if n >= 0)
    failed = [sid for sid, n in built.items() if n < 0]
    lines = [f"  {sid}: built {n} node(s)" for sid, n in built.items()]
    out = f"optchat catch-up: {total} nodes built across {len(built)} chat(s)."
    if failed:
        out += f" Failed: {', '.join(failed)}."
    return out


def _cmd_optchat_status(raw_args: str = "", session_id: str = "") -> str:
    """Slash handler for ``/optchat-status``: show the summary-tree frontier/backlog per open chat."""
    from .core import _STORES, _STORES_LOCK
    with _STORES_LOCK:
        items = list(_STORES.items())
    if not items:
        return "optchat: no open session stores."
    lines = []
    for sid, store in items:
        st = store.status()
        ro = " [read-only]" if store.read_only else ""
        lines.append(f"  {sid}: T={st['T']} covered={st['covered']} "
                     f"backlog={st['T'] - st['covered']} view={st['view_bytes']}B{ro}")
    return "optchat status:\n" + "\n".join(lines)


def register(ctx) -> None:
    from .engine import OptChatEngine
    from .prompt import PROTOCOL

    try:
        ctx.register_auxiliary_task(
            AUX_TASK,
            display_name="OptChat compaction",
            description="Background 512-byte line summaries for the OptChat memory tree.",
            inherit_from="compression",
        )
    except Exception as exc:  # already registered (hot reload / re-discovery)
        logger.debug("optchat: aux task already present (%s)", exc)

    # The protocol joins the system prompt only while this engine is selected,
    # so turns and compactions share one system prompt (cache invariant).
    try:
        ctx.register_system_prompt_section(
            SECTION_ID,
            lambda _info: PROTOCOL if _engine_selected() else "",
            max_chars=6000,
        )
    except Exception as exc:
        logger.debug("optchat: system prompt section already present (%s)", exc)

    ctx.register_context_engine(OptChatEngine(llm=ctx.llm))
    logger.info("optchat: context engine registered (compaction task '%s')", AUX_TASK)

    # On-demand catch-up + status slash commands. register_command is forwarded to the global
    # plugin command registry; guard so a re-registration (hot reload) never aborts register().
    try:
        ctx.register_command("/optchat-catchup", _cmd_optchat_catchup,
                             description="Drain missed optchat compaction for every open chat",
                             args_hint="")
        ctx.register_command("/optchat-status", _cmd_optchat_status,
                             description="Show optchat summary-tree frontier/backlog per open chat",
                             args_hint="")
    except Exception as exc:
        logger.debug("optchat: commands already present (%s)", exc)