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