"""OptChat dashboard API — read-only status for the memory engine.

Mounted at /api/plugins/optchat/. Runs in the dashboard/web process, which is
NOT the gateway process that owns the stores: everything here just reads the
durable files the engine writes under ``<HERMES_HOME>/optchat/<session>/``
plus ``config.yaml`` for the enablement truth. No agent imports.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


def _home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


def _root() -> Path:
    return _home() / "optchat"


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _session_status(session: str) -> Optional[dict]:
    d = _root() / session
    meta = _read_json(d / "meta.json")
    if meta is None:
        return None
    meta["session"] = session
    age = meta.get("last_event_ts") or 0
    meta["age_seconds"] = round(max(0.0, time.time() - age), 1)
    return meta


def _all_sessions() -> list:
    root = _root()
    if not root.is_dir():
        return []
    out = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            st = _session_status(child.name)
            if st:
                out.append(st)
    out.sort(key=lambda s: s.get("last_event_ts") or 0, reverse=True)
    return out


def _engine_config() -> Dict[str, Any]:
    truth: Dict[str, Any] = {"engine_selected": False, "plugin_enabled": False,
                             "plugins_enabled": [], "context_engine": None}
    try:
        import yaml
        cfg = yaml.safe_load((_home() / "config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        return truth
    truth["context_engine"] = ((cfg.get("context") or {}).get("engine"))
    truth["engine_selected"] = truth["context_engine"] == "optchat"
    enabled = ((cfg.get("plugins") or {}).get("enabled")) or []
    truth["plugins_enabled"] = [str(x) for x in enabled]
    truth["plugin_enabled"] = "optchat" in truth["plugins_enabled"]
    return truth


@router.get("/engine")
def engine() -> Dict[str, Any]:
    return _engine_config()


@router.get("/sessions")
def sessions(limit: int = Query(25, ge=1, le=200)) -> Dict[str, Any]:
    return {"sessions": _all_sessions()[:limit], "active": _engine_config()}


@router.get("/status")
def status(session: Optional[str] = Query(None)) -> Dict[str, Any]:
    st = _session_status(session) if session else None
    if st is None:
        live = _all_sessions()
        if not live:
            raise HTTPException(status_code=404, detail="no optchat sessions yet")
        st = live[0]
    st["engine"] = _engine_config()
    return st


@router.get("/view")
def view(session: Optional[str] = Query(None), tail: int = Query(120, ge=1, le=2000)) -> Dict[str, Any]:
    session = session or (_all_sessions() or [{}])[0].get("session")
    if not session:
        raise HTTPException(status_code=404, detail="no optchat sessions yet")
    d = _root() / session
    data = _read_json(d / "view.json") or {}
    lines = data.get("lines") or []
    sizes: Dict[str, int] = {}
    try:
        with (d / "tree.jsonl").open("r", encoding="utf-8") as fh:
            for raw in fh:
                try:
                    rec = json.loads(raw)
                    sizes[f"{rec['l']}:{rec['i']}"] = rec["size"]
                except (json.JSONDecodeError, KeyError):
                    continue
    except OSError:
        pass
    out = []
    for l, i in lines[-tail:]:
        out.append({"l": l, "i": i, "name": f"{int(i) << int(l)}+{1 << int(l)}"})
    return {"session": session, "covered": data.get("covered", 0),
            "lines": out, "sizes": sizes}


@router.get("/line")
def line(session: Optional[str] = Query(None), l: int = Query(..., ge=0),
         i: int = Query(..., ge=0)) -> Dict[str, Any]:
    session = session or (_all_sessions() or [{}])[0].get("session")
    if not session:
        raise HTTPException(status_code=404, detail="no optchat sessions yet")
    try:
        with (_root() / session / "tree.jsonl").open("r", encoding="utf-8") as fh:
            for raw in fh:
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if rec.get("l") == l and rec.get("i") == i:
                    return rec
    except OSError:
        pass
    raise HTTPException(status_code=404, detail=f"node({l},{i}) not built yet")