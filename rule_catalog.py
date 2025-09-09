from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - optional in non-Streamlit contexts
    st = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - dependency may be missing
    yaml = None  # type: ignore


def _repo_root() -> Path:
    # Assume this file lives at repo root
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_rule_catalog(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Read validation_rules.yaml and return a normalized catalog dict.

    Returns a dict with keys:
      - "by_code": map of {code -> full rule dict}
      - "list": list of rule dicts in file order
      - "parameters": dict of global parameters from YAML if present
      - "version": string version if present
    """
    catalog: Dict[str, Any] = {
        "by_code": {},
        "list": [],
        "parameters": {},
        "version": "",
    }

    try:
        if yaml is None:
            if st:
                st.info("PyYAML is not installed. Install requirements to enable YAML-driven rules.")
            return catalog

        yaml_path = Path(path) if path is not None else (_repo_root() / "validation_rules.yaml")
        if not yaml_path.exists():
            if st:
                st.info(f"validation_rules.yaml not found at {yaml_path}. The app will continue with default messaging.")
            return catalog

        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            if st:
                st.warning("validation_rules.yaml is not a dict. Unable to load rule catalog.")
            return catalog

        rules = data.get("rules") or []
        if not isinstance(rules, list):
            if st:
                st.warning("validation_rules.yaml missing 'rules' list. No rules to display.")
            rules = []

        version = data.get("version") or ""
        parameters = data.get("parameters") or {}

        by_code: Dict[str, Any] = {}
        ordered_list = []
        for r in rules:
            if not isinstance(r, dict):
                continue
            code = str(r.get("code") or "").strip()
            if not code:
                continue
            ordered_list.append(r)
            by_code[code] = r

        catalog["by_code"] = by_code
        catalog["list"] = ordered_list
        catalog["parameters"] = parameters if isinstance(parameters, dict) else {}
        catalog["version"] = str(version) if version is not None else ""

        return catalog
    except Exception as e:  # graceful failure
        if st:
            st.warning(f"Failed to read validation_rules.yaml: {e}")
        return catalog


def rule_meta(code: str) -> Dict[str, Any]:
    """Return the rule dict for a given rule code or {} if not found."""
    try:
        code_norm = str(code or "").strip()
        if not code_norm:
            return {}
        return load_rule_catalog(_repo_root() / "validation_rules.yaml")["by_code"].get(code_norm, {})
    except Exception:
        return {}


def severity_badge_text(sev: Optional[str]) -> str:
    """
    Return a compact label string for tables by severity level.
    Critical -> "🔴 Critical"
    High -> "🟠 High"
    Medium -> "🟡 Medium"
    Low -> "🟢 Low"
    Fallback -> "⚪ Unknown"
    """
    if not sev:
        return "⚪ Unknown"
    s = str(sev).strip().lower()
    if s == "critical":
        return "🔴 Critical"
    if s == "high":
        return "🟠 High"
    if s == "medium":
        return "🟡 Medium"
    if s == "low":
        return "🟢 Low"
    return "⚪ Unknown"


