import json
import os

import requests
import streamlit as st

ACCOUNT_FILE = "account.json"
GIST_FILENAME = "tteolsaopal_account.json"
GIST_HEADERS = {"Accept": "application/vnd.github+json"}

DEFAULT_ACCOUNT: dict = {
    "seed": 10000,
    "strategy": "Pro3",
    "cycle_number": 1,
    "cycle_start_date": "",
    "initial_asset": 10000.0,
    "holdings": [],
    "profit_log": [],
    "strategy_mismatch_since": None,
    "filter_streak": 0,
    "strategy_switch_date": None,
}


def _inject_defaults(acct: dict) -> dict:
    strategy = acct.get("strategy", DEFAULT_ACCOUNT["strategy"])
    for h in acct.get("holdings", []):
        h.setdefault("slot_type", "normal")
        h.setdefault("strategy_at_entry", strategy)
    acct.setdefault("strategy_mismatch_since", None)
    acct.setdefault("filter_streak", 0)
    acct.setdefault("strategy_switch_date", None)
    return acct


def _creds() -> tuple[str, str]:
    return st.secrets.get("GITHUB_PAT", ""), st.secrets.get("GIST_ID", "")


def load_account() -> dict:
    pat, gist_id = _creds()
    if pat and gist_id:
        try:
            resp = requests.get(
                f"https://api.github.com/gists/{gist_id}",
                headers={**GIST_HEADERS, "Authorization": f"token {pat}"},
                timeout=5,
            )
            resp.raise_for_status()
            acct = json.loads(resp.json()["files"][GIST_FILENAME]["content"])
            return _inject_defaults(acct)
        except Exception as e:
            st.warning(f"Gist 로드 실패, 기본값으로 대체합니다: {e}")
            return dict(DEFAULT_ACCOUNT)

    if os.path.exists(ACCOUNT_FILE):
        try:
            with open(ACCOUNT_FILE, encoding="utf-8") as f:
                return _inject_defaults(json.load(f))
        except Exception as e:
            st.warning(f"account.json 로드 실패, 기본값으로 대체합니다: {e}")
            return dict(DEFAULT_ACCOUNT)

    return dict(DEFAULT_ACCOUNT)


def save_account(acct: dict) -> None:
    pat, gist_id = _creds()
    if pat and gist_id:
        try:
            resp = requests.patch(
                f"https://api.github.com/gists/{gist_id}",
                headers={**GIST_HEADERS, "Authorization": f"token {pat}"},
                json={"files": {GIST_FILENAME: {"content": json.dumps(acct, ensure_ascii=False, indent=2)}}},
                timeout=5,
            )
            resp.raise_for_status()
        except Exception as e:
            st.error(f"Gist 저장 실패: {e}")
        return

    try:
        with open(ACCOUNT_FILE, "w", encoding="utf-8") as f:
            json.dump(acct, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"account.json 저장 실패: {e}")
