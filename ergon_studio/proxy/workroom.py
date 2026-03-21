from __future__ import annotations

AD_HOC_WORKROOM_ID = "ad-hoc-workroom"


def is_ad_hoc_workroom(workroom_id: str | None) -> bool:
    return workroom_id == AD_HOC_WORKROOM_ID
