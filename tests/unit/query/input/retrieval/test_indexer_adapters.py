# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""Tests for read_indexer_reports community-level roll-up."""

import pandas as pd
import pytest

from graphrag.query.indexer_adapters import read_indexer_reports


def _make_communities(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal final_communities DataFrame."""
    return pd.DataFrame(rows)


def _make_reports(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal final_community_reports DataFrame."""
    return pd.DataFrame(rows)


def test_rollup_selects_max_community_level_per_entity():
    """Each entity should map to its highest community level only.

    Entity 'e1' belongs to community 0 (level 0) and community 2 (level 1).
    Entity 'e2' belongs to community 1 (level 0) only.
    Without the fix (groupby title), all three community IDs 0, 1, 2 survive.
    With the fix (groupby entity_ids), only community 2 and 1 survive because
    e1's max community is 2 and e2's max community is 1.
    """
    communities = _make_communities(
        [
            # level-0 community for e1
            {"community": 0, "level": 0, "title": "Community 0", "entity_ids": ["e1"]},
            # level-0 community for e2
            {"community": 1, "level": 0, "title": "Community 1", "entity_ids": ["e2"]},
            # level-1 (higher) community that also contains e1
            {"community": 2, "level": 1, "title": "Community 2", "entity_ids": ["e1"]},
        ]
    )

    reports = _make_reports(
        [
            {
                "id": "r0",
                "community": 0,
                "level": 0,
                "title": "Report 0",
                "summary": "",
                "full_content": "",
                "rank": 1.0,
                "rank_explanation": "",
                "findings": [],
            },
            {
                "id": "r1",
                "community": 1,
                "level": 0,
                "title": "Report 1",
                "summary": "",
                "full_content": "",
                "rank": 1.0,
                "rank_explanation": "",
                "findings": [],
            },
            {
                "id": "r2",
                "community": 2,
                "level": 1,
                "title": "Report 2",
                "summary": "",
                "full_content": "",
                "rank": 1.0,
                "rank_explanation": "",
                "findings": [],
            },
        ]
    )

    result = read_indexer_reports(
        final_community_reports=reports,
        final_communities=communities,
        community_level=None,
        dynamic_community_selection=False,
    )

    result_ids = {r.id for r in result}
    # community 0 must be excluded: e1 has a higher-level community (2)
    assert "r0" not in result_ids, (
        "Report for community 0 should be excluded because e1 has a higher community (2)"
    )
    assert "r1" in result_ids, "Report for community 1 should be included (e2's only community)"
    assert "r2" in result_ids, "Report for community 2 should be included (e1's max-level community)"


def test_rollup_no_duplicates_when_entity_in_one_community():
    """Single-level graph: each entity in exactly one community. All reports survive."""
    communities = _make_communities(
        [
            {"community": 0, "level": 0, "title": "Community 0", "entity_ids": ["e1", "e2"]},
            {"community": 1, "level": 0, "title": "Community 1", "entity_ids": ["e3"]},
        ]
    )

    reports = _make_reports(
        [
            {
                "id": "r0",
                "community": 0,
                "level": 0,
                "title": "Report 0",
                "summary": "",
                "full_content": "",
                "rank": 1.0,
                "rank_explanation": "",
                "findings": [],
            },
            {
                "id": "r1",
                "community": 1,
                "level": 0,
                "title": "Report 1",
                "summary": "",
                "full_content": "",
                "rank": 1.0,
                "rank_explanation": "",
                "findings": [],
            },
        ]
    )

    result = read_indexer_reports(
        final_community_reports=reports,
        final_communities=communities,
        community_level=None,
        dynamic_community_selection=False,
    )

    result_ids = {r.id for r in result}
    assert result_ids == {"r0", "r1"}
