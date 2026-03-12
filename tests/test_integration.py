"""Integration test — full pipeline with real API.

Run with: pytest tests/test_integration.py -m integration
Requires MISTRAL_API_KEY to be set.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skip(reason="Requires real API key and dataset — enable manually")
def test_full_pipeline():
    """Run the full evaluation pipeline end-to-end.

    TODO: Implement once dataset is explored (ML-T004) and pipeline is wired (ML-T007).
    """
    pass
