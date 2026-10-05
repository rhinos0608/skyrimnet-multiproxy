"""Focused mapping tests: Vertex drift fixes (no catalog/dashboard edits)."""

from proxy_internal.providers.vertex import _VERTEX_MODEL_ALIASES, _resolve_vertex_model


def test_vertex_retired_aliases_removed():
    for retired in (
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemma-3-27b",
    ):
        assert retired not in _VERTEX_MODEL_ALIASES, retired
    # Unknown names pass through as-is
    assert _resolve_vertex_model("vertex:gemini-2.5-pro") == "gemini-2.5-pro"


def test_vertex_ga_models_present():
    for m in (
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ):
        assert m in _VERTEX_MODEL_ALIASES, m
        assert _resolve_vertex_model(f"vertex:{m}") == _VERTEX_MODEL_ALIASES[m]
