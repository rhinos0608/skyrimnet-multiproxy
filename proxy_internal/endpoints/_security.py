"""Shared request validation helpers for sensitive endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request


def require_json_object(data: Any) -> dict:
    """Ensure a parsed JSON body is an object."""
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    return data


async def read_json_object(request: Request) -> dict:
    """Parse a request body as JSON and ensure it is an object."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    return require_json_object(data)


def require_dashboard_csrf(request: Request) -> None:
    """Require the dashboard CSRF cookie for state-changing config requests."""
    import proxy

    token = request.cookies.get(proxy.ADMIN_CSRF_COOKIE)
    if token != proxy.ADMIN_CSRF_TOKEN:
        raise HTTPException(status_code=403, detail="Dashboard session required")


def require_string_field(data: dict, field: str, *, allow_empty: bool = False) -> str:
    """Read a required string field from a JSON object."""
    value = data.get(field)
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail=f"{field} must be a string")
    value = value.strip()
    if not value and not allow_empty:
        raise HTTPException(status_code=400, detail=f"{field} required")
    return value


def optional_string_field(data: dict, field: str) -> str:
    """Read an optional string field from a JSON object."""
    value = data.get(field)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail=f"{field} must be a string")
    return value.strip()


def require_bool_field(data: dict, field: str) -> bool:
    """Read a required boolean field from a JSON object."""
    value = data.get(field)
    if not isinstance(value, bool):
        raise HTTPException(status_code=400, detail=f"{field} must be a boolean")
    return value


def require_int_field(data: dict, field: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """Read and validate an integer field from a JSON object."""
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise HTTPException(status_code=400, detail=f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise HTTPException(status_code=400, detail=f"{field} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise HTTPException(status_code=400, detail=f"{field} must be at most {maximum}")
    return value


def require_float_field(data: dict, field: str) -> float:
    """Read and validate a numeric field from a JSON object."""
    value = data.get(field)
    if isinstance(value, bool):
        raise HTTPException(status_code=400, detail=f"{field} must be a number")
    if isinstance(value, (int, float)):
        return float(value)
    raise HTTPException(status_code=400, detail=f"{field} must be a number")
