from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DesignTokens:
    theme: str
    colors: Mapping[str, str]
    typography: Mapping[str, str]
    radius: Mapping[str, str]
    border: Mapping[str, str]
    shadows: Mapping[str, str]
    spacing: Mapping[str, str]
    motion: Mapping[str, str]
    blur: Mapping[str, str]


def get_design_tokens(theme: str = "dark") -> DesignTokens:
    normalized_theme = "light" if str(theme).strip().lower() == "light" else "dark"
    accent = "#7C3AED"

    if normalized_theme == "dark":
        colors = {
            "ink": "#0B0D12",
            "paper": "#F7F8FA",
            "accent": accent,
            "danger": "#FF3B30",
            "success": "#34C759",
            "glass": "rgba(255,255,255,0.08)",
            "glass_strong": "rgba(255,255,255,0.14)",
            "stroke": "rgba(255,255,255,0.18)",
            "text_primary": "#F7F8FA",
            "text_secondary": "#D6DAE2",
            "text_muted": "#B3BBCA",
            "surface": "rgba(12,14,20,0.62)",
            "surface_alt": "rgba(22,24,33,0.7)",
            "outline_focus": "rgba(124,58,237,0.85)",
        }
        shadows = {
            "brutal": "8px 8px 0 rgba(0,0,0,0.45)",
            "brutal_hover": "10px 10px 0 rgba(0,0,0,0.52)",
            "button": "6px 6px 0 rgba(0,0,0,0.52)",
            "button_hover": "7px 7px 0 rgba(0,0,0,0.58)",
            "soft": "0 18px 45px rgba(8, 10, 16, 0.38)",
            "inner": "inset 0 1px 0 rgba(255,255,255,0.12)",
        }
    else:
        colors = {
            "ink": "#0B0D12",
            "paper": "#F7F8FA",
            "accent": accent,
            "danger": "#FF3B30",
            "success": "#34C759",
            "glass": "rgba(255,255,255,0.72)",
            "glass_strong": "rgba(255,255,255,0.88)",
            "stroke": "rgba(0,0,0,0.10)",
            "text_primary": "#0B0D12",
            "text_secondary": "#202633",
            "text_muted": "#4A5365",
            "surface": "rgba(255,255,255,0.78)",
            "surface_alt": "rgba(245,247,251,0.88)",
            "outline_focus": "rgba(124,58,237,0.76)",
        }
        shadows = {
            "brutal": "8px 8px 0 rgba(0,0,0,0.65)",
            "brutal_hover": "10px 10px 0 rgba(0,0,0,0.72)",
            "button": "6px 6px 0 rgba(0,0,0,0.58)",
            "button_hover": "7px 7px 0 rgba(0,0,0,0.66)",
            "soft": "0 16px 40px rgba(15, 23, 42, 0.18)",
            "inner": "inset 0 1px 0 rgba(255,255,255,0.56)",
        }

    typography = {
        "heading": '"Inter", system-ui, -apple-system, "Segoe UI", sans-serif',
        "body": '"Inter", system-ui, -apple-system, "Segoe UI", sans-serif',
        "mono": 'ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace',
    }
    radius = {
        "card": "18px",
        "input": "14px",
        "button": "12px",
        "sm": "10px",
        "pill": "999px",
    }
    border = {"regular": "2px", "strong": "3px"}
    spacing = {
        "xxs": "0.25rem",
        "xs": "0.5rem",
        "sm": "0.75rem",
        "md": "1rem",
        "lg": "1.5rem",
        "xl": "2rem",
    }
    motion = {
        "fast": "110ms",
        "normal": "140ms",
        "slow": "220ms",
        "ease_out": "cubic-bezier(0.2, 0.85, 0.32, 1)",
    }
    blur = {"glass": "16px"}

    return DesignTokens(
        theme=normalized_theme,
        colors=colors,
        typography=typography,
        radius=radius,
        border=border,
        shadows=shadows,
        spacing=spacing,
        motion=motion,
        blur=blur,
    )


__all__ = ["DesignTokens", "get_design_tokens"]
