from __future__ import annotations

from textwrap import dedent

import streamlit as st

from ui.tokens import get_design_tokens


def _build_global_css(theme: str) -> str:
    tokens = get_design_tokens(theme)
    c = tokens.colors
    t = tokens.typography
    r = tokens.radius
    b = tokens.border
    s = tokens.shadows
    sp = tokens.spacing
    m = tokens.motion
    blur = tokens.blur
    is_dark = tokens.theme == "dark"

    if is_dark:
        bg_base = c["ink"]
        bg_layer_a = "rgba(124,58,237,0.25)"
        bg_layer_b = "rgba(124,58,237,0.14)"
        noise_opacity = "0.08"
        accent_deep = "#5B21B6"
        hero_secondary = "rgba(110,63,227,0.95)"
        hero_tertiary = "rgba(56,25,125,0.88)"
    else:
        bg_base = c["paper"]
        bg_layer_a = "rgba(124,58,237,0.18)"
        bg_layer_b = "rgba(124,58,237,0.10)"
        noise_opacity = "0.035"
        accent_deep = "#6D28D9"
        hero_secondary = "rgba(124,58,237,0.92)"
        hero_tertiary = "rgba(91,33,182,0.86)"

    return dedent(
        f"""
        <style>
        :root {{
            --ui-ink: {c["ink"]};
            --ui-paper: {c["paper"]};
            --ui-accent: {c["accent"]};
            --ui-accent-deep: {accent_deep};
            --ui-danger: {c["danger"]};
            --ui-success: {c["success"]};
            --ui-glass: {c["glass"]};
            --ui-glass-strong: {c["glass_strong"]};
            --ui-stroke: {c["stroke"]};
            --ui-surface: {c["surface"]};
            --ui-surface-alt: {c["surface_alt"]};
            --ui-text-primary: {c["text_primary"]};
            --ui-text-secondary: {c["text_secondary"]};
            --ui-text-muted: {c["text_muted"]};
            --ui-focus: {c["outline_focus"]};

            --font-heading: {t["heading"]};
            --font-body: {t["body"]};
            --font-mono: {t["mono"]};

            --radius-card: {r["card"]};
            --radius-input: {r["input"]};
            --radius-button: {r["button"]};
            --radius-sm: {r["sm"]};
            --radius-pill: {r["pill"]};

            --border-regular: {b["regular"]};
            --border-strong: {b["strong"]};

            --shadow-brutal: {s["brutal"]};
            --shadow-brutal-hover: {s["brutal_hover"]};
            --shadow-button: {s["button"]};
            --shadow-button-hover: {s["button_hover"]};
            --shadow-soft: {s["soft"]};
            --shadow-inner: {s["inner"]};

            --space-xxs: {sp["xxs"]};
            --space-xs: {sp["xs"]};
            --space-sm: {sp["sm"]};
            --space-md: {sp["md"]};
            --space-lg: {sp["lg"]};
            --space-xl: {sp["xl"]};

            --motion-fast: {m["fast"]};
            --motion-normal: {m["normal"]};
            --motion-slow: {m["slow"]};
            --motion-ease: {m["ease_out"]};
            --blur-glass: {blur["glass"]};
        }}

        html, body, .stApp {{
            font-family: var(--font-body);
            color: var(--ui-text-primary);
            -webkit-font-smoothing: antialiased;
            text-rendering: optimizeLegibility;
        }}

        h1, h2, h3, h4, h5, h6 {{
            font-family: var(--font-heading);
            color: var(--ui-text-primary);
            font-weight: 800;
            letter-spacing: -0.01em;
        }}

        p, li, label, span {{
            color: inherit;
        }}

        code, pre {{
            font-family: var(--font-mono);
        }}

        hr {{
            border: 0;
            border-top: var(--border-regular) solid var(--ui-stroke);
            margin: var(--space-lg) 0;
        }}

        a {{
            color: var(--ui-accent);
            text-decoration-thickness: 2px;
            text-underline-offset: 2px;
        }}

        /* Streamlit layout selectors changed between versions. Keep stable testid + fallback. */
        [data-testid="stAppViewContainer"],
        section.main {{
            min-height: 100vh;
            background:
                radial-gradient(1200px circle at 15% 10%, {bg_layer_a}, transparent 55%),
                radial-gradient(900px circle at 80% 30%, {bg_layer_b}, transparent 60%),
                {bg_base};
            position: relative;
        }}

        [data-testid="stAppViewContainer"]::before,
        section.main::before {{
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            opacity: {noise_opacity};
            background-image:
                radial-gradient(circle at 1px 1px, rgba(255, 255, 255, 0.55) 1px, transparent 0);
            background-size: 3px 3px;
        }}

        [data-testid="stAppViewContainer"] > .main,
        section.main > div {{
            position: relative;
            z-index: 1;
        }}

        [data-testid="stAppViewBlockContainer"],
        .block-container {{
            padding-top: calc(var(--space-lg) - 0.1rem);
            padding-bottom: var(--space-xl);
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stSidebar"] > div:first-child {{
            background: var(--ui-surface-alt);
            border-right: var(--border-strong) solid var(--ui-stroke);
            box-shadow: var(--shadow-brutal), var(--shadow-inner);
            backdrop-filter: blur(var(--blur-glass));
            -webkit-backdrop-filter: blur(var(--blur-glass));
        }}

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {{
            color: var(--ui-text-secondary);
        }}

        [data-testid="stMetric"],
        .stMetric,
        .stContainer,
        [data-testid="stFileUploader"],
        .stDataFrame,
        [data-testid="stTable"],
        [data-testid="stAlert"],
        [data-testid="stForm"] {{
            background: var(--ui-glass);
            border: var(--border-strong) solid var(--ui-stroke);
            border-radius: var(--radius-card);
            box-shadow: var(--shadow-brutal), var(--shadow-soft), var(--shadow-inner);
            backdrop-filter: blur(var(--blur-glass));
            -webkit-backdrop-filter: blur(var(--blur-glass));
        }}

        /* Optional :has selector for modern engines, safe to ignore where unsupported. */
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMetric"]) {{
            background: transparent;
            border-radius: var(--radius-card);
        }}

        [data-testid="stMetric"] {{
            padding: 0.8rem 0.95rem;
        }}

        div[data-testid="stMetricLabel"] p {{
            color: var(--ui-text-muted) !important;
            text-transform: uppercase;
            letter-spacing: 0.045em;
            font-size: 0.72rem !important;
            font-weight: 700 !important;
        }}

        div[data-testid="stMetricValue"] {{
            color: var(--ui-text-primary) !important;
            font-weight: 800 !important;
            font-size: clamp(1.24rem, 2vw, 1.65rem);
        }}

        div[data-testid="stMetricDelta"] {{
            color: var(--ui-accent) !important;
            font-weight: 700 !important;
        }}

        .stButton > button,
        [data-testid="stDownloadButton"] > button {{
            border: var(--border-strong) solid var(--ui-stroke);
            border-radius: var(--radius-button);
            padding: 0.58rem 1rem;
            background: linear-gradient(155deg, var(--ui-accent) 0%, var(--ui-accent-deep) 100%);
            color: #FFFFFF;
            font-family: var(--font-heading);
            font-weight: 700;
            letter-spacing: 0.01em;
            box-shadow: var(--shadow-button), var(--shadow-inner);
        }}

        .stButton > button[kind="secondary"],
        .stButton > button[kind="tertiary"],
        button[data-testid="stBaseButton-secondary"],
        button[data-testid="stBaseButton-tertiary"] {{
            background: var(--ui-glass-strong);
            color: var(--ui-text-primary);
        }}

        .stButton > button:disabled,
        [data-testid="stDownloadButton"] > button:disabled {{
            opacity: 0.6;
            cursor: not-allowed;
            box-shadow: none;
        }}

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stDateInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-baseweb="select"] > div,
        div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {{
            background: var(--ui-glass-strong) !important;
            color: var(--ui-text-primary) !important;
            border: var(--border-regular) solid var(--ui-stroke) !important;
            border-radius: var(--radius-input) !important;
            box-shadow: var(--shadow-inner);
            min-height: 2.6rem;
        }}

        div[data-baseweb="select"] input {{
            color: var(--ui-text-primary) !important;
        }}

        div[data-testid="stTextInput"] input::placeholder,
        div[data-testid="stNumberInput"] input::placeholder,
        div[data-testid="stTextArea"] textarea::placeholder {{
            color: var(--ui-text-muted);
            opacity: 0.82;
        }}

        div[data-testid="stTextInput"] input:hover,
        div[data-testid="stNumberInput"] input:hover,
        div[data-testid="stDateInput"] input:hover,
        div[data-testid="stTextArea"] textarea:hover,
        div[data-baseweb="select"] > div:hover {{
            border-color: rgba(124, 58, 237, 0.64) !important;
        }}

        [data-testid="stFileUploader"] {{
            padding: var(--space-sm);
        }}

        [data-testid="stFileUploaderDropzone"] {{
            border: var(--border-strong) dashed var(--ui-stroke) !important;
            border-radius: var(--radius-card) !important;
            background: var(--ui-glass-strong) !important;
        }}

        [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: rgba(124, 58, 237, 0.7) !important;
        }}

        .stDataFrame,
        [data-testid="stTable"] {{
            overflow: hidden;
        }}

        .stDataFrame [role="columnheader"],
        [data-testid="stTable"] th {{
            background: rgba(124, 58, 237, 0.17) !important;
            color: var(--ui-text-primary) !important;
            font-weight: 700 !important;
        }}

        .stDataFrame [role="gridcell"],
        [data-testid="stTable"] td {{
            color: var(--ui-text-secondary) !important;
            background: transparent !important;
        }}

        [data-testid="stTabs"] [data-baseweb="tab-list"] {{
            gap: var(--space-xs);
            border: var(--border-regular) solid var(--ui-stroke);
            border-radius: var(--radius-card);
            background: var(--ui-glass);
            padding: var(--space-xxs);
            box-shadow: var(--shadow-inner);
        }}

        [data-testid="stTabs"] button {{
            border: var(--border-regular) solid transparent !important;
            border-radius: var(--radius-button) !important;
            color: var(--ui-text-secondary) !important;
            font-weight: 700 !important;
            padding: 0.48rem 0.9rem !important;
            background: transparent !important;
        }}

        [data-testid="stTabs"] button[aria-selected="true"] {{
            background: linear-gradient(155deg, var(--ui-accent), var(--ui-accent-deep)) !important;
            color: #FFFFFF !important;
            box-shadow: 5px 5px 0 rgba(0, 0, 0, 0.38);
            border-color: rgba(255,255,255,0.18) !important;
        }}

        [data-testid="stAlert"] {{
            border-left: var(--border-strong) solid var(--ui-accent);
            padding: 0.7rem 0.85rem;
        }}

        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {{
            color: var(--ui-text-secondary);
            margin: 0;
        }}

        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p strong {{
            color: var(--ui-text-primary);
        }}

        [data-testid="stCaptionContainer"],
        .stCaption {{
            color: var(--ui-text-muted) !important;
        }}

        [data-testid="stRadio"] [role="radiogroup"] > label {{
            border: var(--border-regular) solid var(--ui-stroke);
            border-radius: var(--radius-pill);
            background: var(--ui-glass-strong);
            padding: 0.24rem 0.7rem;
        }}

        [data-testid="stSlider"] [role="slider"] {{
            border: 2px solid rgba(255,255,255,0.35);
            box-shadow: 0 0 0 4px rgba(124,58,237,0.24);
        }}

        .hero-card {{
            background: linear-gradient(145deg, var(--ui-accent) 0%, {hero_secondary} 55%, {hero_tertiary} 100%);
            border: var(--border-strong) solid rgba(255,255,255,0.24);
            border-radius: var(--radius-card);
            padding: 1.35rem 1.6rem;
            margin-bottom: var(--space-md);
            box-shadow: var(--shadow-brutal), var(--shadow-soft), var(--shadow-inner);
            backdrop-filter: blur(var(--blur-glass));
            -webkit-backdrop-filter: blur(var(--blur-glass));
        }}

        .hero-card h2,
        .hero-card h3,
        .hero-card p {{
            color: #FFFFFF !important;
            margin: 0;
        }}

        .hero-card p {{
            opacity: 0.95;
            margin-top: 0.38rem;
        }}

        .pattern-card,
        .usecase-card {{
            height: 100%;
            background: var(--ui-glass);
            border: var(--border-strong) solid var(--ui-stroke);
            border-radius: var(--radius-card);
            padding: 0.9rem 1rem;
            box-shadow: var(--shadow-brutal), var(--shadow-inner);
            backdrop-filter: blur(var(--blur-glass));
            -webkit-backdrop-filter: blur(var(--blur-glass));
        }}

        .pattern-card h4,
        .usecase-card h4 {{
            margin: 0 0 0.35rem 0;
            font-size: 0.94rem;
            color: var(--ui-text-primary);
        }}

        .pattern-card p,
        .usecase-card p {{
            margin: 0;
            font-size: 0.85rem;
            line-height: 1.45;
            color: var(--ui-text-secondary);
        }}

        .info-badge {{
            display: inline-block;
            padding: 0.34rem 0.8rem;
            border: var(--border-regular) solid rgba(124,58,237,0.58);
            border-radius: var(--radius-pill);
            background: rgba(124,58,237,0.16);
            color: var(--ui-text-primary);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin: 0.45rem 0 0.62rem;
            box-shadow: var(--shadow-inner);
        }}

        .highlight-box {{
            margin: 0.65rem 0;
            padding: 0.74rem 0.9rem;
            border: var(--border-regular) solid rgba(124,58,237,0.54);
            border-left: var(--border-strong) solid var(--ui-accent);
            border-radius: var(--radius-sm);
            background: rgba(124,58,237,0.11);
            color: var(--ui-text-primary);
            box-shadow: var(--shadow-inner);
        }}

        .score-bar-container {{
            position: relative;
            height: 30px;
            margin: 0.45rem 0;
            overflow: hidden;
            border: var(--border-regular) solid var(--ui-stroke);
            border-radius: var(--radius-input);
            background: var(--ui-glass-strong);
            box-shadow: var(--shadow-inner);
        }}

        .score-bar-fill {{
            height: 100%;
            border-right: var(--border-regular) solid rgba(255,255,255,0.45);
            transition: width var(--motion-slow) var(--motion-ease);
        }}

        .score-bar-label {{
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--ui-text-primary);
            font-size: 0.83rem;
            font-weight: 800;
            text-shadow: 0 1px 2px rgba(0,0,0,0.32);
        }}

        .savings-metric {{
            text-align: center;
            background: rgba(52,199,89,0.14);
            border: var(--border-strong) solid rgba(52,199,89,0.58);
            border-radius: var(--radius-card);
            padding: 0.92rem;
            box-shadow: var(--shadow-brutal), var(--shadow-inner);
        }}

        .savings-metric .number {{
            color: var(--ui-text-primary);
            font-size: 1.55rem;
            font-weight: 800;
            line-height: 1.15;
        }}

        .savings-metric .label {{
            color: var(--ui-text-secondary);
            font-size: 0.79rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 700;
        }}

        .stButton > button:focus-visible,
        [data-testid="stDownloadButton"] > button:focus-visible,
        [data-testid="stTabs"] button:focus-visible,
        div[data-testid="stTextInput"] input:focus-visible,
        div[data-testid="stNumberInput"] input:focus-visible,
        div[data-testid="stDateInput"] input:focus-visible,
        div[data-testid="stTextArea"] textarea:focus-visible,
        div[data-baseweb="select"] *:focus-visible,
        [data-testid="stFileUploaderDropzone"]:focus-within {{
            outline: 3px solid var(--ui-focus) !important;
            outline-offset: 2px;
        }}

        @media (prefers-reduced-motion: no-preference) {{
            .stButton > button,
            [data-testid="stMetric"],
            .pattern-card,
            .usecase-card,
            .hero-card,
            [data-testid="stFileUploader"],
            [data-testid="stAlert"],
            .stDataFrame,
            [data-testid="stTable"] {{
                transition:
                    transform var(--motion-normal) var(--motion-ease),
                    box-shadow var(--motion-normal) var(--motion-ease),
                    border-color var(--motion-fast) ease,
                    background-color var(--motion-fast) ease;
            }}

            [data-testid="stMetric"]:hover,
            .pattern-card:hover,
            .usecase-card:hover,
            [data-testid="stFileUploader"]:hover,
            [data-testid="stAlert"]:hover,
            .stDataFrame:hover,
            [data-testid="stTable"]:hover {{
                transform: translateY(-2px);
                box-shadow: var(--shadow-brutal-hover), var(--shadow-soft), var(--shadow-inner);
            }}

            .stButton > button:hover,
            [data-testid="stDownloadButton"] > button:hover {{
                transform: translateY(-1px);
                box-shadow: var(--shadow-button-hover), var(--shadow-inner);
            }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            *,
            *::before,
            *::after {{
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }}
        }}

        @media (max-width: 900px) {{
            [data-testid="stAppViewBlockContainer"],
            .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}

            [data-testid="stTabs"] [data-baseweb="tab-list"] {{
                flex-wrap: wrap;
            }}

            .hero-card {{
                padding: 1rem 1.05rem;
            }}

            [data-testid="stMetric"] {{
                box-shadow: 5px 5px 0 rgba(0, 0, 0, 0.45), var(--shadow-inner);
            }}
        }}
        </style>
        """
    ).strip()


def inject_global_styles(theme: str = "dark") -> None:
    st.markdown(_build_global_css(theme), unsafe_allow_html=True)


__all__ = ["inject_global_styles"]
