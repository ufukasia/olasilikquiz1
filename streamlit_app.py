# Ã‡alÄ±ÅŸtÄ±rmak iÃ§in terminale ÅŸu komutu yazÄ±nÄ±z: python -m streamlit run streamlit_app.py
"""
Streamlit tabanlı, Öğrenci numarasina göre kişiselleştirilmiş 5 soruluk quiz.
- Soru senaryolari 2-hafta-olasılık sunumundaki temalara dayanir.
- Sayısal değerler Öğrenci numarasindan deterministik olarak üretilir.
- Cevaplar %5 göreli tolerans ile puanlanir.
- Sonuçlar outputs/quiz_results.csv dosyasina kaydedilir.
"""

from __future__ import annotations

import io
import base64
import hmac
import hashlib
import html
import json
import math
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import qrcode
import streamlit as st
import streamlit.components.v1 as components
from matplotlib import patches
from matplotlib import pyplot as plt

ENV_PATH = Path(".env")
TAB_MONITOR_COMPONENT_PATH = Path(__file__).resolve().parent / "components" / "tab_monitor"
TEACHER_CODE_HASH_KEY = "TEACHER_CODE_HASH"
TEACHER_CODE_KEY = "TEACHER_CODE"
PUBLIC_BASE_URL_KEY = "PUBLIC_BASE_URL"
APP_TIMEZONE_KEY = "APP_TIMEZONE"
DEFAULT_APP_TIMEZONE = "Europe/Istanbul"
APP_LANGUAGE_KEY = "APP_LANGUAGE"
DEFAULT_UI_LANGUAGE = "tr"
SUPPORTED_UI_LANGUAGES = ("tr", "en")
UI_LANGUAGE_STATE_KEY = "_ui_language"
QUIZ_DURATION_OPTIONS = [1, 3, 5, 10, 15, 20, 30]
TEACHER_OPTIONS = [
    "Prof. Dr. Yalçın ATA",
    "Prof. Dr. Arif DEMİR",
    "Dr. Öğr. Üyesi Emel GÜVEN",
    "Dr. Öğr. Üyesi Haydar KILIÇ",
    "Dr. Öğr. Üyesi Ufuk ASIL",
    "Dr. Öğr. Üyesi Muhammed ELMNEFI",
    "Öğr. Gör. Sema ÇİFTÇİ",
]
ANSWER_REL_TOL = 0.05
RESULTS_PATH = Path("outputs/quiz_results.csv")
QUIZ_CONTROL_PATH = Path("outputs/quiz_control.json")
QUIZ_ANSWER_DRAFTS_PATH = Path("outputs/quiz_answer_drafts.csv")
COMMENT_DRAFT_SCOPE_STATE_KEY = "_comment_draft_scope"
COMMENT_DRAFT_VALUES_STATE_KEY = "_comment_draft_values"
DEFAULT_QUIZ_CONTROL: dict[str, Any] = {
    "is_open": False,
    "active_session": "",
    "opened_at": "",
    "closed_at": "",
    "session_start": "",
    "session_end": "",
    "comment_end": "",
    "quiz_duration_minutes": 0,
    "tab_violation_enabled": False,
}
TAB_MONITOR_COMPONENT = components.declare_component(
    "tab_monitor",
    path=str(TAB_MONITOR_COMPONENT_PATH),
)


def _now_iso() -> str:
    return now_in_app_timezone().isoformat(timespec="seconds")


def _now_session_id() -> str:
    return now_in_app_timezone().strftime("%Y%m%d-%H%M%S")


def hash_secret(secret: str) -> str:
    """Girilen gizli değeri SHA-256 ile ozetler."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _parse_env_lines(lines: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            values[key] = value
    return values


def load_env_values(path: Path | None = None) -> dict[str, str]:
    """Basit .env dosyasini okur (script dizini + çalışma dizini fallback)."""
    if path is not None:
        candidates = [path]
    else:
        app_dir_env = Path(__file__).resolve().parent / ".env"
        candidates = [app_dir_env, ENV_PATH]

    values: dict[str, str] = {}
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        try:
            file_values = _parse_env_lines(candidate.read_text(encoding="utf-8").splitlines())
        except OSError:
            continue
        values.update(file_values)
    return values


def get_secret_or_env(key: str) -> str:
    """Değeri secrets, ortam degiskeni veya .env dosyasindan alir."""
    try:
        secret_value = str(st.secrets.get(key, "")).strip()
    except Exception:
        secret_value = ""
    if secret_value:
        return secret_value

    env_file_values = load_env_values()
    return str(os.getenv(key) or env_file_values.get(key, "")).strip()


def get_teacher_code_hash() -> str:
    """Öğretmen kod hash de?erini alir (hash veya düz metin koddan)."""
    raw_hash = get_secret_or_env(TEACHER_CODE_HASH_KEY).lower()
    if raw_hash:
        return raw_hash

    plain_code = get_secret_or_env(TEACHER_CODE_KEY)
    if plain_code:
        return hash_secret(plain_code.strip())
    return ""


def get_public_base_url() -> str:
    """Public quiz URL de?erini secrets, ortam veya .env dosyasindan alir."""
    return get_secret_or_env(PUBLIC_BASE_URL_KEY)


def normalize_ui_language(value: Any) -> str:
    """Desteklenen dil kodunu normalize eder."""
    normalized = str(value or "").strip().lower()
    if normalized in SUPPORTED_UI_LANGUAGES:
        return normalized
    return ""


def _query_lang_param() -> str:
    """URL query parametresinden lang de?erini okur."""
    try:
        raw = st.query_params.get("lang", "")
    except Exception:
        return ""
    if isinstance(raw, list):
        return str(raw[0] if raw else "")
    return str(raw)


def resolve_default_ui_language() -> str:
    """Varsayilan arayuz dilini (query -> env -> default) belirler."""
    from_query = normalize_ui_language(_query_lang_param())
    if from_query:
        return from_query
    from_env = normalize_ui_language(get_secret_or_env(APP_LANGUAGE_KEY))
    if from_env:
        return from_env
    return DEFAULT_UI_LANGUAGE


def set_ui_language(language: str) -> str:
    """Arayuz dilini session state ve URL query parametresine yazar."""
    normalized = normalize_ui_language(language) or DEFAULT_UI_LANGUAGE
    st.session_state[UI_LANGUAGE_STATE_KEY] = normalized
    try:
        current_query_lang = normalize_ui_language(st.query_params.get("lang", ""))
        if current_query_lang != normalized:
            st.query_params["lang"] = normalized
    except Exception:
        pass
    return normalized


def get_ui_language() -> str:
    """Aktif arayuz dilini dondurur."""
    from_state = normalize_ui_language(st.session_state.get(UI_LANGUAGE_STATE_KEY, ""))
    if from_state:
        return from_state
    return set_ui_language(resolve_default_ui_language())


def tr(tr_text: str, en_text: str, **kwargs: Any) -> str:
    """Aktif dile g?re metin secimi yapar."""
    text = en_text if get_ui_language() == "en" else tr_text
    if kwargs:
        return text.format(**kwargs)
    return text


def render_language_selector() -> str:
    """Sidebar'da dil secimini g?sterir."""
    options = [("Turkce (TR)", "tr"), ("English (EN)", "en")]
    labels = [label for label, _ in options]
    label_to_code = {label: code for label, code in options}
    current = get_ui_language()
    current_label = next((label for label, code in options if code == current), labels[0])
    selected_label = st.selectbox(
        "Dil / Language",
        options=labels,
        index=labels.index(current_label),
        key="ui_language_select",
    )
    return set_ui_language(label_to_code[selected_label])


def resolve_app_timezone() -> tuple[ZoneInfo, str, str]:
    """Uygulamanin kullanacagi saat dilimini cozer.

    Donus: (timezone_obj, effective_timezone_name, invalid_requested_name)
    """
    requested_name = get_secret_or_env(APP_TIMEZONE_KEY) or DEFAULT_APP_TIMEZONE
    try:
        return ZoneInfo(requested_name), requested_name, ""
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC"), "UTC", requested_name


def now_in_app_timezone() -> datetime:
    """Uygulamanin baz aldigi saat diliminde simdiki zaman."""
    tz, _, _ = resolve_app_timezone()
    return datetime.now(tz)


def _utc_offset_text(value: datetime) -> str:
    raw = value.strftime("%z")
    if len(raw) == 5:
        return f"{raw[:3]}:{raw[3:]}"
    return raw


def format_dt_obj_for_ui(value: datetime) -> str:
    """Datetime nesnesini arayuzde okunur ve saat dilimi belirtili formatta yazar."""
    offset = _utc_offset_text(value)
    return f"{value.strftime('%d.%m.%Y %H:%M:%S')} (UTC{offset})"


def time_basis_for_ui() -> str:
    """Sistemin hangi saat dilimini baz aldigini metin olarak verir."""
    now = now_in_app_timezone()
    _, effective_name, invalid_requested = resolve_app_timezone()
    offset = _utc_offset_text(now)
    base = f"{effective_name} (UTC{offset})"
    if invalid_requested:
        return f"{base} | APP_TIMEZONE geçersiz: {invalid_requested}"
    return base


class OcrShieldRng:
    """Mulberry32 tabanlı deterministik PRNG - Python surumu."""

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFF

    def next(self) -> float:
        """0-1 arasi float dondurur."""
        self._state = (self._state + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((self._state ^ (self._state >> 15)) * (1 | self._state)) & 0xFFFFFFFF
        t = (t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF) ^ t) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0


# Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â
# OCR / VLM KALKAN SİSTEMİ  (7 katman)
# ──────────────────────────────────────────────────────────────────
# K1 Homoglyph ikamesi       : Copy-paste / metin cikarmayi bozar
# K2 Renk yarma (clip-path)  : OCR karakter sinir tespitini bozar
# K3 Hayalet karakter         : Rakam yanina farklı düşük-opak harf
# K4 Phantom (sahte) sayilar : Gercek sayinin yanina yanlis deger
# K5 Kelime grubu bozma      : Rotation + dikey kayma + font-size
# K6 Zero-width Unicode      : Gorunmez karakterler segmentasyonu bozar
# K7 CSS gorsel gurultu      : text-shadow + pattern (CSS tarafinda)
# Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â

# K1: Gorunusu ayni, Unicode kod noktasi farklı -> copy-paste bozuk
_HOMOGLYPHS = {
    'a': '\u0430', 'c': '\u0441', 'e': '\u0435', 'o': '\u043e',
    'p': '\u0440', 'x': '\u0445', 'y': '\u0443',
    'A': '\u0410', 'B': '\u0412', 'C': '\u0421', 'E': '\u0415',
    'H': '\u041d', 'K': '\u041a', 'M': '\u041c', 'O': '\u041e',
    'P': '\u0420', 'T': '\u0422', 'X': '\u0425',
}

# K3: Rakam/harfin yaninda gorunecek hayalet adaylari
_GHOST_MAP = {
    '0': '869', '1': '7l', '2': 'Z7', '3': '85', '4': '91',
    '5': '6S', '6': '80', '7': '1T', '8': '06', '9': '40',
}

# K2: Radikal renk paleti
_OCR_WARM = ["#ff9e6c", "#ffb347", "#ff7eb3", "#ffa07a", "#f0c040"]
_OCR_COOL = ["#6cc4ff", "#47d1b3", "#7eb3ff", "#40e0d0", "#8be0ff"]

# K6: Gorunmez Unicode
_ZWCHARS = ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"]


def _compute_seed_from_id(student_id: str) -> int:
    """Öğrenci numarasindan deterministik seed uretir."""
    digest = hashlib.sha256(student_id.strip().encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _ocr_shield_text(text: str, rng: OcrShieldRng) -> str:
    """K1+K2+K3: Harf bazinda koruma."""
    parts: list[str] = []
    for ch in text:
        if ch in (" ", "\n", "\t", "\r"):
            parts.append(ch)
            continue

        # K1: Homoglyph ikamesi (~25%)
        display_ch = ch
        if ch in _HOMOGLYPHS and rng.next() < 0.25:
            display_ch = _HOMOGLYPHS[ch]

        safe_ch = html.escape(display_ch)
        r1, r2, r3, r4, r5 = (rng.next() for _ in range(5))

        # K2: ~40% renk yarma
        if r1 < 0.40:
            clip_pct = 42 + int(r2 * 16)
            wi = int(r4 * len(_OCR_WARM)) % len(_OCR_WARM)
            ci = int(r5 * len(_OCR_COOL)) % len(_OCR_COOL)
            if r3 < 0.5:
                tc, bc = _OCR_WARM[wi], _OCR_COOL[ci]
            else:
                tc, bc = _OCR_COOL[ci], _OCR_WARM[wi]
            sx = f"{r1 * 0.5 - 0.25:.2f}"
            ch_html = (
                f'<span class="ocr-w">'
                f'<span class="ocr-t" style="clip-path:inset(0 0 {100-clip_pct}% 0);color:{tc}">{safe_ch}</span>'
                f'<span class="ocr-b" style="clip-path:inset({clip_pct}% 0 0 0);color:{bc};'
                f'transform:translateX({sx}px)">{safe_ch}</span>'
                f'</span>'
            )
        else:
            dx = f"{r2 * 0.6 - 0.3:.2f}"
            dy = f"{r3 * 0.4 - 0.2:.2f}"
            ls = f"{r4 * 0.4 - 0.2:.2f}"
            stl = f"letter-spacing:{ls}px"
            if abs(float(dx)) > 0.08 or abs(float(dy)) > 0.08:
                stl += f";position:relative;transform:translate({dx}px,{dy}px)"
            ch_html = f'<span class="ocr-g" style="{stl}">{safe_ch}</span>'

        # K3: Hayalet karakter (~20% rakam/harflerde)
        if ch in _GHOST_MAP and rng.next() < 0.20:
            cands = _GHOST_MAP[ch]
            ghost = cands[int(rng.next() * len(cands)) % len(cands)]
            gx = f"{rng.next() * 4 - 2:.1f}"
            gy = f"{rng.next() * 3 - 1.5:.1f}"
            gop = f"{0.10 + rng.next() * 0.10:.2f}"
            ch_html = (
                f'<span class="ocr-cw">'
                f'{ch_html}'
                f'<span class="ocr-ghost" style="'
                f'transform:translate({gx}px,{gy}px);'
                f'opacity:{gop}">{html.escape(ghost)}</span>'
                f'</span>'
            )

        parts.append(ch_html)

    return "".join(parts)


def _extract_number(word: str) -> str | None:
    """Kelimeden sayı cikarir."""
    stripped = word.strip(".,;:!()[]{}\"'%")
    if not stripped:
        return None
    try:
        float(stripped.replace(",", "."))
        return stripped
    except ValueError:
        return None


def _perturb_number(num_str: str, rng: OcrShieldRng) -> str:
    """Gercek sayiya yakin ama YANLIS bir sayı uretir."""
    try:
        val = float(num_str.replace(",", "."))
        offset = rng.next() * 0.4 + 0.15
        if rng.next() < 0.5:
            offset = -offset
        new_val = max(0.0, val + offset)
        if "." in num_str or "," in num_str:
            sep = "." if "." in num_str else ","
            dec = len(num_str.split(sep)[1]) if sep in num_str else 2
            result = f"{new_val:.{dec}f}"
            return result.replace(".", ",") if sep == "," else result
        return str(max(0, int(new_val)))
    except (ValueError, IndexError):
        return num_str


def _ocr_shield_full(text: str, rng: OcrShieldRng) -> str:
    """Tam OCR kalkan pipeline: K1-K6."""
    words = text.split(" ")
    word_htmls: list[str] = []

    for w in words:
        if not w:
            word_htmls.append("")
            continue

        wh = _ocr_shield_text(w, rng)

        # K4: Phantom sayilar (~70% sayilarda)
        num = _extract_number(w)
        if num is not None and rng.next() < 0.70:
            wr1, wr2 = _perturb_number(num, rng), _perturb_number(num, rng)
            o1 = f"{0.08 + rng.next() * 0.07:.2f}"
            o2 = f"{0.06 + rng.next() * 0.06:.2f}"
            x1 = f"{rng.next() * 6 - 3:.1f}"
            y1 = f"{rng.next() * 4 - 2:.1f}"
            x2 = f"{rng.next() * 6 - 3:.1f}"
            y2 = f"{rng.next() * 4 - 2:.1f}"
            wh = (
                f'<span class="ocr-nw">{wh}'
                f'<span class="ocr-phantom" style="opacity:{o1};'
                f'transform:translate({x1}px,{y1}px)">{html.escape(wr1)}</span>'
                f'<span class="ocr-phantom" style="opacity:{o2};'
                f'transform:translate({x2}px,{y2}px)">{html.escape(wr2)}</span>'
                f'</span>'
            )

        word_htmls.append(wh)

    # K5: Kelime grubu efektleri
    result_parts: list[str] = []
    i = 0
    while i < len(word_htmls):
        gs = min(1 + int(rng.next() * 3), len(word_htmls) - i)
        gh = " ".join(word_htmls[i:i + gs])
        r1, r2, r3, r4 = rng.next(), rng.next(), rng.next(), rng.next()
        ang = (r1 * 2.0 - 1.0) * 1.2
        if abs(ang) < 0.3:
            ang = 0.3 if ang >= 0 else -0.3
        dy = (r2 * 2.0 - 1.0) * 1.0
        fs = 97 + r3 * 6
        stl = (
            f"display:inline-block;"
            f"transform:rotate({ang:.2f}deg) translateY({dy:.1f}px);"
            f"font-size:{fs:.1f}%;"
            f"transform-origin:center center"
        )
        # K6: Zero-width Unicode
        zw = _ZWCHARS[int(r4 * len(_ZWCHARS)) % len(_ZWCHARS)]
        result_parts.append(f'<span class="ocr-wg" style="{stl}">{gh}</span>')
        if i + gs < len(word_htmls):
            result_parts.append(f" {zw}")
        i += gs

    return "".join(result_parts)


def inject_student_id_overlay(student_id: str) -> None:
    """Öğrenci numarasini session state'e kaydeder (render için)."""
    st.session_state["_ocr_student_id"] = student_id.strip()


def _get_ocr_rng() -> OcrShieldRng | None:
    """Aktif Öğrenci numarası varsa RNG dondurur."""
    sid = st.session_state.get("_ocr_student_id", "")
    if not sid:
        return None
    return OcrShieldRng(_compute_seed_from_id(sid))


def is_teacher_code_configured() -> bool:
    """Öğretmen kod hash ayarı var mı"""
    value = get_teacher_code_hash()
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def verify_teacher_code(input_code: str) -> bool:
    """Öğretmen kodunu hash ile dogrular."""
    stored_hash = get_teacher_code_hash()
    if not (input_code or "").strip() or not is_teacher_code_configured():
        return False
    input_hash = hash_secret(input_code.strip())
    return hmac.compare_digest(input_hash, stored_hash)


def parse_iso_datetime(value: str) -> datetime | None:
    """ISO metnini datetime nesnesine cevirir."""
    if not value:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    tz, _, _ = resolve_app_timezone()
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=tz)
    return parsed.astimezone(tz)


def format_dt_for_ui(value: str) -> str:
    """ISO datetime metnini arayuzde okunur formata cevirir."""
    parsed = parse_iso_datetime(value)
    if parsed is None:
        return "-"
    return format_dt_obj_for_ui(parsed)


def _format_countdown_hhmmss(total_seconds: int) -> str:
    """Saniyeyi HH:MM:SS veya MM:SS metnine cevirir."""
    safe_seconds = max(0, int(total_seconds))
    hours, rem = divmod(safe_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def render_live_countdown(
    end_at: datetime,
    *,
    language: str = DEFAULT_UI_LANGUAGE,
    label_tr: str,
    label_en: str,
    ended_tr: str,
    ended_en: str,
) -> None:
    """Tarayici tarafinda saniye saniye guncellenen geri sayim satiri."""
    is_en = normalize_ui_language(language) == "en"
    label = label_en if is_en else label_tr
    ended_text = ended_en if is_en else ended_tr

    remaining_seconds = int((end_at - now_in_app_timezone()).total_seconds())
    if remaining_seconds <= 0:
        st.caption(ended_text)
        return

    initial_text = f"{label}: {_format_countdown_hhmmss(remaining_seconds)}"
    element_id = f"countdown-{hashlib.sha1(f'{label}-{end_at.isoformat()}'.encode('utf-8')).hexdigest()[:10]}"
    countdown_html = f"""
    <!doctype html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="margin:0;padding:0;background:transparent;">
        <div id="{element_id}" style="color:#cbd5e1;font-size:0.86rem;line-height:1.35;font-family:'Space Grotesk',sans-serif;">
            {html.escape(initial_text)}
        </div>
        <script>
        (function() {{
            const el = document.getElementById({json.dumps(element_id)});
            const endAtMs = Date.parse({json.dumps(end_at.isoformat())});
            const label = {json.dumps(label)};
            const ended = {json.dumps(ended_text)};
            if (!el || Number.isNaN(endAtMs)) {{
                return;
            }}

            const pad = (value) => String(value).padStart(2, "0");
            const formatCountdown = (totalSeconds) => {{
                const safe = Math.max(0, Math.floor(totalSeconds));
                const hours = Math.floor(safe / 3600);
                const minutes = Math.floor((safe % 3600) / 60);
                const seconds = safe % 60;
                if (hours > 0) {{
                    return `${{pad(hours)}}:${{pad(minutes)}}:${{pad(seconds)}}`;
                }}
                return `${{pad(minutes)}}:${{pad(seconds)}}`;
            }};

            const render = () => {{
                const remaining = Math.floor((endAtMs - Date.now()) / 1000);
                if (!Number.isFinite(remaining) || remaining <= 0) {{
                    el.textContent = ended;
                    return false;
                }}
                el.textContent = `${{label}}: ${{formatCountdown(remaining)}}`;
                return true;
            }};

            if (!render()) {{
                return;
            }}
            const timer = setInterval(() => {{
                if (!render()) {{
                    clearInterval(timer);
                }}
            }}, 250);
        }})();
        </script>
    </body>
    </html>
    """
    components.html(countdown_html, height=34)


def auto_close_quiz_if_expired(control: dict[str, Any]) -> dict[str, Any]:
    """Yorum süresi de dahil tum s?reler dolduysa oturumu otomatik kapatir."""
    if not control.get("is_open"):
        return control

    now = now_in_app_timezone()
    # Yorum süresi varsa, kapanma kriteri comment_end'dir
    comment_end_at = parse_iso_datetime(str(control.get("comment_end") or ""))
    end_at = parse_iso_datetime(str(control.get("session_end") or ""))

    # Kapatma: comment_end varsa ona bak, yoksa session_end'e bak
    close_at = comment_end_at or end_at
    if close_at is None:
        return control

    if now >= close_at:
        control["is_open"] = False
        control["closed_at"] = _now_iso()
        save_quiz_control(control)
    return control


def load_quiz_control() -> dict[str, Any]:
    """Quiz açık/kapalı durumunu diskten okur."""
    control = dict(DEFAULT_QUIZ_CONTROL)
    if QUIZ_CONTROL_PATH.exists():
        try:
            loaded = json.loads(QUIZ_CONTROL_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                control.update(loaded)
        except (json.JSONDecodeError, OSError):
            pass

    control["is_open"] = bool(control.get("is_open", False))
    control["active_session"] = str(control.get("active_session") or "")
    control["opened_at"] = str(control.get("opened_at") or "")
    control["closed_at"] = str(control.get("closed_at") or "")
    control["session_start"] = str(control.get("session_start") or "")
    control["session_end"] = str(control.get("session_end") or "")
    control["comment_end"] = str(control.get("comment_end") or "")
    raw_duration = control.get("quiz_duration_minutes", 0)
    try:
        control["quiz_duration_minutes"] = max(0, int(raw_duration))
    except (TypeError, ValueError):
        control["quiz_duration_minutes"] = 0
    control["tab_violation_enabled"] = bool(control.get("tab_violation_enabled", True))
    return control


def save_quiz_control(control: dict[str, Any]) -> None:
    """Quiz açık/kapalı durumunu diske yazar."""
    QUIZ_CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "is_open": bool(control.get("is_open", False)),
        "active_session": str(control.get("active_session") or ""),
        "opened_at": str(control.get("opened_at") or ""),
        "closed_at": str(control.get("closed_at") or ""),
        "session_start": str(control.get("session_start") or ""),
        "session_end": str(control.get("session_end") or ""),
        "comment_end": str(control.get("comment_end") or ""),
        "quiz_duration_minutes": max(0, int(control.get("quiz_duration_minutes", 0))),
        "tab_violation_enabled": bool(control.get("tab_violation_enabled", True)),
    }
    QUIZ_CONTROL_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def rerun_app() -> None:
    """Streamlit surumune göre uygun rerun cagrisi."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def load_results_df() -> pd.DataFrame:
    """Sonuc tablosunu guvenli sekilde okur."""
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(RESULTS_PATH, encoding="utf-8")
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


def clear_results_for_teacher(teacher_name: str) -> int:
    """Seçilen ??retmene ait tum sonuc kay?tlarini siler."""
    if not RESULTS_PATH.exists():
        return 0

    df = load_results_df()
    if df.empty or "teacher_name" not in df.columns:
        return 0

    teacher_clean = teacher_name.strip()
    mask = df["teacher_name"].astype(str).str.strip() == teacher_clean
    removed_count = int(mask.sum())
    if removed_count == 0:
        return 0

    kept_df = df[~mask].copy()
    # Dosya yapisini korumak için kolonlari ayni sekilde yazar.
    if kept_df.empty:
        kept_df = df.iloc[0:0].copy()
    kept_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    return removed_count


def get_existing_submission(student_id: str, quiz_session: str) -> dict[str, Any] | None:
    """Ayni ??rencinin ayni oturumdaki kaydini bulur."""
    if not quiz_session:
        return None

    df = load_results_df()
    if df.empty or "quiz_session" not in df.columns or "student_id" not in df.columns:
        return None

    sid = student_id.strip()
    mask = (df["quiz_session"].astype(str) == quiz_session) & (df["student_id"].astype(str).str.strip() == sid)
    if not mask.any():
        return None

    latest = df[mask].sort_values("timestamp", ascending=False).iloc[0]
    return latest.to_dict()


def submission_count_for_session(quiz_session: str, teacher_name: str | None = None) -> int:
    """Bir oturumdaki toplam teslim sayisi (istege bagli Öğretmen filtresiyle)."""
    if not quiz_session:
        return 0

    df = load_results_df()
    if df.empty or "quiz_session" not in df.columns:
        return 0

    mask = df["quiz_session"].astype(str).str.strip() == quiz_session.strip()
    if teacher_name and "teacher_name" in df.columns:
        mask = mask & (df["teacher_name"].astype(str).str.strip() == teacher_name.strip())
    return int(mask.sum())


def rng_from_student(student_id: str, quiz_session: str) -> random.Random:
    """Ayni Öğrenci ve ayni oturum için ayni soru sayilarini uretir."""
    digest = hashlib.sha256(f"{student_id.strip()}::{quiz_session.strip()}".encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16)
    return random.Random(seed)


def question_bank(
    student_id: str,
    quiz_session: str,
    language: str = DEFAULT_UI_LANGUAGE,
) -> list[dict[str, Any]]:
    """5 soruluk soru listesi."""
    r = rng_from_student(student_id, quiz_session)
    questions: list[dict[str, Any]] = []
    is_en = normalize_ui_language(language) == "en"

    # Q1 - Textile
    p_u = round(r.uniform(0.08, 0.14), 3)
    p_d = round(r.uniform(0.04, 0.08), 3)
    p_ud = round(r.uniform(max(0.005, 0.5 * min(p_u, p_d)), 0.9 * min(p_u, p_d)), 3)
    questions.append(
        {
            "title": "Textile Factory" if is_en else "Tekstil Fabrikası",
            "text": (
                (
                    f"The probability of failing the length test is P(U)={p_u:.3f}, the probability of texture defect is P(D)={p_d:.3f}, "
                    f"and the joint probability is P(U\u2229D)={p_ud:.3f}. "
                    "What is the probability that a strip that failed the length test also has a texture defect (P(D|U))?"
                )
                if is_en
                else (
                    f"Uzunluk testinden kalma olasılığı P(U)={p_u:.3f}, doku hatası P(D)={p_d:.3f}, "
                    f"iki hatanın birlikte olasılığı P(U\u2229D)={p_ud:.3f}. "
                    "Uzunluk testinden kalan bir seridin doku hatalı olma olasılığı nedir (P(D|U))?"
                )
            ),
            "answer": p_ud / p_u,
            "tolerance": ANSWER_REL_TOL,
            "visual": {
                "kind": "venn_prob",
                "left_label": "U",
                "right_label": "D",
                "left": p_u,
                "right": p_d,
                "both": p_ud,
            },
        }
    )

    # Q2 - Vehicle maintenance
    p_y = round(r.uniform(0.20, 0.32), 3)
    p_f = round(r.uniform(0.32, 0.50), 3)
    p_yf = round(r.uniform(max(0.06, 0.4 * min(p_y, p_f)), 0.9 * min(p_y, p_f)), 3)
    questions.append(
        {
            "title": "Vehicle Service" if is_en else "Araç Bakım Servisi",
            "text": (
                (
                    f"The probability of oil change is P(Y)={p_y:.3f}, the probability of filter change is P(F)={p_f:.3f}, "
                    f"and the joint probability is P(Y\u2229F)={p_yf:.3f}. "
                    "What is the probability that a vehicle requiring oil change also requires filter change (P(F|Y))?"
                )
                if is_en
                else (
                    f"Yağ değişimi olasılığı P(Y)={p_y:.3f}, filtre değişimi olasılığı P(F)={p_f:.3f}, "
                    f"birlikte olasılık P(Y\u2229F)={p_yf:.3f}. "
                    "Yağ değişimi gereken bir aracın filtre değişimine de ihtiyacı olma olasılığı nedir (P(F|Y))?"
                )
            ),
            "answer": p_yf / p_y,
            "tolerance": ANSWER_REL_TOL,
            "visual": {
                "kind": "venn_prob",
                "left_label": "Y",
                "right_label": "F",
                "left": p_y,
                "right": p_f,
                "both": p_yf,
            },
        }
    )

    # Q3 - Student survey
    total = 500
    smokers = r.randint(170, 260)
    alcohol = r.randint(220, 290)
    both = r.randint(int(0.4 * min(smokers, alcohol)), int(0.75 * min(smokers, alcohol)))
    questions.append(
        {
            "title": "Student Survey" if is_en else "Öğrenci Anketi",
            "text": (
                (
                    f"In a class of {total} students, {smokers} students smoke, {alcohol} students consume alcohol, "
                    f"and {both} students do both. What is the probability that an alcohol-consuming student also smokes (P(S|A))?"
                )
                if is_en
                else (
                    f"{total} kişilik sınıfta sigara içen sayısı {smokers}, alkol kullanan sayısı {alcohol}, "
                    f"her ikisi {both}. Alkol kullanan bir öğrencinin sigara da içme olasılığı nedir (P(S|A))?"
                )
            ),
            "answer": both / alcohol,
            "tolerance": ANSWER_REL_TOL,
            "visual": {
                "kind": "venn_count",
                "left_label": "S",
                "right_label": "A",
                "left": smokers,
                "right": alcohol,
                "both": both,
                "total": total,
            },
        }
    )

    # Q4 - Serum production flow
    r1 = round(r.uniform(0.08, 0.14), 3)
    r2 = round(r.uniform(0.06, 0.12), 3)
    questions.append(
        {
            "title": "Serum Quality Control" if is_en else "Serum Kalite Kontrol",
            "text": (
                (
                    f"Rejection rates are r1={r1:.3f} and r2={r2:.3f}. "
                    "Assume the department-2 rejection rate applies to batches that passed department 1 "
                    "(i.e., no extra dependency between departments). "
                    "What is the probability that a batch passes department 1 and is rejected in department 2?"
                )
                if is_en
                else (
                    f"Reddetme oranları r1={r1:.3f}, r2={r2:.3f}. "
                    "2. departman red oranının 1. departmanı geçen partilere uygulandığını "
                    "(departmanlar arasında ek bağımlılık olmadığını) varsayın. "
                    "Bir partinin 1. departmanı geçip 2. departmanda reddedilme olasılığı nedir?"
                )
            ),
            "answer": (1 - r1) * r2,
            "tolerance": ANSWER_REL_TOL,
            "visual": {"kind": "serum_flow", "r1": r1, "r2": r2},
        }
    )

    # Q5 - Series / parallel circuit
    p_a = round(r.uniform(0.85, 0.97), 3)
    p_b = round(r.uniform(0.85, 0.97), 3)
    p_c = round(r.uniform(0.70, 0.90), 3)
    p_d = round(r.uniform(0.70, 0.90), 3)
    p_system = p_a * p_b * (1 - (1 - p_c) * (1 - p_d))
    questions.append(
        {
            "title": "Electrical Circuit" if is_en else "Elektrik Devresi",
            "text": (
                (
                    f"A and B are in series, C and D are in parallel, then both groups are connected in series. "
                    f"P(A)={p_a:.3f}, P(B)={p_b:.3f}, P(C)={p_c:.3f}, P(D)={p_d:.3f}. "
                    "Assume component failures are statistically independent. "
                    "What is the operating probability of the whole system?"
                )
                if is_en
                else (
                    f"A ve B seri, C ve D paralel, sonra iki grup tekrar seri. "
                    f"P(A)={p_a:.3f}, P(B)={p_b:.3f}, P(C)={p_c:.3f}, P(D)={p_d:.3f}. "
                    "Bileşen arızalarının istatistiksel olarak bağımsız olduğunu varsayın. "
                    "Tüm sistemin çalışma olasılığı nedir?"
                )
            ),
            "answer": p_system,
            "tolerance": ANSWER_REL_TOL,
            "visual": {"kind": "series_parallel", "p_a": p_a, "p_b": p_b, "p_c": p_c, "p_d": p_d},
        }
    )

    return questions
def check_answer(user_value: float, correct_value: float, rel_tol: float) -> bool:
    """% tabanlı toleransli kontrol."""
    return math.isclose(user_value, correct_value, rel_tol=rel_tol, abs_tol=1e-4)


def _sanitize_answer_value(value: Any) -> float:
    """Ham değeri guvenli ve 0-1 araliginda cevaba cevirir."""
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(normalized):
        return 0.0
    return min(1.0, max(0.0, normalized))


def _score_quiz_answers(
    questions: list[dict[str, Any]],
    given_values: list[float],
) -> tuple[int, list[dict[str, Any]]]:
    """Verilen cevap listesini puanlar ve kayda uygun formatta dondurur."""
    normalized_values = [_sanitize_answer_value(val) for val in given_values]
    if len(normalized_values) < len(questions):
        normalized_values.extend([0.0] * (len(questions) - len(normalized_values)))
    else:
        normalized_values = normalized_values[: len(questions)]

    score = 0
    scored: list[dict[str, Any]] = []
    for given, q in zip(normalized_values, questions):
        is_correct = check_answer(given, q["answer"], q["tolerance"])
        scored.append(
            {
                "given": given,
                "correct": q["answer"],
                "is_correct": is_correct,
                "explanation": "",
            }
        )
        if is_correct:
            score += 20
    return score, scored


def _parse_quiz_scope(scope: str) -> tuple[str, str]:
    """session:student scope metnini ayristirir."""
    normalized = str(scope or "").strip()
    if ":" not in normalized:
        return "", ""
    session_id, student_id = normalized.split(":", 1)
    return session_id.strip(), student_id.strip()


def _load_quiz_answer_drafts_df() -> pd.DataFrame:
    """Quiz cevap taslak CSV'sini guvenli bicimde okur."""
    if not QUIZ_ANSWER_DRAFTS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(QUIZ_ANSWER_DRAFTS_PATH, encoding="utf-8")
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


def _read_quiz_answer_snapshot_record(scope: str) -> dict[str, Any] | None:
    """Scope'a ait son quiz taslagini satir olarak okur."""
    session_id, student_id = _parse_quiz_scope(scope)
    if not session_id or not student_id:
        return None

    df = _load_quiz_answer_drafts_df()
    if df.empty or "quiz_session" not in df.columns or "student_id" not in df.columns:
        return None

    mask = (
        (df["quiz_session"].astype(str).str.strip() == session_id)
        & (df["student_id"].astype(str).str.strip() == student_id)
    )
    if not mask.any():
        return None

    latest = df[mask].sort_values("updated_at", ascending=False).iloc[0]
    return latest.to_dict()


def _store_quiz_answer_snapshot(
    scope: str,
    answers: list[dict[str, Any]],
    student_name: str = "",
    teacher_name: str = "",
) -> None:
    """Quiz cevaplarini Öğrenci+oturum bazli sunucu taslagi olarak saklar."""
    session_id, student_id = _parse_quiz_scope(scope)
    if not session_id or not student_id:
        return

    row: dict[str, Any] = {
        "updated_at": _now_iso(),
        "quiz_session": session_id,
        "student_id": student_id,
        "student_name": student_name.strip(),
        "teacher_name": teacher_name.strip(),
    }
    for idx, item in enumerate(answers, start=1):
        row[f"q{idx}_given"] = _sanitize_answer_value(item.get("given", 0.0))

    QUIZ_ANSWER_DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if QUIZ_ANSWER_DRAFTS_PATH.exists():
        old_df = _load_quiz_answer_drafts_df()
        if old_df.empty:
            new_df.to_csv(QUIZ_ANSWER_DRAFTS_PATH, index=False, encoding="utf-8")
            return

        for col in new_df.columns:
            if col not in old_df.columns:
                old_df[col] = pd.NA
        for col in old_df.columns:
            if col not in new_df.columns:
                new_df[col] = pd.NA

        mask = (
            (old_df["quiz_session"].astype(str).str.strip() == session_id)
            & (old_df["student_id"].astype(str).str.strip() == student_id)
        )
        merged = pd.concat([old_df[~mask], new_df], ignore_index=True)
        merged.to_csv(QUIZ_ANSWER_DRAFTS_PATH, index=False, encoding="utf-8")
        return

    new_df.to_csv(QUIZ_ANSWER_DRAFTS_PATH, index=False, encoding="utf-8")


def _read_quiz_answer_snapshot(scope: str, question_count: int) -> list[float] | None:
    """Scope'a ait quiz taslagindan cevap listesini okur."""
    record = _read_quiz_answer_snapshot_record(scope)
    if record is None:
        return None

    answers: list[float] = []
    for idx in range(1, question_count + 1):
        raw = record.get(f"q{idx}_given", 0.0)
        if pd.isna(raw):
            answers.append(0.0)
        else:
            answers.append(_sanitize_answer_value(raw))
    return answers


def _clear_quiz_answer_snapshot(scope: str | None = None) -> None:
    """Sunucu tarafindaki quiz taslak kaydini temizler."""
    if scope is None:
        return

    session_id, student_id = _parse_quiz_scope(scope)
    if not session_id or not student_id or not QUIZ_ANSWER_DRAFTS_PATH.exists():
        return

    df = _load_quiz_answer_drafts_df()
    if df.empty or "quiz_session" not in df.columns or "student_id" not in df.columns:
        return

    mask = (
        (df["quiz_session"].astype(str).str.strip() == session_id)
        & (df["student_id"].astype(str).str.strip() == student_id)
    )
    if not mask.any():
        return

    kept_df = df[~mask].copy()
    if kept_df.empty:
        QUIZ_ANSWER_DRAFTS_PATH.unlink(missing_ok=True)
        return
    kept_df.to_csv(QUIZ_ANSWER_DRAFTS_PATH, index=False, encoding="utf-8")


def _auto_submit_expired_quiz_from_snapshot(
    student_id: str,
    student_name: str,
    teacher_name: str,
    quiz_session: str,
    questions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Süre doldu?unda, varsa sakli cevaplari otomatik puanlayip kaydeder."""
    session_clean = quiz_session.strip()
    student_id_clean = student_id.strip()
    if not session_clean or not student_id_clean:
        return None

    existing = get_existing_submission(student_id_clean, session_clean)
    if existing is not None:
        return existing

    scope = f"{session_clean}:{student_id_clean}"
    snapshot_record = _read_quiz_answer_snapshot_record(scope)
    if snapshot_record is None:
        return None

    snapshot = _read_quiz_answer_snapshot(scope, len(questions))
    if snapshot is None:
        return None

    resolved_student_name = str(snapshot_record.get("student_name") or student_name).strip()
    resolved_teacher_name = str(snapshot_record.get("teacher_name") or teacher_name).strip()

    score, scored = _score_quiz_answers(questions, snapshot)
    record_result(
        student_id_clean,
        resolved_student_name,
        resolved_teacher_name,
        session_clean,
        score,
        scored,
    )
    _clear_quiz_answer_snapshot(scope)
    return get_existing_submission(student_id_clean, session_clean)


def _store_comment_draft(scope: str, explanations: dict[int, str]) -> None:
    """Yorum aşamasındaki açıklamaları session_state'te taslak olarak saklar."""
    cleaned: dict[int, str] = {}
    for idx, text in explanations.items():
        try:
            q_idx = int(idx)
        except (TypeError, ValueError):
            continue
        if q_idx <= 0:
            continue
        cleaned[q_idx] = str(text or "").strip()

    st.session_state[COMMENT_DRAFT_SCOPE_STATE_KEY] = scope.strip()
    st.session_state[COMMENT_DRAFT_VALUES_STATE_KEY] = cleaned


def _read_comment_draft(scope: str) -> dict[int, str] | None:
    """Scope eslesirse kaydedilen yorum taslagini okur."""
    saved_scope = str(st.session_state.get(COMMENT_DRAFT_SCOPE_STATE_KEY) or "").strip()
    if saved_scope != scope.strip():
        return None

    raw_values = st.session_state.get(COMMENT_DRAFT_VALUES_STATE_KEY)
    if not isinstance(raw_values, dict):
        return None

    cleaned: dict[int, str] = {}
    for idx, text in raw_values.items():
        try:
            q_idx = int(idx)
        except (TypeError, ValueError):
            continue
        if q_idx <= 0:
            continue
        cleaned[q_idx] = str(text or "").strip()
    return cleaned


def _clear_comment_draft(scope: str | None = None) -> None:
    """Yorum taslak verisini temizler."""
    if scope is not None:
        saved_scope = str(st.session_state.get(COMMENT_DRAFT_SCOPE_STATE_KEY) or "").strip()
        if saved_scope != scope.strip():
            return
    st.session_state.pop(COMMENT_DRAFT_SCOPE_STATE_KEY, None)
    st.session_state.pop(COMMENT_DRAFT_VALUES_STATE_KEY, None)


def _auto_submit_expired_comments_from_draft(student_id: str, quiz_session: str) -> bool:
    """Yorum süresi biterse taslagi otomatik olarak sonuca yazar."""
    session_clean = quiz_session.strip()
    student_id_clean = student_id.strip()
    if not session_clean or not student_id_clean:
        return False

    existing_submission = get_existing_submission(student_id_clean, session_clean)
    if existing_submission is None:
        return False

    scope = f"{session_clean}:{student_id_clean}"
    draft = _read_comment_draft(scope)
    if draft is None:
        return False

    _update_explanations(student_id_clean, session_clean, draft)
    _clear_comment_draft(scope)
    return True


def non_empty_line_count(value: str) -> int:
    """Metindeki bos olmayan satir sayisini verir."""
    return sum(1 for line in (value or "").splitlines() if line.strip())


def _base_figure(width: float = 6.0, height: float = 3.2):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.axis("off")
    return fig, ax


def plot_venn(
    left_label: str,
    right_label: str,
    left_value: float,
    right_value: float,
    both_value: float,
    *,
    as_probability: bool,
    total: int | None = None,
):
    """Iki olayli Venn benzeri gorsel."""
    fig, ax = _base_figure(5.8, 3.4)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 5)

    left_only = max(left_value - both_value, 0)
    right_only = max(right_value - both_value, 0)

    c1 = patches.Circle((2.8, 2.5), 1.45, facecolor="#4fc3f7", alpha=0.45, edgecolor="white", linewidth=1.6)
    c2 = patches.Circle((4.2, 2.5), 1.45, facecolor="#ffca28", alpha=0.45, edgecolor="white", linewidth=1.6)
    ax.add_patch(c1)
    ax.add_patch(c2)

    if as_probability:
        fmt = lambda x: f"{x:.3f}"
        footer = "Olasılık diyagramı"
    else:
        fmt = lambda x: f"{int(round(x))}"
        footer = f"Toplam Öğrenci = {total}" if total is not None else "Sayım diyagramı"

    ax.text(2.0, 4.2, left_label, color="white", fontsize=11, weight="bold", ha="center")
    ax.text(5.0, 4.2, right_label, color="white", fontsize=11, weight="bold", ha="center")

    ax.text(2.15, 2.5, fmt(left_only), color="white", fontsize=11, weight="bold", ha="center", va="center")
    ax.text(3.50, 2.5, fmt(both_value), color="white", fontsize=11, weight="bold", ha="center", va="center")
    ax.text(4.85, 2.5, fmt(right_only), color="white", fontsize=11, weight="bold", ha="center", va="center")

    ax.text(3.5, 0.55, footer, color="#cbd5e1", fontsize=9, ha="center")
    return fig


def plot_serum_flow(r1: float, r2: float):
    """Departman bazli akis g?rseli."""
    fig, ax = _base_figure(6.6, 3.4)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    def draw_box(cx: float, cy: float, w: float, h: float, text: str, color: str):
        rect = patches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, text, color="black", fontsize=9, weight="bold", ha="center", va="center")

    def arrow(p1: tuple[float, float], p2: tuple[float, float], label: str, tx: float, ty: float):
        ax.annotate(
            "",
            xy=p2,
            xytext=p1,
            arrowprops={"arrowstyle": "->", "color": "#e2e8f0", "lw": 2.2, "shrinkA": 0, "shrinkB": 0},
        )
        ax.text(tx, ty, label, color="#e2e8f0", fontsize=8.8, ha="center")

    draw_box(2.2, 2.8, 1.8, 0.9, "Departman 1", "#93c5fd")
    draw_box(5.2, 2.8, 1.8, 0.9, "Departman 2", "#86efac")
    draw_box(5.2, 1.0, 2.0, 0.9, "Red 2", "#fca5a5")

    ax.scatter([0.9], [2.8], s=65, color="#fde047", edgecolor="white", linewidth=1.0, zorder=3)
    ax.text(0.7, 2.8, "Giriş", color="white", fontsize=9, ha="right", va="center")

    arrow((1.0, 2.8), (1.3, 2.8), "", 0, 0)
    arrow((3.1, 2.8), (4.3, 2.8), f"Geçiş1 = {1-r1:.3f}", 3.7, 3.15)
    arrow((5.2, 2.35), (5.2, 1.45), f"Red2 = {r2:.3f}", 5.95, 1.95)

    ax.text(5.4, 4.25, f"Hedef yol olasılığı: (1-r1)*r2 = {(1-r1)*r2:.4f}", color="#f8fafc", fontsize=9, ha="center")
    return fig


def plot_series_parallel(p_a: float, p_b: float, p_c: float, p_d: float):
    """Baglantilari birlesik, net seri-paralel devre cizimi."""
    fig, ax = _base_figure(7.2, 3.8)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    wire = {"color": "#e2e8f0", "lw": 2.8, "solid_capstyle": "round"}

    def draw_component(x0: float, x1: float, y: float, label: str, p: float, fill: str):
        body_h = 0.70
        rect = patches.FancyBboxPatch(
            (x0, y - body_h / 2),
            x1 - x0,
            body_h,
            boxstyle="round,pad=0.03,rounding_size=0.1",
            facecolor=fill,
            edgecolor="white",
            linewidth=1.6,
        )
        ax.add_patch(rect)
        ax.text((x0 + x1) / 2, y, label, color="black", fontsize=10, weight="bold", ha="center", va="center")
        ax.text((x0 + x1) / 2, y - 0.78, f"P({label})={p:.3f}", color="white", fontsize=8.5, ha="center")

    y_main = 2.6
    y_top = 3.9
    y_bottom = 1.3

    x_in = 0.8
    ax_x0, ax_x1 = 1.5, 2.3
    bx_x0, bx_x1 = 2.9, 3.7
    x_split = 4.5
    cx_x0, cx_x1 = 5.3, 6.1
    dx_x0, dx_x1 = 5.3, 6.1
    x_join = 6.9
    x_out = 9.0

    # Main serial line to split node
    ax.plot([x_in, ax_x0], [y_main, y_main], **wire)
    ax.plot([ax_x1, bx_x0], [y_main, y_main], **wire)
    ax.plot([bx_x1, x_split], [y_main, y_main], **wire)

    # Split node and parallel branches
    ax.plot([x_split, x_split], [y_bottom, y_top], **wire)
    ax.plot([x_split, cx_x0], [y_top, y_top], **wire)
    ax.plot([cx_x1, x_join], [y_top, y_top], **wire)
    ax.plot([x_split, dx_x0], [y_bottom, y_bottom], **wire)
    ax.plot([dx_x1, x_join], [y_bottom, y_bottom], **wire)

    # Join node and output line
    ax.plot([x_join, x_join], [y_bottom, y_top], **wire)
    ax.plot([x_join, x_out], [y_main, y_main], **wire)

    draw_component(ax_x0, ax_x1, y_main, "A", p_a, "#bae6fd")
    draw_component(bx_x0, bx_x1, y_main, "B", p_b, "#bae6fd")
    draw_component(cx_x0, cx_x1, y_top, "C", p_c, "#bbf7d0")
    draw_component(dx_x0, dx_x1, y_bottom, "D", p_d, "#bbf7d0")

    # Junction dots to make continuity explicit
    for x, y in [
        (x_split, y_main),
        (x_split, y_top),
        (x_split, y_bottom),
        (x_join, y_main),
        (x_join, y_top),
        (x_join, y_bottom),
    ]:
        ax.add_patch(patches.Circle((x, y), 0.06, facecolor="#e2e8f0", edgecolor="none"))

    ax.scatter([x_in, x_out], [y_main, y_main], s=70, color="#fde047", edgecolor="white", linewidth=1.0, zorder=3)
    ax.text(x_in - 0.2, y_main, "Giriş", color="white", fontsize=9, ha="right", va="center")
    ax.text(x_out + 0.2, y_main, "Çıkış", color="white", fontsize=9, ha="left", va="center")
    ax.text(5.0, 4.7, "AB seri, CD paralel, sonra seri", color="#f8fafc", fontsize=10, ha="center")
    return fig


def render_question_visual(question: dict[str, Any]):
    """Soruya göre uygun gorsel olusturur."""
    visual = question.get("visual")
    if not visual:
        return None

    kind = visual.get("kind")
    if kind == "venn_prob":
        return plot_venn(
            visual["left_label"],
            visual["right_label"],
            visual["left"],
            visual["right"],
            visual["both"],
            as_probability=True,
        )
    if kind == "venn_count":
        return plot_venn(
            visual["left_label"],
            visual["right_label"],
            visual["left"],
            visual["right"],
            visual["both"],
            as_probability=False,
            total=visual.get("total"),
        )
    if kind == "serum_flow":
        return plot_serum_flow(visual["r1"], visual["r2"])
    if kind == "series_parallel":
        return plot_series_parallel(visual["p_a"], visual["p_b"], visual["p_c"], visual["p_d"])
    return None


def record_result(
    student_id: str,
    student_name: str,
    teacher_name: str,
    quiz_session: str,
    score: int,
    answers: list[dict[str, Any]],
):
    """Sonuclari CSV dosyasina ekler."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = {
        "timestamp": _now_iso(),
        "student_id": student_id,
        "student_name": student_name,
        "teacher_name": teacher_name,
        "quiz_session": quiz_session,
        "score": score,
    }
    for i, item in enumerate(answers, start=1):
        row[f"q{i}_given"] = item["given"]
        row[f"q{i}_correct"] = item["correct"]
        row[f"q{i}_is_correct"] = item["is_correct"]
        row[f"q{i}_explanation"] = str(item.get("explanation") or "").strip()

    new_df = pd.DataFrame([row])
    if RESULTS_PATH.exists():
        old_df = pd.read_csv(RESULTS_PATH, encoding="utf-8")
        for col in new_df.columns:
            if col not in old_df.columns:
                old_df[col] = pd.NA
        for col in old_df.columns:
            if col not in new_df.columns:
                new_df[col] = pd.NA
        # Tek satir ekleme concat warning'ini onler ve kolon hizasini korur.
        ordered_cols = old_df.columns.tolist()
        old_df.loc[len(old_df)] = new_df.reindex(columns=ordered_cols).iloc[0]
        old_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    else:
        new_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")


def evaluate_quiz_availability(control: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Quizin o an öğrenciye açık olup olmadığını belirler.

    Dönüş: (phase, message, normalized_control)
      phase = "quiz"    -> cevap girme süresi
      phase = "comment" -> yorum/açıklama süresi
      phase = "closed"  -> quiz tamamen kapalı
    """
    normalized = dict(control)
    now = now_in_app_timezone()

    end_at = parse_iso_datetime(str(normalized.get("session_end") or ""))
    comment_end_at = parse_iso_datetime(str(normalized.get("comment_end") or ""))

    close_at = comment_end_at or end_at
    if normalized.get("is_open") and close_at is not None and now >= close_at:
        normalized["is_open"] = False
        normalized["closed_at"] = _now_iso()
        save_quiz_control(normalized)
        return (
            "closed",
            tr(
                "Quiz sonlandı. Quiz ve yorum süresi doldu. Oturum otomatik kapatıldı.",
                "Quiz ended. Quiz and comment periods are over. Session was closed automatically.",
            ),
            normalized,
        )

    active_session = str(normalized.get("active_session") or "")
    if not normalized.get("is_open") or not active_session:
        return (
            "closed",
            tr(
                "Quiz şu an kapalı. Baz alınan saat: {time_basis} | Sistem saati: {now}",
                "Quiz is currently closed. Time basis: {time_basis} | System time: {now}",
                time_basis=time_basis_for_ui(),
                now=format_dt_obj_for_ui(now),
            ),
            normalized,
        )

    start_at = parse_iso_datetime(str(normalized.get("session_start") or ""))
    if start_at is not None and now < start_at:
        return (
            "closed",
            tr(
                "Quiz henüz başlamadı. Sistem saati: {now} | Başlangıç: {start}",
                "Quiz has not started yet. System time: {now} | Start: {start}",
                now=format_dt_obj_for_ui(now),
                start=format_dt_for_ui(normalized["session_start"]),
            ),
            normalized,
        )

    if end_at is not None and now >= end_at:
        if comment_end_at is not None and now < comment_end_at:
            remaining = int((comment_end_at - now).total_seconds())
            mins = max(1, math.ceil(remaining / 60))
            return (
                "comment",
                tr(
                    "Quiz sonlandı. Yorum yazma süresi devam ediyor (kalan: ~{mins} dk).",
                    "Quiz ended. Comment period is still open (~{mins} min left).",
                    mins=mins,
                ),
                normalized,
            )
        return "closed", tr("Quiz sonlandı. Süre doldu.", "Quiz ended. Time is up."), normalized

    return "quiz", "", normalized


def render_session_report(df: pd.DataFrame, active_session: str):
    """Oturum bazlı özet metrikler."""
    st.markdown(f"### {tr('Oturum Bazlı Rapor', 'Session Report')}")
    if "quiz_session" not in df.columns:
        st.info(tr("Rapor için quiz_session kolonu bulunamadı.", "quiz_session column was not found for reporting."))
        return

    sessions = [s for s in df["quiz_session"].astype(str).str.strip().unique().tolist() if s]
    if not sessions:
        st.info(tr("Raporlanacak oturum kaydı yok.", "No session records available for reporting."))
        return

    sessions = sorted(sessions, reverse=True)
    default_idx = 0
    if active_session and active_session in sessions:
        default_idx = sessions.index(active_session)

    selected_session = st.selectbox(
        tr("Rapor Oturumu", "Report Session"),
        options=sessions,
        index=default_idx,
        key="report_session_select",
    )

    session_df = df[df["quiz_session"].astype(str).str.strip() == selected_session].copy()
    if session_df.empty:
        st.info(tr("Seçilen oturumda kayıt yok.", "No records found in the selected session."))
        return

    if "student_id" in session_df.columns:
        participants = session_df["student_id"].astype(str).str.strip().replace("", pd.NA).dropna().nunique()
    else:
        participants = len(session_df)

    score_series = pd.to_numeric(session_df.get("score"), errors="coerce").dropna()
    avg_score = f"{score_series.mean():.1f}" if not score_series.empty else "-"
    max_score = f"{score_series.max():.1f}" if not score_series.empty else "-"
    min_score = f"{score_series.min():.1f}" if not score_series.empty else "-"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(tr("Katılan", "Participants"), int(participants))
    m2.metric(tr("Ortalama", "Average"), avg_score)
    m3.metric(tr("En Yüksek", "Highest"), max_score)
    m4.metric(tr("En Düşük", "Lowest"), min_score)

    visible_cols = ["timestamp", "student_id", "student_name", "teacher_name", "score"]
    existing_cols = [col for col in visible_cols if col in session_df.columns]
    if existing_cols:
        st.dataframe(session_df.sort_values("timestamp", ascending=False)[existing_cols], width="stretch")


def render_qr(url: str) -> None:
    """Verilen URL için QR görseli üretip ekrana basar."""
    qr = qrcode.QRCode(border=2, box_size=6)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.image(
        buf.getvalue(),
        caption=tr("Öğrenciler bu QR ile quize girer.", "Students enter the quiz via this QR code."),
        width=220,
    )


def render_student_entry_qr() -> None:
    """Öğrenci erişim linki ve QR bilgisini gösterir."""
    base_url = get_public_base_url()
    if base_url:
        st.markdown(f"#### {tr('Öğrenci Erişim', 'Student Access')}")
        st.code(base_url, language="text")
        render_qr(base_url)
    else:
        st.info(tr("QR için PUBLIC_BASE_URL secret giriniz.", "Set PUBLIC_BASE_URL in secrets to enable QR."))


def teacher_view():
    """Öğretmen paneli."""
    st.subheader(tr("Öğretmen Modu", "Teacher Mode"))

    if not is_teacher_code_configured():
        st.error(
            tr(
                "Secrets/.env ayarı eksik: TEACHER_CODE_HASH tanımlı değil veya geçersiz.",
                "Missing secrets/.env config: TEACHER_CODE_HASH is not defined or invalid.",
            )
        )
        st.code(
            tr(
                "TEACHER_CODE_HASH=<sha256_hex_değeri>\n# veya\nTEACHER_CODE=<düz-metin-kod>",
                "TEACHER_CODE_HASH=<sha256_hex_value>\n# or\nTEACHER_CODE=<plain-text-code>",
            ),
            language="bash",
        )
        st.caption(
            tr(
                "Hash üretmek için: python -c \"import hashlib; print(hashlib.sha256('yeni-kod'.encode()).hexdigest())\"",
                "Generate hash with: python -c \"import hashlib; print(hashlib.sha256('new-code'.encode()).hexdigest())\"",
            )
        )
        st.caption(
            tr(
                "Streamlit Cloud'da bu değeri App Settings -> Secrets bölümüne eklemelisiniz.",
                "On Streamlit Cloud, add this value under App Settings -> Secrets.",
            )
        )
        return

    select_placeholder = tr("Seçiniz...", "Select...")
    selected_teacher = st.selectbox(
        tr("Öğretmen Seçimi", "Teacher Selection"),
        options=[select_placeholder] + TEACHER_OPTIONS,
        key="teacher_login_select",
    )
    code = st.text_input(tr("Öğretmen kodu", "Teacher code"), type="password", key="teacher_code_input")
    if selected_teacher == select_placeholder or not code.strip():
        st.info(
            tr(
                "Öğretmeninizi seçip kodu girdiğinizde sadece kendi öğrencilerinizi görürsünüz.",
                "After selecting your name and entering the code, you will only see your own students.",
            )
        )
        return
    if not verify_teacher_code(code):
        st.error(tr("Öğretmen kodu hatalı.", "Invalid teacher code."))
        return

    st.caption(tr("Giriş yapan Öğretmen: {teacher}", "Logged-in Teacher: {teacher}", teacher=selected_teacher))
    if "teacher_clear_message" in st.session_state:
        st.success(st.session_state.pop("teacher_clear_message"))

    with st.expander(tr("Veri Temizleme", "Data Cleanup"), expanded=False):
        st.warning(
            tr(
                "Bu işlem seçili öğretmene ait tüm öğrenci kayıtlarını kalıcı olarak siler.",
                "This operation permanently deletes all student records of the selected teacher.",
            )
        )
        confirm_clear = st.checkbox(
            tr(
                "Seçili öğretmene ait tüm kayıtları silmek istiyorum.",
                "I want to delete all records of the selected teacher.",
            ),
            key="confirm_teacher_data_clear",
        )
        if st.button(
            tr("Seçili Öğretmenin Verilerini Temizle", "Clear Selected Teacher Data"),
            use_container_width=True,
            key="clear_teacher_data_btn",
        ):
            if not confirm_clear:
                st.error(tr("Lütfen önce onay kutusunu işaretleyin.", "Please check the confirmation box first."))
            else:
                removed = clear_results_for_teacher(selected_teacher)
                st.session_state["teacher_clear_message"] = (
                    tr(
                        "{teacher} için {removed} kayıt silindi.",
                        "Deleted {removed} records for {teacher}.",
                        teacher=selected_teacher,
                        removed=removed,
                    )
                    if removed > 0
                    else tr(
                        "{teacher} için silinecek kayıt bulunamadı.",
                        "No records found to delete for {teacher}.",
                        teacher=selected_teacher,
                    )
                )
                rerun_app()

    control = auto_close_quiz_if_expired(load_quiz_control())
    now = now_in_app_timezone()

    st.markdown(f"### {tr('Quiz Yönetimi', 'Quiz Management')}")
    st.caption(tr("Baz alınan saat dilimi: {tz}", "Time basis: {tz}", tz=time_basis_for_ui()))
    st.caption(tr("Sunucu sistem saati: {now}", "Server system time: {now}", now=format_dt_obj_for_ui(now)))
    if control["is_open"] and control["active_session"]:
        st.success(
            tr(
                "Durum: AÇIK | Oturum: {session}",
                "Status: OPEN | Session: {session}",
                session=control["active_session"],
            )
        )
        st.caption(tr("Başlangıç: {start}", "Start: {start}", start=format_dt_for_ui(control["session_start"])))
        duration_minutes = int(control.get("quiz_duration_minutes", 0) or 0)
        if duration_minutes <= 0:
            start_at_for_duration = parse_iso_datetime(control["session_start"])
            end_at_for_duration = parse_iso_datetime(control["session_end"])
            if start_at_for_duration is not None and end_at_for_duration is not None and end_at_for_duration > start_at_for_duration:
                duration_minutes = max(
                    1,
                    int(round((end_at_for_duration - start_at_for_duration).total_seconds() / 60)),
                )
        if duration_minutes > 0:
            st.caption(
                tr(
                    "Quiz süresi: {minutes} dakika",
                    "Quiz duration: {minutes} minutes",
                    minutes=duration_minutes,
                )
            )
        tab_rule_state = tr("Açık", "Enabled") if bool(control.get("tab_violation_enabled", True)) else tr("Kapalı", "Disabled")
        st.caption(
            tr(
                "Sekme değişikliği cezası: {state}",
                "Tab-switch penalty: {state}",
                state=tab_rule_state,
            )
        )
        if control["opened_at"]:
            st.caption(tr("Açılış zamanı: {opened}", "Opened at: {opened}", opened=format_dt_for_ui(control["opened_at"])))

        start_at = parse_iso_datetime(control["session_start"])
        end_at = parse_iso_datetime(control["session_end"])
        comment_end_at = parse_iso_datetime(str(control.get("comment_end") or ""))
        if start_at is not None and now < start_at:
            st.info(tr("Oturum planlandı ancak henüz başlamadı.", "Session is scheduled but has not started yet."))
        elif end_at is not None and now < end_at:
            render_live_countdown(
                end_at,
                language=get_ui_language(),
                label_tr="Quiz cevap süresi kalan",
                label_en="Quiz answer time left",
                ended_tr="Quiz cevap süresi doldu.",
                ended_en="Quiz answer time is over.",
            )
        elif end_at is not None and now >= end_at and comment_end_at is not None and now < comment_end_at:
            render_live_countdown(
                comment_end_at,
                language=get_ui_language(),
                label_tr="Yorum yazma süresi kalan",
                label_en="Comment-writing time left",
                ended_tr="Yorum yazma süresi doldu.",
                ended_en="Comment-writing period is over.",
            )

        st.metric(
            tr("Bu oturumda teslim", "Submissions in this session"),
            submission_count_for_session(control["active_session"], selected_teacher),
        )
        render_student_entry_qr()
        if st.button(tr("Quizi Kapat", "Close Quiz"), use_container_width=True, key="close_quiz_btn"):
            control["is_open"] = False
            control["closed_at"] = _now_iso()
            save_quiz_control(control)
            rerun_app()
    else:
        st.warning(tr("Durum: KAPALI", "Status: CLOSED"))
        if control["closed_at"]:
            st.caption(tr("Son kapanış zamanı: {closed}", "Last closed at: {closed}", closed=format_dt_for_ui(control["closed_at"])))

        default_start = now.replace(second=0, microsecond=0)
        default_duration_minutes = 30
        default_start_time = default_start.time().replace(tzinfo=None)
        with st.expander(tr("Yeni Oturum Zaman Penceresi", "New Session Time Window"), expanded=True):
            start_date = st.date_input(tr("Başlangıç tarihi", "Start date"), value=default_start.date(), key="session_start_date")
            start_time = st.time_input(
                tr("Başlangıç saati", "Start time"),
                value=default_start_time,
                key="session_start_time",
                step=timedelta(minutes=1),
            )
            selected_duration = st.selectbox(
                tr("Sunum/Quiz süresi (dk)", "Lecture/Quiz duration (min)"),
                options=QUIZ_DURATION_OPTIONS,
                index=QUIZ_DURATION_OPTIONS.index(default_duration_minutes),
                key="quiz_duration_select",
            )
            tab_violation_enabled = st.toggle(
                tr(
                    "Sekme/uygulama değişikliği tespiti aktif (ihlalde quiz sonlansın)",
                    "Enable tab/app switch monitoring (auto-end on violation)",
                ),
                value=False,
                key="tab_violation_toggle",
            )
            comment_minutes = st.number_input(
                tr(
                    "Yorum süresi (dk) - Quiz bittikten sonra açıklama yazma için ek süre",
                    "Comment duration (min) - extra time after quiz for explanations",
                ),
                min_value=0,
                max_value=60,
                value=5,
                step=1,
                key="comment_duration_input",
            )

        tz, _, _ = resolve_app_timezone()
        start_at = datetime.combine(start_date, start_time).replace(tzinfo=tz)
        end_at = start_at + timedelta(minutes=int(selected_duration))
        comment_end_at = end_at + timedelta(minutes=comment_minutes) if comment_minutes > 0 else None
        st.caption(tr("Seçilen başlangıç: {value}", "Selected start: {value}", value=format_dt_obj_for_ui(start_at)))
        st.caption(
            tr(
                "Seçilen sunum/quiz süresi: {duration} dakika",
                "Selected lecture/quiz duration: {duration} minutes",
                duration=selected_duration,
            )
        )
        st.caption(
            tr("Sekme değişikliği cezası: ", "Tab-switch penalty: ")
            + (
                tr("Açık (ihlalde quiz sonlanır)", "Enabled (quiz ends on violation)")
                if tab_violation_enabled
                else tr("Kapalı", "Disabled")
            )
        )
        if comment_end_at:
            st.caption(
                tr("Seçilen yorum bitiş: {value}", "Selected comment end: {value}", value=format_dt_obj_for_ui(comment_end_at))
            )
        if st.button(tr("Quizi Aç (Yeni Oturum)", "Open Quiz (New Session)"), type="primary", use_container_width=True, key="open_quiz_btn"):
            save_quiz_control(
                {
                    "is_open": True,
                    "active_session": _now_session_id(),
                    "opened_at": _now_iso(),
                    "closed_at": "",
                    "session_start": start_at.isoformat(timespec="seconds"),
                    "session_end": end_at.isoformat(timespec="seconds"),
                    "comment_end": comment_end_at.isoformat(timespec="seconds") if comment_end_at else "",
                    "quiz_duration_minutes": int(selected_duration),
                    "tab_violation_enabled": bool(tab_violation_enabled),
                }
            )
            rerun_app()

    df_all = load_results_df()
    if df_all.empty:
        st.warning(tr("Henüz kayıt yok.", "No records yet."))
        return

    if "student_name" not in df_all.columns:
        df_all["student_name"] = ""
    if "teacher_name" not in df_all.columns:
        df_all["teacher_name"] = ""
    if "quiz_session" not in df_all.columns:
        df_all["quiz_session"] = ""

    df_teacher = df_all[df_all["teacher_name"].astype(str).str.strip() == selected_teacher].copy()
    render_session_report(df_teacher, str(control.get("active_session") or ""))

    query = st.text_input(
        tr("Kendi öğrencilerinizde ara (numara, isim veya oturum)", "Search your students (number, name, or session)"),
        key="teacher_search",
    )
    filtered = df_teacher
    if query:
        query_str = query.strip()
        mask = filtered["student_id"].astype(str).str.fullmatch(query_str) | filtered["student_name"].astype(
            str
        ).str.contains(query_str, case=False, na=False)
        mask = mask | filtered["quiz_session"].astype(str).str.contains(query_str, case=False, na=False)
        filtered = filtered[mask]

    if filtered.empty:
        st.info(tr("Eşleşen kayıt bulunamadı.", "No matching records found."))
        return

    st.dataframe(filtered.sort_values("timestamp", ascending=False), width="stretch")
def _generate_bg_svg(rng: OcrShieldRng, width: int = 600, height: int = 200) -> str:
    """K8: Arka plan SVG gurultusu.

    Soru kartinin background'una uygulanan deterministik SVG:
    - Rastgele noktalar (farklı boyut ve renk)
    - Ince cizgiler (farklı aci ve renk)
    - Sahte (yanlis) sayilar düşük opasitede
    """
    elements: list[str] = []

    # Rastgele noktalar (40-60 adet, belirgin)
    dot_count = 40 + int(rng.next() * 20)
    dot_colors = ["#ff9e6c", "#6cc4ff", "#ffb347", "#47d1b3", "#ff7eb3", "#40e0d0"]
    for _ in range(dot_count):
        cx = int(rng.next() * width)
        cy = int(rng.next() * height)
        r = 1.5 + rng.next() * 4
        color = dot_colors[int(rng.next() * len(dot_colors)) % len(dot_colors)]
        opacity = f"{0.10 + rng.next() * 0.12:.2f}"
        elements.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" '
            f'fill="{color}" opacity="{opacity}"/>'
        )

    # Cizgiler (12-18 adet, kalin)
    line_count = 12 + int(rng.next() * 6)
    for _ in range(line_count):
        x1 = int(rng.next() * width)
        y1 = int(rng.next() * height)
        x2 = int(rng.next() * width)
        y2 = int(rng.next() * height)
        color = dot_colors[int(rng.next() * len(dot_colors)) % len(dot_colors)]
        opacity = f"{0.08 + rng.next() * 0.10:.2f}"
        elements.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="1.0" opacity="{opacity}"/>'
        )

    # Sahte sayilar (20-30 adet, belirgin)
    fake_count = 20 + int(rng.next() * 10)
    for _ in range(fake_count):
        fx = int(rng.next() * (width - 40))
        fy = 10 + int(rng.next() * (height - 15))
        fake_val = f"{rng.next():.2f}"
        font_size = 10 + int(rng.next() * 8)
        color = dot_colors[int(rng.next() * len(dot_colors)) % len(dot_colors)]
        opacity = f"{0.12 + rng.next() * 0.13:.2f}"
        angle = int((rng.next() * 2 - 1) * 20)
        elements.append(
            f'<text x="{fx}" y="{fy}" font-size="{font_size}" '
            f'fill="{color}" opacity="{opacity}" '
            f'transform="rotate({angle},{fx},{fy})">{fake_val}</text>'
        )

    # Sahte P(...) ifadeleri (8-14 adet, belirgin)
    expr_count = 8 + int(rng.next() * 6)
    expr_templates = ["P(A)", "P(B)", "P(A∩B)", "P(A|B)", "P(B|A)", "P(A∪B)"]
    for _ in range(expr_count):
        fx = int(rng.next() * (width - 60))
        fy = 10 + int(rng.next() * (height - 15))
        expr = expr_templates[int(rng.next() * len(expr_templates)) % len(expr_templates)]
        fake_val = f"{rng.next():.2f}"
        label = f"{expr}={fake_val}"
        font_size = 9 + int(rng.next() * 6)
        color = dot_colors[int(rng.next() * len(dot_colors)) % len(dot_colors)]
        opacity = f"{0.10 + rng.next() * 0.12:.2f}"
        angle = int((rng.next() * 2 - 1) * 15)
        elements.append(
            f'<text x="{fx}" y="{fy}" font-size="{font_size}" '
            f'fill="{color}" opacity="{opacity}" '
            f'transform="rotate({angle},{fx},{fy})">{html.escape(label)}</text>'
        )

    svg_content = "\n".join(elements)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'{svg_content}</svg>'
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"url('data:image/svg+xml;base64,{b64}')"


def render_question_card(
    title: str,
    body: str,
    index: int,
    language: str = DEFAULT_UI_LANGUAGE,
):
    body = (
        body.replace("âˆ©", "∩")
        .replace("Ã¢Ë†Â©", "∩")
        .replace("P(U@D)", "P(U∩D)")
        .replace("P(Y@F)", "P(Y∩F)")
    )
    question_prefix = "Question" if normalize_ui_language(language) == "en" else "Soru"
    rng = _get_ocr_rng()
    if rng is not None:
        rendered_title = _ocr_shield_full(f"{question_prefix} {index}: {title}", rng)
        rendered_body = _ocr_shield_full(body, rng)
        bg_svg = _generate_bg_svg(rng)
        card_style = (
            f"background-image:{bg_svg};"
            f"background-size:cover;background-repeat:no-repeat;"
        )
    else:
        rendered_title = f"{question_prefix} {index}: {html.escape(title)}"
        rendered_body = html.escape(body)
        card_style = ""

    st.markdown(
        f"""
        <div class="q-card" style="{card_style}">
            <div class="q-title">{rendered_title}</div>
            <div class="q-body">{rendered_body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2family=Space+Grotesk:wght@400;600;700&display=swap');
        :root {
            --bg-main: #0b132b;
            --bg-soft: #1c2541;
            --ink: #f8fafc;
            --muted: #cbd5e1;
            --edge: rgba(255,255,255,0.12);
            --glass: rgba(255,255,255,0.07);
            --background-color: #0b132b;
            --secondary-background-color: #121f40;
            --text-color: #f8fafc;
            --primary-color: #60a5fa;
        }
        html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            color-scheme: dark !important;
        }
        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(900px 500px at 15% 0%, #243b68 0%, rgba(36,59,104,0.0) 70%),
                radial-gradient(700px 380px at 100% 10%, #2a416f 0%, rgba(42,65,111,0.0) 65%),
                linear-gradient(160deg, #0b132b 0%, #101a35 45%, #0c1630 100%);
            color: var(--ink);
            position: relative;
            isolation: isolate;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #121f40 0%, #0f1834 100%);
            border-left: 1px solid var(--edge);
        }
        [data-testid="stSidebarContent"] {
            height: 100vh;
            overflow-y: auto;
            overflow-x: hidden;
            overscroll-behavior: contain;
            padding-bottom: 20px;
        }
        [data-testid="stSidebar"] > div:first-child {
            height: 100vh;
            overflow-y: auto;
            overflow-x: hidden;
            overscroll-behavior: contain;
        }
        [data-testid="stSidebar"] ::-webkit-scrollbar {
            width: 10px;
        }
        [data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
            background: rgba(203, 213, 225, 0.35);
            border-radius: 999px;
        }
        [data-testid="stSidebar"] ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.06);
        }
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        .stDateInput input,
        .stTimeInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            background: rgba(15, 23, 42, 0.92) !important;
            color: #f8fafc !important;
            border-color: rgba(255, 255, 255, 0.20) !important;
        }
        .stCheckbox label, .stRadio label, .stSelectbox label,
        .stTextInput label, .stNumberInput label, .stDateInput label, .stTimeInput label, .stTextArea label {
            color: #f8fafc !important;
        }
        .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
            color: #f8fafc !important;
        }
        [data-testid="stHeader"] {
            background: rgba(11, 19, 43, 0.88) !important;
        }
        /* ==== OCR/VLM Kalkan Stilleri (K1-K8) ==== */
        .ocr-w { display:inline-block; position:relative; user-select:text; line-height:1; }
        .ocr-t { display:inline-block; user-select:text; }
        .ocr-b { display:inline-block; position:absolute; left:0; top:0; width:100%; pointer-events:none; user-select:none; }
        .ocr-g { display:inline; user-select:text; }
        .ocr-wg { display:inline-block; user-select:text; vertical-align:baseline; }
        .ocr-cw { display:inline-block; position:relative; }
        .ocr-ghost { position:absolute; left:0; top:0; pointer-events:none; user-select:none; filter:blur(0.3px); color:rgba(255,255,255,0.9); }
        .ocr-nw { display:inline-block; position:relative; }
        .ocr-phantom { position:absolute; left:0; top:0; white-space:nowrap; pointer-events:none; user-select:none; color:rgba(255,255,255,1); filter:blur(0.5px); }
        /* Soru kartlari */
        .q-card {
            background: linear-gradient(140deg, rgba(255,255,255,0.10), rgba(255,255,255,0.03));
            border: 1px solid var(--edge);
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 10px;
            box-shadow: 0 18px 35px rgba(0,0,0,0.30);
            position: relative;
            overflow: hidden;
        }
        .q-card::after {
            content: '';
            position: absolute;
            inset: 0;
            background-image:
                repeating-linear-gradient(45deg, transparent, transparent 2px, rgba(255,255,255,0.012) 2px, rgba(255,255,255,0.012) 4px),
                repeating-linear-gradient(-30deg, transparent, transparent 3px, rgba(180,200,255,0.008) 3px, rgba(180,200,255,0.008) 5px);
            pointer-events: none;
            z-index: 1;
            border-radius: inherit;
        }
        .q-title {
            color: var(--ink);
            font-weight: 700;
            font-size: 1.02rem;
            margin-bottom: 6px;
            position: relative; z-index: 2;
        }
        .q-body {
            color: var(--ink);
            line-height: 1.55;
            font-size: 0.95rem;
            position: relative; z-index: 2;
            text-shadow:
                0.4px 0.3px 0px rgba(255,150,100,0.07),
                -0.3px 0.4px 0px rgba(100,180,255,0.06),
                0.2px -0.3px 0px rgba(150,255,130,0.05);
        }
        .uni-brand {
            margin: 2px 0 16px 0;
            padding: 12px 16px;
            border: 1px solid var(--edge);
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(96,165,250,0.18), rgba(255,255,255,0.03));
            text-align: center;
            box-shadow: 0 10px 24px rgba(0,0,0,0.22);
        }
        .uni-brand__line {
            color: #f8fafc;
            line-height: 1.25;
            letter-spacing: 0.02em;
            font-weight: 700;
        }
        .uni-brand__line.sub {
            font-size: 1.02rem;
            opacity: 0.95;
        }
        /* Tab-switch violation banner */
        .tab-violation-banner {
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
            border: 2px solid #fca5a5;
            border-radius: 12px;
            padding: 24px;
            margin: 20px 0;
            text-align: center;
            box-shadow: 0 8px 32px rgba(220, 38, 38, 0.4);
        }
        .tab-violation-banner h2 {
            color: #fff;
            margin: 0 0 8px 0;
            font-size: 1.4rem;
        }
        .tab-violation-banner p {
            color: #fecaca;
            margin: 0;
            font-size: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _update_explanations(student_id: str, quiz_session: str, explanations: dict[int, str]) -> None:
    """Mevcut CSV kaydindaki açıklama alanlarini gunceller."""
    if not RESULTS_PATH.exists():
        return
    df = pd.read_csv(RESULTS_PATH, encoding="utf-8")
    mask = (
        (df["student_id"].astype(str).str.strip() == student_id)
        & (df["quiz_session"].astype(str).str.strip() == quiz_session)
    )
    if not mask.any():
        return
    for q_idx, text in explanations.items():
        col = f"q{q_idx}_explanation"
        if col not in df.columns:
            df[col] = ""
        df.loc[mask, col] = text.strip()
    df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")


# Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â
# SEKME / UYGULAMA DEGİSİKLİGİ TESPİT SİSTEMİ
# Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â

def _check_tab_violation(
    student_id: str,
    quiz_session: str,
    *,
    deadline_at: datetime | None = None,
    monitor_enabled: bool = True,
) -> bool:
    """Custom component'ten gelen sekme ihlali bilgisini okur."""
    payload = TAB_MONITOR_COMPONENT(
        student_id=student_id.strip(),
        quiz_session=quiz_session.strip(),
        monitor_enabled=bool(monitor_enabled),
        deadline_iso=deadline_at.isoformat(timespec="seconds") if deadline_at else "",
        key=f"tab_monitor_{quiz_session.strip()}_{student_id.strip()}",
        default={"violated": False},
    )
    if isinstance(payload, dict):
        return bool(payload.get("violated"))
    return bool(payload)


def _record_tab_violation(
    student_id: str,
    student_name: str,
    teacher_name: str,
    quiz_session: str,
    questions: list[dict[str, Any]],
    language: str = DEFAULT_UI_LANGUAGE,
) -> None:
    """Sekme değişikliği ihlalini 0 puanla CSV'ye kaydeder."""
    is_en = normalize_ui_language(language) == "en"
    explanation_text = (
        "[VIOLATION] Screen/tab switch detected - quiz was auto-terminated."
        if is_en
        else "[IHLAL] Ekran/sekme değişikliği - quiz otomatik sonlandırıldı."
    )
    scored: list[dict[str, Any]] = []
    for q in questions:
        scored.append(
            {
                "given": 0.0,
                "correct": q["answer"],
                "is_correct": False,
                "explanation": explanation_text,
            }
        )
    record_result(student_id, student_name, teacher_name, quiz_session, 0, scored)
    _clear_quiz_answer_snapshot(f"{quiz_session.strip()}:{student_id.strip()}")


def _render_violation_block(language: str = DEFAULT_UI_LANGUAGE) -> None:
    """İhlal durumunda öğrenciye gösterilen uyarı bloğu."""
    is_en = normalize_ui_language(language) == "en"
    title = "Quiz Terminated" if is_en else "Quiziniz Sonlandırıldı"
    line1 = (
        "A switch to another app or browser tab was detected during the quiz."
        if is_en
        else "Quiz sırasında başka bir uygulamaya veya sekmeye geçtiğiniz tespit edildi."
    )
    line2 = (
        "To preserve exam integrity, your quiz was recorded as <strong>0 points</strong>."
        if is_en
        else "Kopya ihtimaline karşı quiziniz <strong>0 puan</strong> olarak kaydedilmiştir."
    )
    line3 = (
        "This event has been reported to your instructor."
        if is_en
        else "Bu durum öğretmeninize bildirilmiştir."
    )
    st.markdown(
        f"""
        <div class="tab-violation-banner">
            <h2>&#9888; {title}</h2>
            <p>
                {line1}<br>
                {line2}<br>
                {line3}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error(
        (
            "Per quiz rules, switching screens/tabs automatically ends the quiz and records 0 points."
            if is_en
            else "Quiz kurallarına göre ekran/sekme değişikliği yapmanız durumunda quiziniz otomatik olarak sonlandırılır ve 0 puan kaydedilir."
        )
    )


def main():
    st.set_page_config(page_title="Personalized Probability Quiz", page_icon="Q", layout="wide")
    inject_styles()

    with st.sidebar:
        selected_language = render_language_selector()
        st.header(tr("Kontrol Paneli", "Control Panel"))
        st.markdown(
            tr(
                "- Tolerans: +/- %5 (göreli)\n- Her soru 20 puan\n- Sorular öğrenciye özeldir",
                "- Tolerance: +/- 5% (relative)\n- Each question is 20 points\n- Questions are personalized by student",
            )
        )
        teacher_view()

    university_line = "Ostim Technical University" if selected_language == "en" else "Ostim Teknik Üniversitesi"
    faculty_line = "Faculty of Engineering" if selected_language == "en" else "Mühendislik Fakültesi"
    st.markdown(
        f"""
        <div class="uni-brand">
            <div class="uni-brand__line">{university_line}</div>
            <div class="uni-brand__line sub">{faculty_line}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title(tr("Kişiye Özel 5 Soruluk Olasılık Quizi", "Personalized 5-Question Probability Quiz"))
    st.caption(
        tr(
            "Sayılar öğrenci numarasına göre değişir. Her oturumda her öğrenci bir kez teslim yapabilir.",
            "Numbers are generated from the student ID. Each student can submit only once per session.",
        )
    )

    quiz_phase, quiz_message, quiz_control = evaluate_quiz_availability(load_quiz_control())
    active_session = str(quiz_control.get("active_session") or "")

    if quiz_phase == "quiz":
        st.info(
            tr(
                "Aktif quiz oturumu: {session} | Başlangıç: {start} | Quiz bitiş: {end}",
                "Active quiz session: {session} | Start: {start} | Quiz end: {end}",
                session=active_session,
                start=format_dt_for_ui(quiz_control["session_start"]),
                end=format_dt_for_ui(quiz_control["session_end"]),
            )
        )
        now = now_in_app_timezone()
        end_at = parse_iso_datetime(quiz_control["session_end"])
        if end_at is not None and now < end_at:
            render_live_countdown(
                end_at,
                language=selected_language,
                label_tr="Cevap süresi kalan",
                label_en="Answer time left",
                ended_tr="Cevap süresi doldu.",
                ended_en="Answer time is over.",
            )
    elif quiz_phase == "comment":
        st.warning(quiz_message)
    else:
        st.warning(quiz_message)

    st.caption(
        tr(
            "Baz alınan saat dilimi: {tz} | Sistem saati: {now}",
            "Time basis: {tz} | System time: {now}",
            tz=time_basis_for_ui(),
            now=format_dt_obj_for_ui(now_in_app_timezone()),
        )
    )

    student_name = st.text_input(tr("Ad Soyad", "Full Name"))
    student_id = st.text_input(
        tr("Öğrenci Numarası (9 rakam)", "Student ID (9 digits)"),
        max_chars=9,
        placeholder="210205901",
    )
    teacher_placeholder = tr("Seçiniz...", "Select...")
    with st.expander(tr("Öğretmen Bilgisi (Aç/Kapat)", "Teacher Info (Expand/Collapse)"), expanded=False):
        teacher_name = st.selectbox(
            tr("Öğretmeninizi seçiniz", "Select your teacher"),
            options=[teacher_placeholder] + TEACHER_OPTIONS,
            key="teacher_name_select",
        )

    student_name_clean = student_name.strip()
    student_id_clean = student_id.strip()
    teacher_name_clean = teacher_name.strip()

    if student_id_clean:
        if not student_id_clean.isdigit():
            st.error(tr("Öğrenci numarası sadece rakamlardan oluşmalıdır.", "Student ID must contain only digits."))
            st.stop()
        if len(student_id_clean) != 9:
            st.error(tr("Öğrenci numarası tam olarak 9 rakam olmalıdır.", "Student ID must be exactly 9 digits."))
            st.stop()
        inject_student_id_overlay(student_id_clean)

    if not student_name_clean or not student_id_clean:
        st.info(tr("Lütfen ad soyad ve öğrenci numarası giriniz.", "Please enter full name and student ID."))
        st.stop()
    if teacher_name == teacher_placeholder:
        st.info(tr("Lütfen öğretmeninizi seçiniz.", "Please select your teacher."))
        st.stop()

    questions = question_bank(student_id_clean, active_session, language=selected_language)

    if quiz_phase == "closed":
        existing_submission = get_existing_submission(student_id_clean, active_session)
        quiz_auto_submitted = False
        if existing_submission is None:
            existing_submission = _auto_submit_expired_quiz_from_snapshot(
                student_id_clean,
                student_name_clean,
                teacher_name_clean,
                active_session,
                questions,
            )
            quiz_auto_submitted = existing_submission is not None

        if quiz_auto_submitted:
            st.success(
                tr(
                    "Quiz süresi bittiği için son kaydedilen cevaplarınız otomatik gönderildi.",
                    "Quiz time ended, so your latest saved answers were auto-submitted.",
                )
            )

        comments_auto_submitted = _auto_submit_expired_comments_from_draft(student_id_clean, active_session)
        existing_submission = existing_submission or get_existing_submission(student_id_clean, active_session)
        comment_end_at = parse_iso_datetime(str(quiz_control.get("comment_end") or ""))
        comment_period_ended = comment_end_at is not None and now_in_app_timezone() >= comment_end_at
        if (comments_auto_submitted or comment_period_ended) and existing_submission is not None:
            st.success(
                tr(
                    "Yorum süresi doldu. Yorumlarınız gönderilmiştir.",
                    "Comment period ended. Your comments have been submitted.",
                )
            )

        if existing_submission is not None and pd.notna(existing_submission.get("score")):
            st.info(
                tr(
                    "Kayıtlı puanınız: {score}/100",
                    "Recorded score: {score}/100",
                    score=int(float(existing_submission["score"])),
                )
            )
        st.stop()

    if quiz_phase == "quiz":
        tab_violation_enabled = bool(quiz_control.get("tab_violation_enabled", True))
        violation_scope = f"{active_session}:{student_id_clean}"
        if st.session_state.get("_tab_violation_scope") != violation_scope:
            st.session_state["_tab_violation"] = False
            st.session_state["_tab_violation_scope"] = violation_scope

        quiz_end_at = parse_iso_datetime(str(quiz_control.get("session_end") or ""))

        tab_violated = _check_tab_violation(
            student_id_clean,
            active_session,
            deadline_at=quiz_end_at,
            monitor_enabled=tab_violation_enabled,
        )
        if tab_violation_enabled and tab_violated:
            st.session_state["_tab_violation"] = True
        elif not tab_violation_enabled:
            st.session_state["_tab_violation"] = False

        if st.session_state.get("_tab_violation"):
            existing_submission = get_existing_submission(student_id_clean, active_session)
            if existing_submission is None:
                _record_tab_violation(
                    student_id_clean,
                    student_name_clean,
                    teacher_name_clean,
                    active_session,
                    questions,
                    language=selected_language,
                )
            _render_violation_block(language=selected_language)
            st.stop()

        existing_submission = get_existing_submission(student_id_clean, active_session)
        if existing_submission is not None:
            _clear_quiz_answer_snapshot(f"{active_session}:{student_id_clean}")
            st.error(
                tr(
                    "Bu oturumda daha önce teslim yaptınız. Quiz süresi bittiğinde yorumlarınızı yazabilirsiniz.",
                    "You already submitted in this session. You can write comments after quiz time ends.",
                )
            )
            if pd.notna(existing_submission.get("score")):
                st.info(
                    tr(
                        "Kayıtlı puanınız: {score}/100",
                        "Recorded score: {score}/100",
                        score=int(float(existing_submission["score"])),
                    )
                )
            st.stop()

        if tab_violation_enabled:
            st.markdown(
                tr(
                    "> **Quiz Aşaması:** Sadece sayısal cevapları giriniz. Quiz süresi bittikten sonra açıklama/yorum yazmak için ek süre verilecektir.\n\n> &#9888; **Uyarı:** Quiz sırasında başka bir uygulamaya veya sekmeye geçerseniz quiziniz **otomatik olarak sonlandırılır** ve **0 puan** kaydedilir.",
                    "> **Quiz Phase:** Enter numeric answers only. After the quiz ends, you will get extra time for explanations/comments.\n\n> &#9888; **Warning:** If you switch to another app/tab during the quiz, your quiz will be **auto-terminated** and recorded as **0 points**.",
                )
            )
        else:
            st.markdown(
                tr(
                    "> **Quiz Aşaması:** Sadece sayısal cevapları giriniz. Quiz süresi bittikten sonra açıklama/yorum yazmak için ek süre verilecektir.",
                    "> **Quiz Phase:** Enter numeric answers only. After the quiz ends, you will get extra time for explanations/comments.",
                )
            )

        saved_answers = _read_quiz_answer_snapshot(violation_scope, len(questions))
        answers: list[dict[str, Any]] = []
        cols = st.columns(2, gap="large")
        for idx, q in enumerate(questions, start=1):
            with cols[(idx - 1) % 2]:
                render_question_card(q["title"], q["text"], idx, language=selected_language)
                default_value = 0.0
                if saved_answers is not None and idx - 1 < len(saved_answers):
                    default_value = _sanitize_answer_value(saved_answers[idx - 1])
                user_value = st.number_input(
                    tr("Cevabın (0-1 arası)", "Your answer (0-1)"),
                    key=f"ans_{active_session}_{student_id_clean}_{idx}",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.001,
                    format="%.4f",
                    value=float(default_value),
                )
                answers.append(
                    {
                        "given": user_value,
                        "correct": q["answer"],
                        "tolerance": q["tolerance"],
                        "explanation": "",
                    }
                )

                fig = render_question_visual(q)
                if fig is not None:
                    st.pyplot(fig, clear_figure=True, width="stretch")
                    plt.close(fig)

        _store_quiz_answer_snapshot(
            f"{active_session}:{student_id_clean}",
            answers,
            student_name_clean,
            teacher_name_clean,
        )
        st.caption(tr("Cevaplarınız otomatik olarak kaydediliyor.", "Your answers are saved automatically."))

        confirm_submit = st.checkbox(
            tr(
                "Cevaplarımı göndermek istediğime eminim. (Gönderdikten sonra değiştiremezsiniz.)",
                "I confirm that I want to submit my answers. (You cannot change them after submission.)",
            ),
            key=f"confirm_submit_checkbox_{active_session}_{student_id_clean}",
        )
        if st.button(tr("Cevapları Gönder ve Puanla", "Submit Answers and Score"), type="primary", disabled=not confirm_submit):
            if get_existing_submission(student_id_clean, active_session) is not None:
                st.error(tr("Bu oturum için kaydınız zaten alınmış.", "Your submission for this session is already recorded."))
                st.stop()

            score, scored = _score_quiz_answers(questions, [ans["given"] for ans in answers])

            st.success(tr("Toplam puan: {score}/100", "Total score: {score}/100", score=score))
            st.info(
                tr(
                    "Sonucunuz kaydedildi. Quiz süresi bittikten sonra açıklama/yorum yazabileceksiniz.",
                    "Your result has been saved. You can write explanations/comments after quiz time ends.",
                )
            )

            record_result(student_id_clean, student_name_clean, teacher_name_clean, active_session, score, scored)
            _clear_quiz_answer_snapshot(f"{active_session}:{student_id_clean}")
            st.balloons()

    elif quiz_phase == "comment":
        comment_end_at = parse_iso_datetime(str(quiz_control.get("comment_end") or ""))
        if comment_end_at is not None:
            _check_tab_violation(
                student_id_clean,
                active_session,
                deadline_at=comment_end_at,
                monitor_enabled=False,
            )

        existing_submission = get_existing_submission(student_id_clean, active_session)
        if existing_submission is None:
            existing_submission = _auto_submit_expired_quiz_from_snapshot(
                student_id_clean,
                student_name_clean,
                teacher_name_clean,
                active_session,
                questions,
            )
            if existing_submission is None:
                st.error(
                    tr(
                        "Quiz sürecinde cevap göndermediniz. Yorum aşamasında yeni cevap kabul edilmez.",
                        "You did not submit answers during the quiz. New answers are not accepted in comment phase.",
                    )
                )
                st.stop()
            st.success(
                tr(
                    "Quiz süresi dolduğu için mevcut cevaplarınız otomatik gönderildi ve puanlandı.",
                    "Quiz time ended, so your current answers were auto-submitted and scored.",
                )
            )

        comment_scope = f"{active_session}:{student_id_clean}"
        draft_explanations = _read_comment_draft(comment_scope) or {}

        if pd.notna(existing_submission.get("score")):
            st.info(
                tr(
                    "Kayıtlı puanınız: {score}/100",
                    "Recorded score: {score}/100",
                    score=int(float(existing_submission["score"])),
                )
            )

        st.markdown(
            tr(
                "> **Yorum Aşaması:** Quiz süresi doldu. Cevaplarınız kilitlenmiştir. Aşağıda her soru için çözüm açıklamanızı yazabilirsiniz.",
                "> **Comment Phase:** Quiz time is over. Your answers are locked. You can write your solution explanations below.",
            )
        )

        explanations: dict[int, str] = {}
        cols = st.columns(2, gap="large")
        for idx, q in enumerate(questions, start=1):
            with cols[(idx - 1) % 2]:
                render_question_card(q["title"], q["text"], idx, language=selected_language)

                given_val = existing_submission.get(f"q{idx}_given")
                locked_answer_label = tr(
                    "Soru {idx} cevabın (kilitli)",
                    "Question {idx} answer (locked)",
                    idx=idx,
                )
                if pd.notna(given_val):
                    st.text_input(
                        locked_answer_label,
                        value=f"{float(given_val):.4f}",
                        disabled=True,
                        key=f"locked_ans_{idx}",
                    )
                else:
                    st.text_input(
                        locked_answer_label,
                        value="-",
                        disabled=True,
                        key=f"locked_ans_{idx}",
                    )

                prev_exp = draft_explanations.get(idx)
                if prev_exp is None:
                    prev_exp = str(existing_submission.get(f"q{idx}_explanation") or "")
                explanation = st.text_area(
                    tr("Çözümünü 2-3 satırla açıkla", "Explain your solution in 2-3 lines"),
                    key=f"exp_{active_session}_{student_id_clean}_{idx}",
                    height=88,
                    value=prev_exp if prev_exp else "",
                    placeholder=tr(
                        "Kullandığınız formül ve adımları kısaca yazın.",
                        "Briefly write the formula and steps you used.",
                    ),
                )
                explanations[idx] = explanation.strip()

                fig = render_question_visual(q)
                if fig is not None:
                    st.pyplot(fig, clear_figure=True, width="stretch")
                    plt.close(fig)

        _store_comment_draft(comment_scope, explanations)

        if st.button(tr("Yorumları Kaydet", "Save Comments"), type="primary"):
            _update_explanations(student_id_clean, active_session, explanations)
            _clear_comment_draft(comment_scope)
            st.success(tr("Yorumlarınız kaydedildi!", "Your comments have been saved!"))
            st.balloons()
if __name__ == "__main__":
    main()

