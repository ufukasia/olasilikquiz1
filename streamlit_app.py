# Calistirmak icin terminale su komutu yaziniz: python -m streamlit run streamlit_app.py
"""
Streamlit tabanli, ogrenci numarasina gore kisilestirilmis 5 soruluk quiz.
- Soru senaryolari 2-hafta-olasilik sunumundaki temalara dayanir.
- Sayisal degerler ogrenci numarasindan deterministik olarak uretilir.
- Cevaplar %5 goreli tolerans ile puanlanir.
- Sonuclar outputs/quiz_results.csv dosyasina kaydedilir.
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
QUIZ_DURATION_OPTIONS = [1, 3, 5, 10, 15, 20, 30]
TEACHER_OPTIONS = [
    "Prof. Dr. Yalçın ATA",
    "Prof. Dr. Arif DEMİR",
    "Dr. Öğr. Üyesi Emel GÜVEN",
    "Dr. Öğr. Üyesi Haydar KILIÇ",
    "Dr. Öğr. Üyesi Ufuk ASIL",
    "Öğr. Gör. Sema ÇİFTÇİ",
]
ANSWER_REL_TOL = 0.05
RESULTS_PATH = Path("outputs/quiz_results.csv")
QUIZ_CONTROL_PATH = Path("outputs/quiz_control.json")
DEFAULT_QUIZ_CONTROL: dict[str, Any] = {
    "is_open": False,
    "active_session": "",
    "opened_at": "",
    "closed_at": "",
    "session_start": "",
    "session_end": "",
    "comment_end": "",
    "quiz_duration_minutes": 0,
    "tab_violation_enabled": True,
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
    """Girilen gizli degeri SHA-256 ile ozetler."""
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
    """Basit .env dosyasini okur (script dizini + calisma dizini fallback)."""
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
    """Degeri secrets, ortam degiskeni veya .env dosyasindan alir."""
    try:
        secret_value = str(st.secrets.get(key, "")).strip()
    except Exception:
        secret_value = ""
    if secret_value:
        return secret_value

    env_file_values = load_env_values()
    return str(os.getenv(key) or env_file_values.get(key, "")).strip()


def get_teacher_code_hash() -> str:
    """Ogretmen kod hash degerini alir (hash veya duz metin koddan)."""
    raw_hash = get_secret_or_env(TEACHER_CODE_HASH_KEY).lower()
    if raw_hash:
        return raw_hash

    plain_code = get_secret_or_env(TEACHER_CODE_KEY)
    if plain_code:
        return hash_secret(plain_code.strip())
    return ""


def get_public_base_url() -> str:
    """Public quiz URL degerini secrets, ortam veya .env dosyasindan alir."""
    return get_secret_or_env(PUBLIC_BASE_URL_KEY)


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
        return f"{base} | APP_TIMEZONE gecersiz: {invalid_requested}"
    return base


class OcrShieldRng:
    """Mulberry32 tabanli deterministik PRNG - Python surumu."""

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFF

    def next(self) -> float:
        """0-1 arasi float dondurur."""
        self._state = (self._state + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((self._state ^ (self._state >> 15)) * (1 | self._state)) & 0xFFFFFFFF
        t = (t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF) ^ t) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0


# ══════════════════════════════════════════════════════════════════
# OCR / VLM KALKAN SİSTEMİ  (7 katman)
# ──────────────────────────────────────────────────────────────────
# K1 Homoglyph ikamesi       : Copy-paste / metin cikarmayi bozar
# K2 Renk yarma (clip-path)  : OCR karakter sinir tespitini bozar
# K3 Hayalet karakter         : Rakam yanina farkli dusuk-opak harf
# K4 Phantom (sahte) sayilar : Gercek sayinin yanina yanlis deger
# K5 Kelime grubu bozma      : Rotation + dikey kayma + font-size
# K6 Zero-width Unicode      : Gorunmez karakterler segmentasyonu bozar
# K7 CSS gorsel gurultu      : text-shadow + pattern (CSS tarafinda)
# ══════════════════════════════════════════════════════════════════

# K1: Gorunusu ayni, Unicode kod noktasi farkli -> copy-paste bozuk
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
    """Ogrenci numarasindan deterministik seed uretir."""
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
    """Kelimeden sayi cikarir."""
    stripped = word.strip(".,;:!?()[]{}\"'%")
    if not stripped:
        return None
    try:
        float(stripped.replace(",", "."))
        return stripped
    except ValueError:
        return None


def _perturb_number(num_str: str, rng: OcrShieldRng) -> str:
    """Gercek sayiya yakin ama YANLIS bir sayi uretir."""
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
    """Ogrenci numarasini session state'e kaydeder (render icin)."""
    st.session_state["_ocr_student_id"] = student_id.strip()


def _get_ocr_rng() -> OcrShieldRng | None:
    """Aktif ogrenci numarasi varsa RNG dondurur."""
    sid = st.session_state.get("_ocr_student_id", "")
    if not sid:
        return None
    return OcrShieldRng(_compute_seed_from_id(sid))


def is_teacher_code_configured() -> bool:
    """Ogretmen kod hash ayari var mi?"""
    value = get_teacher_code_hash()
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def verify_teacher_code(input_code: str) -> bool:
    """Ogretmen kodunu hash ile dogrular."""
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


def auto_close_quiz_if_expired(control: dict[str, Any]) -> dict[str, Any]:
    """Yorum suresi de dahil tum sureler dolduysa oturumu otomatik kapatir."""
    if not control.get("is_open"):
        return control

    now = now_in_app_timezone()
    # Yorum suresi varsa, kapanma kriteri comment_end'dir
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
    """Quiz acik/kapali durumunu diskten okur."""
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
    """Quiz acik/kapali durumunu diske yazar."""
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
    """Streamlit surumune gore uygun rerun cagrisi."""
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
    """Secilen ogretmene ait tum sonuc kayitlarini siler."""
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
    # Dosya yapisini korumak icin kolonlari ayni sekilde yazar.
    if kept_df.empty:
        kept_df = df.iloc[0:0].copy()
    kept_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    return removed_count


def get_existing_submission(student_id: str, quiz_session: str) -> dict[str, Any] | None:
    """Ayni ogrencinin ayni oturumdaki kaydini bulur."""
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
    """Bir oturumdaki toplam teslim sayisi (istege bagli ogretmen filtresiyle)."""
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
    """Ayni ogrenci ve ayni oturum icin ayni soru sayilarini uretir."""
    digest = hashlib.sha256(f"{student_id.strip()}::{quiz_session.strip()}".encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16)
    return random.Random(seed)


def question_bank(student_id: str, quiz_session: str) -> list[dict[str, Any]]:
    """5 soruluk soru listesi."""
    r = rng_from_student(student_id, quiz_session)
    questions: list[dict[str, Any]] = []

    # Q1 - Tekstil
    p_u = round(r.uniform(0.08, 0.14), 3)
    p_d = round(r.uniform(0.04, 0.08), 3)
    p_ud = round(r.uniform(max(0.005, 0.5 * min(p_u, p_d)), 0.9 * min(p_u, p_d)), 3)
    questions.append(
        {
            "title": "Tekstil Fabrikasi",
            "text": (
                f"Uzunluk testinden kalma olasiligi P(U)={p_u:.3f}, doku hatasi P(D)={p_d:.3f}, "
                f"iki hatanin birlikte olasiligi P(U\u2229D)={p_ud:.3f}. "
                "Uzunluk testinden kalan bir seridin doku hatali olma olasiligi nedir (P(D|U))?"
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

    # Q2 - Arac bakim
    p_y = round(r.uniform(0.20, 0.32), 3)
    p_f = round(r.uniform(0.32, 0.50), 3)
    p_yf = round(r.uniform(max(0.06, 0.4 * min(p_y, p_f)), 0.9 * min(p_y, p_f)), 3)
    questions.append(
        {
            "title": "Arac Bakim Servisi",
            "text": (
                f"Yag degisimi olasiligi P(Y)={p_y:.3f}, filtre degisimi olasiligi P(F)={p_f:.3f}, "
                f"birlikte olasilik P(Y\u2229F)={p_yf:.3f}. "
                "Yag degisimi gereken aracin filtre degisimine de ihtiyaci olma olasiligi nedir (P(F|Y))?"
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

    # Q3 - Ogrenci anketi
    total = 500
    smokers = r.randint(170, 260)
    alcohol = r.randint(220, 290)
    both = r.randint(int(0.4 * min(smokers, alcohol)), int(0.75 * min(smokers, alcohol)))
    questions.append(
        {
            "title": "Ogrenci Anketi",
            "text": (
                f"{total} kisilik sinifta sigara icen sayisi {smokers}, alkol kullanan sayisi {alcohol}, "
                f"her ikisi {both}. Alkol kullanan bir ogrencinin sigara da icme olasiligi nedir (P(S|A))?"
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

    # Q4 - Serum uretimi
    r1 = round(r.uniform(0.08, 0.14), 3)
    r2 = round(r.uniform(0.06, 0.12), 3)
    questions.append(
        {
            "title": "Serum Kalite Kontrol",
            "text": (
                f"Reddetme oranlari r1={r1:.3f}, r2={r2:.3f}. "
                "Bir partinin 1. departmani gecip 2. departmanda reddedilme olasiligi nedir?"
            ),
            "answer": (1 - r1) * r2,
            "tolerance": ANSWER_REL_TOL,
            "visual": {"kind": "serum_flow", "r1": r1, "r2": r2},
        }
    )

    # Q5 - Seri/paralel devre
    p_a = round(r.uniform(0.85, 0.97), 3)
    p_b = round(r.uniform(0.85, 0.97), 3)
    p_c = round(r.uniform(0.70, 0.90), 3)
    p_d = round(r.uniform(0.70, 0.90), 3)
    p_system = p_a * p_b * (1 - (1 - p_c) * (1 - p_d))
    questions.append(
        {
            "title": "Elektrik Devresi",
            "text": (
                f"A ve B seri, C ve D paralel, sonra iki grup tekrar seri. "
                f"P(A)={p_a:.3f}, P(B)={p_b:.3f}, P(C)={p_c:.3f}, P(D)={p_d:.3f}. "
                "Tum sistemin calisma olasiligi nedir?"
            ),
            "answer": p_system,
            "tolerance": ANSWER_REL_TOL,
            "visual": {"kind": "series_parallel", "p_a": p_a, "p_b": p_b, "p_c": p_c, "p_d": p_d},
        }
    )

    return questions


def check_answer(user_value: float, correct_value: float, rel_tol: float) -> bool:
    """% tabanli toleransli kontrol."""
    return math.isclose(user_value, correct_value, rel_tol=rel_tol, abs_tol=1e-4)


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
        footer = "Olasilik diyagrami"
    else:
        fmt = lambda x: f"{int(round(x))}"
        footer = f"Toplam ogrenci = {total}" if total is not None else "Sayim diyagrami"

    ax.text(2.0, 4.2, left_label, color="white", fontsize=11, weight="bold", ha="center")
    ax.text(5.0, 4.2, right_label, color="white", fontsize=11, weight="bold", ha="center")

    ax.text(2.15, 2.5, fmt(left_only), color="white", fontsize=11, weight="bold", ha="center", va="center")
    ax.text(3.50, 2.5, fmt(both_value), color="white", fontsize=11, weight="bold", ha="center", va="center")
    ax.text(4.85, 2.5, fmt(right_only), color="white", fontsize=11, weight="bold", ha="center", va="center")

    ax.text(3.5, 0.55, footer, color="#cbd5e1", fontsize=9, ha="center")
    return fig


def plot_serum_flow(r1: float, r2: float):
    """Departman bazli akis gorseli."""
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
    ax.text(0.7, 2.8, "Giris", color="white", fontsize=9, ha="right", va="center")

    arrow((1.0, 2.8), (1.3, 2.8), "", 0, 0)
    arrow((3.1, 2.8), (4.3, 2.8), f"Gecis1 = {1-r1:.3f}", 3.7, 3.15)
    arrow((5.2, 2.35), (5.2, 1.45), f"Red2 = {r2:.3f}", 5.95, 1.95)

    ax.text(5.4, 4.25, f"Hedef yol olasiligi: (1-r1)*r2 = {(1-r1)*r2:.4f}", color="#f8fafc", fontsize=9, ha="center")
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
    ax.text(x_in - 0.2, y_main, "Giris", color="white", fontsize=9, ha="right", va="center")
    ax.text(x_out + 0.2, y_main, "Cikis", color="white", fontsize=9, ha="left", va="center")
    ax.text(5.0, 4.7, "AB seri, CD paralel, sonra seri", color="#f8fafc", fontsize=10, ha="center")
    return fig


def render_question_visual(question: dict[str, Any]):
    """Soruya gore uygun gorsel olusturur."""
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
        pd.concat([old_df, new_df], ignore_index=True).to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    else:
        new_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")


def evaluate_quiz_availability(control: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Quizin o an ogrenciye acik olup olmadigini belirler.

    Donus: (phase, message, normalized_control)
      phase = "quiz"    -> cevap girme suresi (telefona az dokunma)
      phase = "comment" -> yorum/aciklama suresi (telefona dokunabilir)
      phase = "closed"  -> quiz tamamen kapali
    """
    normalized = dict(control)
    now = now_in_app_timezone()

    end_at = parse_iso_datetime(str(normalized.get("session_end") or ""))
    comment_end_at = parse_iso_datetime(str(normalized.get("comment_end") or ""))

    # Tum sureler dolduysa oturumu kapat
    close_at = comment_end_at or end_at
    if normalized.get("is_open") and close_at is not None and now >= close_at:
        normalized["is_open"] = False
        normalized["closed_at"] = _now_iso()
        save_quiz_control(normalized)
        return "closed", "Quiz ve yorum suresi doldu. Oturum otomatik kapatildi.", normalized

    active_session = str(normalized.get("active_session") or "")
    if not normalized.get("is_open") or not active_session:
        return (
            "closed",
            f"Quiz su an kapali. Baz alinan saat: {time_basis_for_ui()} | Sistem saati: {format_dt_obj_for_ui(now)}",
            normalized,
        )

    start_at = parse_iso_datetime(str(normalized.get("session_start") or ""))
    if start_at is not None and now < start_at:
        return (
            "closed",
            "Quiz henuz baslamadi. "
            f"Sistem saati: {format_dt_obj_for_ui(now)} | "
            f"Baslangic: {format_dt_for_ui(normalized['session_start'])}",
            normalized,
        )

    # Quiz cevap suresi hala devam ediyor mu?
    if end_at is not None and now >= end_at:
        # Quiz suresi doldu ama yorum suresi var ve devam ediyor
        if comment_end_at is not None and now < comment_end_at:
            remaining = int((comment_end_at - now).total_seconds())
            mins = max(1, math.ceil(remaining / 60))
            return (
                "comment",
                f"Quiz suresi doldu. Yorum yazma suresi devam ediyor (kalan: ~{mins} dk).",
                normalized,
            )
        # comment_end yoksa veya o da dolduysa kapali (yukaridaki blokta yakalanir ama guvenlik icin)
        return "closed", "Quiz suresi doldu.", normalized

    return "quiz", "", normalized


def render_session_report(df: pd.DataFrame, active_session: str):
    """Oturum bazli ozet metrikler."""
    st.markdown("### Oturum Bazli Rapor")
    if "quiz_session" not in df.columns:
        st.info("Rapor icin quiz_session kolonu bulunamadi.")
        return

    sessions = [s for s in df["quiz_session"].astype(str).str.strip().unique().tolist() if s]
    if not sessions:
        st.info("Raporlanacak oturum kaydi yok.")
        return

    sessions = sorted(sessions, reverse=True)
    default_idx = 0
    if active_session and active_session in sessions:
        default_idx = sessions.index(active_session)

    selected_session = st.selectbox(
        "Rapor Oturumu",
        options=sessions,
        index=default_idx,
        key="report_session_select",
    )

    session_df = df[df["quiz_session"].astype(str).str.strip() == selected_session].copy()
    if session_df.empty:
        st.info("Secilen oturumda kayit yok.")
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
    m1.metric("Katilan", int(participants))
    m2.metric("Ortalama", avg_score)
    m3.metric("En Yuksek", max_score)
    m4.metric("En Dusuk", min_score)

    visible_cols = ["timestamp", "student_id", "student_name", "teacher_name", "score"]
    existing_cols = [col for col in visible_cols if col in session_df.columns]
    if existing_cols:
        st.dataframe(session_df.sort_values("timestamp", ascending=False)[existing_cols], width="stretch")


def render_qr(url: str) -> None:
    """Verilen URL icin QR gorseli uretip ekrana basar."""
    qr = qrcode.QRCode(border=2, box_size=6)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.image(buf.getvalue(), caption="Ogrenciler bu QR ile quize girer.", width=220)


def render_student_entry_qr() -> None:
    """Ogrenci erisim linki ve QR bilgisini gosterir."""
    base_url = get_public_base_url()
    if base_url:
        st.markdown("#### Ogrenci Erisim")
        st.code(base_url, language="text")
        render_qr(base_url)
    else:
        st.info("QR icin PUBLIC_BASE_URL secret giriniz.")


def teacher_view():
    """Ogretmen paneli."""
    st.subheader("Ogretmen Modu")

    if not is_teacher_code_configured():
        st.error("Secrets/.env ayari eksik: TEACHER_CODE_HASH tanimli degil veya gecersiz.")
        st.code("TEACHER_CODE_HASH=<sha256_hex_degeri>\n# veya\nTEACHER_CODE=<duz-metin-kod>", language="bash")
        st.caption(
            "Hash uretmek icin: python -c \"import hashlib; print(hashlib.sha256('yeni-kod'.encode()).hexdigest())\""
        )
        st.caption("Streamlit Cloud'da bu degeri App Settings -> Secrets bolumune eklemelisiniz.")
        return

    selected_teacher = st.selectbox(
        "Ogretmen Secimi",
        options=["Seciniz..."] + TEACHER_OPTIONS,
        key="teacher_login_select",
    )
    code = st.text_input("Ogretmen kodu", type="password", key="teacher_code_input")
    if selected_teacher == "Seciniz..." or not code.strip():
        st.info("Ogretmeninizi secip kodu girdiginizde sadece kendi ogrencilerinizi gorursunuz.")
        return
    if not verify_teacher_code(code):
        st.error("Ogretmen kodu hatali.")
        return

    st.caption(f"Giris yapan ogretmen: {selected_teacher}")
    if "teacher_clear_message" in st.session_state:
        st.success(st.session_state.pop("teacher_clear_message"))

    with st.expander("Veri Temizleme", expanded=False):
        st.warning("Bu islem secili ogretmene ait tum ogrenci kayitlarini kalici olarak siler.")
        confirm_clear = st.checkbox(
            "Secili ogretmene ait tum kayitlari silmek istiyorum.",
            key="confirm_teacher_data_clear",
        )
        if st.button(
            "Secili Ogretmenin Verilerini Temizle",
            use_container_width=True,
            key="clear_teacher_data_btn",
        ):
            if not confirm_clear:
                st.error("Lutfen once onay kutusunu isaretleyin.")
            else:
                removed = clear_results_for_teacher(selected_teacher)
                st.session_state["teacher_clear_message"] = (
                    f"{selected_teacher} icin {removed} kayit silindi."
                    if removed > 0
                    else f"{selected_teacher} icin silinecek kayit bulunamadi."
                )
                rerun_app()

    control = auto_close_quiz_if_expired(load_quiz_control())
    now = now_in_app_timezone()

    st.markdown("### Quiz Yonetimi")
    st.caption(f"Baz alinan saat dilimi: {time_basis_for_ui()}")
    st.caption(f"Sunucu sistem saati: {format_dt_obj_for_ui(now)}")
    if control["is_open"] and control["active_session"]:
        st.success(f"Durum: ACIK | Oturum: {control['active_session']}")
        st.caption(f"Baslangic: {format_dt_for_ui(control['session_start'])}")
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
            st.caption(f"Quiz suresi: {duration_minutes} dakika")
        tab_rule_state = "Acik" if bool(control.get("tab_violation_enabled", True)) else "Kapali"
        st.caption(f"Sekme degisikligi cezasi: {tab_rule_state}")
        if control["opened_at"]:
            st.caption(f"Acilis zamani: {format_dt_for_ui(control['opened_at'])}")

        start_at = parse_iso_datetime(control["session_start"])
        end_at = parse_iso_datetime(control["session_end"])
        comment_end_at = parse_iso_datetime(str(control.get("comment_end") or ""))
        if start_at is not None and now < start_at:
            st.info("Oturum planlandi ancak henuz baslamadi.")
        elif end_at is not None and now < end_at:
            remaining_seconds = int((end_at - now).total_seconds())
            remaining_minutes = max(1, math.ceil(remaining_seconds / 60))
            st.info(f"Quiz cevap suresi kalan: ~{remaining_minutes} dakika")
        elif end_at is not None and now >= end_at and comment_end_at is not None and now < comment_end_at:
            remaining_seconds = int((comment_end_at - now).total_seconds())
            remaining_minutes = max(1, math.ceil(remaining_seconds / 60))
            st.info(f"Yorum yazma suresi kalan: ~{remaining_minutes} dakika")

        st.metric(
            "Bu oturumda teslim",
            submission_count_for_session(control["active_session"], selected_teacher),
        )
        render_student_entry_qr()
        if st.button("Quizi Kapat", use_container_width=True, key="close_quiz_btn"):
            control["is_open"] = False
            control["closed_at"] = _now_iso()
            save_quiz_control(control)
            rerun_app()
    else:
        st.warning("Durum: KAPALI")
        if control["closed_at"]:
            st.caption(f"Son kapanis zamani: {format_dt_for_ui(control['closed_at'])}")

        default_start = now.replace(second=0, microsecond=0)
        default_duration_minutes = 15
        default_start_time = default_start.time().replace(tzinfo=None)
        with st.expander("Yeni Oturum Zaman Penceresi", expanded=True):
            start_date = st.date_input("Baslangic tarihi", value=default_start.date(), key="session_start_date")
            start_time = st.time_input(
                "Baslangic saati",
                value=default_start_time,
                key="session_start_time",
                step=timedelta(minutes=1),
            )
            selected_duration = st.selectbox(
                "Sunum/Quiz suresi (dk)",
                options=QUIZ_DURATION_OPTIONS,
                index=QUIZ_DURATION_OPTIONS.index(default_duration_minutes),
                key="quiz_duration_select",
            )
            tab_violation_enabled = st.toggle(
                "Sekme/uygulama degisikligi tespiti aktif (ihlalde quiz sonlansin)",
                value=True,
                key="tab_violation_toggle",
            )
            comment_minutes = st.number_input(
                "Yorum suresi (dk) — Quiz bittikten sonra aciklama yazma icin ek sure",
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
        st.caption(f"Secilen baslangic: {format_dt_obj_for_ui(start_at)}")
        st.caption(f"Secilen sunum/quiz suresi: {selected_duration} dakika")
        st.caption(
            "Sekme degisikligi cezasi: "
            + ("Acik (ihlalde quiz sonlanir)" if tab_violation_enabled else "Kapali")
        )
        if comment_end_at:
            st.caption(f"Secilen yorum bitis: {format_dt_obj_for_ui(comment_end_at)}")
        if st.button("Quizi Ac (Yeni Oturum)", type="primary", use_container_width=True, key="open_quiz_btn"):
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
        st.warning("Henuz kayit yok.")
        return

    if "student_name" not in df_all.columns:
        df_all["student_name"] = ""
    if "teacher_name" not in df_all.columns:
        df_all["teacher_name"] = ""
    if "quiz_session" not in df_all.columns:
        df_all["quiz_session"] = ""

    df_teacher = df_all[df_all["teacher_name"].astype(str).str.strip() == selected_teacher].copy()
    render_session_report(df_teacher, str(control.get("active_session") or ""))

    query = st.text_input("Kendi ogrencilerinizde ara (numara, isim veya oturum)", key="teacher_search")
    filtered = df_teacher
    if query:
        query_str = query.strip()
        mask = filtered["student_id"].astype(str).str.fullmatch(query_str) | filtered["student_name"].astype(
            str
        ).str.contains(query_str, case=False, na=False)
        mask = mask | filtered["quiz_session"].astype(str).str.contains(query_str, case=False, na=False)
        filtered = filtered[mask]

    if filtered.empty:
        st.info("Eslesen kayit bulunamadi.")
        return

    st.dataframe(filtered.sort_values("timestamp", ascending=False), width="stretch")


def _generate_bg_svg(rng: OcrShieldRng, width: int = 600, height: int = 200) -> str:
    """K8: Arka plan SVG gurultusu.

    Soru kartinin background'una uygulanan deterministik SVG:
    - Rastgele noktalar (farkli boyut ve renk)
    - Ince cizgiler (farkli aci ve renk)
    - Sahte (yanlis) sayilar dusuk opasitede
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


def render_question_card(title: str, body: str, index: int):
    body = (
        body.replace("âˆ©", "∩")
        .replace("Ã¢Ë†Â©", "∩")
        .replace("P(U@D)", "P(U∩D)")
        .replace("P(Y@F)", "P(Y∩F)")
    )
    rng = _get_ocr_rng()
    if rng is not None:
        rendered_title = _ocr_shield_full(f"Soru {index}: {title}", rng)
        rendered_body = _ocr_shield_full(body, rng)
        bg_svg = _generate_bg_svg(rng)
        card_style = (
            f"background-image:{bg_svg};"
            f"background-size:cover;background-repeat:no-repeat;"
        )
    else:
        rendered_title = f"Soru {index}: {html.escape(title)}"
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
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
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
    """Mevcut CSV kaydindaki aciklama alanlarini gunceller."""
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


# ══════════════════════════════════════════════════════════════════
# SEKME / UYGULAMA DEGİSİKLİGİ TESPİT SİSTEMİ
# ══════════════════════════════════════════════════════════════════

def _check_tab_violation(student_id: str, quiz_session: str) -> bool:
    """Custom component'ten gelen sekme ihlali bilgisini okur."""
    payload = TAB_MONITOR_COMPONENT(
        student_id=student_id.strip(),
        quiz_session=quiz_session.strip(),
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
) -> None:
    """Sekme degisikligi ihlalini 0 puanla CSV'ye kaydeder."""
    scored: list[dict[str, Any]] = []
    for q in questions:
        scored.append({
            "given": 0.0,
            "correct": q["answer"],
            "is_correct": False,
            "explanation": "[IHLAL] Ekran/sekme degisikligi - quiz otomatik sonlandirildi.",
        })
    record_result(student_id, student_name, teacher_name, quiz_session, 0, scored)


def _render_violation_block() -> None:
    """Ihlal durumunda ogrenciye gosterilen uyari blogu."""
    st.markdown(
        """
        <div class="tab-violation-banner">
            <h2>&#9888; Quiziniz Sonlandirildi</h2>
            <p>
                Quiz sirasinda baska bir uygulamaya veya sekmeye gectiginiz tespit edildi.<br>
                Kopya ihtimaline karsi quiziniz <strong>0 puan</strong> olarak kaydedilmistir.<br>
                Bu durum ogretmeninize bildirilmistir.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error(
        "Quiz kurallarina gore ekran/sekme degisikligi yapmaniz durumunda "
        "quiziniz otomatik olarak sonlandirilir ve 0 puan kaydedilir."
    )


def main():
    st.set_page_config(page_title="Kisiye Ozel Olasilik Quizi", page_icon="Q", layout="wide")
    inject_styles()

    st.title("Kisiye Ozel 5 Soruluk Olasilik Quizi")
    st.caption("Sayilar ogrenci numarasina gore degisir. Her oturumda her ogrenci tek kez teslim yapabilir.")

    with st.sidebar:
        st.header("Kontrol Paneli")
        st.markdown("- Tolerans: +/- %5 (goreli)\n- Her soru 20 puan\n- Sorular ogrenciye ozeldir")
        teacher_view()

    quiz_phase, quiz_message, quiz_control = evaluate_quiz_availability(load_quiz_control())
    active_session = str(quiz_control.get("active_session") or "")
    if quiz_phase == "closed":
        st.warning(quiz_message)
        st.stop()

    # Faz bilgisi
    if quiz_phase == "quiz":
        st.info(
            f"Aktif quiz oturumu: {active_session} | "
            f"Baslangic: {format_dt_for_ui(quiz_control['session_start'])} | "
            f"Quiz bitis: {format_dt_for_ui(quiz_control['session_end'])}"
        )
        now = now_in_app_timezone()
        end_at = parse_iso_datetime(quiz_control["session_end"])
        if end_at is not None and now < end_at:
            remaining = max(1, math.ceil((end_at - now).total_seconds() / 60))
            st.caption(f"Cevap suresi kalan: ~{remaining} dk | "
                       f"Saat dilimi: {time_basis_for_ui()} | Sistem saati: {format_dt_obj_for_ui(now)}")
    elif quiz_phase == "comment":
        st.warning(quiz_message)

    st.caption(f"Baz alinan saat dilimi: {time_basis_for_ui()} | Sistem saati: {format_dt_obj_for_ui(now_in_app_timezone())}")

    student_name = st.text_input("Ad Soyad")
    student_id = st.text_input("Ogrenci Numarasi (9 rakam)", max_chars=9, placeholder="210205901")
    with st.expander("Ogretmen Bilgisi (Ac/Kapat)", expanded=False):
        teacher_name = st.selectbox(
            "Ogretmeninizi seciniz",
            options=["Seciniz..."] + TEACHER_OPTIONS,
            key="teacher_name_select",
        )

    student_name_clean = student_name.strip()
    student_id_clean = student_id.strip()
    teacher_name_clean = teacher_name.strip()

    # Ogrenci numarasi validasyonu
    if student_id_clean:
        if not student_id_clean.isdigit():
            st.error("Ogrenci numarasi sadece rakamlardan olusmalidir.")
            st.stop()
        if len(student_id_clean) != 9:
            st.error("Ogrenci numarasi tam olarak 9 rakam olmalidir.")
            st.stop()
        inject_student_id_overlay(student_id_clean)

    if not student_name_clean or not student_id_clean:
        st.info("Lutfen ad soyad ve ogrenci numarasi giriniz.")
        st.stop()
    if teacher_name == "Seciniz...":
        st.info("Lutfen ogretmeninizi seciniz.")
        st.stop()

    questions = question_bank(student_id_clean, active_session)

    # ================================================================
    # FAZ 1: QUIZ — sadece sayisal cevap, telefona minimum dokunma
    # ================================================================
    if quiz_phase == "quiz":
        tab_violation_enabled = bool(quiz_control.get("tab_violation_enabled", True))
        violation_scope = f"{active_session}:{student_id_clean}"
        if st.session_state.get("_tab_violation_scope") != violation_scope:
            st.session_state["_tab_violation"] = False
            st.session_state["_tab_violation_scope"] = violation_scope

        # --- Sekme/uygulama degisikligi kontrolu ---
        if tab_violation_enabled:
            tab_violated = _check_tab_violation(student_id_clean, active_session)
            if tab_violated:
                # Session state'e kaydet (kalici)
                st.session_state["_tab_violation"] = True
        else:
            st.session_state["_tab_violation"] = False

        if st.session_state.get("_tab_violation"):
            # Ihlal varsa: once kayit kontrol et, kayit yoksa 0 puan yaz
            existing_submission = get_existing_submission(student_id_clean, active_session)
            if existing_submission is None:
                _record_tab_violation(
                    student_id_clean,
                    student_name_clean,
                    teacher_name_clean,
                    active_session,
                    questions,
                )
            _render_violation_block()
            st.stop()

        existing_submission = get_existing_submission(student_id_clean, active_session)
        if existing_submission is not None:
            st.error("Bu oturumda daha once teslim yaptiniz. Quiz suresi bittiginde yorumlarinizi yazabilirsiniz.")
            if pd.notna(existing_submission.get("score")):
                st.info(f"Kayitli puaniniz: {int(float(existing_submission['score']))}/100")
            st.stop()

        if tab_violation_enabled:
            st.markdown(
                "> **Quiz Asamasi:** Sadece sayisal cevaplari giriniz. "
                "Quiz suresi bittikten sonra aciklama/yorum yazmak icin ek sure verilecektir.\n\n"
                "> &#9888; **Uyari:** Quiz sirasinda baska bir uygulamaya veya sekmeye gecerseniz "
                "quiziniz **otomatik olarak sonlandirilir** ve **0 puan** kaydedilir."
            )
        else:
            st.markdown(
                "> **Quiz Asamasi:** Sadece sayisal cevaplari giriniz. "
                "Quiz suresi bittikten sonra aciklama/yorum yazmak icin ek sure verilecektir."
            )

        answers: list[dict[str, Any]] = []
        cols = st.columns(2, gap="large")
        for idx, q in enumerate(questions, start=1):
            with cols[(idx - 1) % 2]:
                render_question_card(q["title"], q["text"], idx)
                user_value = st.number_input(
                    "Cevabin (0-1 arasi)",
                    key=f"ans_{idx}",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.001,
                    format="%.4f",
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

        confirm_submit = st.checkbox(
            "Cevaplarimi gondermek istedigime eminim. (Gonderdikten sonra degistiremezsiniz.)",
            key="confirm_submit_checkbox",
        )
        if st.button("Cevaplari Gonder ve Puanla", type="primary", disabled=not confirm_submit):
            if get_existing_submission(student_id_clean, active_session) is not None:
                st.error("Bu oturum icin kaydiniz zaten alinmis.")
                st.stop()

            scored: list[dict[str, Any]] = []
            score = 0
            for ans, q in zip(answers, questions):
                ok = check_answer(ans["given"], q["answer"], q["tolerance"])
                scored_item = {
                    "given": ans["given"],
                    "correct": q["answer"],
                    "is_correct": ok,
                    "explanation": "",
                }
                scored.append(scored_item)
                if ok:
                    score += 20

            st.success(f"Toplam puan: {score}/100")
            st.info("Sonucunuz kaydedildi. Quiz suresi bittikten sonra aciklama/yorum yazabileceksiniz.")

            record_result(student_id_clean, student_name_clean, teacher_name_clean, active_session, score, scored)
            st.balloons()

    # ================================================================
    # FAZ 2: YORUM — cevaplar kilitli, aciklama kutulari acik
    # ================================================================
    elif quiz_phase == "comment":
        existing_submission = get_existing_submission(student_id_clean, active_session)
        if existing_submission is None:
            st.error("Quiz surecinde cevap gondermediniz. Yorum asamasinda yeni cevap kabul edilmez.")
            st.stop()

        if pd.notna(existing_submission.get("score")):
            st.info(f"Kayitli puaniniz: {int(float(existing_submission['score']))}/100")

        st.markdown(
            "> **Yorum Asamasi:** Quiz suresi doldu. Cevaplarin kilitlenmistir. "
            "Asagida her soru icin cozum aciklamanizi yazabilirsiniz."
        )

        explanations: dict[int, str] = {}
        cols = st.columns(2, gap="large")
        for idx, q in enumerate(questions, start=1):
            with cols[(idx - 1) % 2]:
                render_question_card(q["title"], q["text"], idx)

                # Kilitli cevap gosterimi
                given_val = existing_submission.get(f"q{idx}_given")
                if pd.notna(given_val):
                    st.text_input(
                        f"Soru {idx} cevabin (kilitli)",
                        value=f"{float(given_val):.4f}",
                        disabled=True,
                        key=f"locked_ans_{idx}",
                    )
                else:
                    st.text_input(
                        f"Soru {idx} cevabin (kilitli)",
                        value="—",
                        disabled=True,
                        key=f"locked_ans_{idx}",
                    )

                # Onceki aciklama varsa goster
                prev_exp = str(existing_submission.get(f"q{idx}_explanation") or "")
                explanation = st.text_area(
                    "Cozumunu 2-3 satirla acikla",
                    key=f"exp_{idx}",
                    height=88,
                    value=prev_exp if prev_exp else "",
                    placeholder="Kullandiginiz formul ve adimlari kisaca yazin.",
                )
                explanations[idx] = explanation.strip()

                fig = render_question_visual(q)
                if fig is not None:
                    st.pyplot(fig, clear_figure=True, width="stretch")
                    plt.close(fig)

        if st.button("Yorumlari Kaydet", type="primary"):
            _update_explanations(student_id_clean, active_session, explanations)
            st.success("Yorumlariniz kaydedildi!")
            st.balloons()


if __name__ == "__main__":
    main()

