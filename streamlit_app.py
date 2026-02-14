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
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import qrcode
import streamlit as st
from matplotlib import patches
from matplotlib import pyplot as plt

ENV_PATH = Path(".env")
TEACHER_CODE_HASH_KEY = "TEACHER_CODE_HASH"
TEACHER_CODE_KEY = "TEACHER_CODE"
PUBLIC_BASE_URL_KEY = "PUBLIC_BASE_URL"
APP_TIMEZONE_KEY = "APP_TIMEZONE"
DEFAULT_APP_TIMEZONE = "Europe/Istanbul"
TEACHER_OPTIONS = [
    "Prof. Dr. Yalçın ATA",
    "Prof. Dr. Arif DEMİR",
    "Dr. Öğr. Üyesi Emel GÜVEN",
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
}


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


def _sanitize_overlay_text(student_id: str) -> str:
    """Overlay icin guvenli metin olusturur."""
    cleaned = "".join(ch for ch in student_id.strip() if ch.isalnum() or ch in "-_.")
    return cleaned[:32] if cleaned else "ID"


def build_student_id_overlay_data_uri(student_id: str) -> str:
    """Ogrenci numarasini 58 derece watermark SVG data URI'sine donusturur."""
    token = html.escape(_sanitize_overlay_text(student_id))
    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='980' height='560' viewBox='0 0 980 560'>
  <rect width='100%' height='100%' fill='transparent'/>
  <g transform='rotate(58 490 280)'>
    <text x='140' y='320'
          font-family='monospace'
          font-size='68'
          font-weight='800'
          letter-spacing='3'
          fill='rgba(255, 82, 82, 0.27)'>{token}</text>
    <text x='145' y='325'
          font-family='monospace'
          font-size='68'
          font-weight='800'
          letter-spacing='3'
          fill='rgba(73, 229, 255, 0.25)'>{token}</text>
    <text x='136' y='315'
          font-family='monospace'
          font-size='68'
          font-weight='800'
          letter-spacing='3'
          fill='rgba(255, 221, 107, 0.24)'>{token}</text>
  </g>
</svg>
""".strip()
    return f"data:image/svg+xml;utf8,{quote(svg, safe='')}"


def inject_student_id_overlay(student_id: str) -> None:
    """Ogrenci numarasina bagli OCR zorlastirici overlay'i aktif eder."""
    data_uri = build_student_id_overlay_data_uri(student_id)
    st.markdown(
        f"""
        <style>
        :root {{
            --student-id-overlay: url("{data_uri}");
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    """Bitis zamani gecen aktif oturumu otomatik kapatir."""
    end_at = parse_iso_datetime(str(control.get("session_end") or ""))
    if not control.get("is_open") or end_at is None:
        return control

    if now_in_app_timezone() >= end_at:
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
                f"iki hatanin birlikte olasiligi P(Uâˆ©D)={p_ud:.3f}. "
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
                f"birlikte olasilik P(Yâˆ©F)={p_yf:.3f}. "
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


def evaluate_quiz_availability(control: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    """Quizin o an ogrenciye acik olup olmadigini belirler."""
    normalized = dict(control)
    now = now_in_app_timezone()
    end_at = parse_iso_datetime(str(normalized.get("session_end") or ""))
    if normalized.get("is_open") and end_at is not None and now >= end_at:
        normalized["is_open"] = False
        normalized["closed_at"] = _now_iso()
        save_quiz_control(normalized)
        return False, "Quiz suresi doldu. Oturum otomatik kapatildi.", normalized

    active_session = str(normalized.get("active_session") or "")
    if not normalized.get("is_open") or not active_session:
        return (
            False,
            f"Quiz su an kapali. Baz alinan saat: {time_basis_for_ui()} | Sistem saati: {format_dt_obj_for_ui(now)}",
            normalized,
        )

    start_at = parse_iso_datetime(str(normalized.get("session_start") or ""))
    if start_at is not None and now < start_at:
        return (
            False,
            "Quiz henuz baslamadi. "
            f"Sistem saati: {format_dt_obj_for_ui(now)} | "
            f"Baslangic: {format_dt_for_ui(normalized['session_start'])}",
            normalized,
        )
    return True, "", normalized


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
        st.caption(f"Bitis: {format_dt_for_ui(control['session_end'])}")
        if control["opened_at"]:
            st.caption(f"Acilis zamani: {format_dt_for_ui(control['opened_at'])}")

        start_at = parse_iso_datetime(control["session_start"])
        end_at = parse_iso_datetime(control["session_end"])
        if start_at is not None and now < start_at:
            st.info("Oturum planlandi ancak henuz baslamadi.")
        elif end_at is not None:
            remaining_seconds = int((end_at - now).total_seconds())
            if remaining_seconds > 0:
                remaining_minutes = max(1, math.ceil(remaining_seconds / 60))
                st.info(f"Kalan sure: yaklasik {remaining_minutes} dakika")

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

        default_start = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        default_end = default_start + timedelta(minutes=45)
        default_start_time = default_start.time().replace(tzinfo=None)
        default_end_time = default_end.time().replace(tzinfo=None)
        with st.expander("Yeni Oturum Zaman Penceresi", expanded=True):
            start_date = st.date_input("Baslangic tarihi", value=default_start.date(), key="session_start_date")
            start_time = st.time_input("Baslangic saati", value=default_start_time, key="session_start_time")
            end_date = st.date_input("Bitis tarihi", value=default_end.date(), key="session_end_date")
            end_time = st.time_input("Bitis saati", value=default_end_time, key="session_end_time")

        tz, _, _ = resolve_app_timezone()
        start_at = datetime.combine(start_date, start_time).replace(tzinfo=tz)
        end_at = datetime.combine(end_date, end_time).replace(tzinfo=tz)
        st.caption(f"Secilen baslangic: {format_dt_obj_for_ui(start_at)}")
        st.caption(f"Secilen bitis: {format_dt_obj_for_ui(end_at)}")
        if end_at <= start_at:
            st.error("Bitis zamani, baslangic zamanindan sonra olmalidir.")
        elif st.button("Quizi Ac (Yeni Oturum)", type="primary", use_container_width=True, key="open_quiz_btn"):
            save_quiz_control(
                {
                    "is_open": True,
                    "active_session": _now_session_id(),
                    "opened_at": _now_iso(),
                    "closed_at": "",
                    "session_start": start_at.isoformat(timespec="seconds"),
                    "session_end": end_at.isoformat(timespec="seconds"),
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


def render_question_card(title: str, body: str, index: int):
    st.markdown(
        f"""
        <div class="q-card">
            <div class="q-title">Soru {index}: {title}</div>
            <div class="q-body">{body}</div>
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
            --student-id-overlay: none;
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
        @keyframes ocr-luma-jitter {
            0% {
                background-position: 0 0, 0 0, 0 0, 0 0;
            }
            25% {
                background-position: 0.8px -0.6px, -0.5px 0.7px, 1px 0.5px, -0.6px -0.8px;
            }
            50% {
                background-position: -0.7px 0.5px, 0.6px -0.7px, -0.8px 0.4px, 0.7px -0.6px;
            }
            75% {
                background-position: 0.6px 0.7px, -0.7px -0.5px, 0.5px -0.9px, -0.5px 0.8px;
            }
            100% {
                background-position: 0 0, 0 0, 0 0, 0 0;
            }
        }
        @keyframes ocr-line-jitter {
            0% {
                transform: translate3d(0, 0, 0);
            }
            33% {
                transform: translate3d(0.5px, -0.3px, 0);
            }
            66% {
                transform: translate3d(-0.4px, 0.4px, 0);
            }
            100% {
                transform: translate3d(0, 0, 0);
            }
        }
        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 2147482999;
            mix-blend-mode: soft-light;
            opacity: 0.32;
            background-image:
                repeating-linear-gradient(
                    0deg,
                    rgba(255, 255, 255, 0.050) 0px,
                    rgba(255, 255, 255, 0.050) 1px,
                    rgba(255, 255, 255, 0.0) 1px,
                    rgba(255, 255, 255, 0.0) 6px
                ),
                repeating-linear-gradient(
                    90deg,
                    rgba(0, 0, 0, 0.050) 0px,
                    rgba(0, 0, 0, 0.050) 1px,
                    rgba(0, 0, 0, 0.0) 1px,
                    rgba(0, 0, 0, 0.0) 6px
                ),
                radial-gradient(circle at 23% 27%, rgba(255, 255, 255, 0.080) 1px, rgba(255, 255, 255, 0.0) 2px),
                radial-gradient(circle at 74% 72%, rgba(0, 0, 0, 0.080) 1px, rgba(0, 0, 0, 0.0) 2px);
            background-size: 6px 6px, 6px 6px, 14px 14px, 16px 16px;
            animation: ocr-luma-jitter 1.15s steps(3, end) infinite;
            will-change: background-position;
        }
        [data-testid="stAppViewContainer"]::after {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 2147483000;
            mix-blend-mode: normal;
            background-image:
                var(--student-id-overlay),
                repeating-linear-gradient(
                    20deg,
                    rgba(255, 255, 255, 0.20) 0px,
                    rgba(255, 255, 255, 0.20) 2px,
                    rgba(255, 255, 255, 0.0) 2px,
                    rgba(255, 255, 255, 0.0) 16px
                ),
                repeating-linear-gradient(
                    -20deg,
                    rgba(4, 10, 28, 0.16) 0px,
                    rgba(4, 10, 28, 0.16) 2px,
                    rgba(4, 10, 28, 0.0) 2px,
                    rgba(4, 10, 28, 0.0) 19px
                );
            background-size: 980px 560px, 16px 16px, 19px 19px;
            background-repeat: repeat, repeat, repeat;
            opacity: 1;
            animation: ocr-line-jitter 1.6s steps(2, end) infinite;
            will-change: transform;
        }
        .q-card {
            background: linear-gradient(140deg, rgba(255,255,255,0.10), rgba(255,255,255,0.03));
            border: 1px solid var(--edge);
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 10px;
            box-shadow: 0 18px 35px rgba(0,0,0,0.30);
        }
        .q-title {
            color: var(--ink);
            font-weight: 700;
            font-size: 1.02rem;
            margin-bottom: 6px;
        }
        .q-body {
            color: var(--ink);
            line-height: 1.55;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
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

    quiz_open, quiz_message, quiz_control = evaluate_quiz_availability(load_quiz_control())
    active_session = str(quiz_control.get("active_session") or "")
    if not quiz_open:
        st.warning(quiz_message)
        st.stop()

    st.info(
        f"Aktif quiz oturumu: {active_session} | "
        f"Baslangic: {format_dt_for_ui(quiz_control['session_start'])} | "
        f"Bitis: {format_dt_for_ui(quiz_control['session_end'])}"
    )
    st.caption(f"Baz alinan saat dilimi: {time_basis_for_ui()} | Sistem saati: {format_dt_obj_for_ui(now_in_app_timezone())}")

    student_name = st.text_input("Ad Soyad")
    student_id = st.text_input("Ogrenci Numarasi / ID")
    with st.expander("Ogretmen Bilgisi (Ac/Kapat)", expanded=False):
        teacher_name = st.selectbox(
            "Ogretmeninizi seciniz",
            options=["Seciniz..."] + TEACHER_OPTIONS,
            key="teacher_name_select",
        )

    student_name_clean = student_name.strip()
    student_id_clean = student_id.strip()
    teacher_name_clean = teacher_name.strip()
    if student_id_clean:
        inject_student_id_overlay(student_id_clean)

    if not student_name_clean or not student_id_clean:
        st.info("Lutfen ad soyad ve ogrenci numarasi giriniz.")
        st.stop()
    if teacher_name == "Seciniz...":
        st.info("Lutfen ogretmeninizi seciniz.")
        st.stop()

    existing_submission = get_existing_submission(student_id_clean, active_session)
    if existing_submission is not None:
        st.error("Bu oturumda daha once teslim yaptiniz. Tekrar puan degistiremezsiniz.")
        if pd.notna(existing_submission.get("score")):
            st.info(f"Kayitli puaniniz: {int(float(existing_submission['score']))}/100")
        st.stop()

    questions = question_bank(student_id_clean, active_session)

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
            answers.append({"given": user_value, "correct": q["answer"], "tolerance": q["tolerance"]})

            fig = render_question_visual(q)
            if fig is not None:
                st.pyplot(fig, clear_figure=True, width="stretch")
                plt.close(fig)

    if st.button("Cevaplari Gonder ve Puanla", type="primary"):
        if get_existing_submission(student_id_clean, active_session) is not None:
            st.error("Bu oturum icin kaydiniz zaten alinmis.")
            st.stop()

        scored: list[dict[str, Any]] = []
        score = 0
        for ans, q in zip(answers, questions):
            ok = check_answer(ans["given"], q["answer"], q["tolerance"])
            scored_item = {"given": ans["given"], "correct": q["answer"], "is_correct": ok}
            scored.append(scored_item)
            if ok:
                score += 20

        st.success(f"Toplam puan: {score}/100")
        st.info("Sonucunuz kaydedildi. Guvenlik nedeniyle dogru cevaplar ogrenci ekraninda gosterilmez.")

        record_result(student_id_clean, student_name_clean, teacher_name_clean, active_session, score, scored)
        st.balloons()


if __name__ == "__main__":
    main()
