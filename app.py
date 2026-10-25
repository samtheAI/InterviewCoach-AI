from __future__ import annotations

import hashlib
import html
import json
import os
import uuid
from datetime import datetime
from typing import Any

import streamlit as st

from interview_engine import (
    CHAT_MODEL,
    STT_MODEL,
    TTS_MODEL,
    InterviewConfig,
    build_final_report,
    evaluate_answer,
    export_sessions,
    generate_question,
    get_client,
    import_sessions,
    safe_error_message,
    session_averages,
    speech_for_text,
    transcribe_audio,
    utc_now_iso,
)


st.set_page_config(
    page_title="InterviewOS · AI Voice Coach",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
  --violet: #8b5cf6;
  --cyan: #22d3ee;
  --pink: #f472b6;
  --ink: #f8fafc;
  --muted: #9ba8bd;
  --panel: rgba(15, 18, 32, .62);
  --line: rgba(255,255,255,.10);
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
  color: var(--ink);
  background:
    radial-gradient(900px 600px at 8% -10%, rgba(139,92,246,.24), transparent 62%),
    radial-gradient(760px 520px at 100% 10%, rgba(34,211,238,.15), transparent 62%),
    radial-gradient(680px 520px at 72% 96%, rgba(244,114,182,.11), transparent 65%),
    #070912;
}
.stApp::before {
  content:""; position:fixed; inset:0; pointer-events:none; opacity:.16;
  background-image: linear-gradient(rgba(255,255,255,.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.03) 1px, transparent 1px);
  background-size: 48px 48px;
  mask-image: linear-gradient(to bottom, black, transparent 78%);
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1280px; padding-top: 2.1rem; padding-bottom: 5rem; }

h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; }
p { color:#c7d0df; }
[data-testid="stSidebar"] { background: rgba(8,10,20,.76); border-right:1px solid var(--line); backdrop-filter:blur(24px); }
[data-testid="stSidebar"] .block-container { padding-top:1.6rem; }

.brand { display:flex; align-items:center; gap:.75rem; margin-bottom:.4rem; }
.brand-mark { width:38px;height:38px;border-radius:13px;display:grid;place-items:center;font-weight:800;
  background:linear-gradient(135deg,var(--violet),var(--cyan));box-shadow:0 0 35px rgba(139,92,246,.42); }
.brand-name { font-family:'Manrope';font-weight:800;font-size:1.08rem;letter-spacing:-.03em; }
.eyebrow { text-transform:uppercase;letter-spacing:.16em;font-size:.69rem;font-weight:700;color:#a9b5ca;margin-bottom:.7rem; }
.gradient-text { background:linear-gradient(100deg,#fff 8%,#c8b9ff 52%,#79e8f4);-webkit-background-clip:text;color:transparent; }
.hero-title { font-family:'Manrope';font-size:clamp(2.8rem,6.2vw,5.8rem);font-weight:800;line-height:.97;letter-spacing:-.065em;margin:.3rem 0 1.2rem;max-width:980px; }
.hero-sub { max-width:720px;font-size:1.08rem;line-height:1.65;color:#aeb9cc;margin-bottom:1.8rem; }

.glass, div[data-testid="stForm"] {
  border:1px solid var(--line); border-radius:24px;
  background:linear-gradient(145deg,rgba(255,255,255,.075),rgba(255,255,255,.025));
  backdrop-filter:blur(24px); -webkit-backdrop-filter:blur(24px);
  box-shadow:0 28px 80px rgba(0,0,0,.26), inset 0 1px 0 rgba(255,255,255,.08);
}
div[data-testid="stForm"] { padding:1.2rem 1.2rem .4rem; }
.status-row {display:flex;gap:.55rem;flex-wrap:wrap;margin:1rem 0 2rem;}
.pill {display:inline-flex;align-items:center;gap:.45rem;border:1px solid var(--line);background:rgba(255,255,255,.045);padding:.47rem .72rem;border-radius:999px;color:#cbd5e1;font-size:.78rem;font-weight:600;}
.dot {width:7px;height:7px;border-radius:50%;background:#34d399;box-shadow:0 0 12px #34d399;}
.section-kicker { color:#a78bfa;font-size:.7rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;margin-bottom:.45rem; }
.section-title { font-family:'Manrope';font-weight:750;font-size:1.65rem;letter-spacing:-.04em;margin-bottom:.3rem; }
.section-copy { color:#93a0b5;font-size:.92rem;margin-bottom:1rem; }

div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea,
div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] input {
  background:rgba(7,10,20,.62) !important;border:1px solid rgba(255,255,255,.11) !important;color:#eef2ff !important;border-radius:13px !important;
}
div[data-testid="stTextArea"] textarea:focus, div[data-testid="stTextInput"] input:focus {border-color:rgba(139,92,246,.75)!important;box-shadow:0 0 0 3px rgba(139,92,246,.12)!important;}
label, .stRadio label { color:#d8deeb !important;font-weight:600!important; }

.stButton > button, .stFormSubmitButton > button {
  border:1px solid rgba(255,255,255,.12); border-radius:13px; font-weight:700; min-height:2.9rem;
  background:rgba(255,255,255,.055); color:#eef2ff; transition:all .2s ease;
}
.stButton > button:hover { transform:translateY(-1px); border-color:rgba(139,92,246,.65); color:white; box-shadow:0 12px 28px rgba(0,0,0,.2); }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  color:white; border:0; background:linear-gradient(105deg,#7c3aed,#8b5cf6 48%,#06b6d4); box-shadow:0 14px 36px rgba(124,58,237,.28);
}
.stButton > button[kind="primary"]:hover { box-shadow:0 18px 44px rgba(124,58,237,.4); }

.question-card { position:relative;overflow:hidden;padding:1.55rem 1.65rem;border-radius:24px;border:1px solid rgba(167,139,250,.22);background:linear-gradient(145deg,rgba(139,92,246,.14),rgba(34,211,238,.035));box-shadow:0 28px 80px rgba(0,0,0,.24); }
.question-card::after {content:"";position:absolute;width:230px;height:230px;border-radius:50%;right:-90px;top:-120px;background:rgba(34,211,238,.12);filter:blur(30px);}
.q-meta { display:flex;gap:.55rem;align-items:center;color:#b9c5d8;font-size:.75rem;text-transform:uppercase;letter-spacing:.09em;font-weight:700;margin-bottom:.9rem; }
.q-number {color:#67e8f9;}
.q-text {font-family:'Manrope';font-size:clamp(1.45rem,3.1vw,2.35rem);font-weight:700;line-height:1.25;letter-spacing:-.04em;color:white;max-width:940px;}
.listen {margin-top:1rem;color:#aab6c9;font-size:.88rem;}

.metric-card {padding:1rem 1.05rem;border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.045);height:100%;}
.metric-label {color:#93a0b5;font-size:.72rem;text-transform:uppercase;letter-spacing:.11em;font-weight:700;}
.metric-value {font-family:'Manrope';font-size:1.75rem;font-weight:800;margin-top:.25rem;}
.score-ring {width:128px;height:128px;border-radius:50%;display:grid;place-items:center;margin:.4rem auto 1.1rem;background:conic-gradient(#8b5cf6 calc(var(--score)*1%),rgba(255,255,255,.08) 0);position:relative;box-shadow:0 0 55px rgba(139,92,246,.2);}
.score-ring::before {content:"";position:absolute;inset:10px;border-radius:50%;background:#0d1020;}
.score-ring span {position:relative;font-family:'Manrope';font-size:2rem;font-weight:800;}
.feedback-card {border:1px solid var(--line);background:rgba(255,255,255,.038);border-radius:20px;padding:1.2rem 1.3rem;margin:.8rem 0;}
.feedback-card strong {color:white;}
.feedback-list {margin:.55rem 0 0;padding-left:1.15rem;color:#c9d2e2;}
.feedback-list li {margin:.4rem 0;line-height:1.5;}
.chat-answer {border-left:2px solid #22d3ee;background:rgba(34,211,238,.045);padding:.95rem 1rem;border-radius:0 14px 14px 0;color:#cbd5e1;font-size:.92rem;line-height:1.58;}

.session-card {border:1px solid var(--line);border-radius:14px;padding:.7rem .8rem;background:rgba(255,255,255,.035);margin:.45rem 0;}
.session-role {font-weight:700;color:#eef2ff;font-size:.88rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.session-meta {font-size:.71rem;color:#8e9bb0;margin-top:.18rem;}
.progress-line {height:5px;background:rgba(255,255,255,.07);border-radius:99px;overflow:hidden;margin:.5rem 0 .25rem;}
.progress-line span {display:block;height:100%;background:linear-gradient(90deg,#8b5cf6,#22d3ee);border-radius:99px;}

[data-testid="stAudioInput"] {border:1px solid var(--line);border-radius:18px;padding:.65rem;background:rgba(255,255,255,.035);}
[data-testid="stAudio"] {border-radius:14px;overflow:hidden;}
div[data-testid="stAlert"] {border-radius:16px;background:rgba(255,255,255,.055);border:1px solid var(--line);}
.footer {text-align:center;color:#6f7c92;font-size:.76rem;margin-top:4rem;padding-top:1.3rem;border-top:1px solid rgba(255,255,255,.07);}
#MainMenu, footer {visibility:hidden;}

@media (max-width:700px){.block-container{padding:1.2rem .9rem 4rem}.hero-title{font-size:3rem}.question-card{padding:1.2rem}.q-text{font-size:1.45rem}}
</style>
"""


st.markdown(APP_CSS, unsafe_allow_html=True)


SAMPLE_JD = """We are hiring a Senior Product Manager to own our AI productivity platform.
You will define product strategy, discover high-value customer problems, partner with engineering
and design, prioritize an ambiguous roadmap, and drive adoption. The ideal candidate has 5+ years
of product experience, strong analytical judgment, excellent stakeholder communication, and a
track record of shipping B2B SaaS products with measurable business impact. Experience with AI/ML
products, experimentation, enterprise customers, and responsible AI is preferred."""

VOICE_LABELS = {
    "Autumn · warm": "autumn",
    "Diana · composed": "diana",
    "Hannah · clear": "hannah",
    "Austin · confident": "austin",
    "Daniel · grounded": "daniel",
    "Troy · energetic": "troy",
}


def e(value: Any) -> str:
    return html.escape(str(value))


def configured_api_key() -> str | None:
    """Read Streamlit Secrets when present, otherwise let the SDK use the env."""

    if os.getenv("GROQ_API_KEY"):
        return os.getenv("GROQ_API_KEY")
    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def init_state() -> None:
    defaults = {
        "sessions": {},
        "active_session_id": None,
        "screen": "setup",
        "draft_answer": "",
        "last_audio_digest": None,
        "pending_import": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def config_from_dict(data: dict[str, Any]) -> InterviewConfig:
    return InterviewConfig(
        role=str(data.get("role", "")),
        company=str(data.get("company", "")),
        job_description=str(data.get("job_description", "")),
        candidate_context=str(data.get("candidate_context", "")),
        interview_type=str(data.get("interview_type", "Mixed interview")),
        difficulty=str(data.get("difficulty", "Realistic")),
        question_count=int(data.get("question_count", 5)),
        voice=str(data.get("voice", "autumn")),
    )


def current_session() -> dict[str, Any] | None:
    session_id = st.session_state.active_session_id
    return st.session_state.sessions.get(session_id) if session_id else None


def save_session(session: dict[str, Any]) -> None:
    session["updated_at"] = utc_now_iso()
    st.session_state.sessions[session["id"]] = session


def make_session(config: InterviewConfig, question: dict[str, str]) -> dict[str, Any]:
    session_id = uuid.uuid4().hex[:10]
    now = utc_now_iso()
    return {
        "id": session_id,
        "config": config.__dict__,
        "status": "active",
        "created_at": now,
        "updated_at": now,
        "turns": [],
        "current_question": question,
        "question_audio": None,
        "final_report": None,
    }


def session_label(session: dict[str, Any]) -> str:
    config = session.get("config", {})
    return str(config.get("role") or "Untitled interview")


def question_audio(session: dict[str, Any], client: Any) -> bytes | None:
    cached = session.get("question_audio")
    if isinstance(cached, bytes):
        return cached
    try:
        data = speech_for_text(
            client,
            session.get("current_question", {}).get("question", ""),
            session.get("config", {}).get("voice", "autumn"),
        )
        session["question_audio"] = data
        save_session(session)
        return data
    except Exception:
        return None


def render_brand() -> None:
    st.markdown(
        '<div class="brand"><div class="brand-mark">◈</div><div><div class="brand-name">InterviewOS</div>'
        '<div style="font-size:.66rem;color:#77849a;letter-spacing:.11em;text-transform:uppercase">Voice practice studio</div></div></div>',
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        render_brand()
        st.caption("Practice room")
        if st.button("＋ New interview", use_container_width=True, type="primary"):
            st.session_state.active_session_id = None
            st.session_state.screen = "setup"
            st.session_state.draft_answer = ""
            st.rerun()

        sessions = sorted(
            st.session_state.sessions.values(),
            key=lambda item: item.get("updated_at", ""),
            reverse=True,
        )
        if sessions:
            st.markdown('<div class="eyebrow" style="margin-top:1.4rem">Your sessions</div>', unsafe_allow_html=True)
        for session in sessions:
            turns = len(session.get("turns", []))
            total = max(1, int(session.get("config", {}).get("question_count", 5)))
            width = min(100, int(turns / total * 100))
            role = e(session_label(session))
            status = e(session.get("status", "active").title())
            st.markdown(
                f'<div class="session-card"><div class="session-role">{role}</div>'
                f'<div class="session-meta">{turns}/{total} answered · {status}</div>'
                f'<div class="progress-line"><span style="width:{width}%"></span></div></div>',
                unsafe_allow_html=True,
            )
            if st.button("Open", key=f"open_{session['id']}", use_container_width=True):
                st.session_state.active_session_id = session["id"]
                st.session_state.screen = "report" if session.get("status") == "complete" else "interview"
                st.session_state.draft_answer = ""
                st.rerun()

        st.markdown('<div class="eyebrow" style="margin-top:1.7rem">Session vault</div>', unsafe_allow_html=True)
        if st.session_state.sessions:
            st.download_button(
                "↓ Export sessions",
                export_sessions(st.session_state.sessions),
                file_name=f"interviewos-sessions-{datetime.now().date()}.json",
                mime="application/json",
                use_container_width=True,
            )
        uploaded = st.file_uploader("Import sessions", type=["json"], label_visibility="collapsed")
        if uploaded and st.button("Import file", use_container_width=True):
            try:
                imported = import_sessions(uploaded.getvalue().decode("utf-8"))
                st.session_state.sessions.update(imported)
                st.success(f"Imported {len(imported)} session(s).")
                st.rerun()
            except Exception as exc:
                st.error(safe_error_message(exc))
        st.caption("Export sessions before closing your browser to keep a portable backup.")


def render_setup() -> None:
    st.markdown('<div class="eyebrow">AI interview simulator · voice enabled</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title"><span class="gradient-text">Turn the job description</span><br>into your next offer.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">A focused practice room that interviews you like the real panel, listens to every answer, and shows exactly how to improve.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="status-row"><span class="pill"><span class="dot"></span> Live AI coach</span>'
        '<span class="pill">◉ Voice answers</span><span class="pill">◌ Spoken questions</span>'
        '<span class="pill">⌁ Session history</span></div>',
        unsafe_allow_html=True,
    )

    with st.form("setup_form"):
        st.markdown('<div class="section-kicker">01 · Target</div><div class="section-title">Build your interview room</div><div class="section-copy">Paste the real posting. The coach will map questions, signals, and feedback to it.</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1.25, 1])
        with c1:
            role = st.text_input("Target role", placeholder="e.g. Senior Product Manager")
        with c2:
            company = st.text_input("Company (optional)", placeholder="e.g. Acme AI")
        job_description = st.text_area(
            "Job description",
            value=st.session_state.get("sample_jd", ""),
            height=210,
            placeholder="Paste responsibilities, qualifications, and company context…",
        )
        candidate_context = st.text_area(
            "Your background (optional)",
            height=110,
            placeholder="Paste a resume summary or the experiences you want the coach to use…",
        )

        st.markdown('<div class="section-kicker" style="margin-top:1rem">02 · Format</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1.2, 1, .8, 1.05])
        with c1:
            interview_type = st.selectbox(
                "Interview style",
                ["Mixed interview", "Behavioral", "Hiring manager", "Technical / role skills", "Case & strategy", "HR screening"],
            )
        with c2:
            difficulty = st.selectbox("Difficulty", ["Supportive", "Realistic", "Challenging"], index=1)
        with c3:
            question_count = st.selectbox("Questions", [3, 5, 7, 10], index=1)
        with c4:
            voice_label = st.selectbox("Interviewer voice", list(VOICE_LABELS), index=0)

        submitted = st.form_submit_button("Launch practice room →", type="primary", use_container_width=True)

    c1, _ = st.columns([1, 3])
    with c1:
        if st.button("Use a sample role", use_container_width=True):
            st.session_state.sample_jd = SAMPLE_JD
            st.rerun()

    if submitted:
        if not role.strip() or len(job_description.strip()) < 80:
            st.error("Add a target role and a fuller job description (at least 80 characters).")
            return
        config = InterviewConfig(
            role=role.strip(), company=company.strip(), job_description=job_description.strip(),
            candidate_context=candidate_context.strip(), interview_type=interview_type,
            difficulty=difficulty, question_count=int(question_count), voice=VOICE_LABELS[voice_label],
        )
        try:
            with st.spinner("Designing your interview…"):
                client = get_client(configured_api_key())
                first_question = generate_question(client, config, [])
                session = make_session(config, first_question)
                save_session(session)
                st.session_state.active_session_id = session["id"]
                st.session_state.screen = "interview"
                st.session_state.sample_jd = ""
                st.rerun()
        except Exception as exc:
            st.error(safe_error_message(exc), icon="⚠️")


def feedback_list(title: str, items: list[str], icon: str) -> None:
    rows = "".join(f"<li>{e(item)}</li>" for item in items)
    st.markdown(f'<div class="feedback-card"><strong>{icon} {e(title)}</strong><ul class="feedback-list">{rows}</ul></div>', unsafe_allow_html=True)


def render_turn_feedback(turn: dict[str, Any]) -> None:
    fb = turn.get("feedback", {})
    score = int(fb.get("overall_score", 0))
    st.markdown('<div class="section-kicker">Coach feedback</div>', unsafe_allow_html=True)
    c0, c1 = st.columns([.7, 2.3])
    with c0:
        st.markdown(f'<div class="score-ring" style="--score:{score}"><span>{score}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center;font-weight:700;color:#d8deeb">{e(fb.get("verdict", ""))}</div>', unsafe_allow_html=True)
    with c1:
        cols = st.columns(4)
        for col, key in zip(cols, ("relevance", "structure", "evidence", "clarity")):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{key}</div><div class="metric-value">{int(fb.get("scores", {}).get(key, 0))}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="feedback-card"><strong>Coach read</strong><div style="color:#c7d0df;margin-top:.55rem;line-height:1.6">{e(fb.get("coach_note", ""))}</div></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        feedback_list("What landed", fb.get("strengths", []), "↗")
    with c2:
        feedback_list("Make it sharper", fb.get("improvements", []), "⌁")
    if fb.get("improved_answer"):
        st.markdown(
            f'<div class="feedback-card"><strong>Stronger answer</strong>'
            f'<div style="color:#c7d0df;margin-top:.55rem;line-height:1.65">{e(fb.get("improved_answer", ""))}</div></div>',
            unsafe_allow_html=True,
        )
    if fb.get("follow_up_probe"):
        st.caption(f"A real interviewer may ask: {fb['follow_up_probe']}")


def finish_session(session: dict[str, Any], client: Any) -> None:
    if not session.get("turns"):
        st.warning("Answer at least one question before finishing.")
        return
    with st.spinner("Writing your final debrief…"):
        session["final_report"] = build_final_report(
            client, config_from_dict(session["config"]), session["turns"]
        )
        session["status"] = "complete"
        save_session(session)
        st.session_state.screen = "report"
        st.rerun()


def render_interview() -> None:
    session = current_session()
    if not session:
        st.session_state.screen = "setup"
        st.rerun()
    config = config_from_dict(session["config"])
    turns = session.get("turns", [])
    number = len(turns) + 1
    total = config.question_count

    top1, top2 = st.columns([3, 1])
    with top1:
        st.markdown(f'<div class="eyebrow">{e(config.company or "Practice room")} · {e(config.role)}</div>', unsafe_allow_html=True)
        st.markdown(f"## Interview in progress")
    with top2:
        st.metric("Progress", f"{min(number, total)} / {total}")

    question = session.get("current_question", {})
    st.markdown(
        f'<div class="question-card"><div class="q-meta"><span class="q-number">Question {number:02d}</span><span>·</span><span>{e(question.get("competency", ""))}</span></div>'
        f'<div class="q-text">{e(question.get("question", ""))}</div>'
        f'<div class="listen">Interviewer is listening for: {e(question.get("listen_for", ""))}</div></div>',
        unsafe_allow_html=True,
    )

    try:
        client = get_client(configured_api_key())
    except Exception as exc:
        st.error(safe_error_message(exc))
        return

    audio = question_audio(session, client)
    if audio:
        st.audio(audio, format="audio/wav")
    else:
        st.caption("Voice playback is temporarily unavailable; the written question is ready.")

    if turns:
        with st.expander(f"Feedback from question {len(turns)}", expanded=True):
            latest = turns[-1]
            st.markdown(f'<div class="chat-answer"><strong>Your answer</strong><br>{e(latest.get("answer", ""))}</div>', unsafe_allow_html=True)
            render_turn_feedback(latest)

    st.markdown('<div class="section-kicker" style="margin-top:1.7rem">Your answer</div><div class="section-title">Speak naturally. Be specific.</div><div class="section-copy">Record a response or type one. Voice is transcribed automatically before scoring.</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Answer mode",
        ["Voice", "Type"],
        horizontal=True,
        label_visibility="collapsed",
    )
    answer = ""
    recorded = None
    if mode == "Voice":
        if hasattr(st, "audio_input"):
            recorded = st.audio_input("Record your answer", sample_rate=16000)
        else:  # Compatibility fallback for older self-hosted Streamlit releases.
            recorded = st.file_uploader(
                "Upload a recorded answer",
                type=["wav", "mp3", "m4a", "ogg", "webm"],
            )
        if recorded:
            digest = hashlib.sha256(recorded.getvalue()).hexdigest()
            if digest != st.session_state.last_audio_digest:
                st.info("Recording ready. Submit when you are happy with it.")
    else:
        answer = st.text_area("Answer", value=st.session_state.draft_answer, height=180, placeholder="Use STAR: situation, task, action, result…")

    c1, c2 = st.columns([3, 1])
    with c1:
        submit = st.button("Submit answer for coaching →", type="primary", use_container_width=True)
    with c2:
        finish = st.button("Finish session", use_container_width=True)

    if finish:
        try:
            finish_session(session, client)
        except Exception as exc:
            st.error(safe_error_message(exc), icon="⚠️")

    if submit:
        try:
            with st.spinner("Listening and coaching…"):
                if mode == "Voice":
                    if not recorded:
                        st.warning("Record an answer first, or switch to Type.")
                        return
                    raw = recorded.getvalue()
                    answer = transcribe_audio(client, raw, recorded.name or "answer.wav")
                    st.session_state.last_audio_digest = hashlib.sha256(raw).hexdigest()
                else:
                    answer = answer.strip()
                if len(answer) < 20:
                    st.warning("Give a fuller answer so the coach has enough evidence to score.")
                    return

                feedback = evaluate_answer(client, config, question, answer)
                turn = {"question": question, "answer": answer, "feedback": feedback, "answered_at": utc_now_iso()}
                session["turns"].append(turn)
                session["question_audio"] = None
                if len(session["turns"]) >= total:
                    session["final_report"] = build_final_report(client, config, session["turns"])
                    session["status"] = "complete"
                    save_session(session)
                    st.session_state.screen = "report"
                else:
                    session["current_question"] = generate_question(client, config, session["turns"])
                    save_session(session)
                    st.session_state.draft_answer = ""
                st.rerun()
        except Exception as exc:
            st.error(safe_error_message(exc), icon="⚠️")


def render_report() -> None:
    session = current_session()
    if not session:
        st.session_state.screen = "setup"
        st.rerun()
    config = config_from_dict(session["config"])
    turns = session.get("turns", [])
    report = session.get("final_report") or {}
    averages = report.get("averages") or session_averages(turns)

    st.markdown('<div class="eyebrow">Final debrief · session complete</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-title" style="font-size:clamp(2.7rem,5vw,4.8rem)"><span class="gradient-text">{e(report.get("readiness", "Interview debrief"))}</span><br>for {e(config.role)}.</div>', unsafe_allow_html=True)
    if report.get("headline"):
        st.markdown(f'<div class="hero-sub">{e(report["headline"])}</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    labels = [("overall", "Overall"), ("relevance", "Relevance"), ("structure", "Structure"), ("evidence", "Evidence"), ("clarity", "Clarity")]
    for col, (key, label) in zip(cols, labels):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{int(averages.get(key,0))}</div></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="feedback-card"><strong>↗ Strongest signal</strong><div style="margin-top:.6rem;color:#cbd5e1;line-height:1.6">{e(report.get("strongest_signal", "Complete more questions to reveal your strongest signal."))}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="feedback-card"><strong>⌁ Top priority</strong><div style="margin-top:.6rem;color:#cbd5e1;line-height:1.6">{e(report.get("top_priority", "Keep practicing concise, evidence-led answers."))}</div></div>', unsafe_allow_html=True)

    feedback_list("Your 3-step action plan", report.get("action_plan", []), "◈")
    if report.get("closing_note"):
        st.info(report["closing_note"])

    st.markdown("### Answer review")
    for index, turn in enumerate(turns, start=1):
        score = turn.get("feedback", {}).get("overall_score", 0)
        with st.expander(f"{index:02d} · {turn.get('question', {}).get('competency', 'Question')} · {score}/100"):
            st.markdown(f"**{turn.get('question', {}).get('question', '')}**")
            st.write(turn.get("answer", ""))
            render_turn_feedback(turn)

    transcript = {
        "product": "InterviewOS",
        "role": config.role,
        "company": config.company,
        "config": session["config"],
        "final_report": report,
        "turns": turns,
    }
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("↓ Download this debrief", json.dumps(transcript, indent=2), f"{config.role.lower().replace(' ','-')}-debrief.json", "application/json", use_container_width=True)
    with c2:
        if st.button("Start another interview", type="primary", use_container_width=True):
            st.session_state.active_session_id = None
            st.session_state.screen = "setup"
            st.rerun()


def main() -> None:
    init_state()
    render_sidebar()
    screen = st.session_state.screen
    if screen == "interview":
        render_interview()
    elif screen == "report":
        render_report()
    else:
        render_setup()
    st.markdown(
        f'<div class="footer">InterviewOS · {e(CHAT_MODEL)} · STT {e(STT_MODEL)} · TTS {e(TTS_MODEL)}<br>AI feedback is practice guidance—not a hiring decision.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
