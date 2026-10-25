"""Core AI and session utilities for InterviewOS.

The Streamlit UI lives in ``app.py``. Keeping the API and data logic here makes
the important behaviour easy to test without starting a web server.
"""

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Iterable

from groq import Groq


CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "qwen/qwen3.8-27b")
STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
TTS_MODEL = os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")
MAX_TTS_CHARS = 190


class InterviewCoachError(RuntimeError):
    """A safe, user-facing error raised by the interview engine."""


@dataclass(frozen=True)
class InterviewConfig:
    role: str
    company: str
    job_description: str
    candidate_context: str
    interview_type: str
    difficulty: str
    question_count: int
    voice: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_client(api_key: str | None = None) -> Groq:
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise InterviewCoachError(
            "Groq is not configured. Add GROQ_API_KEY to Streamlit Secrets."
        )
    return Groq(api_key=key)


def _content_from_response(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError) as exc:
        raise InterviewCoachError("The coach returned an empty response. Please retry.") from exc


def parse_json_object(value: str) -> dict[str, Any]:
    """Parse an object even when a model wraps JSON in prose or code fences."""

    if not value or not value.strip():
        raise InterviewCoachError("The coach returned an empty response. Please retry.")

    cleaned = value.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise InterviewCoachError("The coach response was not valid JSON. Please retry.")
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise InterviewCoachError("The coach response was not valid JSON. Please retry.") from exc

    if not isinstance(parsed, dict):
        raise InterviewCoachError("The coach response had an unexpected format. Please retry.")
    return parsed


def _chat_json(
    client: Groq,
    *,
    system: str,
    user: str,
    temperature: float = 0.35,
    max_completion_tokens: int = 1800,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        response_format={"type": "json_object"},
    )
    return parse_json_object(_content_from_response(response))


def normalize_question(payload: dict[str, Any]) -> dict[str, str]:
    question = str(payload.get("question", "")).strip()
    if not question:
        raise InterviewCoachError("The coach did not create a question. Please retry.")
    return {
        "question": question,
        "competency": str(payload.get("competency", "Role fit")).strip() or "Role fit",
        "intent": str(payload.get("intent", "Assess job readiness")).strip(),
        "listen_for": str(payload.get("listen_for", "Specific evidence and clear reasoning")).strip(),
    }


def generate_question(
    client: Groq,
    config: InterviewConfig,
    previous_turns: Iterable[dict[str, Any]],
) -> dict[str, str]:
    completed = []
    for turn in previous_turns:
        completed.append(
            {
                "question": turn.get("question", {}).get("question", ""),
                "answer": str(turn.get("answer", ""))[:900],
                "score": turn.get("feedback", {}).get("overall_score"),
            }
        )

    system = """You are InterviewOS, an elite but fair interview coach and hiring manager.
Create exactly one realistic interview question tailored to the supplied job description.
Adapt to the candidate's previous answers. Do not repeat prior questions. Prefer questions
that reveal evidence, judgment, ownership, and role-specific depth. Return only JSON with:
question (string), competency (2-5 words), intent (one sentence), listen_for (one sentence).
Keep the spoken question under 175 characters so it sounds natural in text-to-speech."""

    user = json.dumps(
        {
            "role": config.role,
            "company": config.company,
            "interview_type": config.interview_type,
            "difficulty": config.difficulty,
            "job_description": config.job_description[:9000],
            "candidate_context": config.candidate_context[:5000],
            "completed_turns": completed,
            "question_number": len(completed) + 1,
            "total_questions": config.question_count,
        },
        ensure_ascii=False,
    )
    return normalize_question(_chat_json(client, system=system, user=user, max_completion_tokens=600))


def evaluate_answer(
    client: Groq,
    config: InterviewConfig,
    question: dict[str, str],
    answer: str,
) -> dict[str, Any]:
    system = """You are a rigorous interview coach. Evaluate the answer against the job
description and the question. Be candid, specific, and constructive. Reward concrete evidence,
clear structure, ownership, metrics, and trade-off awareness. Do not invent facts the candidate
did not say. Return only JSON with exactly these keys:
overall_score (integer 0-100), verdict (one of Strong, Solid, Developing, Needs work),
scores (object with relevance, structure, evidence, clarity; integers 0-100),
strengths (array of 2 concise strings), improvements (array of 2 concise strings),
coach_note (2-3 sentences), improved_answer (a strong first-person sample answer, max 170 words),
follow_up_probe (one short question)."""

    user = json.dumps(
        {
            "role": config.role,
            "company": config.company,
            "interview_type": config.interview_type,
            "difficulty": config.difficulty,
            "job_description": config.job_description[:9000],
            "candidate_context": config.candidate_context[:4000],
            "question": question,
            "candidate_answer": answer[:8000],
        },
        ensure_ascii=False,
    )
    payload = _chat_json(client, system=system, user=user, max_completion_tokens=1600)
    return normalize_feedback(payload)


def _bounded_score(value: Any) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def normalize_feedback(payload: dict[str, Any]) -> dict[str, Any]:
    raw_scores = payload.get("scores") if isinstance(payload.get("scores"), dict) else {}
    scores = {
        key: _bounded_score(raw_scores.get(key, 0))
        for key in ("relevance", "structure", "evidence", "clarity")
    }
    overall = _bounded_score(payload.get("overall_score", mean(scores.values())))
    verdict = str(payload.get("verdict", "Developing")).strip()
    if verdict not in {"Strong", "Solid", "Developing", "Needs work"}:
        verdict = "Strong" if overall >= 85 else "Solid" if overall >= 70 else "Developing" if overall >= 50 else "Needs work"

    def string_list(name: str) -> list[str]:
        value = payload.get(name, [])
        if not isinstance(value, list):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()][:3]

    return {
        "overall_score": overall,
        "verdict": verdict,
        "scores": scores,
        "strengths": string_list("strengths"),
        "improvements": string_list("improvements"),
        "coach_note": str(payload.get("coach_note", "")).strip(),
        "improved_answer": str(payload.get("improved_answer", "")).strip(),
        "follow_up_probe": str(payload.get("follow_up_probe", "")).strip(),
    }


def transcribe_audio(client: Groq, audio_bytes: bytes, filename: str = "answer.wav") -> str:
    if not audio_bytes:
        raise InterviewCoachError("No recording was received. Please record your answer again.")
    stream = io.BytesIO(audio_bytes)
    stream.name = filename
    result = client.audio.transcriptions.create(
        file=(filename, stream.read()),
        model=STT_MODEL,
        response_format="json",
        language="en",
        temperature=0.0,
        prompt="A professional job interview answer with role-specific terminology.",
    )
    text = str(getattr(result, "text", "")).strip()
    if not text:
        raise InterviewCoachError("I could not hear a clear answer. Please try recording again.")
    return text


def speech_for_text(client: Groq, text: str, voice: str = "autumn") -> bytes:
    spoken = re.sub(r"\s+", " ", text).strip()
    if not spoken:
        raise InterviewCoachError("There is no question to speak.")
    if len(spoken) > MAX_TTS_CHARS:
        spoken = spoken[:MAX_TTS_CHARS].rsplit(" ", 1)[0].rstrip(".,;:") + "."
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=voice,
        input=f"[professionally] {spoken}",
        response_format="wav",
    )
    if hasattr(response, "read"):
        data = response.read()
    elif hasattr(response, "content"):
        data = response.content
    else:
        try:
            data = bytes(response)
        except Exception as exc:  # pragma: no cover - SDK version fallback
            raise InterviewCoachError("The interviewer voice could not be generated.") from exc
    if not data:
        raise InterviewCoachError("The interviewer voice could not be generated.")
    return data


def session_averages(turns: Iterable[dict[str, Any]]) -> dict[str, int]:
    collected = list(turns)
    dimensions = ("overall", "relevance", "structure", "evidence", "clarity")
    if not collected:
        return {name: 0 for name in dimensions}

    def values(name: str) -> list[int]:
        if name == "overall":
            return [_bounded_score(t.get("feedback", {}).get("overall_score", 0)) for t in collected]
        return [_bounded_score(t.get("feedback", {}).get("scores", {}).get(name, 0)) for t in collected]

    return {name: int(round(mean(values(name)))) for name in dimensions}


def build_final_report(client: Groq, config: InterviewConfig, turns: list[dict[str, Any]]) -> dict[str, Any]:
    averages = session_averages(turns)
    compact_turns = [
        {
            "question": turn.get("question", {}).get("question", ""),
            "answer": str(turn.get("answer", ""))[:1200],
            "feedback": turn.get("feedback", {}),
        }
        for turn in turns
    ]
    system = """You are an executive interview coach writing a concise final debrief.
Use only the evidence supplied. Return only JSON with: headline (one sentence),
readiness (one of Ready, Nearly ready, More practice needed), strongest_signal (string),
top_priority (string), action_plan (array of exactly 3 practical strings),
closing_note (2 encouraging but candid sentences)."""
    user = json.dumps(
        {
            "role": config.role,
            "company": config.company,
            "job_description": config.job_description[:7000],
            "score_averages": averages,
            "interview_turns": compact_turns,
        },
        ensure_ascii=False,
    )
    report = _chat_json(client, system=system, user=user, max_completion_tokens=900)
    report["averages"] = averages
    return report


def safe_error_message(exc: Exception) -> str:
    message = str(exc)
    lower = message.lower()
    if "api_key" in lower or "authentication" in lower or "401" in lower:
        return "Groq rejected the configured API key. Check the Streamlit secret and retry."
    if "rate limit" in lower or "rate_limit" in lower or "429" in lower:
        return "Groq is receiving requests too quickly. Wait about 30 seconds, then retry."
    if "timeout" in lower or "timed out" in lower:
        return "The AI service took too long to respond. Please retry in a moment."
    if isinstance(exc, InterviewCoachError):
        return message
    return "Something went wrong while contacting the interview coach. Please retry."


def export_sessions(sessions: dict[str, Any]) -> str:
    def portable(value: Any) -> Any:
        if isinstance(value, bytes):
            return None
        if isinstance(value, dict):
            return {str(key): portable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [portable(item) for item in value]
        return value

    return json.dumps(
        {"schema_version": 1, "exported_at": utc_now_iso(), "sessions": portable(sessions)},
        ensure_ascii=False,
        indent=2,
    )


def import_sessions(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise InterviewCoachError("That session file is not valid JSON.") from exc
    sessions = payload.get("sessions") if isinstance(payload, dict) else None
    if not isinstance(sessions, dict):
        raise InterviewCoachError("That file does not contain InterviewOS sessions.")
    cleaned: dict[str, Any] = {}
    for key, session in sessions.items():
        if isinstance(session, dict) and isinstance(session.get("config"), dict):
            cleaned[str(key)] = session
    if not cleaned:
        raise InterviewCoachError("No usable interview sessions were found in that file.")
    return cleaned
