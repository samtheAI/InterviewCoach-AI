import json

import pytest

from interview_engine import (
    InterviewCoachError,
    export_sessions,
    import_sessions,
    normalize_feedback,
    parse_json_object,
    session_averages,
)


def test_parse_json_object_accepts_fenced_json():
    assert parse_json_object('```json\n{"question": "Tell me about a launch."}\n```')["question"] == "Tell me about a launch."


def test_parse_json_object_extracts_embedded_object():
    assert parse_json_object('Result: {"score": 82} done')["score"] == 82


def test_feedback_scores_are_normalized_and_bounded():
    feedback = normalize_feedback(
        {
            "overall_score": 108,
            "verdict": "unexpected",
            "scores": {"relevance": 90, "structure": -4, "evidence": "75", "clarity": 88.4},
            "strengths": ["Specific example"],
            "improvements": "Add a metric",
        }
    )
    assert feedback["overall_score"] == 100
    assert feedback["scores"] == {"relevance": 90, "structure": 0, "evidence": 75, "clarity": 88}
    assert feedback["verdict"] == "Strong"
    assert feedback["improvements"] == ["Add a metric"]


def test_session_averages():
    turns = [
        {"feedback": {"overall_score": 60, "scores": {"relevance": 70, "structure": 50, "evidence": 40, "clarity": 80}}},
        {"feedback": {"overall_score": 80, "scores": {"relevance": 90, "structure": 70, "evidence": 60, "clarity": 100}}},
    ]
    assert session_averages(turns) == {"overall": 70, "relevance": 80, "structure": 60, "evidence": 50, "clarity": 90}


def test_import_sessions_rejects_unrelated_json():
    with pytest.raises(InterviewCoachError):
        import_sessions(json.dumps({"hello": "world"}))


def test_import_sessions_keeps_valid_entries():
    payload = {"sessions": {"abc": {"config": {"role": "PM"}, "turns": []}, "bad": "nope"}}
    assert list(import_sessions(json.dumps(payload))) == ["abc"]


def test_export_sessions_omits_cached_audio_bytes():
    exported = export_sessions({"abc": {"config": {"role": "PM"}, "question_audio": b"wave"}})
    assert json.loads(exported)["sessions"]["abc"]["question_audio"] is None
