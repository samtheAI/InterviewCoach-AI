# InterviewOS

InterviewOS is a voice-first AI interview coach built with Streamlit and Groq. Paste a job description, choose an interview style, and practice a realistic interview with spoken questions, voice transcription, evidence-based scoring, and a final readiness report.

**Live app:** [interviewos-ai-coach.streamlit.app](https://interviewos-ai-coach.streamlit.app/)

## Features

- Job-description-aware questions that adapt to earlier answers
- Voice answers using Whisper Large V3 Turbo speech-to-text
- Natural interviewer playback using Canopy Labs Orpheus TTS
- Qwen-powered scoring across relevance, structure, evidence, and clarity
- Improved-answer examples and realistic follow-up probes
- Multi-session manager with JSON export/import
- Responsive glassmorphism interface with typed-answer fallback
- Clean handling for authentication, timeout, and rate-limit errors

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="your-key"
streamlit run app.py
```

For Streamlit Community Cloud, add `GROQ_API_KEY` under **App settings → Secrets**.

## AI models

- Interview generation and coaching: `qwen/qwen3.8-27b`
- Speech to text: `whisper-large-v3-turbo`
- Text to speech: `canopylabs/orpheus-v1-english`

Model IDs can be overridden with the `GROQ_CHAT_MODEL`, `GROQ_STT_MODEL`, and `GROQ_TTS_MODEL` environment variables.

## Privacy

Recordings are sent to Groq only when the candidate submits an answer. Session data lives in the current Streamlit browser session; use the export button to keep a portable backup. API keys are never committed to the repository.

## Tests

```bash
pytest -q
```
