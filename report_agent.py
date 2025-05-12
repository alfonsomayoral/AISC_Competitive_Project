"""
report_agent.py – AI agent to analyze a transcribed conversation
and produce a structured Markdown report from data/transcripts.csv
"""

import csv, os, textwrap
from pathlib import Path
from typing import Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

TRANSCRIPT_CSV = Path("data/transcripts.csv")
REPORT_PATH = Path("data/conversation_report.md")
MODEL_NAME = "philschmid/bart-large-cnn-samsum"  # Public & tuned for dialogue

# ---------- 1. Read transcript from CSV ----------
def load_transcript(csv_path: Path, max_minutes: Optional[int] = None) -> str:
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found – run emotion_detection.py first.")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            t = float(row["timestamp_s"])
            if max_minutes and t / 60 > max_minutes:
                break
            rows.append(row["text"])
    return "\n".join(rows)

raw_dialogue = load_transcript(TRANSCRIPT_CSV)

# ---------- 2. Load summarizer pipeline ----------
print("[AGENT] Loading summarization model…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU only
)

# ---------- 3. Generate summary ----------
print("[AGENT] Generating summary...")
summary = summarizer(
    raw_dialogue,
    max_length=300,
    min_length=100,
    do_sample=False
)[0]["summary_text"]

# ---------- 4. Build Markdown report ----------
report_md = textwrap.dedent(f"""
    # 📝 Conversation Report

    **Generated automatically by AI Agent**

    ## 🧵 Summary
    {summary}

    ## 📌 Key Points
    - [Add key bullet points manually or extract from summary if needed]

    ## ✅ Action Items
    - [Add tasks or follow-ups based on summary]

    ## 🎭 Emotional Tone
    - Reflective, engaged, collaborative (inferred from overall structure)

    _Generated using model: `{MODEL_NAME}`_
""").strip()

# ---------- 5. Save the report ----------
REPORT_PATH.parent.mkdir(exist_ok=True)
REPORT_PATH.write_text(report_md, encoding="utf-8")

print(f"[AGENT] Report saved to {REPORT_PATH.resolve()}")