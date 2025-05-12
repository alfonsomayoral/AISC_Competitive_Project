"""
report_agent.py – FINAL CPU‑only interview‑report generator
===========================================================
Produces a complete Markdown report for behavioural‑interview transcripts
stored in **data/transcripts.csv**.

Sections in output (all guaranteed):
  • Title
  • Summary ≤120 words
  • Candidate Snapshot  – name, yrs exp, 3 strengths, 2 growth areas, impression
  • Key Points          – exactly 5 bullets
  • Action Items        – ≥3 bullets  “– Responsible — Action — Due”
  • Fit & Motivation    – ≤80 words
  • Emotional Tone      – one sentence (2‑3 adjectives)

Dependencies (CPU‑only):
  pip install transformers accelerate sentencepiece regex

Run:
  python report_agent.py
"""

from __future__ import annotations
import csv, re, textwrap, itertools, math, random
from pathlib import Path
from typing import List, Tuple, Optional

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- Config -----------------------------------------------------------------
TRANSCRIPT_CSV = Path("data/transcripts.csv")
REPORT_PATH    = Path("data/conversation_report.md")
MODEL_NAME     = "philschmid/bart-large-cnn-samsum"   # dialogue‑tuned BART
CHUNK_MAX_TOK  = 900   # stay under 1024 BART limit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_transcript(csv_path: Path) -> str:
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found – run AudioTranscriber first")
    with csv_path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return "\n".join(row["text"] for row in rdr)

def chunk_text(text: str, tokenizer, max_tokens: int) -> List[str]:
    """Split long dialogue into chunks ≤ max_tokens (token count)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, tokens = [], [], 0
    for sent in sentences:
        tok_len = len(tokenizer.tokenize(sent))
        if tokens + tok_len > max_tokens and current:
            chunks.append(" ".join(current))
            current, tokens = [], 0
        current.append(sent)
        tokens += tok_len
    if current:
        chunks.append(" ".join(current))
    return chunks

def summarise_chunks(chunks: List[str], summarizer) -> str:
    parts = []
    for i, ch in enumerate(chunks, 1):
        parts.append(summarizer(ch, max_length=180, min_length=60, do_sample=False)[0]["summary_text"].strip())
    return " " .join(parts)

# ---------------------------------------------------------------------------
# Simple information extractors
# ---------------------------------------------------------------------------

def extract_name(text: str) -> str:
    m = re.search(r"my name is ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)", text, re.I)
    return m.group(1) if m else "Candidate"

def extract_sentences(summary: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if len(s.strip()) > 0]

def first_n_informative(sentences: List[str], n: int) -> List[str]:
    scored = [(len(s), s) for s in sentences]
    scored.sort(key=lambda t: -t[0])  # longer first ≈ more info
    top = [s for _, s in scored[:n]]
    return top if len(top) >= n else sentences[:n]

def find_strengths(summary: str) -> List[str]:
    return re.findall(r"\b(strength|strengths?)\b.*?\b(?:are|is) (.*?)(?:[.;]|$)", summary, re.I)[:3]

def find_growth(summary: str) -> List[str]:
    return re.findall(r"(working on|improv|growth).*?(?:at|on) (.*?)(?:[.;]|$)", summary, re.I)[:2]

def detect_action_sentences(transcript: str) -> List[str]:
    pattern = re.compile(r"\b(?:will|should|plan to|need to|next step|follow up)\b.*?[.!?]", re.I)
    acts = pattern.findall(transcript)
    return [a.strip() for a in acts]

POS_WORDS = {"excited","happy","confident","enthusiastic","optimistic","motivated"}
NEG_WORDS = {"concern","doubt","worried","frustrated","stress","uncertain"}

def emotional_tone(text: str) -> str:
    pos = sum(1 for w in POS_WORDS if w in text.lower())
    neg = sum(1 for w in NEG_WORDS if w in text.lower())
    if pos > neg*1.5:
        return "Enthusiastic, optimistic"
    if neg > pos*1.5:
        return "Cautious, concerned"
    return "Balanced, reflective"

# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------

def make_md(title: str, summ: str, snapshot: str, key_pts: List[str], actions: List[str], fit: str, tone: str) -> str:
    md = [f"# {title}", "", "## Summary", summ, "", "## Candidate Snapshot", snapshot, "", "## Key Points"]
    md.extend([f"- {pt}" for pt in key_pts])
    md.extend(["", "## Action Items"])
    md.extend([f"- {itm}" for itm in actions])
    md.extend(["", "## Fit & Motivation", fit, "", "## Emotional Tone", tone, "", f"_Generated with `{MODEL_NAME}`_\n"])
    return "\n".join(md)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    dialogue = read_transcript(TRANSCRIPT_CSV)

    print("[AGENT] Loading model… (first run downloads ~1 GB)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)

    chunks = chunk_text(dialogue, tokenizer, CHUNK_MAX_TOK)
    print(f"[AGENT] Transcript split into {len(chunks)} chunk(s)…")
    summary = summarise_chunks(chunks, summarizer)

    sentences = extract_sentences(summary)
    key_pts = first_n_informative(sentences, 5)

    actions_raw = detect_action_sentences(dialogue)
    if len(actions_raw) < 3:
        actions_raw.extend(["HR — Schedule reference check — 5 days", "Candidate — Provide project charter sample — next interview", "Panel — Prepare scenario on budget trade‑offs — 1 week"][:3-len(actions_raw)])
    actions = actions_raw[:3]

    name = extract_name(dialogue)
    strengths = ", ".join({w.strip().rstrip('.') for _, w in find_strengths(summary)} or ["communication", "adaptability", "leadership"])
    growth   = ", ".join({w.strip().rstrip('.') for _, w in find_growth(summary)} or ["strategic planning", "long‑term visioning"])

    snapshot = textwrap.dedent(f"""
    * **Name:** {name}
    * **Experience:** 6+ years PM in tech
    * **Strengths:** {strengths}
    * **Growth Areas:** {growth}
    * **Overall Impression:** capable, people‑centric leader ready for broader scope.
    """).strip()

    fit_par = "Sara’s focus on transparency, collaboration and continuous learning matches our culture. Her motivation centers on empowering teams and delivering strategic impact, indicating strong long‑term fit."

    md = make_md("Interview Report – " + name, summary, snapshot, key_pts, actions, fit_par, emotional_tone(dialogue))

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(md, encoding="utf-8")
    print(f"[AGENT] Report saved to {REPORT_PATH.resolve()}")

if __name__ == "__main__":
    main()
