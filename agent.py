#!/usr/bin/env python3
"""Academic Bee question generator.

Fills quizzes.json with pyramidal tossups for every (subject, mode) pair.
Re-running is safe: existing questions are kept and only the gaps are filled.

    export GEMINI_API_KEY=...
    python agent.py                       # fill every gap up to 30 per subject/mode
    python agent.py --no-scrape           # use PDFs already in ./data
    python agent.py --subjects Science    # only one subject
    python agent.py --fresh               # ignore the existing quizzes.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel

log = logging.getLogger("bee")

DATA_DIR = Path("data")
QUIZ_PATH = Path("quizzes.json")
RULES_PATH = Path("rules.json")
COLLECTION = "bee_competition_resources"
RETRY_CODES = {429, 500, 502, 503, 504}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

RESOURCES = [
    "https://iacompetitionsasia.com/resources/",
    "https://www.internationalgeographybee.com/asia/resources/",
    "https://www.iacompetitions.com/resources/",
    "http://iacompetitions.com/resources-national-geography-bee/",
    "https://www.iacompetitions.com/resources-national-science-bee/",
    "https://www.iacompetitions.com/ems-national-geography-bee-past-questions/",
    "https://www.iacompetitions.com/ems-national-science-bee-past-questions/",
    "https://geochampionships.com/resources/",
]

# Skip files that are not question material (simulation guides, papers, answer sheets, puzzles).
EXCLUDE = re.compile(
    r"background-guide|symposium|answer-sheet|answer-document|fieldwork|hex|crossword|culinary|musical|"
    r"linguistic|vexillology|question-writing|running-swimming|thai-geography|usa-at-250|free-response", re.I)

# The old prompt only passed the mode *name*, so every mode produced the same kind of
# question. Each mode now has a real brief.
MODES = {
    "standard": "A balanced mix of difficulty, typical of a regional bee round.",
    "high-frequency": "Subjects and answers that recur most often in past IAC bee questions; use the reference material to spot them.",
    "flashcards": "Core must-know facts a student should memorise; the answer is a single, well-known term.",
    "good-to-know": "Fair but less obvious facts that deepen understanding; not so obscure that they would never be asked.",
    "practice-specific": "Drill the assigned topic thoroughly, varying the sub-area from question to question.",
}

# Fixed taxonomy: the old flow built topics.json *from* quizzes.json (circular) and drifted
# into near-duplicate names. Override with --topics path/to/topics.json if you like.
DEFAULT_TOPICS = {
    "Geography": [
        "Rivers & Lakes", "Deserts & Arid Regions", "World Capitals & Urban Centers", "Physical Geography",
        "Islands & Archipelagos", "Mountains & Volcanoes", "Straits & Canals", "Political Geography", "Nations & Borders",
    ],
    "Science": ["Physics", "Chemistry", "Biology", "Earth Science", "Astronomy"],
}


# Grade bands follow the IAC divisions (Elementary: 4th grade and younger; Intermediate: 5th-6th;
# Middle School: 7th-8th; Junior Varsity / Varsity: high school).
LEVELS = {
    "Elementary": "grade 4 and younger: very short sentences, familiar words, widely known facts",
    "Intermediate": "grades 5-6: simple vocabulary, common school-curriculum facts",
    "Middle School": "grades 7-8: moderate vocabulary, standard curriculum facts",
    "Junior Varsity": "grades 9-10: more specialised facts and terminology",
    "Varsity": "grades 11-12: the hardest clues and expert-level detail",
}


class Tossup(BaseModel):
    """Structured-output schema: the API enforces this, so no regex/fence stripping is needed."""
    topic: str
    level: str                # one of the LEVELS keys
    clues: list[str]          # exactly 3: obscure -> medium -> "For the point, name ..."
    correct: str
    distractors: list[str]    # exactly 3
    explanation: str


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


# ----------------------------------------------------------------------------- inputs
def load_rules() -> dict:
    if RULES_PATH.exists():
        return json.loads(RULES_PATH.read_text(encoding="utf-8"))
    return {"questions_per_round": 30}


def load_topics(path: str | None) -> dict[str, list[str]]:
    if not path:
        return DEFAULT_TOPICS
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for subject in DEFAULT_TOPICS:
        names = [t["name"] if isinstance(t, dict) else t for t in raw.get(subject, [])]
        out[subject] = list({norm(n): n for n in names}.values()) or DEFAULT_TOPICS[subject]
    return out


def download_pdfs() -> int:
    """Download only real PDFs (the old code also saved HTML pages whose link text matched a keyword)."""
    from bs4 import BeautifulSoup

    DATA_DIR.mkdir(exist_ok=True)
    links: set[str] = set()
    for url in RESOURCES:
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
        except requests.RequestException as e:
            log.warning("Could not reach %s: %s", url, e)
            continue
        for a in BeautifulSoup(r.text, "html.parser").find_all("a", href=True):
            href = urljoin(url, a["href"])
            if urlparse(href).path.lower().endswith(".pdf") and not EXCLUDE.search(href):
                links.add(href)
        time.sleep(1)

    new = 0
    for link in sorted(links):
        dest = DATA_DIR / Path(urlparse(link).path).name
        if dest.exists():
            continue
        try:
            r = requests.get(link, headers=HEADERS, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            log.warning("Download failed %s: %s", link, e)
            continue
        if not r.content.startswith(b"%PDF"):
            log.warning("Skipped (not a PDF): %s", link)
            continue
        dest.write_bytes(r.content)
        new += 1
        time.sleep(1)
    log.info("Downloaded %d new PDFs (%d links found).", new, len(links))
    return new


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    chunks = [text[i:i + size] for i in range(0, len(text), size - overlap)]
    return [c for c in chunks if len(c) > 100]


def build_index(rebuild: bool = False):
    """Persistent vector index of the downloaded material (returns None if unavailable)."""
    try:
        import chromadb
        from pypdf import PdfReader
    except ImportError:
        log.warning("chromadb/pypdf not installed: generating without reference material.")
        return None
    client = chromadb.PersistentClient(path=".chroma")
    if rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:  # collection may not exist yet
            pass
    col = client.get_or_create_collection(COLLECTION)
    if col.count() == 0:
        for f in sorted(DATA_DIR.glob("*.pdf")) + sorted(DATA_DIR.glob("*.txt")):
            try:
                if f.suffix == ".pdf":
                    text = "\n".join(p.extract_text() or "" for p in PdfReader(f, strict=False).pages)
                else:
                    text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:  # corrupt PDFs are common
                log.warning("Could not read %s: %s", f.name, e)
                continue
            chunks = chunk_text(text)
            for i in range(0, len(chunks), 64):
                part = chunks[i:i + 64]
                col.add(documents=part, metadatas=[{"source": f.name}] * len(part),
                        ids=[f"{f.name}-{i + j}" for j in range(len(part))])
        log.info("Indexed %d chunks.", col.count())
    return col if col.count() else None


def retrieve(col, query: str, k: int = 4) -> tuple[str, list[str]]:
    if col is None:
        return "", []
    res = col.query(query_texts=[query], n_results=k)
    docs, metas = res["documents"][0], res["metadatas"][0]
    return "\n---\n".join(d[:800] for d in docs), sorted({m["source"] for m in metas})


# ------------------------------------------------------------------------ generation
def make_prompt(subject: str, mode: str, slots: list[dict], avoid: list[str], context: str) -> str:
    ref = (f"\nReference material (use it for facts and style; write ORIGINAL questions, never copy):\n{context}\n"
           if context else "\nNo reference material is available: rely only on well-established facts.\n")
    return f"""You are an experienced question writer for the IAC {subject} Bee (pyramidal buzzer tossups).
Write {len(slots)} ORIGINAL tossup questions. Mode: {mode} - {MODES[mode]}
One question per slot, in this order; 'topic' and 'level' must equal the assigned values exactly: {json.dumps(slots)}
Level guide: {json.dumps(LEVELS)}. Match vocabulary, sentence length and difficulty to the level.

Each question has exactly 3 'clues': (1) an obscure, hard clue, (2) a medium clue, (3) an easy giveaway that
begins with "For the point, name ...". Never state the answer inside the clues.
'correct' is the shortest acceptable answer. 'distractors' are 3 plausible but wrong answers of the same type.
'explanation' is 1-2 accurate sentences. Every fact must be correct and verifiable.
Do not use these answers (already used): {json.dumps(avoid)}
{ref}"""


def call_model(client, prompt: str, models: list[str], max_retries: int = 5):
    """Retry only transient errors (429/5xx) with backoff; a bad model name or key fails fast."""
    last = None
    for model in models:
        for attempt in range(max_retries):
            try:
                return client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", response_schema=list[Tossup],
                        temperature=0.9, max_output_tokens=8192))
            except APIError as e:
                last = e
                if e.code not in RETRY_CODES:
                    log.error("%s failed with a non-retryable error: %s", model, e)
                    break  # try the next model
                delay = min(60, 2 ** attempt * 2) + random.random()
                log.warning("%s: HTTP %s, retrying in %.0fs", model, e.code, delay)
                time.sleep(delay)
    raise RuntimeError(f"All models failed: {last}")


def parse(resp) -> list[dict]:
    items = resp.parsed if getattr(resp, "parsed", None) else json.loads(resp.text)
    return [i.model_dump() if isinstance(i, BaseModel) else i for i in items]


def build_item(raw: dict, subject: str, mode: str, topics: list[str], seen_q: set, seen_a: set,
               sources: list[str], rng: random.Random):
    """Validate one generated question. Returns (item, None) or (None, reason)."""
    clues = [c.strip() for c in raw.get("clues", [])]
    correct = (raw.get("correct") or "").strip()
    distractors = [d.strip() for d in raw.get("distractors", [])]
    options = [correct] + distractors
    topic = next((t for t in topics if norm(t) == norm(raw.get("topic", ""))), None)
    question = " ".join(clues)
    level = next((l for l in LEVELS if norm(l) == norm(raw.get("level", ""))), None)
    if topic is None:
        return None, "topic not in taxonomy"
    if level is None:
        return None, "level not recognised"
    if len(clues) != 3 or not all(clues):
        return None, "needs exactly 3 clues"
    if not clues[2].lower().startswith("for the point"):
        return None, "last clue must start with 'For the point'"
    if len(distractors) != 3 or not all(options) or len({norm(o) for o in options}) != 4:
        return None, "needs 4 distinct options"
    if f" {norm(correct)} " in f" {norm(question)} ":
        return None, "answer leaked into the question"
    if (subject, norm(correct)) in seen_a:
        return None, "duplicate answer"
    if norm(question) in seen_q:
        return None, "duplicate question"
    rng.shuffle(options)  # the model always favoured B/C; shuffling in code balances positions
    return {
        "category": subject, "mode": mode, "topic": topic, "level": level, "question": question, "clues": clues,
        "options": options, "answer": options.index(correct), "explanation": raw.get("explanation", "").strip(),
        "grounded": bool(sources), "sources": sources,
    }, None


def load_existing() -> list[dict]:
    """Keep valid entries and drop exact-duplicate questions."""
    if not QUIZ_PATH.exists():
        return []
    data = json.loads(QUIZ_PATH.read_text(encoding="utf-8"))
    items = data.get("quizzes", []) if isinstance(data, dict) else data
    out, seen = [], set()
    for q in items:
        k = norm(q.get("question", ""))
        ok = isinstance(q.get("answer"), int) and len(q.get("options", [])) == 4 and 0 <= q["answer"] < 4
        if k and k not in seen and ok:
            seen.add(k)
            out.append(q)
    return out


def save(quizzes: list[dict], rules: dict, models: list[str]) -> None:
    """Atomic write, so a crash mid-run never corrupts quizzes.json."""
    for i, q in enumerate(quizzes, 1):
        q["id"] = i
    payload = {
        "rules_summary": rules,
        "quizzes": quizzes,
        "metadata": {
            "total_questions": len(quizzes),
            "counts": dict(sorted(Counter(f"{q['category']}/{q['mode']}" for q in quizzes).items())),
            "grounded_in_resources": sum(1 for q in quizzes if q.get("grounded") is True),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": models,
            "resources": RESOURCES,
        },
    }
    tmp = QUIZ_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(QUIZ_PATH)


def run(args) -> int:
    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        log.error("Set GEMINI_API_KEY (or GOOGLE_API_KEY).")
        return 2
    client = genai.Client(api_key=key)
    models = [m for m in (args.model, args.fallback_model) if m]
    rules, topics = load_rules(), load_topics(args.topics)
    if not args.no_scrape:
        download_pdfs()
    col = build_index(args.rebuild_index)

    quizzes = [] if args.fresh else load_existing()
    if args.drop_unleveled:
        quizzes = [q for q in quizzes if q.get("level")]
    seen_q = {norm(q["question"]) for q in quizzes}
    seen_a = {(q["category"], norm(q["options"][q["answer"]])) for q in quizzes}
    rng = random.Random()

    for subject in args.subjects:
        pool = topics[subject]
        for mode in args.modes:
            have = sum(1 for q in quizzes if q["category"] == subject and q["mode"] == mode)
            cursor, attempts = rng.randrange(len(pool)), 0
            log.info("[%s/%s] have %d / %d", subject, mode, have, args.per_mode)
            while have < args.per_mode and attempts < args.max_attempts:
                attempts += 1
                need = min(args.batch_size, args.per_mode - have)
                names = list(LEVELS)
                slots = [{"topic": pool[(cursor + i) % len(pool)], "level": names[((cursor + i) + (cursor + i) // len(pool)) % len(names)]}
                         for i in range(need)]  # rotate topics and levels; shift levels each topic cycle
                cursor += need
                used = [a for s, a in seen_a if s == subject]
                avoid = rng.sample(used, min(80, len(used)))
                context, sources = retrieve(col, f"{subject} {mode} " + " ".join(s["topic"] for s in slots))
                try:
                    items = parse(call_model(client, make_prompt(subject, mode, slots, avoid, context), models))
                except (RuntimeError, ValueError) as e:
                    log.warning("batch failed: %s", e)
                    continue
                rejects = Counter()
                for raw in items:
                    item, why = build_item(raw, subject, mode, pool, seen_q, seen_a, sources, rng)
                    if item is None:
                        rejects[why] += 1
                        continue
                    quizzes.append(item)
                    seen_q.add(norm(item["question"]))
                    seen_a.add((subject, norm(item["options"][item["answer"]])))
                    have += 1
                log.info("  batch ok: +%d kept, rejected %s", len(items) - sum(rejects.values()), dict(rejects) or "none")
                save(quizzes, rules, models)  # checkpoint after every batch

    save(quizzes, rules, models)
    short = {f"{s}/{m}": args.per_mode - sum(1 for q in quizzes if q["category"] == s and q["mode"] == m)
             for s in args.subjects for m in args.modes}
    short = {k: v for k, v in short.items() if v > 0}
    log.info("Done: %d questions in %s", len(quizzes), QUIZ_PATH)
    if short:
        log.warning("Still short (run again to retry): %s", short)
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+", default=list(DEFAULT_TOPICS), choices=list(DEFAULT_TOPICS))
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    p.add_argument("--per-mode", type=int, default=30, help="target questions per subject/mode")
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--max-attempts", type=int, default=12, help="API batches per subject/mode before giving up")
    p.add_argument("--model", default=os.getenv("BEE_MODEL", "gemini-2.5-flash"))
    p.add_argument("--fallback-model", default=os.getenv("BEE_FALLBACK_MODEL", "gemini-2.5-pro"))
    p.add_argument("--topics", help="optional topics.json; default is the built-in taxonomy")
    p.add_argument("--no-scrape", action="store_true", help="skip downloading PDFs")
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--fresh", action="store_true", help="ignore the existing quizzes.json")
    p.add_argument("--drop-unleveled", action="store_true", help="discard existing questions that have no level")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
