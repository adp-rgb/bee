import json
import os
import time
from pathlib import Path
from bs4 import BeautifulSoup
import chromadb
from google import genai
from google.genai import types
from google.genai.errors import APIError, ServerError
from pypdf import PdfReader
import requests


def load_rules():
    rules_path = Path("rules.json")
    if not rules_path.exists():
        return {"questions_per_round": 300}
    with open(rules_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_topics():
    topics_path = Path("topics.json")
    if not topics_path.exists():
        return {
            "Geography": [{"name": "World Capitals & Urban Centers"}],
            "Science": [{"name": "Velocity"}],
        }
    with open(topics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def download_fresh_pdfs():
    """Scrapes resources for Science Bee and Geography Bee materials."""
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    resources = [
        {
            "url": "https://iacompetitionsasia.com/resources/",
            "name": "IAC Asia Resources",
            "keywords": ["science", "geography", "bee", "competition", "question", "practice"],
        },
        {
            "url": "https://www.internationalgeographybee.com/asia/resources/",
            "name": "International Geography Bee - Asia",
            "keywords": ["geography", "bee", "competition", "past", "question"],
        },
        {
            "url": "https://www.iacompetitions.com/resources/",
            "name": "IAC Competitions Resources",
            "keywords": ["science", "geography", "bee", "competition", "question"],
        },
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    all_pdf_links = []
    downloaded_count = 0

    for resource in resources:
        try:
            response = requests.get(resource["url"], headers=headers, timeout=15)
            response.raise_for_status()
            time.sleep(1)
        except Exception as e:
            print(f"   ❌ Failed to reach {resource['name']}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        pdf_links = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            text = a_tag.get_text().lower()
            if href.lower().endswith(".pdf") or any(kw in text for kw in resource["keywords"]):
                if not href.startswith("http"):
                    href = requests.compat.urljoin(resource["url"], href)
                pdf_links.append(href)

        all_pdf_links.extend(pdf_links)

    all_pdf_links = list(set(all_pdf_links))

    for pdf_url in all_pdf_links:
        try:
            filename = pdf_url.split("/")[-1].split("?")[0]
            if not filename or len(filename) < 3:
                filename = f"resource_{len(list(data_dir.glob('*.pdf')))}.pdf"

            file_path = data_dir / filename
            if file_path.exists():
                continue

            pdf_res = requests.get(pdf_url, headers=headers, timeout=30)
            pdf_res.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(pdf_res.content)

            downloaded_count += 1
            time.sleep(1)
        except Exception as e:
            continue

    print(f"✅ Downloaded {downloaded_count} new resources.")
    return downloaded_count


def build_vector_store():
    """Build vector store from downloaded materials."""
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="bee_competition_resources")

    data_dir = Path("data")
    all_files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.txt"))

    if not all_files:
        return collection

    doc_id = 0
    for file_path in all_files:
        text = ""
        if file_path.suffix == ".pdf":
            try:
                reader = PdfReader(file_path, strict=False)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            except Exception:
                continue
        elif file_path.suffix == ".txt":
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception:
                continue

        chunks = [text[i : i + 1000] for i in range(0, len(text), 1000) if len(text[i : i + 1000]) > 100]
        for chunk in chunks:
            try:
                collection.add(
                    documents=[chunk],
                    metadatas=[{"source": file_path.name}],
                    ids=[f"doc_{doc_id}"],
                )
                doc_id += 1
            except Exception:
                continue

    return collection


def generate_with_retry(client, prompt_text, primary_model="gemini-2.5-flash", fallback_model="gemini-1.5-pro", max_retries=5):
    """Generates quiz questions with fallback retry handling."""
    models_to_try = [primary_model, fallback_model]

    for model_name in models_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        max_output_tokens=8192,
                    ),
                )
                return response
            except (ServerError, APIError) as e:
                wait_time = attempt * 3
                time.sleep(wait_time)

    raise RuntimeError("Failed to generate content after exhausting retries.")


def run_ai_agent():
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")

    print("🐝 Starting Academic Bee AI Generator...\n")
    client = genai.Client(api_key=api_key)
    rules = load_rules()
    topics_data = load_topics()

    download_fresh_pdfs()
    collection = build_vector_store()

    modes = ["standard", "high-frequency", "flashcards", "good-to-know", "practice-specific"]
    subjects = ["Geography", "Science"]
    
    # Batch size of 5 Qs per request (6 batches = 30 Qs per mode/subject)
    questions_per_batch = 5
    batches_per_mode = 6

    all_quizzes = []

    for subject in subjects:
        for mode in modes:
            print(f"\n⚙️ Generating 30 Qs for [{subject}] - Mode [{mode}]...")
            for batch_num in range(batches_per_mode):
                prompt_text = f"""
You are an official question writer for Science Bee and Geography Bee competitions.
Generate EXACTLY {questions_per_batch} tossup questions for:
- Subject: {subject}
- Mode: {mode}

Topic taxonomy reference: {json.dumps(topics_data.get(subject, []))}

JSON Requirements:
Return a raw JSON array containing exactly {questions_per_batch} objects:
[
  {{
    "category": "{subject}",
    "mode": "{mode}",
    "topic": "Specific Topic Name",
    "question": "Pyramidal question text starting with hard clues and ending with 'For the point, name...'",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": 0,
    "explanation": "Brief context and explanation."
  }}
]
"""
                try:
                    res = generate_with_retry(client, prompt_text)
                    raw_text = res.text.strip()
                    if raw_text.startswith("```"):
                        lines = raw_text.splitlines()
                        lines = [l for l in lines if not l.startswith("```")]
                        raw_text = "\n".join(lines).strip()

                    items = json.loads(raw_text)
                    for item in items:
                        item["category"] = subject
                        item["mode"] = mode
                    all_quizzes.extend(items)
                    print(f"   ✓ Batch {batch_num + 1}/{batches_per_mode} complete (+{len(items)} Qs)")
                except Exception as e:
                    print(f"   ❌ Batch failed: {e}")
                    continue

    if not all_quizzes:
        print("❌ No questions were generated.")
        return

    for idx, q in enumerate(all_quizzes, start=1):
        q["id"] = idx

    output_payload = {
        "rules_summary": rules,
        "quizzes": all_quizzes,
        "metadata": {
            "total_questions": len(all_quizzes),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    with open("quizzes.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Total {len(all_quizzes)} questions generated and saved to quizzes.json!")


if __name__ == "__main__":
    run_ai_agent()
