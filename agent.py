import os
import json
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import chromadb
from pypdf import PdfReader
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError

def load_rules():
    rules_path = Path("rules.json")
    if not rules_path.exists():
        raise FileNotFoundError("rules.json not found.")
    with open(rules_path, "r", encoding="utf-8") as f:
        return json.load(f)

def download_fresh_pdfs():
    """Scrapes multiple resources for Science Bee and Geography Bee materials."""
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Resources to scrape
    resources = [
        {
            "url": "https://www.internationalgeographybee.com/asia/resources/",
            "name": "International Geography Bee - Asia",
            "keywords": ["geography", "bee", "competition", "past", "question"]
        },
        {
            "url": "https://www.iacompetitions.com/resources/",
            "name": "IAC Competitions Resources",
            "keywords": ["science", "geography", "bee", "competition", "question", "practice"]
        },
        {
            "url": "https://www.internationalgeographybee.com/europe/resources/",
            "name": "International Geography Bee - Europe",
            "keywords": ["geography", "bee", "competition", "past", "question"]
        }
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    all_pdf_links = []
    downloaded_count = 0

    for resource in resources:
        print(f"\n📍 Scraping {resource['name']}...")
        print(f"   URL: {resource['url']}")
        
        try:
            response = requests.get(resource['url'], headers=headers, timeout=15)
            response.raise_for_status()
        except Exception as e:
            print(f"   ❌ Failed to reach {resource['name']}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find all links (both PDFs and potential document links)
        pdf_links = []
        
        # Method 1: Direct PDF links
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            text = a_tag.get_text().lower()
            
            # Check if it's a PDF or contains keywords
            is_pdf = href.lower().endswith(".pdf")
            has_keywords = any(kw in text for kw in resource['keywords'])
            
            if is_pdf or (has_keywords and ("pdf" in href.lower() or "download" in text)):
                if not href.startswith("http"):
                    href = requests.compat.urljoin(resource['url'], href)
                pdf_links.append(href)
        
        # Method 2: Look for document containers and extract links
        for div in soup.find_all("div", class_=["resource", "document", "material", "content"]):
            for link in div.find_all("a", href=True):
                href = link["href"]
                if href.lower().endswith(".pdf") or "download" in href.lower():
                    if not href.startswith("http"):
                        href = requests.compat.urljoin(resource['url'], href)
                    pdf_links.append(href)

        # Deduplicate
        pdf_links = list(set(pdf_links))
        print(f"   ✓ Found {len(pdf_links)} PDF resources")
        all_pdf_links.extend(pdf_links)

    print(f"\n📥 Total PDF resources found: {len(all_pdf_links)}")

    # Download all collected PDFs
    for pdf_url in all_pdf_links:
        try:
            filename = pdf_url.split("/")[-1].split("?")[0]
            if not filename or len(filename) < 3:
                filename = f"resource_{len(list(data_dir.glob('*.pdf')))}.pdf"
            
            file_path = data_dir / filename

            # Skip if already exists
            if file_path.exists():
                print(f"   ⊘ Already downloaded: {filename}")
                continue

            print(f"   ⬇️  Downloading: {filename}...")
            pdf_res = requests.get(pdf_url, headers=headers, timeout=30)
            pdf_res.raise_for_status()
            
            with open(file_path, "wb") as f:
                f.write(pdf_res.content)
            
            downloaded_count += 1
            print(f"      ✓ Saved: {filename}")
            
        except Exception as e:
            print(f"   ❌ Failed to download {pdf_url}: {e}")
            continue

    print(f"\n✅ Successfully downloaded {downloaded_count} new PDF resources!")
    return downloaded_count

def build_vector_store():
    """Build vector store from downloaded Science Bee and Geography Bee materials."""
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="bee_competition_resources")
    
    data_dir = Path("data")
    all_files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.txt"))
    
    if len(all_files) == 0:
        print("⚠️  No PDF files found in data/ directory. Using fallback retrieval.")
        return collection
    
    print(f"\n🗂️  Indexing {len(all_files)} competition resources into vector database...")
    
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
            except Exception as e:
                print(f"   ⚠️  Skipping corrupted PDF {file_path.name}: {e}")
                continue
        elif file_path.suffix == ".txt":
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception as e:
                print(f"   ⚠️  Skipping corrupted TXT {file_path.name}: {e}")
                continue

        if len(text.strip()) < 100:
            print(f"   ⊘ Skipping {file_path.name} (insufficient content)")
            continue

        # Split into chunks
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000) if len(text[i:i+1000]) > 100]
        
        for chunk in chunks:
            try:
                collection.add(
                    documents=[chunk],
                    metadatas=[{"source": file_path.name}],
                    ids=[f"doc_{doc_id}"]
                )
                doc_id += 1
            except Exception as e:
                print(f"   ⚠️  Error adding chunk from {file_path.name}: {e}")
                continue
        
        print(f"   ✓ Indexed {file_path.name} ({len(chunks)} chunks)")
            
    print(f"✅ Successfully indexed {doc_id} text chunks from competition resources!")
    return collection

def generate_with_retry(client, prompt_text, primary_model='gemini-2.5-flash', fallback_model='gemini-1.5-flash', max_retries=5):
    """Generate content with retry logic for API failures."""
    models_to_try = [primary_model, fallback_model]
    
    for model_name in models_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        max_output_tokens=8192
                    )
                )
                return response
            except (ServerError, APIError) as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    wait_time = attempt * 5
                    print(f"   ⏱️  High demand detected. Retrying in {wait_time}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise e
        print(f"   Switching to fallback model...")

    raise RuntimeError("Failed to generate content after exhausting model retries.")

def run_ai_agent():
    """Main agent function: Scrape, Index, and Generate Science/Geography Bee Quizzes."""
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")

    print("🐝 Academic Bee AI Agent Starting...\n")
    print("=" * 60)

    client = genai.Client(api_key=api_key)
    rules = load_rules()

    # Step 1: Scrape and download fresh PDFs from official resources
    print("📥 STEP 1: Downloading Science & Geography Bee Resources")
    print("-" * 60)
    download_fresh_pdfs()
    
    # Step 2: Build local vector database from downloaded PDFs
    print("\n🗂️  STEP 2: Building Vector Database")
    print("-" * 60)
    collection = build_vector_store()

    # Step 3: Query relevant context for questions
    print("\n🔍 STEP 3: Retrieving Competition Context")
    print("-" * 60)
    
    total_requested = rules.get("questions_per_round", 30)
    batch_size = 10
    num_batches = (total_requested + batch_size - 1) // batch_size

    # Multiple query strategies
    queries = [
        "geography bee tossup pyramidal question clues",
        "science bee competition question format",
        "geography bee past competition questions",
        "science competition practice questions"
    ]
    
    all_retrieved_context = []
    for query in queries:
        try:
            query_results = collection.query(
                query_texts=[query],
                n_results=5
            )
            if query_results["documents"] and query_results["documents"][0]:
                all_retrieved_context.extend(query_results["documents"][0])
        except Exception as e:
            print(f"   ⚠️  Query failed for '{query}': {e}")
    
    retrieved_context = "\n---\n".join(all_retrieved_context[:10]) if all_retrieved_context else ""
    
    if not retrieved_context:
        print("   ⚠️  No competition materials found. Using template-based generation.")
        retrieved_context = "Using official Science Bee and Geography Bee format guidelines."
    else:
        print(f"   ✓ Retrieved {len(all_retrieved_context)} relevant context passages")

    # Step 4: Generate questions in batches
    print(f"\n🤖 STEP 4: Generating {total_requested} Pyramidal Tossup Questions")
    print("-" * 60)

    all_quizzes = []

    for batch_num in range(num_batches):
        current_count = min(batch_size, total_requested - len(all_quizzes))
        print(f"\n   Batch {batch_num + 1}/{num_batches}: Generating {current_count} questions...")

        prompt_text = f"""
You are an official item writer for Science Bee and Geography Bee competitions.
Your task is to create pyramidal tossup questions using the provided competition materials.

REFERENCE MATERIALS FROM OFFICIAL SOURCES:
{retrieved_context}

RULES:
- Competition: {rules.get('competition_name', 'Science & Geography Bee')}
- Question Structure: {rules.get('question_structure', 'Pyramidal Tossup')}
- Format: Each question has 3-4 clues progressing from obscure/specific to obvious/general
- Final clue must end with "For the point, name..." or similar format
- Content: Focus on factual, verifiable information from academic sources
- Category: Alternate between GEOGRAPHY and SCIENCE topics

GENERATE EXACTLY {current_count} pyramidal tossup questions.

Output STRICT JSON array matching this format (NO OTHER TEXT):
[
  {{
    "id": 1,
    "category": "Geography",
    "question": "Clue 1: [obscure reference]... Clue 2: [medium difficulty]... Clue 3: [accessible]... For the point, name...",
    "options": ["Answer A", "Answer B", "Answer C", "Answer D"],
    "answer": 0,
    "explanation": "Factual explanation of why this is correct, citing the source material."
  }},
  {{
    "id": 2,
    "category": "Science",
    "question": "Clue 1: [specific fact]... Clue 2: [related concept]... Clue 3: [common knowledge]... For the point, name...",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": 1,
    "explanation": "Scientific explanation of the correct answer."
  }}
]

IMPORTANT: 
- Only return valid JSON array
- Each question must have exactly 4 options
- Answer index must be 0-3
- Questions must be grounded in the competition materials provided
- Alternate between Geography (odd IDs) and Science (even IDs) categories
"""

        try:
            response = generate_with_retry(client, prompt_text)
            batch_data = json.loads(response.text)
            all_quizzes.extend(batch_data)
            print(f"   ✓ Generated {len(batch_data)} questions successfully")
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON parsing error: {e}")
            print(f"   Response preview: {response.text[:200]}")
            continue
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            continue

    # Step 5: Finalize and save
    print(f"\n💾 STEP 5: Saving Quiz Data")
    print("-" * 60)
    
    # Renumber questions
    for idx, q in enumerate(all_quizzes, start=1):
        q["id"] = idx

    scoring_rules = rules.get("scoring", {})
    output_payload = {
        "rules_summary": {
            "total_questions": len(all_quizzes),
            "max_correct_per_player": scoring_rules.get("max_correct_per_player", 6),
            "early_penalty": scoring_rules.get("early_incorrect_penalty", -1),
            "bonus_table": scoring_rules.get("bonus_structure", [])
        },
        "quizzes": all_quizzes,
        "metadata": {
            "source": "Science Bee & Geography Bee Official Resources",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "resources": [
                "https://www.internationalgeographybee.com/asia/resources/",
                "https://www.iacompetitions.com/resources/",
                "https://www.internationalgeographybee.com/europe/resources/"
            ]
        }
    }

    with open("quizzes.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    # Calculate category counts - FIX: Use separate variables
    geo_count = len([q for q in all_quizzes if 'Geography' in q.get('category', '')])
    sci_count = len([q for q in all_quizzes if 'Science' in q.get('category', '')])
    
    print("\n" + "=" * 60)
    print(f"✅ SUCCESS! Generated {len(all_quizzes)} pyramidal tossup questions")
    print(f"📄 Saved to: quizzes.json")
    print(f"📊 Categories: {geo_count} Geography, {sci_count} Science")
    print("=" * 60)

if __name__ == "__main__":
    run_ai_agent()
