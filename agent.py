import os
import json
import time
from pathlib import Path
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

def build_vector_store():
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="bee_past_questions")
    
    data_dir = Path("data")
    all_files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.txt"))
    
    print(f"Indexing {len(all_files)} files into local vector database...")
    
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
                print(f"Skipping corrupted PDF {file_path.name}: {e}")
                continue
        elif file_path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        chunks = [text[i:i+1000] for i in range(0, len(text), 1000) if len(text[i:i+1000]) > 100]
        for chunk in chunks:
            collection.add(
                documents=[chunk],
                metadatas=[{"source": file_path.name}],
                ids=[f"doc_{doc_id}"]
            )
            doc_id += 1
            
    print(f"Indexed {doc_id} text chunks successfully!")
    return collection

def generate_with_retry(client, prompt_text, primary_model='gemini-3.6-flash', fallback_model='gemini-2.5-flash', max_retries=5):
    """Generates content with exponential backoff and fallback model handling for 503 errors."""
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
                    print(f"[{model_name}] 503 High Demand detected. Retrying in {wait_time}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise e
        print(f"Switching from {model_name} to fallback model...")

    raise RuntimeError("Failed to generate content after exhausting model retries due to high demand.")

def run_ai_agent():
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")

    client = genai.Client(api_key=api_key)
    rules = load_rules()
    
    total_requested = rules.get("questions_per_round", 30)
    batch_size = 10
    num_batches = (total_requested + batch_size - 1) // batch_size

    collection = build_vector_store()

    query_results = collection.query(
        query_texts=["pyramidal geography science bee tossup question for the point name"],
        n_results=10
    )
    retrieved_context = "\n---\n".join(query_results["documents"][0]) if query_results["documents"] else ""

    all_quizzes = []

    for batch_num in range(num_batches):
        current_count = min(batch_size, total_requested - len(all_quizzes))
        print(f"Generating batch {batch_num + 1}/{num_batches} ({current_count} questions)...")

        prompt_text = f"""
        You are an official item writer for {rules.get('competition_name', 'IAC Bee')}.
        
        Reference sample questions:
        {retrieved_context}

        Generate EXACTLY {current_count} pyramidal tossup questions following these rules:
        - Question Structure: {rules.get('question_structure', 'Pyramidal tossup')}
        - Clues go from obscure to accessible giveaway ending in 'For the point, name...'

        Output STRICT JSON array matching this format:
        [
          {{
            "id": 1,
            "category": "Geography / Earth Science",
            "question": "Pyramidal question text...",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": 0,
            "explanation": "Brief explanation..."
          }}
        ]
        """

        response = generate_with_retry(client, prompt_text)
        batch_data = json.loads(response.text)
        all_quizzes.extend(batch_data)

    for idx, q in enumerate(all_quizzes, start=1):
        q["id"] = idx

    scoring_rules = rules.get("scoring", {})
    output_payload = {
        "rules_summary": {
            "total_questions": len(all_quizzes),
            "max_correct_per_player": scoring_rules.get("max_correct_per_player", 6),
            "early_penalty": scoring_rules.get("early_incorrect_penalty", -1),
            "bonus_table": scoring_rules.get("bonus_structure", {})
        },
        "quizzes": all_quizzes
    }

    with open("quizzes.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"Successfully generated {len(all_quizzes)} questions in quizzes.json!")

if __name__ == "__main__":
    run_ai_agent()
