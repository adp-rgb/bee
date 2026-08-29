import os
import json
from pathlib import Path
import chromadb
from pypdf import PdfReader
from google import genai
from google.genai import types

def load_rules():
    rules_path = Path("rules.json")
    if not rules_path.exists():
        raise FileNotFoundError("rules.json not found.")
    with open(rules_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_vector_store():
    """Extracts text from all PDFs and text files in data/ into ChromaDB."""
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
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")
                continue
        elif file_path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        # Break long text into smaller chunks for quick retrieval
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

def run_ai_agent():
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")

    client = genai.Client(api_key=api_key)
    rules = load_rules()
    num_questions = rules.get("questions_per_round", 30)

    # 1. Index local PDFs and web text
    collection = build_vector_store()

    # 2. Retrieve representative sample clues across subjects
    query_results = collection.query(
        query_texts=["pyramidal geography science bee tossup question for the point name"],
        n_results=15
    )
    retrieved_context = "\n---\n".join(query_results["documents"][0])

    # 3. Dynamic Prompt adhering strictly to rules.json
    prompt = f"""
    You are an official item writer for the {rules['competition_name']}.
    
    Below is a reference sample of style and questions extracted from past official exams and the IAC website:
    {retrieved_context}

    Generate EXACTLY {num_questions} high-quality, pyramidal-style quiz tossups following these rules:
    - Question Structure: {rules['question_structure']}
    - Maximum correct answers allowed per player per round: {rules['scoring']['max_correct_per_player']}
    - Early incorrect answer penalty: {rules['scoring']['early_incorrect_penalty']} point

    Output STRICT JSON matching this exact structure:
    [
      {{
        "id": 1,
        "category": "Geography / Earth Science",
        "question": "Pyramidal question text starting with obscure clues and ending with giveaway starting with 'For the point, name...'",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "answer": 0,
        "explanation": "Fact summary explaining the correct answer."
      }}
    ]
    """

    print(f"Generating {num_questions} questions via Gemini API...")
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    quiz_data = json.loads(response.text)
    
    output_payload = {
        "rules_summary": {
            "total_questions": len(quiz_data),
            "max_correct_per_player": rules["scoring"]["max_correct_per_player"],
            "early_penalty": rules["scoring"]["early_incorrect_penalty"],
            "bonus_table": rules["scoring"]["bonus_structure"]
        },
        "quizzes": quiz_data
    }

    with open("quizzes.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"Successfully generated {len(quiz_data)} questions in quizzes.json!")

if __name__ == "__main__":
    run_ai_agent()
