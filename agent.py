import os
import json
import time
from pathlib import Path
from google import genai
from google.genai import types

def load_rules():
    rules_path = Path("rules.json")
    if not rules_path.exists():
        raise FileNotFoundError("rules.json not found. Please create the rules configuration file.")
    with open(rules_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_ai_agent():
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing or empty.")

    client = genai.Client(api_key=api_key)

    # 1. Load dynamic rules
    rules = load_rules()
    num_questions = rules.get("questions_per_round", 30)

    # 2. Locate PDF documents in data/
    data_dir = Path("data")
    pdf_files = list(data_dir.glob("*.pdf"))
    
    uploaded_files = []
    file_parts = []

    try:
        if pdf_files:
            selected_pdfs = pdf_files[:5]  # Cap PDF batch size for API context stability
            print(f"Uploading {len(selected_pdfs)} reference PDF(s)...")
            
            for pdf_path in selected_pdfs:
                file_obj = client.files.upload(file=str(pdf_path))
                uploaded_files.append(file_obj)

                while file_obj.state.name == "PROCESSING":
                    time.sleep(2)
                    file_obj = client.files.get(name=file_obj.name)

                if file_obj.state.name == "ACTIVE":
                    file_parts.append(types.Part.from_uri(file_uri=file_obj.uri, mime_type="application/pdf"))

        # 3. Dynamic prompt incorporating rules.json
        prompt_text = f"""
        You are an official item writer for the {rules['competition_name']}.
        Analyze the uploaded PDF documents for style, depth, and subject distribution.

        Generate EXACTLY {num_questions} pyramidal tossup questions following these official rules:
        - Question Structure: {rules['question_structure']}
        - Ensure questions adhere strictly to IAC style rules where clues go from obscure to accessible giveaway.

        Output STRICT JSON array with exactly {num_questions} items matching this structure:
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

        prompt_part = types.Part.from_text(text=prompt_text)
        content_payload = types.Content(role="user", parts=file_parts + [prompt_part])

        print(f"Generating {num_questions} questions per rule specifications...")
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=content_payload,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=8192  # Gives enough token room for 30 detailed questions
            )
        )

        quiz_data = json.loads(response.text)
        
        # Save output JSON along with metadata
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

    finally:
        for file_obj in uploaded_files:
            try:
                client.files.delete(name=file_obj.name)
            except Exception:
                pass

if __name__ == "__main__":
    run_ai_agent()
