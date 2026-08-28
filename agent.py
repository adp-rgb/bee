import os
import json
from pathlib import Path
from google import genai
from google.genai import types

def run_ai_agent():
    # 1. Cleanly retrieve API key
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing or empty.")

    client = genai.Client(api_key=api_key)

    # 2. Locate all PDF files inside the data/ folder
    data_dir = Path("data")
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the 'data/' directory.")

    print(f"Found {len(pdf_files)} PDF file(s) in data/: {[f.name for f in pdf_files]}")

    # 3. Upload all detected PDFs to Gemini
    uploaded_files = []
    try:
        for pdf_path in pdf_files:
            print(f"Uploading {pdf_path.name} to Gemini...")
            file_obj = client.files.upload(file=str(pdf_path))
            uploaded_files.append(file_obj)
            print(f"Uploaded successfully: {file_obj.name}")

        # 4. Formulate prompt referencing all uploaded materials
        prompt = """
        You are an AI research agent for International Academic Competitions (IAC), Science Bowl, and Geography Bees.
        Analyze all uploaded PDF documents containing past competitive questions, formats, and style guidelines.

        Based strictly on the collective style, difficulty, and factual depth present across all provided PDF documents:
        Generate 5 high-quality, pyramidal-style quiz tossups (3-sentence structure: obscure clue -> medium clue -> giveaway starting with 'For the point, name...').

        Topics:
        - 3 Questions on Physical Geography or Earth Science.
        - 2 Questions on General Science.

        Output STRICT JSON matching this exact structure:
        [
          {
            "id": 1,
            "category": "Geography / Earth Science",
            "question": "Pyramidal question text here...",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": 0,
            "explanation": "Fact summary explaining the correct answer for students."
          }
        ]
        """

        # 5. Pass all uploaded file objects along with the prompt
        contents = uploaded_files + [prompt]

        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        # 6. Save output JSON
        quiz_data = json.loads(response.text)
        with open("quizzes.json", "w", encoding="utf-8") as f:
            json.dump(quiz_data, f, indent=2)

        print("Successfully generated quiz set from all PDF sources in quizzes.json!")

    finally:
        # 7. Clean up all uploaded files from Gemini storage
        for file_obj in uploaded_files:
            try:
                client.files.delete(name=file_obj.name)
                print(f"Cleaned up file from Gemini: {file_obj.name}")
            except Exception as e:
                print(f"Error deleting file {file_obj.name}: {e}")

if __name__ == "__main__":
    run_ai_agent()
