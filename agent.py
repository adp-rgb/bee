import os
import json
import time
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

    uploaded_files = []
    file_parts = []

    try:
        # 3. Upload all PDFs and prepare content parts
        for pdf_path in pdf_files:
            print(f"Uploading {pdf_path.name} to Gemini...")
            file_obj = client.files.upload(file=str(pdf_path))
            uploaded_files.append(file_obj)

            # Wait briefly to ensure file processing state is ACTIVE
            while file_obj.state.name == "PROCESSING":
                print(f"Waiting for {pdf_path.name} to finish processing...")
                time.sleep(2)
                file_obj = client.files.get(name=file_obj.name)

            if file_obj.state.name != "ACTIVE":
                raise RuntimeError(f"File {pdf_path.name} failed to process: {file_obj.state.name}")

            # Convert uploaded file URI into a valid Part object
            part = types.Part.from_uri(
                file_uri=file_obj.uri,
                mime_type=file_obj.mime_type
            )
            file_parts.append(part)
            print(f"Ready: {file_obj.name}")

        # 4. Formulate prompt
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

        # 5. Combine file parts and text prompt into contents list
        contents = file_parts + [prompt]

        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        # 6. Parse and save JSON output
        quiz_data = json.loads(response.text)
        with open("quizzes.json", "w", encoding="utf-8") as f:
            json.dump(quiz_data, f, indent=2)

        print("Successfully generated quiz set from all PDF sources in quizzes.json!")

    finally:
        # 7. Clean up files from Gemini storage
        for file_obj in uploaded_files:
            try:
                client.files.delete(name=file_obj.name)
                print(f"Cleaned up file from Gemini: {file_obj.name}")
            except Exception as e:
                print(f"Error deleting file {file_obj.name}: {e}")

if __name__ == "__main__":
    run_ai_agent()
