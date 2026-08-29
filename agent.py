import os
import json
import time
from pathlib import Path
from google import genai
from google.genai import types

def run_ai_agent():
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing or empty.")

    client = genai.Client(api_key=api_key)

    data_dir = Path("data")
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the 'data/' directory.")

    # Limit batch size to 5 files to prevent payload size limits
    selected_pdfs = pdf_files[:5]
    print(f"Processing {len(selected_pdfs)} of {len(pdf_files)} PDF file(s): {[f.name for f in selected_pdfs]}")

    uploaded_files = []
    file_parts = []

    try:
        for pdf_path in selected_pdfs:
            print(f"Uploading {pdf_path.name} to Gemini...")
            file_obj = client.files.upload(file=str(pdf_path))
            uploaded_files.append(file_obj)

            while file_obj.state.name == "PROCESSING":
                print(f"Waiting for {pdf_path.name}...")
                time.sleep(2)
                file_obj = client.files.get(name=file_obj.name)

            if file_obj.state.name != "ACTIVE":
                raise RuntimeError(f"File processing failed: {file_obj.state.name}")

            # Construct explicit Part objects
            part = types.Part.from_uri(
                file_uri=file_obj.uri,
                mime_type="application/pdf"
            )
            file_parts.append(part)
            print(f"Ready: {file_obj.name}")

        prompt = """
        You are an AI research agent for International Academic Competitions (IAC), Science Bowl, and Geography Bees.
        Analyze all attached PDF documents containing past competitive questions, formats, and style guidelines.

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

        # Wrap text prompt in a Part object
        prompt_part = types.Part.from_text(text=prompt)

        # Structure payload into explicit types.Content object
        content_payload = types.Content(
            role="user",
            parts=file_parts + [prompt_part]
        )

        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=content_payload,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        quiz_data = json.loads(response.text)
        with open("quizzes.json", "w", encoding="utf-8") as f:
            json.dump(quiz_data, f, indent=2)

        print("Successfully generated quiz set in quizzes.json!")

    finally:
        for file_obj in uploaded_files:
            try:
                client.files.delete(name=file_obj.name)
                print(f"Cleaned up file from Gemini: {file_obj.name}")
            except Exception as e:
                print(f"Error deleting file {file_obj.name}: {e}")

if __name__ == "__main__":
    run_ai_agent()
