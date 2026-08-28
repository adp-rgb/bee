import os
import json
from google import genai
from google.genai import types

def run_ai_agent():
    # 1. Cleanly retrieve the API key
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing or empty.")

    client = genai.Client(api_key=api_key)

    # 2. Upload the PDF file to the Gemini File API
    pdf_path = "data/"  # Update path to your PDF file
    
    print(f"Uploading {pdf_path} to Gemini...")
    uploaded_file = client.files.upload(file=pdf_path)
    print(f"Uploaded successfully as: {uploaded_file.name}")

    # 3. Formulate the PDF-grounded prompt
    prompt = """
    You are an AI research agent for International Academic Competitions (IAC), Science Bowl, and Geography Bees.
    Analyze the uploaded PDF document containing past competitive questions, formats, and style guidelines.

    Based strictly on the style, difficulty, and factual depth present in the PDF document:
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

    # 4. Generate content passing both the PDF file object and the prompt string
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=[uploaded_file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    # 5. Clean up the uploaded file from Gemini storage
    client.files.delete(name=uploaded_file.name)

    # 6. Save the generated JSON dataset
    quiz_data = json.loads(response.text)
    with open("quizzes.json", "w", encoding="utf-8") as f:
        json.dump(quiz_data, f, indent=2)

    print("Successfully generated PDF-grounded quiz set in quizzes.json!")

if __name__ == "__main__":
    run_ai_agent()
