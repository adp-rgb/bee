import os
import json
from google import genai
from google.genai import types

def run_ai_agent():
    # 1. Strip whitespaces to prevent header encoding issues
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing or empty.")

    # 2. Instantiate client explicitly with api_key parameter
    client = genai.Client(api_key=api_key)

    prompt = """
    You are an AI research agent for International Academic Competitions (IAC), Science Bowl, and Geography Bees.
    Your task is to generate 5 high-quality, pyramidal-style quiz tossups (3-sentence structure: obscure clue -> medium clue -> giveaway starting with 'For the point, name...').

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

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    quiz_data = json.loads(response.text)
    
    with open("quizzes.json", "w", encoding="utf-8") as f:
        json.dump(quiz_data, f, indent=2)

    print("Successfully generated new quiz set in quizzes.json!")

if __name__ == "__main__":
    run_ai_agent()
