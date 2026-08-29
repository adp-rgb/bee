import os
import json
import time
from pathlib import Path
from google import genai
from google.genai import types

def generate_topic_knowledge_base():
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing.")

    client = genai.Client(api_key=api_key)

    # Load existing quizzes to extract unique answer topics
    quizzes_path = Path("quizzes.json")
    if not quizzes_path.exists():
        print("quizzes.json not found. Run agent.py first.")
        return

    with open(quizzes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        quizzes = data.get("quizzes", [])

    # Extract distinct topics from answers and options
    extracted_topics = list(set([q.get("options", [])[q.get("answer", 0)] for q in quizzes if "options" in q]))
    
    # Fallback default topics if dataset is small
    if len(extracted_topics) < 5:
        extracted_topics = ["Acceleration", "Photosynthesis", "Tectonic Plates", "Gravitational Force", "Cell Nucleus"]

    print(f"Generating knowledge base for {len(extracted_topics)} topics across education tiers...")

    topics_db = {}

    for topic in extracted_topics:
        print(f"Processing topic: {topic}...")
        prompt = f"""
        You are an academic curriculum assistant for Geography and Science Bees.
        Provide structured educational information for the topic: "{topic}".

        Provide content tailored to 3 education levels:
        1. Elementary (Simple definitions, relatable real-world examples, 3 core facts)
        2. Middle School (Key tournament facts, formulas/equations if applicable, 5 structured bullet points)
        3. High School (Advanced theoretical concepts, deep clues, historical context)

        Also list 3-5 related academic topics.

        Return STRICT JSON matching this format:
        {{
          "topic": "{topic}",
          "summary": "Short 1-2 sentence overview.",
          "levels": {{
            "elementary": {{
              "overview": "Simple explanation...",
              "facts": ["Fact 1", "Fact 2", "Fact 3"]
            }},
            "middle_school": {{
              "overview": "Standard competition overview...",
              "facts": ["Fact 1", "Fact 2", "Fact 3", "Fact 4", "Fact 5"]
            }},
            "high_school": {{
              "overview": "Advanced competition overview...",
              "facts": ["Fact 1", "Fact 2", "Fact 3", "Fact 4", "Fact 5"]
            }}
          }},
          "related_topics": ["Related Topic 1", "Related Topic 2", "Related Topic 3"]
        }}
        """

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            topic_data = json.loads(response.text)
            topics_db[topic] = topic_data
            time.sleep(1) # Rate limit protection
        except Exception as e:
            print(f"Error processing {topic}: {e}")

    with open("topics.json", "w", encoding="utf-8") as f:
        json.dump(topics_db, f, indent=2)

    print("Saved knowledge base to topics.json successfully!")

if __name__ == "__main__":
    generate_topic_knowledge_base()
