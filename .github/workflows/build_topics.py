import os
import json
import time
from pathlib import Path
from google import genai
from google.genai import types

def build_knowledge_base():
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing.")

    client = genai.Client(api_key=api_key)

    # 1. Gather all unique topics/answers from extracted question files
    extracted_topics = set()
    
    # Check extracted_questions.json (from parsed packets)
    packet_data_path = Path("extracted_questions.json")
    if packet_data_path.exists():
        with open(packet_data_path, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                ans = item.get("answer", "").strip()
                if ans and len(ans) < 50: # filter out super long text answers
                    extracted_topics.add(ans)

    # Check quizzes.json if available
    quizzes_path = Path("quizzes.json")
    if quizzes_path.exists():
        with open(quizzes_path, "r", encoding="utf-8") as f:
            q_data = json.load(f)
            for q in q_data.get("quizzes", []):
                opts = q.get("options", [])
                ans_idx = q.get("answer", 0)
                if opts and ans_idx < len(opts):
                    extracted_topics.add(opts[ans_idx])

    # Default fallback topics if dataset is small
    if len(extracted_topics) < 5:
        extracted_topics = {
            "Acceleration", "Photosynthesis", "Tectonic Plates", 
            "Gravitational Force", "Cell Nucleus", "Velocity", "Newton's Laws"
        }

    topic_list = sorted(list(extracted_topics))
    print(f"Generating knowledge base for {len(topic_list)} topics...")

    topics_db = {}

    for topic in topic_list:
        print(f"Generating details for: {topic}...")
        prompt = f"""
        You are an expert academic bee curriculum developer.
        Generate educational facts for the competition topic: "{topic}".

        Provide:
        1. A clear, 1-2 sentence overview definition.
        2. 8-10 key competition facts (formulas, historical context, specific clues, and principles).
        3. 3-5 closely related academic topics.

        Return STRICT JSON matching this schema:
        {{
          "topic": "{topic}",
          "summary": "1-2 sentence definition",
          "facts": [
            "Fact 1...",
            "Fact 2...",
            "Fact 3..."
          ],
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
            time.sleep(1) # Protect against rate limits
        except Exception as e:
            print(f"Error processing {topic}: {e}")

    with open("topics.json", "w", encoding="utf-8") as f:
        json.dump(topics_db, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated topics.json with {len(topics_db)} items!")

if __name__ == "__main__":
    build_knowledge_base()
