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

# build_topics.py
topics_db = {
    "Geography": [],
    "Science": []
}

for topic in topic_list:
    # Determine category or default based on topic
    category = "Geography" if any(w in topic.lower() for w in ["capital", "river", "mountain", "country", "city"]) else "Science"
    
    topics_db[category].append({
        "name": topic,
        "definition": topic_data.get("summary", ""),
        "key_facts": topic_data.get("facts", []),
        "related_topics": topic_data.get("related_topics", [])
    })

with open("topics.json", "w", encoding="utf-8") as f:
    json.dump(topics_db, f, indent=2, ensure_ascii=False)
        
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
