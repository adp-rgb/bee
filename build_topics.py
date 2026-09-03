import json
import os
import time
from pathlib import Path
from google import genai
from google.genai import types


def build_knowledge_base():
    api_key = (
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    ).strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing.")

    client = genai.Client(api_key=api_key)

    # 1. Gather all unique topics/answers from extracted question files
    extracted_topics = set()

    # Check extracted_questions.json
    packet_data_path = Path("extracted_questions.json")
    if packet_data_path.exists():
        with open(packet_data_path, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                ans = item.get("answer", "").strip()
                if ans and len(ans) < 50:
                    extracted_topics.add(ans)

    # Check quizzes.json
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
            "Acceleration",
            "Photosynthesis",
            "Tectonic Plates",
            "Gravitational Force",
            "Cell Nucleus",
            "Velocity",
            "Newton's Laws",
            "Amazon River",
            "Mount Everest",
            "Tokyo",
        }

    topic_list = sorted(list(extracted_topics))
    print(f"Generating knowledge base for {len(topic_list)} topics...")

    topics_db = {"Geography": [], "Science": []}

    # 2. Query Gemini API for each topic to generate study facts
    for topic in topic_list:
        prompt = f"""
        Provide detailed study guide data for the academic competition topic: "{topic}".
        
        Task:
        1. Classify category as strictly either "Geography" or "Science".
        2. Write a clear summary/definition.
        3. Provide 3-4 distinct high-frequency competition key facts.
        4. List 2-3 related topics.

        Respond ONLY with a JSON object matching this schema:
        {{
            "category": "Geography" or "Science",
            "summary": "A 2-sentence explanation/definition of {topic}.",
            "facts": ["Specific Fact 1", "Specific Fact 2", "Specific Fact 3"],
            "related_topics": ["Related Topic 1", "Related Topic 2"]
        }}
        """

        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )

            # Clean potential Markdown formatting block wrappers
            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()

            topic_data = json.loads(raw_text)

            category = topic_data.get("category", "Science")
            if category not in ["Geography", "Science"]:
                category = "Science"

            topics_db[category].append(
                {
                    "name": topic,
                    "definition": topic_data.get(
                        "summary", f"Overview of {topic}."
                    ),
                    "key_facts": topic_data.get("facts", []),
                    "related_topics": topic_data.get("related_topics", []),
                }
            )

            print(f"   ✓ Generated study guide for: {topic} [{category}]")
            time.sleep(1)  # Rate-limit protection

        except Exception as e:
            print(f"   ❌ Error processing {topic}: {e}")

    # 3. Save knowledge base output
    with open("topics.json", "w", encoding="utf-8") as f:
        json.dump(topics_db, f, indent=2, ensure_ascii=False)

    total_count = len(topics_db["Geography"]) + len(topics_db["Science"])
    print(
        f"\n✅ Successfully generated topics.json with {total_count} unique items across Geography and Science!"
    )


if __name__ == "__main__":
    build_knowledge_base()
