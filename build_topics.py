import json
import time
from pathlib import Path

def generate_300_topics_db():
    modes = [
        "practice-specific",
        "standard",
        "high-frequency",
        "flashcards",
        "good-to-know"
    ]
    
    topics_db = {
        "metadata": {
            "source": "Science Bee & Geography Bee Official Resources",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "categories": [
                "Practice Specific Topics",
                "Start Quiz",
                "Practice Frequently Asked",
                "Flashcards",
                "Good to Know Topics"
            ]
        },
        "Science": [],
        "Geography": []
    }

    quizzes_path = Path("quizzes.json")
    if not quizzes_path.exists():
        raise FileNotFoundError("quizzes.json must exist with 300 questions before generating topics.json.")

    with open(quizzes_path, "r", encoding="utf-8") as f:
        q_data = json.load(f)
        quizzes = q_data.get("quizzes", [])

    for idx, item in enumerate(quizzes):
        subj = item.get("category", "Science")
        if subj not in topics_db:
            topics_db[subj] = []

        # Ensure valid mode assignment even if missing in quiz item
        assigned_mode = item.get("mode") or modes[idx % len(modes)]
        
        topics_db[subj].append({
            "name": item.get("topic", f"Topic {item.get('id', idx + 1)}"),
            "mode": assigned_mode,
            "definition": item.get("explanation", "Study guide explanation."),
            "key_facts": [item.get("question")],
            "related_topics": [item.get("topic", "General")]
        })

    with open("topics.json", "w", encoding="utf-8") as f:
        json.dump(topics_db, f, indent=2, ensure_ascii=False)

    sci_count = len(topics_db.get("Science", []))
    geo_count = len(topics_db.get("Geography", []))
    total = sci_count + geo_count

    print(f"✅ Generated topics.json | Science: {sci_count} | Geography: {geo_count} | Total: {total}")
    
    if total < 300:
        print(f"⚠️ Warning: Output contains only {total}/300 entries. Ensure quizzes.json has 300 items.")

if __name__ == "__main__":
    generate_300_topics_db()
