import requests
from bs4 import BeautifulSoup
from pathlib import Path

def scrape_iac_questions():
    url = "https://www.iacompetitions.com/ems-national-geography-bee-past-questions/"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    print(f"Fetching web data from {url}...")
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to fetch webpage. Status code: {response.status_code}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract text content from question containers or paragraphs
    text_blocks = [p.get_text().strip() for p in soup.find_all(["p", "li", "div"]) if p.get_text().strip()]
    extracted_text = "\n\n".join(text_blocks)

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    output_path = data_dir / "web_past_questions.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)
        
    print(f"Successfully saved scraped web questions to {output_path}")

if __name__ == "__main__":
    scrape_iac_questions()
