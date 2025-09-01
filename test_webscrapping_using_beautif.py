import requests
from bs4 import BeautifulSoup
import json

def scrape_quotes():
    base_url = "http://quotes.toscrape.com/page/{}/"
    page = 1
    all_quotes = []

    while True:
        url = base_url.format(page)
        res = requests.get(url)

        if res.status_code != 200:  # stop when no more pages
            break

        soup = BeautifulSoup(res.text, "html.parser")
        quotes = soup.select(".quote")

        if not quotes:  # no more quotes found
            break

        for q in quotes:
            all_quotes.append({
                "text": q.find("span", class_="text").get_text(strip=True),
                "author": q.find("small", class_="author").get_text(strip=True),
                "tags": [tag.get_text(strip=True) for tag in q.select(".tags a.tag")]
            })

        print(f"✅ Scraped page {page}")
        page += 1

    return all_quotes


if __name__ == "__main__":
    data = scrape_quotes()

    # Save to JSON file
    with open("quotes.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n🎯 Scraped {len(data)} quotes. Saved to quotes.json")
