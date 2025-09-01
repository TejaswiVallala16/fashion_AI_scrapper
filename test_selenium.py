from fastapi import FastAPI
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

app = FastAPI()

def scrape_infinite_scroll(url: str, max_scrolls: int = 5):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")   # run without opening browser
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    all_items = []
    last_height = driver.execute_script("return document.body.scrollHeight")

    for i in range(max_scrolls):
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # wait for new content to load

        # Collect items (example: quotes.toscrape.com/js/)
        quotes = driver.find_elements(By.CLASS_NAME, "quote")
        for q in quotes:
            text = q.find_element(By.CLASS_NAME, "text").text
            author = q.find_element(By.CLASS_NAME, "author").text
            all_items.append({"text": text, "author": author})

        # Check if page height stopped changing (end of content)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    driver.quit()
    return all_items


# 📌 API Endpoint
@app.get("/scrape")
def scrape_api(url: str, max_scrolls: int = 5):
    """
    Example:
    http://127.0.0.1:8000/scrape?url=http://quotes.toscrape.com/js/&max_scrolls=3
    """
    data = scrape_infinite_scroll(url, max_scrolls)
    return {"count": len(data), "data": data[:5]}  # return first 5 items for preview
