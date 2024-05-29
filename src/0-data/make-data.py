import pandas as pd
import requests
from bs4 import BeautifulSoup

base_url = "https://www.airlinequality.com/airline-reviews/british-airways/page/"
pages = 20

contents = []

for page in range(1, pages+1):
    print(f"----Scrapping page {page}----")

    url = base_url + str(page)

    response = requests.get(url)
    content = response.content

    parsed_content = BeautifulSoup(content, 'html.parser')
    title = [title.get_text() for title in parsed_content.find_all("h2", {"class": "text_header"})]
    review = [review.get_text() for review in parsed_content.find_all("div", {"class": "text_content"})]

    reviews = list(map(' '.join, zip(title, review)))
    contents += reviews

    print(f"    {len(reviews)} in page {page}")

df = pd.DataFrame({
    'reviews': contents
})

df.to_parquet('./data/1-raw/ba-reviews.parquet')
