import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from selenium import webdriver
import bs4

from openai import OpenAI
import os
# import pytimer

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# t = pytimer.Timer()

def manual_get_page_text(url: str) -> str:
    """
    Manually grab page text.

    Not used currently.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Define keyword variations (case-insensitive search)
    keywords = ["jobs-description__container"]  
    result_text = []

    # Search all tags for keywords in a case-insensitive manner
    for tag in soup.find_all(['div', 'p', 'span', 'section']):
        tag_text = tag.get_text(strip=True).lower()  # Normalize text to lowercase
        if any(keyword in tag_text for keyword in keywords):
            result_text.append(tag.get_text(strip=True))

    return "\n".join(result_text) if result_text else "No relevant content found."

def scrape_website(url: str) -> str:
    """
    Scrape the website for the job description.
    """
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 AI Interviewer UCI IrvineHacks'})
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    return soup.body.get_text(' ', strip=True)

    # dr = webdriver.Chrome()
    # dr.get(url)
    # content = (dr.page_source, "html.parser")
    # return content

def ai_summarize_url(content: str) -> str|None:
    """
    Use GPT-4o to attempt to summarize the job description.
    """
    prompt = f"Give me the entire job description for the first job listing on the webpage with its text content given below. \
    Ignore any unrelated text to the job description and ignore all job listings after the first and just provide only  \
    the description for this job. Here is the content: {content}"

    # query openAI api
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [{"role": "user", "content": prompt}],
        )
   
    return response.choices[0].message.content

def search_page(url: str) -> str|None:
    """
    Returns None if the request fails, otherwise, if the request is successful,
    returns the job description as summarized by GPT-4o.
    """
    # t.begin_timer()   # debug
    data = scrape_website(url)
    if data is not None:
        text = ai_summarize_url(scrape_website(url))
    else:
        print("No data found.")
        return None
    # t.end_timer()   # debug

    print("retrieved: ", text)
    return text

if __name__ == "__main__":
    url = "https://careers.irvinecompany.com/job/Irvine-Community-Relations-Representative-%28%2420_67-%2425_58%29-CA-92618/705985900/"
    print(search_page(url))