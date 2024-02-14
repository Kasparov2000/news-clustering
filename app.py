import os
import sys

from flask import Flask, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Set your News API key as an environment variable
news_api_key = os.environ.get('NEWS_API_KEY')

import json


import json

def get_news():
    news_api_key = os.environ.get('NEWS_API_KEY')

    if not news_api_key:
        return "Error: News API key not provided."

    base_url = 'https://newsapi.org/v2/everything'
    params = {
        'q': 'music',
        'from': '2024-01-16',
        'sortBy': 'publishedAt',
        'apiKey': news_api_key,
        'language': 'en',
    }

    try:
        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            raise Exception(f"Error fetching news articles. Status code: {response.status_code}")

        articles = response.json().get('articles', [])

        # Save the response to a local JSON file
        with open('articles_response.json', 'w', encoding='utf-8') as json_file:
            json.dump(articles, json_file, ensure_ascii=False)

    except Exception as e:
        print(f"Error during API request: {e}")

        # Try to load from the local JSON file with specified encoding
        try:
            with open('articles_response.json', 'r', encoding='utf-8') as json_file:
                articles = json.load(json_file)
            print("Loaded articles from local file.")
        except FileNotFoundError:
            return f"Error: Could not fetch articles from API, and no local file found."
        except json.JSONDecodeError as e:
            return f"Error decoding local JSON file: {e}"
        except UnicodeDecodeError as e:
            return f"Error decoding local JSON file: {e}"

    # Extract image URL and page URL for each article
    formatted_articles = []
    for article in articles:  # Limit to the first 10 articles
        try:
            title = article['title']
            image_url = article['urlToImage'] if 'urlToImage' in article else ''
            page_url = article['url']
            formatted_articles.append({'title': title, 'image_url': image_url, 'page_url': page_url})
        except TypeError:
            print("Error processing article:", article)

    return formatted_articles

def cluster_articles(articles):

    articles = [article['title'] for article in articles if article['title'].strip()]

    if not articles:
        return []

    vectorizer = TfidfVectorizer(stop_words='english', min_df=2)  # Adjust min_df as needed

    try:
        X = vectorizer.fit_transform(articles)
    except ValueError as e:
        print(f"Error vectorizing articles: {e}")
        return []

    kmeans = KMeans(n_clusters=5)  # Adjust the number of clusters as needed

    try:
        kmeans.fit(X)
        clusters = kmeans.labels_
    except ValueError as e:
        print(f"Error clustering articles: {e}")
        clusters = []

    print("Clusters:", clusters)
    return clusters.tolist()  # Convert numpy array to a regular list


@app.route('/')
def index():
    articles = get_news()
    clusters = cluster_articles(articles)

    if not clusters:
        return "Error clustering articles. Please check the server logs."

    clustered_articles = {cluster: [] for cluster in set(clusters)}

    for article, cluster in zip(articles, clusters):
        clustered_articles[cluster].append(article)

    # Limit each cluster to 10 articles
    clustered_articles_10 = {cluster: articles[:10] for cluster, articles in clustered_articles.items()}

    print("Clustered Articles:", clustered_articles_10, flush=True, file=sys.stderr)

    return render_template('index.html', clustered_articles=clustered_articles_10, enumerate=enumerate)


if __name__ == '__main__':
    app.run(debug=True)
