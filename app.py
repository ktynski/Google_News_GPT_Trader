import streamlit as st
import pandas as pd
from serpapi import GoogleSearch
from newspaper import Article
from newspaper.article import ArticleException
from concurrent.futures import ThreadPoolExecutor
import re
import transformers
import sentence_transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
import ast
from transformers import BertTokenizerFast
import concurrent.futures
import ast
import json
import time
import base64

# Set OpenAI key
openai.api_key = st.secrets["openai_api_key"]  # Set this up in Streamlit secrets
serpapikey = st.secrets["SERPAPI_API_KEY"]

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


def get_search_results(query, num_pages , serpapikey = st.secrets["SERPAPI_API_KEY"]):
    params = {
        "api_key": serpapi.api_key,
        "engine": "google",
        "google_domain": "google.com",
        "q": query,
        "tbm": "nws",
        "tbs": "qdr:w",
        "hl": "en",
        "gl": "us",
        "start": 0
    }

    all_results = []
    for _ in range(num_pages):
        search = GoogleSearch(params)
        results = search.get_dict()

        # Check if there are results for the query
        if not results.get('news_results'):
            print("No more results.")
            break

        all_results.extend(results['news_results'])
        params["start"] += 10

    return all_results


def scrape_article_text(link):
    article = Article(link)
    try:
        article.download()
        article.parse()
        return article.text
    except ArticleException:
        print(f"Failed to download article: {link}")
        return None


def add_article_text_to_df(df):
    with ThreadPoolExecutor(max_workers=10) as executor:
        df['full_text'] = list(executor.map(scrape_article_text, df['link']))
    return df


def clean_text(text):
    try:
        if not isinstance(text, str):
            text = str(text)
        # Remove newline characters
        text = re.sub(r'\n', ' ', text)

        # Remove non-alphanumeric characters and extra whitespaces
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

        return cleaned_text.strip()
    except Exception as e:
        print(f"Error cleaning text: {text}. Error: {e}")
        return text  # return the original text in case of an error


def deduplicate_dataframe(df):
    # Deduplicate based on titles and URLs
    df.drop_duplicates(subset=['title'], inplace=True)

    # Clean all the text columns
    text_columns = ['title', 'snippet']  # Add or remove column names as per your DataFrame
    for column in text_columns:
        df[column] = df[column].apply(clean_text)

    # Replace 'None' with np.nan
    df.replace('None', np.nan, inplace=True)

    # Drop rows with NaN in 'Link Text' or 'Article Text'
    df = df.dropna(subset=['title', 'snippet'])

    return df


def perform_clustering(df):
    # Combine article titles and article texts
    df['Text'] = df['title'] + " " + df['snippet']

    # Drop rows with empty or None article titles
    df.dropna(subset=['snippet'], inplace=True)

    # Get the combined text from the DataFrame
    combined_text = df['Text'].tolist()

    # Embed the combined text using BERT model
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    embeddings = model.encode(combined_text)

    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)

    # Define the similarity threshold
    similarity_threshold = 0.8  # Adjust as per your requirement

    # Initialize the cluster labels
    cluster_labels = [-1] * len(df)

    # Assign items to clusters based on similarity
    num_clusters = 0
    for i in range(len(df)):
        if cluster_labels[i] == -1:
            cluster_labels[i] = num_clusters
            for j in range(i + 1, len(df)):
                if cluster_labels[j] == -1 and similarity_matrix[i, j] >= similarity_threshold:
                    cluster_labels[j] = num_clusters
            num_clusters += 1

    # Update the DataFrame with cluster labels
    df['Cluster'] = cluster_labels

    return df


def call_with_retry(api_call, *args, **kwargs):
    retries = 3
    for i in range(retries):
        try:
            return api_call(*args, **kwargs)
        except Exception as e:  # Catch any exception
            print(f'Error occurred: {e}, retrying in {i + 1} seconds...')
            time.sleep(i + 1)  # sleep for a bit before retrying
    print(f'Failed after {retries} retries. Skipping this one.')
    return None  # return None if the operation failed even after retries


def truncate_text(text, max_tokens=2500):
    text = str(text)
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(tokens)


def summarize_article(article_text, stock):
    article_text = str(article_text)
    article_text = truncate_text(article_text)
    response = call_with_retry(
        openai.ChatCompletion.create,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"""You are an all-knowing stock analyst with a deep background in qualitative analysis and quantitative analysis.
            You are a world-renowned expert at finding bullish and bearish signals from news articles and news stories. These signals could be anything.
             You will be given an article to summarize. You MUST summarize it with your skill as a stock analyst in mind, making sure you include all relevant information for
             a later algo that will parse the summary. You are an AI assistant that summarizes articles. Please provide a summary of the following article and returning your summary only on points related to {stock}:\n"""},
            {"role": "user", "content": article_text}
        ],
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.5
    )

    if response is None:
        return None  # handle skipped operations as necessary
    summary = response['choices'][0]['message']['content'].strip()
    return summary


def evaluate_cluster_summaries(cluster_summaries):
    cluster_summaries = [summary for summary in cluster_summaries if summary is not None]
    if not cluster_summaries:
        print("No summaries to evaluate.")
        return None

    cluster_summaries_str = "\n".join(cluster_summaries)
    truncated_cluster_summaries = truncate_text(cluster_summaries_str)
    response = call_with_retry(
        openai.ChatCompletion.create,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": '''You are an AI assistant that evaluates the bullish/bearish signals in a cluster of article summaries. Please analyze the following summaries and provide your evaluation in exactly the format:

              Example 1: [{{"Stock": "Apple", "Signal Type": "Bullish", "Explanation": "The company has shown great revenue growth.", "Confidence": 80}}]\n\n

              Example 2: [{{"Stock": "Microsoft", "Signal Type": "Bearish", "Explanation": "The company's profit margins have been decreasing.", "Confidence": 60}}]\n\n

              Example 3: [{{"Stock": "Google", "Signal Type": "Bullish", "Explanation": "New product launch expected.", "Confidence": 40}}]\n\n

              Please reform the following string and return only what the JSON formatted as described in the example without any intro or outro text:'''},

            {"role": "user", "content": f"""{truncated_cluster_summaries} \n\n Valid JSON: \n"""}
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.3
    )

    if response is None:
        return None  # handle skipped operations as necessary

    evaluation = response['choices'][0]['message']['content'].strip()
    print(evaluation)
    return evaluation


def reformat_json(json_string):
    response = call_with_retry(
        openai.ChatCompletion.create,
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": '''You are a highly skilled AI assistant trained to reformat strings into valid JSON format and return ONLY valid JSON.
              Please convert the following string into valid JSON format. Make sure that the keys and values are properly enclosed with double quotes (\\"),
              the items are separated by commas, and the entire JSON is enclosed in square brackets.
              Here are three examples of the correct format:\n\n
              Example 1: [{"Stock": "Apple", "Signal Type": "Bullish", "Explanation": "The company has shown great revenue growth.", "Confidence": 80}]\n\n
              Example 2: [{"Stock": "Microsoft", "Signal Type": "Bearish", "Explanation": "The company's profit margins have been decreasing.", "Confidence": 60}]\n\n
              Example 3: [{"Stock": "Google", "Signal Type": "Bullish", "Explanation": "New product launch expected.", "Confidence": 40}]\n\n
              Please reform the following string and return only what the JSON formatted as described in the example without any intro or outro text:'''
            },

            {"role": "user", "content": f"{json_string} \n\n Valid JSON: \n"}
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.3
    )
    if response is None:
        return None  # handle skipped operations as necessary

    result = response['choices'][0]['message']['content'].strip()
    print("Reformatted JSON")
    return result


def evaluate_cluster(cluster_articles, cluster, stock):
    summaries = cluster_articles.apply(lambda x: summarize_article(x, stock)).tolist()
    cluster_evaluation = evaluate_cluster_summaries(summaries)
    try:
        evaluations = ast.literal_eval(cluster_evaluation)
    except Exception as e:
        print(f"Could not parse string to dictionary: {e}")
        print("Attempting to reformat JSON string using GPT-3...")
        cluster_evaluation = reformat_json(cluster_evaluation)
        try:
            evaluations = ast.literal_eval(cluster_evaluation)
        except Exception as e:
            print(f"Could not parse reformatted string to dictionary: {e}")
            evaluations = [{}]
    return cluster, summaries, evaluations


def truncate_text2(text, max_tokens=6000):
    text = str(text)
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(tokens)


def generate_stock_name_variations(stock_name):
    response = call_with_retry(
        openai.ChatCompletion.create,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a smart AI who can generate variations of a given stock name. Please generate all possible variations in a comma-separated list without any brackets for the stock name: {stock_name}."},
        ],
        max_tokens=50,
        n=1,
        temperature=0.3
    )

    if response is None:
        return None  # handle skipped operations as necessary
    stock_name_variations = response['choices'][0]['message']['content'].strip().split(", ")
    return stock_name_variations


def aggregate_signals(df, stock_name):
    stock_name_variations = generate_stock_name_variations(stock_name)
    bullish_signals = []
    bearish_signals = []

    for i in range(1, 3):
        stock_column = f'Signal {i} Stock'
        type_column = f'Signal {i} Type'
        confidence_column = f'Signal {i} Confidence'

        if stock_column in df.columns and type_column in df.columns and confidence_column in df.columns:
            for name in stock_name_variations:
                df[confidence_column] = pd.to_numeric(df[confidence_column], errors='coerce')

                bullish_df = df[(df[stock_column] == name) & (df[type_column] == 'Bullish') & (df[confidence_column] > 40)]
                bearish_df = df[(df[stock_column] == name) & (df[type_column] == 'Bearish') & (df[confidence_column] > 40)]

                bullish_signals.extend(bullish_df['title'] + " " + bullish_df['snippet'] + " " + bullish_df['Summaries'])
                bearish_signals.extend(bearish_df['title'] + " " + bearish_df['snippet'] + " " + bearish_df['Summaries'])

    bullish_signals_str = " ".join(bullish_signals)
    bearish_signals_str = " ".join(bearish_signals)

    return bullish_signals_str, bearish_signals_str


def generate_reports(bullish_signals, bearish_signals, stock_name):
    # Generate Bullish Report
    bullish_signals = truncate_text(bullish_signals)
    bearish_signals = truncate_text(bearish_signals)
    bullish_response = call_with_retry(
        openai.ChatCompletion.create,
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"""You are an AI assistant that generates a detailed bullish report for a given stock.
            The report should be in markdown format and include an analysis of the bullish signals, potential opportunities, and risks.
             You finish off with a detailed call play that leverages the information/edges you found. Here are the bullish signals for {stock_name}:"""},
            {"role": "user", "content": bullish_signals}
        ],
        max_tokens=2500,
        n=1,
        stop=None,
        temperature=0.7
    )
    if bullish_response is None:
        return None  # handle skipped operations as necessary

    bullish_report = bullish_response['choices'][0]['message']['content'].strip()

    # Generate Bearish Report
    bearish_response = call_with_retry(
        openai.ChatCompletion.create,
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"""You are an AI assistant that generates a detailed bearish report for a given stock.
            The report should be in markdown format and include an analysis of the bearish signals, potential threats, and opportunities.
             You finish off with a detailed put play that leverages the information/edges you found. Here are the bearish signals for {stock_name}:"""},
            {"role": "user", "content": bearish_signals}
        ],
        max_tokens=2500,
        n=1,
        stop=None,
        temperature=0.7
    )

    if bearish_response is None:
        return None  # handle skipped operations as necessary

    bearish_report = bearish_response['choices'][0]['message']['content'].strip()

    return bullish_report, bearish_report


def save_dataframe_as_csv(dataframe, filename):
    # Save the DataFrame as CSV with proper handling of special characters using pandas' to_csv method
    csv_file = dataframe.to_csv(index=False, encoding='utf-8-sig', quotechar='"', quoting=1)
    b64 = base64.b64encode(csv_file.encode()).decode()  # Encode the DataFrame as base64 string
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href


# Streamlit App
st.title("Automatic GPT-4 Analysis of Stocks on Twitter")

# User inputs
num_results = st.sidebar.number_input("Number of Tweets to Scrape", min_value=1, max_value=5000, value=20, step=1)
keyword = st.sidebar.text_input("Keyword to Search", value="GME", max_chars=30)

total_tweets = 0  # Initialize total_tweets variable

if st.button("Scrape Tweets"):
    with st.spinner("Scraping Tweets..."):
        # Scrape tweets here
        # TODO: Replace with your code to scrape tweets using the provided inputs
        # The scraped tweets should be stored in a pandas DataFrame called 'tweets_df'
        tweets_df = pd.DataFrame()

        total_tweets = len(tweets_df)

    st.success("Tweets Scrape Complete!")

st.write(f"Total Tweets Scraped: {total_tweets}")

if total_tweets > 0:
    st.markdown("## Download Tweets Data")
    download_link = save_dataframe_as_csv(tweets_df, "tweets_data.csv")
    st.markdown(download_link, unsafe_allow_html=True)
