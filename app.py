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
from transformers import BertTokenizerFast
import concurrent.futures
import ast
import json
import time
import base64

# Set OpenAI key
openai.api_key = st.secrets["openai_api_key"]  # Set this up in Streamlit secrets
GOOGLE_API_KEY = st.secrets["SERPAPI_API_KEY"]


params = {
    "api_key": GOOGLE_API_KEY,
    "engine": "google",
    "google_domain": "google.com",
    "tbm": "nws",
    "tbs": "qdr:w",
    "hl": "en",
    "gl": "us",
    "start": 0
}

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


def get_search_results(query, num_pages):
    params["q"] = query
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
            for j in range(i+1, len(df)):
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
            print(f'Error occurred: {e}, retrying in {i+1} seconds...')
            time.sleep(i+1)  # sleep for a bit before retrying
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
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": f"""You are an all-knowing stock analyst with a deep background in qualitative analysis and quantitative analysis.
            You are a world renowned expert at finding bullish and bearish signals from news articles and news stories. These signals could be anything.
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


import json

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

              Please reform the following string and return only what the json formatted as described in the example without any intro or outro text:'''},

              {"role": "user", "content": f"""{truncated_cluster_summaries} \n\n Valid Json: \n"""}
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

    try:
        parsed_evaluation = json.loads(evaluation)
        return parsed_evaluation
    except json.JSONDecodeError:
        print("Failed to parse evaluation as JSON.")
        return None



def reformat_json(json_string):
    response = call_with_retry(
        openai.ChatCompletion.create,
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": '''You are a highly skilled AI assistant trained to reformat strings into valid JSON format and return ONLY valid json.
              Please convert the following string into a valid JSON format. Make sure that the keys and values are properly enclosed with double quotes (\\"),
              the items are separated by commas, and the entire JSON is enclosed in square brackets.
              Here are three examples of the correct format:\n\n
              Example 1: [{"Stock": "Apple", "Signal Type": "Bullish", "Explanation": "The company has shown great revenue growth.", "Confidence": 80}]\n\n
              Example 2: [{"Stock": "Microsoft", "Signal Type": "Bearish", "Explanation": "The company's profit margins have been decreasing.", "Confidence": 60}]\n\n
              Example 3: [{"Stock": "Google", "Signal Type": "Bullish", "Explanation": "New product launch expected.", "Confidence": 40}]\n\n
              Please reform the following string and return only what the json formatted as described in the example without any intro or outro text:'''
            },

            {"role": "user", "content": f"{json_string} \n\n Valid Json: \n"}
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.3
    )
    if response is None:
        return None  # handle skipped operations as necessary

    result = response['choices'][0]['message']['content'].strip()
    print("Reformatted Json")
    return result

def evaluate_cluster(cluster_articles, cluster, stock):
    summaries = cluster_articles.apply(lambda x: summarize_article(x, stock)).tolist()
    cluster_evaluation = evaluate_cluster_summaries(summaries)
    try:
        if cluster_evaluation is not None:
            evaluations = json.loads(cluster_evaluation)
        else:
            evaluations = [{}]
    except json.JSONDecodeError as e:
        print(f"Could not parse string to dictionary: {e}")
        print("Attempting to reformat JSON string using GPT-3...")
        cluster_evaluation = reformat_json(cluster_evaluation)
        try:
            evaluations = json.loads(cluster_evaluation)
        except json.JSONDecodeError as e:
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
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": f"You are a smart AI who can generate variations of a given stock name. Please generate all possible variations in a comma separated list without any brackets for the stock name: {stock_name}."},
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
             You finish off with a detailed call play that leverages the information/edges you found. Here are the bullish signals for the {stock_name}:"""},
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
             You finish off with a detailed put play that leverages the information/edges you found. Here are the bearish signals for the {stock_name}:"""},
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


def main():
    # Set the page title and icon
    st.set_page_config(page_title="Stock Analysis Web App", page_icon=":chart_with_upwards_trend:")

    # Render the app title
    st.title("Stock Analysis Web App")

    # Render a sidebar to input variables
    st.sidebar.header("Input Variables")

    # Get user input for stock symbol
    stock = st.sidebar.text_input("Enter Stock Symbol", "NVDA")

    # Get user input for number of search result pages
    num_pages = st.sidebar.number_input("Number of Search Result Pages", min_value=1, max_value=10, value=5)

    # Perform the stock analysis when the user clicks the "Run Analysis" button
    if st.sidebar.button("Run Analysis"):
        # Get the search results
        results = get_search_results(stock, num_pages)
        stock_name_variations = generate_stock_name_variations(stock)

        # Assuming the variable 'results' contains your list of dictionaries
        df = pd.DataFrame(results)

        # Assuming 'df' is your DataFrame
        df = add_article_text_to_df(df)

        # Assuming you already have the DataFrame named 'df' from the scraping code
        df_deduplicated = deduplicate_dataframe(df)
        df_clustered = perform_clustering(df_deduplicated)

        # Specify the file path to save the CSV
        csv_file = 'clustered_articles.csv'

        # Save the DataFrame as CSV with proper handling of special characters using pandas' to_csv method
        df_clustered.to_csv(csv_file, index=False, encoding='utf-8-sig', quotechar='"', quoting=1)

        # Assuming 'df_clustered' is your DataFrame with the columns 'Cluster' and 'Article Text'
        clusters = df_clustered['Cluster'].unique()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks to the executor for each cluster
            futures = {executor.submit(evaluate_cluster, df_clustered[df_clustered['Cluster'] == cluster]['full_text'], cluster, stock): cluster for cluster in clusters}

        cluster_evaluations = {}

        for future in concurrent.futures.as_completed(futures):
            cluster, summaries, evaluations = future.result()

            if None in summaries:
                summaries = [summary for summary in summaries if summary is not None]
                print("Some summaries were not obtained.")
            if evaluations is None:
                print("Evaluation for this cluster was not obtained.")

            cluster_evaluations[cluster] = evaluations

        # Assuming 'df_clustered' is your DataFrame with the columns 'Cluster' and 'Article Text'
        df_clustered['Summaries'] = df_clustered['full_text'].apply(lambda x: summarize_article(x, stock))

        # Assuming 'df_clustered' is your DataFrame with the columns 'Cluster' and 'Article Text'
        df_clustered['Summaries'] = df_clustered['full_text'].apply(lambda x: summarize_article(x, stock))

        df_clustered.to_csv(csv_file, index=False, encoding='utf-8-sig', quotechar='"', quoting=1)

        # Assuming 'df_clustered' is your DataFrame with the columns 'Cluster' and 'Summaries'
        df_clustered = pd.read_csv(csv_file)

        bullish_signals, bearish_signals = aggregate_signals(df_clustered, stock)

        bullish_report, bearish_report = generate_reports(bullish_signals, bearish_signals, stock)

        # Render the analysis results
        st.header("Analysis Results")

        st.subheader("Clustered Articles")
        st.write(df_clustered)

        st.subheader("Cluster Evaluations")
        st.write(cluster_evaluations)

        st.subheader("Bullish Signals")
        st.write(bullish_signals)

        st.subheader("Bearish Signals")
        st.write(bearish_signals)

        st.subheader("Bullish Report")
        st.write(bullish_report)

        st.subheader("Bearish Report")
        st.write(bearish_report)


if __name__ == "__main__":
    main()
