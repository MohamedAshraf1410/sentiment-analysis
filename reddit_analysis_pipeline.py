import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('corpus')

import praw

print(nltk.data.path)
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Step 1: Data Collection from Reddit
def collect_reddit_posts(subreddit_name, max_posts, output_file, client_id, client_secret, user_agent):
    reddit = praw.Reddit(client_id="0COYGSoCCvvFMhXZGuAJrg", client_secret="qK3ORw4aXrnTVfGLuxKT5JhQRYNQmg", user_agent="python:reddit_analysis:v1.0")
    subreddit = reddit.subreddit(subreddit_name)
    posts_list = []
    for post in subreddit.hot(limit=max_posts):
        posts_list.append([post.title, post.selftext, post.author.name if post.author else "N/A", str(post.created_utc)])
    posts_df = pd.DataFrame(posts_list, columns=["Title", "Body", "Author", "Date"])
    posts_df.to_csv(output_file, index=False)
    print(f"Data collection complete. Saved {len(posts_df)} posts to {output_file}.")
    return posts_df

# Step 2: Text Preprocessing
def preprocess_text(text):
    # Ensure the text is a string and handle missing values
    if not isinstance(text, str):
        return ""  # If it's not a string, return an empty string
    
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"[^A-Za-z\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert text to lowercase
    nltk.download('punkt')
    tokens = word_tokenize(text)  # Tokenize the text into words
    stop_words = set(stopwords.words("english"))  # Get the set of stop words
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize words
    return " ".join(tokens)  # Join the tokens back into a single string

# Step 2: Preprocess Dataset (applies the text preprocessing to the 'Body' column)
def preprocess_dataset(input_file, output_file):
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("stopwords")
    posts_df = pd.read_csv(input_file)
    posts_df["Processed_Content"] = posts_df["Body"].apply(preprocess_text)  # Apply the text preprocessing
    posts_df.to_csv(output_file, index=False)  # Save the processed data to a new file
    print(f"Text preprocessing complete. Processed data saved to {output_file}.")
    return posts_df    
    

# Step 3: Sentiment Analysis
def analyze_sentiment(text, sia):
    if not isinstance(text, str):  # Check if the text is a string
        text = str(text)  # Convert it to a string if it's not
    sentiment = sia.polarity_scores(text)  # Perform sentiment analysis
    if sentiment["compound"] >= 0.05:
        return "Positive"
    elif sentiment["compound"] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def sentiment_analysis(input_file, output_file):
    nltk.download("vader_lexicon")  # Download the Vader lexicon if it's not already installed
    sia = SentimentIntensityAnalyzer()  # Initialize the SentimentIntensityAnalyzer
    posts_df = pd.read_csv(input_file)
    
    # Ensure that all processed content is treated as a string
    posts_df["Processed_Content"] = posts_df["Processed_Content"].fillna("").astype(str)  # Replace NaNs with empty strings and ensure all are strings
    
    posts_df["Sentiment"] = posts_df["Processed_Content"].apply(lambda x: analyze_sentiment(x, sia))  # Apply sentiment analysis
    posts_df.to_csv(output_file, index=False)  # Save the results to a CSV file
    print(f"Sentiment analysis complete. Results saved to {output_file}.")
    return posts_df

   
    

# Visualization
def visualize_sentiment(posts_df):
    sentiment_counts = posts_df["Sentiment"].value_counts()
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind="pie", autopct="%1.1f%%", startangle=90, colors=["lightblue", "lightgreen", "salmon"])
    plt.title("Sentiment Distribution")
    plt.ylabel("")
    plt.show()

# Main Execution Flow
if __name__ == "__main__":
    # Step 1: Data Collection
    subreddit_name = "Nike"
    max_posts = 10
    raw_file = "reddit_nike_posts.csv"
    client_id = "YOUR_CLIENT_ID"
    client_secret = "YOUR_CLIENT_SECRET"
    user_agent = "YOUR_USER_AGENT"
    collect_reddit_posts(subreddit_name, max_posts, raw_file, client_id, client_secret, user_agent)

    # Step 2: Preprocess Posts
    processed_file = "reddit_nike_posts_processed.csv"
    preprocess_dataset(raw_file, processed_file)

    # Step 3: Sentiment Analysis
    sentiment_file = "reddit_nike_posts_with_sentiment.csv"
    posts_with_sentiment = sentiment_analysis(processed_file, sentiment_file)

    # Step 4: Visualization
    visualize_sentiment(posts_with_sentiment)