import pandas as pd
import nltk
import os
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from tkinter import Tk, filedialog

nltk.download('vader_lexicon')
nltk.download('punkt')

def load_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ Error: File '{file_path}' not found. Please select the correct file.")
        
        Tk().withdraw()
        file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

        if not file_path:
            raise FileNotFoundError("❌ No file selected. Exiting.")

    print(f"✅ Loading dataset from: {file_path}")

    try:
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin1")

    if "review" not in df.columns:
        raise KeyError("❌ The dataset does not contain a 'review' column. Please check the file.")

    df.dropna(subset=["review"], inplace=True)

    return df

file_path = r"C:/Users/user/Desktop/Slashmark/reviews.csv"
df = load_dataset(file_path)

sia = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    if isinstance(text, str):
        score = sia.polarity_scores(text)["compound"]
        return "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"
    return "Neutral"

def get_textblob_sentiment(text):
    if isinstance(text, str):
        score = TextBlob(text).sentiment.polarity
        return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
    return "Neutral"

print("🔍 Performing Sentiment Analysis...")
df["VADER Sentiment"] = df["review"].apply(get_vader_sentiment)
df["TextBlob Sentiment"] = df["review"].apply(get_textblob_sentiment)

print("\n📊 Sample Results:")
print(df.head())

output_file = "restaurant_reviews_with_sentiment.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Sentiment analysis results saved to '{output_file}'")

def plot_sentiment_distribution(df, column_name, title):
    sentiment_counts = df[column_name].value_counts()
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'blue'])

    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2 - 0.1, bar.get_height() + 0.5, 
                 str(bar.get_height()), fontsize=12, fontweight='bold')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

plot_sentiment_distribution(df, "VADER Sentiment", "📊 VADER Sentiment Distribution")
plot_sentiment_distribution(df, "TextBlob Sentiment", "📊 TextBlob Sentiment Distribution")