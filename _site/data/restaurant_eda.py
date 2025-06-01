import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load data
df = pd.read_csv('restaurants_reviews.csv')

# Clean and add sentiment
df = df.dropna(subset=['review_text', 'rating'])
df['sentiment'] = df['review_text'].apply(lambda text: TextBlob(text).sentiment.polarity)

# Aggregate by review period
agg = df.groupby('review_period_description').agg({
    'rating': 'mean',
    'sentiment': 'mean',
    'review_text': 'count'
}).rename(columns={'review_text': 'review_count'}).reset_index()

# Sort if needed
agg['order'] = agg.index
agg = agg.sort_values('order')

# Plot review/sentiment trends
plt.figure(figsize=(10, 5))
plt.plot(agg['review_period_description'], agg['rating'], marker='o', label='Avg Rating')
plt.plot(agg['review_period_description'], agg['sentiment'], marker='x', label='Avg Sentiment')
plt.xticks(rotation=45, ha='right')
plt.title('Rating and Sentiment Over Time')
plt.legend()
plt.tight_layout()
plt.savefig('review_trends.png')
plt.show()

# Plot review count
plt.figure(figsize=(10, 4))
plt.bar(agg['review_period_description'], agg['review_count'], color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title('Number of Reviews Over Time')
plt.tight_layout()
plt.savefig('sentiment_chart.png')
plt.show()

from wordcloud import WordCloud

# Combine all review text into one string
all_text = ' '.join(df['review_text'].dropna())

# Generate the WordCloud
wc = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    colormap='plasma',
    max_words=100
).generate(all_text)

# Plot and save
plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig('review_wordcloud.png')
plt.show()
