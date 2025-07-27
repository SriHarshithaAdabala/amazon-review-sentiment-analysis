import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentiment_utils import analyze_sentiment

st.set_page_config(page_title="Amazon Review Sentiment Analyzer", layout="centered")

st.title("🛍️ Amazon Product Review Sentiment Analyzer")

@st.cache_data
def load_data():
    df = pd.read_csv("reviews.csv")
    df = df[['Score', 'Text']].dropna()
    df = df.rename(columns={'Text': 'Review'})
    return df.head(5000)

data = load_data()
data['Sentiment'] = data['Review'].apply(analyze_sentiment)

st.sidebar.header("🔍 Filter Reviews")
sentiment_filter = st.sidebar.multiselect("Select Sentiment", options=["Positive", "Negative", "Neutral"], default=["Positive", "Negative", "Neutral"])
filtered_data = data[data['Sentiment'].isin(sentiment_filter)]

if st.checkbox("Show Raw Data"):
    st.write(filtered_data)

st.subheader("📊 Sentiment Distribution")
sentiment_counts = filtered_data['Sentiment'].value_counts()
fig, ax = plt.subplots()
ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral', 'gold'])
st.pyplot(fig)

st.subheader("☁️ Word Cloud")
all_text = " ".join(filtered_data['Review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis("off")
st.pyplot(fig_wc)

st.subheader("📝 Try Your Own Review")
user_input = st.text_area("Enter a product review:")
if user_input:
    result = analyze_sentiment(user_input)
    st.markdown(f"**Predicted Sentiment:** `{result}`")
