from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji

extract = URLExtract()


# -------------------- Statistics --------------------
def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]

    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)


def most_busy_users(df):
    top_users = df['user'].value_counts().head()

    percentage_df = (
        (df['user'].value_counts() / df.shape[0]) * 100
    ).round(2).reset_index().rename(columns={'index': 'name', 'user': 'percent'})

    return top_users, percentage_df


# -------------------- WordCloud --------------------
def create_wordcloud(selected_user, df):
    try:
        with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
            stop_words = set(f.read().split())
    except FileNotFoundError:
        stop_words = set()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['message'].notna()]

    if df.empty:
        return None

    text = df['message'].str.cat(sep=' ').strip()

    if not text:
        return None

    wc = WordCloud(
        width=500,
        height=500,
        min_font_size=10,
        background_color='white',
        stopwords=stop_words
    )

    try:
        return wc.generate(text)
    except ValueError:
        return None


# -------------------- Text Analysis --------------------
def most_common_words(selected_user, df):
    try:
        with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
            stop_words = set(f.read().split())
    except FileNotFoundError:
        stop_words = set()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[
        (df['user'] != 'group_notification') &
        (df['message'] != '<Media omitted>\n')
    ]

    words = []
    for message in df['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    return pd.DataFrame(Counter(words).most_common(20), columns=['word', 'count'])


# -------------------- Emoji Analysis --------------------
def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([char for char in message if emoji.is_emoji(char)])

    emoji_count = Counter(emojis).most_common()

    if not emoji_count:
        return pd.DataFrame(columns=['emoji', 'count'])

    return pd.DataFrame(emoji_count, columns=['emoji', 'count'])


# -------------------- Timelines --------------------
def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = (
        df.groupby(['year', 'month_num', 'month'])['message']
        .count()
        .reset_index()
    )

    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)

    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.groupby('only_date')['message'].count().reset_index()


# -------------------- Activity Maps --------------------
def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    heatmap = df.pivot_table(
        index='day_name',
        columns='period',
        values='message',
        aggfunc='count'
    ).fillna(0)

    return heatmap