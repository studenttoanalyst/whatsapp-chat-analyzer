from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji

extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

# def create_wordcloud(selected_user, df):
#
#     # Stop words file
#     with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
#         stop_words = set(f.read().split())
#
#     # Filter user
#     if selected_user != 'Overall':
#         df = df[df['user'] == selected_user]
#
#     # Combine all messages
#     text = df['message'].str.cat(sep=' ')
#
#     # 🔹 Check if there is any text
#     if not text.strip():  # empty or whitespace only
#         return None  # No WordCloud
#
#     wc = WordCloud(
#         width=500,
#         height=500,
#         min_font_size=10,
#         background_color='white',
#         stopwords=stop_words
#     )
#     df_wc = wc.generate(text)
#     return df_wc

# Updated
def create_wordcloud(selected_user, df):
    # Stop words
    try:
        with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
            stop_words = set(f.read().split())
    except FileNotFoundError:
        stop_words = set()

    # Filter user
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Drop empty messages
    df = df[df['message'].notna()]

    if df.empty:
        return None  # no messages at all

    # Combine all messages
    text = df['message'].str.cat(sep=' ').strip()

    # 🔹 Check if there is any text left
    if not text:
        return None  # safe exit

    wc = WordCloud(
        width=500,
        height=500,
        min_font_size=10,
        background_color='white',
        stopwords=stop_words,
        # font_path='NotoNaskhArabic-Regular.ttf'  # Unicode-safe font for live server
    )

    try:
        df_wc = wc.generate(text)
    except ValueError:
        df_wc = None  # fallback if WordCloud fails
    return df_wc
def most_common_words(selected_user, df):

    with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
        stop_words = set(f.read().split())

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    # 🔹 Make sure columns are explicitly named
    most_common_df = pd.DataFrame(Counter(words).most_common(20), columns=['word', 'count'])
    return most_common_df

# def emoji_helper(selected_user,df):
#     if selected_user != 'Overall':
#         df = df[df['user'] == selected_user]
#
#     emojis = []
#     for message in df['message']:
#         emojis.extend([c for c in message if emoji.is_emoji(c)])
#
#     emoji_df=pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
#     return emoji_df
def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if emoji.is_emoji(c)])

    emoji_count = Counter(emojis).most_common()

    # 🔥 IMPORTANT FIX
    if len(emoji_count) == 0:
        return pd.DataFrame(columns=['emoji', 'count'])

    emoji_df = pd.DataFrame(emoji_count, columns=['emoji', 'count'])
    return emoji_df
def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap












