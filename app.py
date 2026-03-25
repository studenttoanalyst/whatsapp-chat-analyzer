import streamlit as st
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import emoji

import Preprocessing
import helper


# -------------------- App Config --------------------
st.sidebar.title("Whatsapp Chat Analyzer")


# -------------------- File Upload --------------------
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt", "zip"])

if uploaded_file is not None:

    # Handle ZIP file
    if uploaded_file.name.endswith('.zip'):
        with zipfile.ZipFile(uploaded_file) as z:
            txt_file = next((f for f in z.namelist() if f.endswith('.txt')), None)

            if txt_file:
                with z.open(txt_file) as f:
                    data = f.read().decode("utf-8")
            else:
                st.error("No TXT file found inside ZIP")
                st.stop()
    else:
        data = uploaded_file.getvalue().decode("utf-8")

    # -------------------- Preprocessing --------------------
    df = Preprocessing.preprocess(data)

    # Remove system messages
    df = df[df['user'] != 'group_notification']

    # User selection
    user_list = sorted(df['user'].dropna().unique().tolist())
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    # -------------------- Run Analysis --------------------
    if st.sidebar.button("Show Analysis"):

        # -------------------- Top Statistics --------------------
        num_messages, words, media_msgs, links = helper.fetch_stats(selected_user, df)

        st.title("Top Statistics")

        st.markdown("""
        <style>
        .stat-number { font-size: 50px; font-weight: bold; }
        .stat-label { font-size: 25px; }
        </style>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        col1.markdown("<p class='stat-label'>Total Messages</p>", unsafe_allow_html=True)
        col1.markdown(f"<p class='stat-number'>{num_messages}</p>", unsafe_allow_html=True)

        col2.markdown("<p class='stat-label'>Total Words</p>", unsafe_allow_html=True)
        col2.markdown(f"<p class='stat-number'>{words}</p>", unsafe_allow_html=True)

        col3.markdown("<p class='stat-label'>Media Shared</p>", unsafe_allow_html=True)
        col3.markdown(f"<p class='stat-number'>{media_msgs}</p>", unsafe_allow_html=True)

        col4.markdown("<p class='stat-label'>Links Shared</p>", unsafe_allow_html=True)
        col4.markdown(f"<p class='stat-number'>{links}</p>", unsafe_allow_html=True)


        # -------------------- Timeline --------------------
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)

        # fig, ax = plt.subplots()
        # ax.plot(timeline['time'], timeline['message'])
        # plt.xticks(rotation=90)
        # st.pyplot(fig)
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(timeline['time'], timeline['message'])

        # 🔥 MAIN FIX (overlap remove)
        ax.set_xticks(range(0, len(timeline['time']), 2))  # har 2nd label
        ax.set_xticklabels(timeline['time'][::2], rotation=45)

        plt.tight_layout()
        st.pyplot(fig)
        st.title("Daily Timeline")
        daily = helper.daily_timeline(selected_user, df)

        fig, ax = plt.subplots()
        ax.plot(daily['only_date'], daily['message'])
        plt.xticks(rotation=90)
        st.pyplot(fig)


        # -------------------- Activity Map --------------------
        st.title("Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Most Active Day")
            busy_day = helper.week_activity_map(selected_user, df)

            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation=90)
            st.pyplot(fig)

        with col2:
            st.subheader("Most Active Month")
            busy_month = helper.month_activity_map(selected_user, df)

            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values)
            plt.xticks(rotation=90)
            st.pyplot(fig)


        # -------------------- Heatmap --------------------
        st.title("Weekly Activity Heatmap")
        heatmap = helper.activity_heatmap(selected_user, df)

        if heatmap.empty:
            st.warning("No activity available")
        else:
            fig, ax = plt.subplots()
            sns.heatmap(heatmap, ax=ax)
            st.pyplot(fig)


        # -------------------- Most Busy Users --------------------
        if selected_user == "Overall":
            st.title("Most Busy Users")

            x, new_df = helper.most_busy_users(df)
            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values)
                plt.xticks(rotation=90)
                st.pyplot(fig)

            with col2:
                st.dataframe(new_df)


        # -------------------- WordCloud --------------------
        st.title("WordCloud")
        wc = helper.create_wordcloud(selected_user, df)

        if wc:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(wc)
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No data for WordCloud")


        # -------------------- Most Common Words --------------------
        st.title("Most Common Words")
        common_df = helper.most_common_words(selected_user, df)

        if not common_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(common_df['word'], common_df['count'])
            ax.set_xlabel("Count")
            ax.set_ylabel("Word")
            st.pyplot(fig)
        else:
            st.info("No words available")


        # -------------------- Emoji Analysis --------------------
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)

        col1, col2 = st.columns(2)

        # Dataframe view (actual emojis)
        with col1:
            st.dataframe(emoji_df)

        # Chart view (safe labels)
        with col2:
            if not emoji_df.empty:
                top_df = emoji_df.head(5).copy()
                top_df['label'] = [f"Emoji {i+1}" for i in range(len(top_df))]

                fig, ax = plt.subplots(figsize=(7, 6))
                sns.barplot(x='label', y='count', data=top_df, ax=ax)

                ax.set_title(f"Top Emojis for {selected_user}")
                ax.set_xlabel("Emoji")
                ax.set_ylabel("Count")

                for i, row in top_df.iterrows():
                    ax.text(i, row['count'], str(row['count']), ha='center')

                st.pyplot(fig)
            else:
                st.info("No emojis found")