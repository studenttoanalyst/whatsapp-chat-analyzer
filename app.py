import streamlit as st
import Preprocessing,helper
import matplotlib.pyplot as plt
import seaborn as sns
import emoji

st.sidebar.title("Whatsapp Chat Analyzer")

# uploaded_file = st.sidebar.file_uploader("Choose a file")
# if uploaded_file is not None:
#     bytes_data = uploaded_file.getvalue()
#     data = bytes_data.decode("utf-8")
#     df = Preprocessing.preprocess(data)
import zipfile
import streamlit as st

uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt", "zip"])

if uploaded_file is not None:

    # If ZIP file
    if uploaded_file.name.endswith('.zip'):
        with zipfile.ZipFile(uploaded_file) as z:
            txt_file = None

            # find first txt file
            for file in z.namelist():
                if file.endswith('.txt'):
                    txt_file = file
                    break

            if txt_file:
                with z.open(txt_file) as f:
                    data = f.read().decode("utf-8")
            else:
                st.error("No TXT file found inside ZIP")
                st.stop()

    # If TXT file
    else:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")

    # Process data
    df = Preprocessing.preprocess(data)

    # fetch unique users
    # remove unwanted users from dataframe first
    df = df[df['user'] != 'group_notification']

    # fetch unique users
    user_list = sorted(df['user'].dropna().unique().tolist())

    # add overall option
    user_list.insert(0, "Overall")

    # sidebar selection
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        st.markdown("""
        <style>
        .stat-number {
            font-size: 50px;
            font-weight: bold;
        }
        .stat-label {
            font-size: 25px;
        }
        </style>
        """, unsafe_allow_html=True)

        with col1:
            st.markdown("<p class='stat-label'>Total Messages</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='stat-number'>{num_messages}</p>", unsafe_allow_html=True)

        with col2:
            st.markdown("<p class='stat-label'>Total Words</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='stat-number'>{words}</p>", unsafe_allow_html=True)

        with col3:
            st.markdown("<p class='stat-label'>Media Shared</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='stat-number'>{num_media_messages}</p>", unsafe_allow_html=True)

        with col4:
            st.markdown("<p class='stat-label'>Links Shared</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='stat-number'>{num_links}</p>", unsafe_allow_html=True)
        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)

        if user_heatmap.empty:
            st.warning("No activity available for this selection 😕")
        else:
            fig, ax = plt.subplots()
            sns.heatmap(user_heatmap, ax=ax)
            st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("WordCloud")
        df_wc = helper.create_wordcloud(selected_user, df)

        if df_wc is not None:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(df_wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No messages available to generate WordCloud for this user")

        # Most common words
        st.title("Most Common Words")

        most_common_df = helper.most_common_words(selected_user, df)

        if not most_common_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(most_common_df['word'], most_common_df['count'])
            ax.set_xlabel("Count", fontsize=14)
            ax.set_ylabel("Word", fontsize=14)
            ax.set_title(f"Top Words for {selected_user}", fontsize=18, fontweight='bold')
            st.pyplot(fig)
        else:
            st.info("No words available to display")
        # # emoji analysis
        # emoji_df = helper.emoji_helper(selected_user,df)
        # st.title("Emoji Analysis")
        #
        # col1,col2 = st.columns(2)
        #
        # with col1:
        #     st.dataframe(emoji_df)
        # # with col2:
        # #     fig,ax = plt.subplots()
        # #     ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
        # #     st.pyplot(fig)
        # # with col2:
        # #     if not emoji_df.empty:
        # #         fig, ax = plt.subplots()
        # #         ax.pie(
        # #             emoji_df['count'].head(),
        # #             labels=emoji_df['emoji'].head(),
        # #             autopct="%0.2f"
        # #         )
        # #         st.pyplot(fig)
        # #     else:
        # #         st.info("No emojis found for this user")
        #
        # with col2:
        #     if not emoji_df.empty:
        #         labels = []
        #         for e in emoji_df['emoji'].head(5):
        #             labels.append(e if e.strip() != "" else "🔹")
        #
        #         fig, ax = plt.subplots()
        #         ax.pie(
        #             emoji_df['count'].head(5),
        #             labels=labels,
        #             autopct="%0.2f",
        #             startangle=90
        #         )
        #         plt.tight_layout()
        #         st.pyplot(fig)
        #     else:
        #         st.info("No emojis found for this user")

        #             Updates code of Emojis analysis
        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)

        with col2:
            if not emoji_df.empty:
                import seaborn as sns
                import matplotlib.pyplot as plt

                # Take top 5 emojis
                top_emoji_df = emoji_df.head(5).copy()

                # 🔹 Replace emojis with text labels for safe plotting
                top_emoji_df['emoji_label'] = [f"Emoji {i + 1}" for i in range(len(top_emoji_df))]

                # Create figure
                fig, ax = plt.subplots(figsize=(7, 6))

                # Barplot using text labels
                sns.barplot(x='emoji_label', y='count', data=top_emoji_df, palette='viridis', ax=ax)

                ax.set_title(f"Top 5 Emojis for {selected_user}", fontsize=16, fontweight='bold')
                ax.tick_params(axis='x', labelsize=14)  # x-axis labels
                ax.tick_params(axis='y', labelsize=14)
                ax.set_xlabel("Emoji", fontsize=14)
                ax.set_ylabel("Count", fontsize=14)

                # Show counts above bars
                for i, row in top_emoji_df.iterrows():
                    ax.text(i, row['count'] + 0.1, str(row['count']), ha='center', va='bottom', fontsize=12)

                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No emojis found for this user")


