"""
app.py
======
WhatsApp Chat Analyzer — Streamlit front-end.

Original features preserved:
  - File upload (TXT / ZIP)
  - Top Statistics
  - Monthly & Daily Timeline
  - Activity Map (day + month)
  - Weekly Heatmap
  - Most Busy Users
  - WordCloud
  - Most Common Words
  - Emoji Analysis

New AI section added (bottom of page):
  - 📊 AI Sentiment Analysis
      • Positive / Negative / Neutral counts & pie chart
      • Sentiment timeline
      • Per-user sentiment ranking (Overall mode)
      • Top positive & negative words
      • AI-generated chat mood summary
"""

import streamlit as st
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import emoji
import pandas as pd

import preprocessing
import helper
import Sentiment as sa          # ← NEW: sentiment analysis module


# ────────────────────────────────────────────────────────────────────────────
# App Config
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
)

st.sidebar.title("💬 WhatsApp Chat Analyzer")


# ────────────────────────────────────────────────────────────────────────────
# File Upload
# ────────────────────────────────────────────────────────────────────────────
uploaded_file = st.sidebar.file_uploader(
    "Choose a file", type=["txt", "zip"]
)

if uploaded_file is not None:

    # Handle ZIP file
    if uploaded_file.name.endswith(".zip"):
        with zipfile.ZipFile(uploaded_file) as z:
            txt_file = next(
                (f for f in z.namelist() if f.endswith(".txt")), None
            )
            if txt_file:
                with z.open(txt_file) as f:
                    data = f.read().decode("utf-8")
            else:
                st.error("No TXT file found inside ZIP")
                st.stop()
    else:
        data = uploaded_file.getvalue().decode("utf-8")

    # ── Preprocessing ────────────────────────────────────────────────────
    df = preprocessing.preprocess(data)
    df = df[df["user"] != "group_notification"]

    if df.empty:
        st.error("No valid messages found. Please check the file format.")
        st.stop()

    # ── User selection ───────────────────────────────────────────────────
    user_list = sorted(df["user"].dropna().unique().tolist())
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    # ── Run Analysis button ──────────────────────────────────────────────
    if st.sidebar.button("Show Analysis"):

        # ════════════════════════════════════════════════════════════════
        # SECTION 1 — Top Statistics
        # ════════════════════════════════════════════════════════════════
        num_messages, words, media_msgs, links = helper.fetch_stats(
            selected_user, df
        )

        st.title("Top Statistics")

        st.markdown(
            """
            <style>
            .stat-number { font-size: 50px; font-weight: bold; }
            .stat-label  { font-size: 25px; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.markdown("<p class='stat-label'>Total Messages</p>", unsafe_allow_html=True)
        col1.markdown(f"<p class='stat-number'>{num_messages}</p>", unsafe_allow_html=True)
        col2.markdown("<p class='stat-label'>Total Words</p>", unsafe_allow_html=True)
        col2.markdown(f"<p class='stat-number'>{words}</p>", unsafe_allow_html=True)
        col3.markdown("<p class='stat-label'>Media Shared</p>", unsafe_allow_html=True)
        col3.markdown(f"<p class='stat-number'>{media_msgs}</p>", unsafe_allow_html=True)
        col4.markdown("<p class='stat-label'>Links Shared</p>", unsafe_allow_html=True)
        col4.markdown(f"<p class='stat-number'>{links}</p>", unsafe_allow_html=True)


        # ════════════════════════════════════════════════════════════════
        # SECTION 2 — Timelines
        # ════════════════════════════════════════════════════════════════
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline["time"], timeline["message"])
        plt.xticks(ticks=range(0, len(timeline), 5), rotation=90)
        st.pyplot(fig)

        st.title("Daily Timeline")
        daily = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily["only_date"], daily["message"])
        plt.xticks(rotation=90)
        st.pyplot(fig)


        # ════════════════════════════════════════════════════════════════
        # SECTION 3 — Activity Map
        # ════════════════════════════════════════════════════════════════
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


        # ════════════════════════════════════════════════════════════════
        # SECTION 4 — Weekly Heatmap
        # ════════════════════════════════════════════════════════════════
        st.title("Weekly Activity Heatmap")
        heatmap = helper.activity_heatmap(selected_user, df)
        if heatmap.empty:
            st.warning("No activity available")
        else:
            fig, ax = plt.subplots()
            sns.heatmap(heatmap, ax=ax)
            st.pyplot(fig)


        # ════════════════════════════════════════════════════════════════
        # SECTION 5 — Most Busy Users (Overall only)
        # ════════════════════════════════════════════════════════════════
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


        # ════════════════════════════════════════════════════════════════
        # SECTION 6 — WordCloud
        # ════════════════════════════════════════════════════════════════
        st.title("WordCloud")
        wc = helper.create_wordcloud(selected_user, df)
        if wc:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("No data for WordCloud")


        # ════════════════════════════════════════════════════════════════
        # SECTION 7 — Most Common Words
        # ════════════════════════════════════════════════════════════════
        st.title("Most Common Words")
        common_df = helper.most_common_words(selected_user, df)
        if not common_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(common_df["word"], common_df["count"])
            ax.set_xlabel("Count")
            ax.set_ylabel("Word")
            st.pyplot(fig)
        else:
            st.info("No words available")


        # ════════════════════════════════════════════════════════════════
        # SECTION 8 — Emoji Analysis
        # ════════════════════════════════════════════════════════════════
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)

        with col2:
            if not emoji_df.empty:
                top_df = emoji_df.head(5).copy()
                top_df["label"] = [f"Emoji {i+1}" for i in range(len(top_df))]
                fig, ax = plt.subplots(figsize=(7, 6))
                sns.barplot(x="label", y="count", data=top_df, ax=ax)
                ax.set_title(f"Top Emojis for {selected_user}")
                ax.set_xlabel("Emoji")
                ax.set_ylabel("Count")
                for i, row in top_df.iterrows():
                    ax.text(i, row["count"], str(row["count"]), ha="center")
                st.pyplot(fig)
            else:
                st.info("No emojis found")


        # ════════════════════════════════════════════════════════════════
        # SECTION 9 — 🤖 AI SENTIMENT ANALYSIS  ← NEW
        # ════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.title("📊 AI Sentiment Analysis")
        st.caption(
            "Powered by VADER + custom Roman Urdu / Hinglish lexicon — "
            "no internet or model download required."
        )

        # ── Score every message ──────────────────────────────────────────
        with st.spinner("Running AI sentiment analysis…"):
            df_sentiment = sa.analyze_sentiment(df)

        # ── Compute aggregated stats ─────────────────────────────────────
        stats = sa.get_sentiment_stats(selected_user, df_sentiment)
        counts = stats["counts"]
        sot    = stats["sentiment_over_time"]
        u_scores = stats["user_scores"]
        pos_words = stats["top_positive_words"]
        neg_words = stats["top_negative_words"]

        # Guard: nothing to show
        total_scored = sum(counts.values())
        if total_scored == 0:
            st.warning("No text messages available for sentiment analysis.")
            st.stop()

        # ── 9-A  Chat Mood Summary ───────────────────────────────────────
        st.subheader("🧠 Chat Mood Summary")
        summary = sa.get_sentiment_summary(counts, u_scores)
        st.markdown(
            f"""
            <div style="
                background:#f0f4ff;
                border-left:5px solid #4a90e2;
                padding:16px 20px;
                border-radius:8px;
                font-size:16px;
                line-height:1.6;
            ">
            {summary}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")   # spacer

        # ── 9-B  Sentiment Counts & Pie Chart ───────────────────────────
        st.subheader("📈 Sentiment Distribution")
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("✅ Positive",  counts["Positive"])
        c2.metric("❌ Negative",  counts["Negative"])
        c3.metric("➖ Neutral",   counts["Neutral"])
        c4.metric("📩 Total",     total_scored)

        # Pie chart
        pie_labels = ["Positive", "Negative", "Neutral"]
        pie_values = [counts[l] for l in pie_labels]
        pie_colors = ["#4caf50", "#f44336", "#9e9e9e"]

        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax_pie.pie(
            pie_values,
            labels=pie_labels,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 12},
        )
        ax_pie.set_title(
            f"Sentiment Distribution — {selected_user}",
            fontsize=14,
            fontweight="bold",
        )
        st.pyplot(fig_pie)

        # ── 9-C  Sentiment Over Time ─────────────────────────────────────
        st.subheader("📅 Sentiment Over Time")

        if sot.empty:
            st.info("Not enough data to plot sentiment timeline.")
        else:
            # Pivot so we get one line per sentiment category
            sot_pivot = sot.pivot_table(
                index="only_date",
                columns="sentiment",
                values="count",
                aggfunc="sum",
            ).fillna(0)

            # Ensure all three columns exist
            for col in ["Positive", "Negative", "Neutral"]:
                if col not in sot_pivot.columns:
                    sot_pivot[col] = 0

            sot_pivot = sot_pivot[["Positive", "Negative", "Neutral"]]

            fig_time, ax_time = plt.subplots(figsize=(12, 4))
            ax_time.plot(
                sot_pivot.index, sot_pivot["Positive"],
                color="#4caf50", label="Positive", linewidth=1.5,
            )
            ax_time.plot(
                sot_pivot.index, sot_pivot["Negative"],
                color="#f44336", label="Negative", linewidth=1.5,
            )
            ax_time.plot(
                sot_pivot.index, sot_pivot["Neutral"],
                color="#9e9e9e", label="Neutral", linewidth=1.5,
                linestyle="--",
            )
            ax_time.set_xlabel("Date")
            ax_time.set_ylabel("Message Count")
            ax_time.set_title("Daily Sentiment Trend")
            ax_time.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_time)

        # ── 9-D  Per-User Sentiment (Overall only) ───────────────────────
        if selected_user == "Overall" and u_scores is not None and not u_scores.empty:
            st.subheader("👥 Per-User Sentiment Ranking")
            st.caption(
                "Mean compound score per user (−1 = most negative, +1 = most positive). "
                "Users with fewer than 5 messages are still included for completeness."
            )

            # Filter users with at least 1 message for cleaner display
            u_display = u_scores[u_scores["message_count"] >= 1].copy()
            u_display.columns = ["User", "Mean Score", "Messages"]
            u_display = u_display.reset_index(drop=True)

            col_pos, col_neg = st.columns(2)

            with col_pos:
                st.markdown("**🌟 Most Positive Users**")
                top5_pos = u_display.head(5)
                fig_pos, ax_pos = plt.subplots(figsize=(5, 3))
                ax_pos.barh(
                    top5_pos["User"],
                    top5_pos["Mean Score"],
                    color="#4caf50",
                )
                ax_pos.set_xlabel("Mean Sentiment Score")
                ax_pos.set_xlim(-1, 1)
                ax_pos.axvline(0, color="black", linewidth=0.8, linestyle="--")
                plt.tight_layout()
                st.pyplot(fig_pos)

            with col_neg:
                st.markdown("**⚠️ Most Negative Users**")
                top5_neg = u_display.tail(5).iloc[::-1]
                fig_neg, ax_neg = plt.subplots(figsize=(5, 3))
                ax_neg.barh(
                    top5_neg["User"],
                    top5_neg["Mean Score"],
                    color="#f44336",
                )
                ax_neg.set_xlabel("Mean Sentiment Score")
                ax_neg.set_xlim(-1, 1)
                ax_neg.axvline(0, color="black", linewidth=0.8, linestyle="--")
                plt.tight_layout()
                st.pyplot(fig_neg)

            # Full table with colour-coded scores
            st.markdown("**📋 Full User Sentiment Table**")

            def _colour_score(val):
                """Colour-code the Mean Score cells."""
                if val > 0.05:
                    return "background-color: #e8f5e9; color: #1b5e20;"
                elif val < -0.05:
                    return "background-color: #ffebee; color: #b71c1c;"
                return "background-color: #f5f5f5;"

            styled = u_display.style.applymap(
                _colour_score, subset=["Mean Score"]
            ).format({"Mean Score": "{:.3f}"})

            st.dataframe(styled, use_container_width=True)

        # ── 9-E  Top Words by Sentiment ──────────────────────────────────
        st.subheader("💬 Top Words by Sentiment")
        col_pw, col_nw = st.columns(2)

        with col_pw:
            st.markdown("**✅ Top Positive Words**")
            if pos_words:
                pos_df = pd.DataFrame(pos_words, columns=["Word", "Count"])
                fig_pw, ax_pw = plt.subplots(figsize=(5, 5))
                ax_pw.barh(
                    pos_df["Word"].iloc[::-1],
                    pos_df["Count"].iloc[::-1],
                    color="#81c784",
                )
                ax_pw.set_xlabel("Frequency")
                plt.tight_layout()
                st.pyplot(fig_pw)
            else:
                st.info("No positive words found.")

        with col_nw:
            st.markdown("**❌ Top Negative Words**")
            if neg_words:
                neg_df = pd.DataFrame(neg_words, columns=["Word", "Count"])
                fig_nw, ax_nw = plt.subplots(figsize=(5, 5))
                ax_nw.barh(
                    neg_df["Word"].iloc[::-1],
                    neg_df["Count"].iloc[::-1],
                    color="#e57373",
                )
                ax_nw.set_xlabel("Frequency")
                plt.tight_layout()
                st.pyplot(fig_nw)
            else:
                st.info("No negative words found.")

        # ── 9-F  Raw sentiment data (expandable) ─────────────────────────
        with st.expander("🔍 View raw sentiment data"):
            display_cols = ["date", "user", "message", "compound", "sentiment"]
            available = [c for c in display_cols if c in df_sentiment.columns]

            if selected_user != "Overall":
                view_df = df_sentiment[df_sentiment["user"] == selected_user][available]
            else:
                view_df = df_sentiment[available]

            st.dataframe(view_df.reset_index(drop=True), use_container_width=True)


