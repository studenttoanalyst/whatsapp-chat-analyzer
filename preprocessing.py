import re
import pandas as pd


def preprocess(data) -> pd.DataFrame:
    """
    Universal WhatsApp chat preprocessing (Android + iPhone)
    Robust for Streamlit Cloud and mixed formats.
    """

    # -------------------- Decode & Normalize --------------------
    if isinstance(data, bytes):
        data = data.decode('utf-8', errors='ignore')

    # Normalize line endings
    data = data.replace('\r\n', '\n').replace('\r', '\n')

    # Clean hidden unicode characters
    data = data.replace('\u202F', ' ')
    data = data.replace('\u200E', '')
    data = data.replace('\u202A', '')
    data = data.replace('\u202C', '')

    # Strip each line (important for iPhone formats)
    data = "\n".join([line.strip() for line in data.split("\n") if line.strip()])

    # -------------------- Patterns --------------------
    pattern_iphone = r'\[\d{1,2}[/-]\d{1,2}[/-]\d{2,4},?\s\d{1,2}:\d{2}(?::\d{2})?(?:[\s\u202F]+)?(?:AM|PM|am|pm)?\]\s'
    pattern_android = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4},?\s\d{1,2}:\d{2}(?:\s)?(?:AM|PM|am|pm)?\s-\s'

    # -------------------- Detect Format --------------------
    if re.search(pattern_iphone, data):
        pattern = pattern_iphone
    elif re.search(pattern_android, data):
        pattern = pattern_android
    else:
        # Fallback to android pattern instead of crashing
        pattern = pattern_android

    # -------------------- Split Data --------------------
    messages = re.split(pattern, data)
    dates = re.findall(pattern, data)

    # Remove first empty split
    messages = messages[1:]

    # -------------------- Clean Dates --------------------
    clean_dates = []
    for d in dates:
        d = d.replace('[', '').replace(']', '')
        d = d.replace(' -', '')
        clean_dates.append(d.strip())

    # -------------------- DataFrame --------------------
    df = pd.DataFrame({
        'user_message': messages,
        'date_str': clean_dates
    })

    # -------------------- Datetime Conversion --------------------
    df['date'] = pd.to_datetime(df['date_str'], errors='coerce')

    # Drop invalid rows safely
    df = df.dropna(subset=['date'])

    df.drop(columns=['date_str'], inplace=True)

    # -------------------- Extract User & Message --------------------
    users = []
    msgs = []

    for message in df['user_message']:
        # Split only on first occurrence of "Name: message"
        parts = re.split(r'^([^:]+):\s', message)

        if len(parts) >= 3:
            users.append(parts[1])
            msgs.append(parts[2])
        else:
            users.append('group_notification')
            msgs.append(parts[0])

    df['user'] = users
    df['message'] = msgs

    df.drop(columns=['user_message'], inplace=True)

    # -------------------- Clean Columns --------------------
    df['user'] = df['user'].fillna("Unknown").astype(str)
    df['message'] = df['message'].fillna("").astype(str)

    # -------------------- Time Features --------------------
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()

    df['hour'] = df['date'].dt.hour.fillna(0).astype(int)
    df['minute'] = df['date'].dt.minute.fillna(0).astype(int)

    # -------------------- Time Period --------------------
    df['period'] = df['hour'].apply(_get_time_period)

    return df


def _get_time_period(hour: int) -> str:
    return f"{hour:02d}-{(hour + 1) % 24:02d}"