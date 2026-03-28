import re
import pandas as pd


def preprocess(data: str) -> pd.DataFrame:
    """
    Universal WhatsApp chat preprocessing (Android + iPhone)
    """

    # -------------------- Clean Unicode --------------------
    data = data.replace('\u202F', ' ')
    data = data.replace('\u200E', '')
    data = data.replace('\u202A', '')
    data = data.replace('\u202C', '')

    # -------------------- Patterns --------------------
    pattern_iphone = r'\[\d{1,2}[/-]\d{1,2}[/-]\d{2,4},?\s\d{1,2}:\d{2}(?::\d{2})?(?:[\s\u202F]+)?(?:AM|PM|am|pm)?\]\s'
    pattern_android = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4},?\s\d{1,2}:\d{2}(?:\s)?(?:AM|PM|am|pm)?\s-\s'

    # -------------------- Detect Format --------------------
    if re.search(pattern_iphone, data):
        pattern = pattern_iphone
        is_iphone = True
    elif re.search(pattern_android, data):
        pattern = pattern_android
        is_iphone = False
    else:
        raise ValueError("Unsupported WhatsApp format")

    # -------------------- Split Data --------------------
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

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

    # -------------------- Convert to Datetime --------------------
    df['date'] = pd.to_datetime(df['date_str'], errors='coerce')

    # 🔥 IMPORTANT: remove invalid dates
    df = df.dropna(subset=['date'])

    df.drop(columns=['date_str'], inplace=True)

    # -------------------- Extract User & Message --------------------
    users = []
    msgs = []

    for message in df['user_message']:
        entry = re.split(r'([^:]+):\s', message)

        if len(entry) > 2:
            users.append(entry[1])
            msgs.append(entry[2])
        else:
            users.append('group_notification')
            msgs.append(entry[0])

    df['user'] = users
    df['message'] = msgs

    df.drop(columns=['user_message'], inplace=True)

    # -------------------- Time Features --------------------
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()

    df['hour'] = df['date'].dt.hour
    df['hour'] = df['hour'].fillna(0).astype(int)

    df['minute'] = df['date'].dt.minute

    # -------------------- Time Period --------------------
    df['period'] = df['hour'].apply(_get_time_period)

    return df


def _get_time_period(hour: int) -> str:
    return f"{hour:02d}-{(hour + 1) % 24:02d}"
