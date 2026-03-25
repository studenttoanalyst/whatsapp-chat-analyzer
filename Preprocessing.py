# import re
# import pandas as pd


# def preprocess(data: str) -> pd.DataFrame:
#     """
#     Preprocess WhatsApp chat text file and return structured DataFrame.
#     """

#     # Regex pattern to capture different WhatsApp date formats
#     pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4},?\s\d{1,2}:\d{2}(?::\d{2})?(?:\u202F|\s)?(?:am|pm)?\s-\s'

#     # Extract messages and dates
#     messages = re.split(pattern, data, flags=re.IGNORECASE)[1:]
#     dates = re.findall(pattern, data, flags=re.IGNORECASE)

#     # Normalize unicode spaces
#     dates = [date.replace('\u202f', ' ') for date in dates]

#     # Create DataFrame
#     df = pd.DataFrame({
#         'user_message': messages,
#         'message_date': dates
#     })

#     # Clean date string
#     df['message_date'] = df['message_date'].str.replace(r' - $', '', regex=True)

#     # Convert to datetime (auto-detect format)
#     df['date'] = pd.to_datetime(
#         df['message_date'],
#         errors='coerce',
#         infer_datetime_format=True
#     )

#     df.drop(columns=['message_date'], inplace=True)

#     # Extract user and message
#     users = []
#     messages = []

#     for message in df['user_message']:
#         entry = re.split(r'([\w\W]+?):\s', message)

#         if entry[1:]:
#             users.append(entry[1])
#             messages.append(" ".join(entry[2:]))
#         else:
#             users.append('group_notification')
#             messages.append(entry[0])

#     df['user'] = users
#     df['message'] = messages

#     df.drop(columns=['user_message'], inplace=True)

#     # Time-based features
#     df['only_date'] = df['date'].dt.date
#     df['year'] = df['date'].dt.year
#     df['month_num'] = df['date'].dt.month
#     df['month'] = df['date'].dt.month_name()
#     df['day'] = df['date'].dt.day
#     df['day_name'] = df['date'].dt.day_name()
#     df['hour'] = df['date'].dt.hour
#     df['minute'] = df['date'].dt.minute

#     # Create time period (hour buckets)
#     df['period'] = df['hour'].apply(_get_time_period)

#     return df


# def _get_time_period(hour: int) -> str:
#     """
#     Convert hour into time range string (e.g., 13-14)
#     """
#     if hour == 23:
#         return "23-00"
#     elif hour == 0:
#         return "00-01"
#     else:
#         return f"{hour:02d}-{hour + 1:02d}"

import re
import pandas as pd
import unicodedata

def preprocess(data: str) -> pd.DataFrame:
    """
    Preprocess WhatsApp chat text file and return structured DataFrame.
    Works safely for any country format and handles AM/PM, special spaces, missing data.
    """

    # Regex pattern to capture different WhatsApp date formats
    pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4},?\s\d{1,2}:\d{2}(?::\d{2})?(?:\u202F|\s)?(?:am|pm)?\s-\s'

    # Extract messages and dates
    messages = re.split(pattern, data, flags=re.IGNORECASE)[1:]
    dates = re.findall(pattern, data, flags=re.IGNORECASE)

    # Normalize unicode spaces
    dates = [unicodedata.normalize("NFKC", date) for date in dates]

    # Create DataFrame
    df = pd.DataFrame({
        'user_message': messages,
        'message_date': dates
    })

    # Ensure message_date is string and safe for .str operations
    df['message_date'] = df['message_date'].fillna('').astype(str)

    # Remove trailing ' - ' safely
    df['message_date'] = df['message_date'].str.replace(r'\s*-\s*$', '', regex=True)

    # Convert to datetime (safe with errors='coerce')
    df['date'] = pd.to_datetime(
        df['message_date'],
        errors='coerce',
        infer_datetime_format=True
    )

    df.drop(columns=['message_date'], inplace=True)

    # Extract user and message
    users = []
    messages_clean = []

    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message, maxsplit=1)

        if entry[1:]:
            users.append(entry[1])
            messages_clean.append(entry[2] if len(entry) > 2 else '')
        else:
            users.append('group_notification')
            messages_clean.append(entry[0])

    df['user'] = users
    df['message'] = messages_clean
    df.drop(columns=['user_message'], inplace=True)

    # Time-based features
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Create time period (hour buckets)
    df['period'] = df['hour'].apply(_get_time_period)

    return df


def _get_time_period(hour: int) -> str:
    """
    Convert hour into time range string (e.g., 13-14)
    """
    if hour == 23:
        return "23-00"
    elif hour == 0:
        return "00-01"
    else:
        return f"{hour:02d}-{hour + 1:02d}"
