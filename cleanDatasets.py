import re
from bs4 import BeautifulSoup
def clean_data_movies(data):
    cleanedReviews = []

    for review in data:
        soup = BeautifulSoup(review, "html.parser")
        cleanText = soup.get_text(separator = "")
        cleanText = cleanText.lower()
        cleanedReviews.append(cleanText)
    return cleanedReviews

def clean_data_steam(data):
    emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags
    u"\U00002700-\U000027BF"  # dingbats
    u"\U0001F900-\U0001F9FF"  # additional symbols
                        "]+", flags=re.UNICODE)
    data = emoji_pattern.sub(r'', data)

    lines = data.splitlines()
    limit = 0.6
    filtered_lines = []
    for line in lines:
        if line.strip():
            non_alnum_count = sum(1 for ch in line if not ch.isalnum() and not ch.isspace())
            ratio = non_alnum_count / len(line) if len(line) > 0 else 0

            # Only include lines that do not exceed the threshold
            if ratio < limit:
                filtered_lines.append(line)
        else:
            filtered_lines.append(line)
    return "\n".join(filtered_lines)
