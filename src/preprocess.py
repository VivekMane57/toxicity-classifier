
import re
import emoji

URL_RE = re.compile(r"http\S+|www\S+")
NON_LETTER_RE = re.compile(r"[^a-zA-Z\s]")
MULTISPACE_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    # demojize → convert emojis to text tokens like :smile:
    text = emoji.demojize(text, delimiters=(" ", " "))
    # keep letters & spaces
    text = NON_LETTER_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text
