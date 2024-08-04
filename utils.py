from textwrap import dedent
from enum import Enum
from dataclasses import dataclass
from bs4 import BeautifulSoup
from bs4.element import Comment
import re
import unicodedata

class Relation(Enum):
    SCHOOLS_ATTENDED = 1
    WORK_FOR = 2
    LIVE_IN = 3
    TOP_MEMBER_EMPLOYEES = 4

    @property
    def str_value(self):
        if self.value == 1:
            return "SCHOOLS_ATTENDED"
        elif self.value == 2:
            return "WORK_FOR"
        elif self.value == 3:
            return "LIVE_IN"
        else:
            return "TOP_MEMBER_EMPLOYEES"

    def __str__(self):
        return f"{self.str_value}({self.value})"

@dataclass
class Input:
    spanbert_mode: bool
    gpt3_mode: bool
    google_api_key: str
    google_engine_id: str
    open_ai_secret_key: str
    relation: "Relation"
    minimum_confidence: float
    seed_query: str
    num_output_tuples: int

    def __str__(self):
        return dedent(f"""
            ======================
            Parameters:
            Client key           = {self.google_api_key}
            Engine key           = {self.google_engine_id}
            OPEN AI key          = {self.open_ai_secret_key}
            Query                = {self.seed_query}
            Relation             = {self.relation}
            Minimum Confidence   = {self.minimum_confidence}
            Num Output Tuples    = {self.num_output_tuples}
            Google Search Results:
            ======================
        """)

def pure_text_from_html(query, body):
    MAX_TEXT_LEN = 10_000

    def wrong_formatted_text(element):
        if "html" == element:
            return False
        if re.match(r"\[[0-9]+\]", element):
            return False
        if re.match(r"\[[a-zA-Z]\]", element):
            return False
        if element == '':
            return False
        return True

    def clease_text(element):
        text = unicodedata.normalize("NFKD", element)
        text = re.sub("[\n]+", "\n", text)
        text = re.sub("[\t]+", "\t", text)
        return text


    soup = BeautifulSoup(body, 'html.parser')
    for data in soup(['style', 'script', 'head', 'title', 'meta', '[document]', 'input', 'nav', 'header']):
        data.decompose()

    texts = list(map(lambda tag: tag.getText(), soup.findAll('p')))
    cleansed_texts = list(map(clease_text, texts))

    cleansed_texts = filter(wrong_formatted_text, cleansed_texts)
    full_text = ' '.join(cleansed_texts)

    return full_text[:MAX_TEXT_LEN]
