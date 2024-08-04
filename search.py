from dataclasses import dataclass

@dataclass
class SearchEntry:
    link: str
    title: str
    snippet: str

    @classmethod
    def from_search_result(cls, entry):
        return cls(link=entry["link"], title=entry["title"], snippet=entry["snippet"])
