from typing import Dict, List

import markdown
from bs4 import BeautifulSoup

from src.backend.corpus_loader import Preprocessor


class MdRemovalPreprocessor(Preprocessor):
    def __init__(
        self, fields: List[str] = ["text"], preserve_raw: bool = True
    ) -> None:
        self.fields = fields

    def __call__(self, doc: Dict) -> Dict:
        for field in self.fields:
            html = markdown.markdown(doc[field])
            soup = BeautifulSoup(html, "html.parser")
            doc[f"{field}_raw"] = doc[field]
            doc[field] = soup.get_text()
        return doc
