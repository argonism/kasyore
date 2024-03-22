import abc
import json
from typing import Dict, Iterator, List, Optional

import psycopg2
from more_itertools import chunked
from psycopg2.extras import RealDictCursor
from psycopg2.sql import SQL, Identifier
from pydantic import BaseModel
from tqdm import tqdm

if False:
    from typing import Type


class Doc(BaseModel):
    doc_id: str
    text: str
    title: str = ""


class Preprocessor(object):
    def __call__(self, doc: Dict) -> Dict:
        raise NotImplementedError


class CorpusLoader(abc.ABC):
    def load(
        self, batch_size: int = 10_000, preprocessores: List[Preprocessor] = []
    ) -> Iterator[List[Dict]]:
        raise NotImplementedError


class JsonlCorpusLoader(CorpusLoader):
    def __init__(
        self,
        path: str,
        preprocessores: List[Preprocessor] = [],
        verbose: bool = True,
    ) -> None:
        self.path = path
        self.verbose = verbose
        self.preprocessores = preprocessores

    def iter_lines(self, path: str) -> Iterator[BaseModel]:
        with open(path) as f:
            for i, line in enumerate(f):
                doc = json.loads(line)
                for preprocessor in self.preprocessores:
                    doc = preprocessor(doc)

                yield doc

    def load(self, batch_size: int = 10_000) -> Iterator[List[BaseModel]]:
        iterator = chunked(self.iter_lines(self.path), batch_size)
        if self.verbose:
            iterator = tqdm(iterator, desc="Loading corpus")
        for chunk in iterator:
            yield chunk


class AdhocCorpusLoader(CorpusLoader):
    def __init__(self, docs: List[Dict]) -> None:
        self.docs = docs

    def dict_to_doc(self, dicts: List[Dict]) -> List[Doc]:
        return [Doc(**doc) for doc in dicts]

    def load(self, batch_size: int = 10_000) -> Iterator[List[BaseModel]]:
        for chunk in chunked(self.docs, batch_size):
            yield self.dict_to_doc(chunk)


class TableMapper(BaseModel):
    table_name: str
    mapper: Dict[str, str]


class PostgresCorpusLoader(CorpusLoader):
    def __init__(
        self,
        table_map: TableMapper,
        host: str,
        password: str = "",
        port: int = 5432,
        user: str = "postgres",
        dbname: str = "kasyore",
        preprocessores: List[Preprocessor] = [],
    ) -> None:
        super().__init__()
        self.table_map = table_map
        self.connection = psycopg2.connect(
            user=user, host=host, password=password, port=port
        )
        self.preprocessores = preprocessores

    def _transform_row(self, row: Dict[str, str]) -> Dict[str, str]:
        mapper = self.table_map.mapper
        for k, v in mapper.items():
            if k in row:
                row[v] = row.pop(k)
        return row

    def map_preprocessores(self, doc: Dict) -> Dict:
        for preprocessor in self.preprocessores:
            doc = preprocessor(doc)
        return doc

    def fetch(self):
        with self.connection.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                SQL("SELECT * FROM {}").format(Identifier(self.table_map.table_name))
            )
            for row in cur:
                row = dict(row)
                record = self._transform_row(row)
                record = self.map_preprocessores(record)
                yield record

    def __del__(self):
        self.connection.close()

    def load(self, batch_size: int = 10_000) -> Iterator[List[Dict]]:
        iterator = chunked(self.fetch(), batch_size)
        for chunk in iterator:
            yield chunk
