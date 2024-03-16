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
    def __call__(self, doc: BaseModel) -> BaseModel:
        raise NotImplementedError


class CorpusLoader(abc.ABC):
    def load(self, batch_size: int = 10_000) -> Iterator[List[BaseModel]]:
        raise NotImplementedError


class JsonlCorpusLoader(CorpusLoader):
    def __init__(
        self,
        path: str,
        preprocessores: List[Preprocessor] = [],
        data_type: BaseModel = Doc,
        verbose: bool = True,
    ) -> None:
        self.path = path
        self.data_type = data_type
        self.verbose = verbose
        self.preprocessores = preprocessores

    def iter_lines(self, path: str) -> Iterator[BaseModel]:
        with open(path) as f:
            for i, line in enumerate(f):
                doc_dict = json.loads(line)
                doc = self.data_type(**doc_dict)
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
    ) -> None:
        self.table_map = table_map
        self.connection = psycopg2.connect(
            user=user, host=host, password=password, port=port
        )

    def _transform_row(self, row: Dict[str, str]) -> Dict[str, str]:
        mapper = self.table_map.mapper
        for k, v in mapper.items():
            if k in row:
                row[v] = row.pop(k)
        return row

    def fetch(self):
        with self.connection.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                SQL("SELECT * FROM {}").format(Identifier(self.table_map.table_name))
            )
            for row in cur:
                row = dict(row)
                transformed = self._transform_row(row)
                record = dict(zip(self.table_map.mapper.keys(), row))
                yield row

    def __del__(self):
        self.connection.close()

    def load(self, batch_size: int = 10_000) -> Iterator[List[BaseModel]]:
        iterator = chunked(self.fetch(), batch_size)
        for chunk in iterator:
            yield chunk
