import json
import time
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, Iterable, List, Optional, Tuple

import elasticsearch
import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

from src.backend.corpus_loader import CorpusLoader
from src.backend.indexer import DenseIndexer, VecRecord
from src.backend.retriever import Retriever
from src.backend.utils import project_dir

logger = getLogger(__name__)


@dataclass
class ElasticsearchConfig:
    host: str
    port: int
    schema: str = "http"
    index_name: str = "fotla_index"
    index_scheme_path: str = project_dir / "vector_indexer/elasticsearch/mappngs.json"


class ElasticsearchIndexManager(object):
    def __init__(self, es: elasticsearch.Elasticsearch, config: ElasticsearchConfig):
        self.config = config
        self.es = elasticsearch.Elasticsearch(
            f"{self.config.schema}://{self.config.host}:{self.config.port}",
        )
        self.config = config

    def gen_index_name(
        self,
        index_name: str,
    ) -> str:
        """Generates a new index name.

        Args:
            index_name: The name of the index.

        Returns:
            The new index name.
        """

        timestamp = str(time.time()).replace(".", "-")

        return f"{index_name}_{timestamp}"

    def read_index_scheme(self) -> dict:
        """Reads the index scheme from the index scheme path.

        Returns:
            The index scheme.
        """
        with open(self.config.index_scheme_path) as f:
            return json.load(f)

    def create_es_index(self, index_name: str) -> None:
        """Creates an index in Elasticsearch for the given vector dimension.

        Args:
            index_name: The name of the index.
        """
        self.es.indices.create(
            index=index_name,
            body=self.read_index_scheme(),
        )

    def create_index_body(
        self, record: BaseModel, fields: Optional[List[str]]
    ) -> Dict:
        vec = None
        if isinstance(record, VecRecord):
            record, vec = record.doc, record.vec

        if fields is None:
            body = {
                "doc_id": record.doc_id,
                "title": record.title,
                "text": record.text,
            }
        else:
            body = {}
            record_dict = record.model_dump()
            for field in fields:
                body[field] = record_dict.get(field, None)

        if vec is not None:
            unit_vec = vec / np.linalg.norm(vec)
            body["vec"] = unit_vec

        return body

    def create_or_update_index_schema(
        self, index_name: str, config: Optional[ElasticsearchConfig] = None
    ) -> None:
        if config is None:
            config = self.config

        new_index_name = self.gen_index_name(index_name)
        self.create_es_index(new_index_name)

        return new_index_name

    def get_latest_index_name(self, index_name: str) -> str:
        """Returns the latest index name.

        Args:
            index_name: The name of the index.

        Returns:
            The latest index name.
        """
        indices = self.es.indices.get_alias(index=index_name)
        return list(indices.keys())[0]

    def switch_alias(self, new_index_name: str, alias_name: str) -> None:
        old_index_name = self.get_latest_index_name(alias_name)
        self.es.indices.delete_alias(index=old_index_name, name=alias_name)
        self.es.indices.put_alias(index=new_index_name, name=alias_name)

    def async_index(self, records: Iterable[BaseModel]) -> None:
        try:
            import asyncio
        except ImportError:
            raise ImportError("asyncio is required for async_index")

        try:
            from elasticsearch.helpers import async_streaming_bulk
        except ImportError:
            raise ImportError("elasticsearch-async is required for async_index")

        from typing import AsyncIterable

        async def async_create_index_body(
            records: Iterable[BaseModel],
        ) -> AsyncIterable[Dict]:
            for record in records:
                body = self.create_index_body(record, self.fields)

                yield {
                    "_op_type": "index",
                    "_index": self.index_name,
                    "_source": body,
                }

        async def main():
            write_count = 0
            async for ok, result in async_streaming_bulk(
                es, async_create_index_body(records)
            ):
                action, result = result.popitem()
                if not ok:
                    print(f"failed to {action} document {result}")
                else:
                    write_count += 1
            return write_count

        es = elasticsearch.AsyncElasticsearch(
            f"{self.config.schema}://{self.config.host}:{self.config.port}",
        )
        loop = asyncio.get_event_loop()
        write_count = loop.run_until_complete(main())
        return write_count

    def index(
        self,
        index_name: str,
        records: Iterable[BaseModel],
        config: Optional[ElasticsearchConfig] = None,
        fields: Optional[List[str]] = None,
        refresh: bool = True,
    ) -> int:
        """Indexes the given vectors.

        Args:
            vectors: The vectors to index.

        Returns:
            The number of vectors fitted.
        """

        new_index_name = self.create_or_update_index_schema(index_name, config)

        write_count = 0
        for record in records:
            body = self.create_index_body(record, fields)

            self.es.index(index=new_index_name, body=body, refresh=refresh)
            write_count += 1

        self.switch_alias(new_index_name, index_name)

        return write_count


class ElasticsearchIndexer(DenseIndexer):
    def __init__(
        self,
        config: ElasticsearchConfig,
        fields: Optional[List[str]] = None,
        recreate_index: bool = False,
    ) -> None:
        self.config = config
        self.fields = fields
        logger.info(f"setting fields: {self.fields}")
        self.es = elasticsearch.Elasticsearch(
            f"{self.config.schema}://{self.config.host}:{self.config.port}",
        )
        self.index_name = self.config.index_name
        logger.info(f"setting index: {self.index_name}")

        self.index_manager = ElasticsearchIndexManager(self.es, self.config)

    def index(self, records: Iterable[BaseModel], refresh: bool = True) -> int:
        """Indexes the given vectors.

        Args:
            vectors: The vectors to index.

        Returns:
            The number of vectors fitted.
        """
        return self.index_manager.index(
            index_name=self.index_name,
            records=records,
            config=self.config,
            fields=self.fields,
            refresh=refresh,
        )

    def query(
        self,
        queries: List[str],
        term_fields: List[str] = [],
        vectors: List[np.ndarray] = [],
        vec_field: str = "vec",
        top_k: int = 10,
        from_: int = 0,
        size: int = 10,
        source: Optional[List[str]] = None,
        operator: str = "and",
    ) -> List[Tuple[str, Dict]]:
        """Returns the top_k most similar vectors to the given vectors.

        Args:
            vectors: The vectors to query.
            top_k: The number of similar vectors to return.

        Returns:
            The indices of the top_k most similar vectors.
        """
        logger.debug(f"Querying {len(queries)} queries.")

        if len(vectors) <= 0 and len(term_fields) <= 0:
            raise ValueError("Either vectors or term_field must be given.")

        if len(vectors) > 0 and len(vectors) != len(queries):
            raise ValueError(
                "The number of vectors must be equal to the number of queries."
            )

        results: List[Tuple[str, Dict]] = []
        for i, query in enumerate(queries):
            logger.debug(f"Retrieving with query: {query}")

            term_query = (
                None
                if len(term_fields) <= 0
                else {
                    "multi_match": {
                        "query": query,
                        "fields": term_fields,
                        "operator": operator,
                        "type": "cross_fields",
                        "boost": 0.5,
                    }
                }
            )

            vec = vectors[i] if len(vectors) > 0 else None
            knn_param = None
            if vec is not None:
                unit_vec = vec / np.linalg.norm(vec)
                knn_param = {
                    "field": "vec",
                    "query_vector": unit_vec,
                    "k": top_k,
                    "num_candidates": top_k * 2,
                    "filter": term_query,
                    "boost": 0.5,
                }

            logger.debug(f"term_query: {term_query}")
            res = self.es.search(
                index=self.index_name,
                knn=knn_param,
                query=term_query,
                from_=from_,
                size=size,
            )
            result = {
                "total": res["hits"]["total"]["value"],
                "hits": res["hits"]["hits"],
            }
            logger.debug(f"query {query} retrieved {len(result['hits'])} results.")
            results.append((query, result))

            logger.debug(f"Retrieved {len(result)} results.")
        return results


class ElasticsearchBM25(Retriever):
    def __init__(
        self,
        es_indexer: ElasticsearchIndexer,
        fields: Optional[List[str]] = ["title", "text"],
    ) -> None:
        self.es_indexer = es_indexer
        self.fields = fields

    def async_index(
        self,
        corpus_loader: CorpusLoader,
        batch_size: int = 10_000,
        total: int = 65613666,
    ) -> None:
        def load_corpus(
            corpus_loader: CorpusLoader, batch_size: int
        ) -> Iterable[BaseModel]:
            for docs_chunk in tqdm(
                corpus_loader.load(batch_size=batch_size),
                desc="loading corpus..",
                total=(total // batch_size) + 1,
            ):
                for doc in docs_chunk:
                    yield doc

        self.es_indexer.async_index(load_corpus(corpus_loader, batch_size))

    def index(
        self,
        corpus_loader: CorpusLoader,
        batch_size: int = 10_000,
        total: int = 65613666,
    ) -> None:
        for docs_chunk in tqdm(
            corpus_loader.load(batch_size=batch_size),
            desc="indexing..",
            total=(total // batch_size) + 1,
        ):
            self.es_indexer.index(
                docs_chunk,
                refresh=False,
            )

    def retrieve(
        self,
        queries: List[str],
        top_k: int,
        search_fields: Optional[List[str]] = None,
        from_: int = 0,
        size: int = 10,
        hybrid: bool = False,
    ) -> List[Tuple[str, Dict]]:
        fields = self.fields if search_fields is None else search_fields
        result = self.es_indexer.query(
            queries, term_fields=fields, top_k=top_k, from_=from_, size=size
        )
        return result
