import abc
from logging import getLogger
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from src.backend.corpus_loader import CorpusLoader, Doc
from src.backend.encoder import DenseEncoder
from src.backend.indexer import DenseIndexer, VecRecord

logger = getLogger(__name__)


class Retriever(abc.ABC):
    @abc.abstractmethod
    def index(self, corpus: CorpusLoader, batch_size: int = 10_000) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def retrieve(
        self,
        queries: List[str],
        from_: int = 0,
        size: int = 10,
        topk: int = 100,
        search_fields: Optional[List[str]] = None,
        **kwargs: Dict,
    ) -> List[Tuple]:
        raise NotImplementedError


def docs_to_texts(docs: Iterable[Doc]) -> Tuple[List[str], List[str]]:
    # docids = []
    texts = []
    for doc in docs:
        # docids.append(doc.doc_id)
        texts.append(doc.text + " " + doc.title)
    return texts


class DenseRetriever(Retriever):
    def __init__(
        self,
        encoder: DenseEncoder,
        vector_indexer: DenseIndexer,
        model_to_texts: Callable[
            [Iterable[Dict]], Tuple[List[str], List[str]]
        ] = docs_to_texts,
    ) -> None:
        self.encoder = encoder
        self.vector_indexer = vector_indexer
        self.model_to_texts = model_to_texts

    def encode_docs(self, models: Iterable[Dict]) -> np.ndarray:
        texts = self.model_to_texts(models)
        return self.encoder.encode_corpus(texts)

    def encode_queries(self, queries: Iterable[str]) -> np.ndarray:
        return self.encoder.encode_queries(queries)

    def async_index(
        self, corpus_loader: CorpusLoader, batch_size: int = 10_000
    ) -> None:
        def yield_doc_vector(embs: np.ndarray, docs_chunk: List[BaseModel]):
            for emb, doc in zip(embs, docs_chunk):
                yield VecRecord(vec=emb, doc=doc)

        write_total = 0
        for docs_chunk in corpus_loader.load(batch_size=batch_size):
            embeddings = self.encode_docs(docs_chunk)
            write_count = self.vector_indexer.async_index(
                yield_doc_vector(embeddings, docs_chunk)
            )
            write_total += write_count
        logger.info(f"Indexed {write_total} documents.")

    def index(
        self,
        corpus_loader: CorpusLoader,
        batch_size: int = 10_000,
    ) -> None:
        def yield_doc_vector(embs: np.ndarray, docs_chunk: List[BaseModel]):
            for emb, doc in zip(embs, docs_chunk):
                doc["vec"] = emb
                yield doc

        write_total = 0
        for docs_chunk in corpus_loader.load(batch_size=batch_size):
            embeddings = self.encode_docs(docs_chunk)
            write_count = self.vector_indexer.index(
                yield_doc_vector(embeddings, docs_chunk)
            )
            write_total += write_count
        logger.info(f"Indexed {write_total} documents.")

    def retrieve(
        self,
        queries: List[str],
        from_: int = 0,
        size: int = 10,
        topk: int = 100,
        search_fields: Optional[List[str]] = None,
        vector_field: str = "vec",
        hybrid: bool = False,
        **kwargs: Dict,
    ) -> List[Tuple]:
        embeddings = self.encode_queries(queries)
        logger.info(f"retrieving with {len(queries)} queries, hybrid={hybrid}")

        return self.vector_indexer.query(
            queries,
            term_fields=search_fields if hybrid else [],
            vectors=embeddings,
            vec_field=vector_field,
            top_k=topk,
            from_=from_,
            size=size,
        )
