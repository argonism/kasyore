import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel

from src.backend.api import start_api
from src.backend.corpus_loader import (
    AdhocCorpusLoader,
    Doc,
    JsonlCorpusLoader,
    Preprocessor,
)
from src.backend.encoder import HFSymetricDenseEncoder
from src.backend.indexer.elasticsearch import (
    ElasticsearchBM25,
    ElasticsearchConfig,
    ElasticsearchIndexer,
)
from src.backend.retriever import DenseRetriever


class KasyoreDoc(BaseModel):
    title: str
    body: str
    updated_at: datetime


def load_indexer(recreate_index: bool = False):
    es_host = os.environ.get("ELASTICSEARCH_HOST", "localhost")
    es_port = os.environ.get("ELASTICSEARCH_PORT", 9200)
    es_config = ElasticsearchConfig(
        es_host,
        es_port,
        index_name="kasyore",
        index_scheme_path="mappings/kasyore.json",
    )
    indexer = ElasticsearchIndexer(
        es_config,
        fields=list(KasyoreDoc.model_fields.keys()),
        recreate_index=recreate_index,
    )
    return indexer


def load_retirever(indexer):
    def sillabi_to_text(docs: List[KasyoreDoc]) -> List[str]:
        return [" ".join([doc.title, doc.body]) for doc in docs]

    # encoder = HFSymetricDenseEncoder("facebook/mcontriever-msmarco")
    # retriever = DenseRetriever(encoder, indexer, model_to_texts=sillabi_to_text)
    retriever = ElasticsearchBM25(indexer)

    return retriever


def index(retriever):
    corpus_loader = JsonlCorpusLoader(
        "corpus/fotla.esa_posts.10.jsonl",
        data_type=KasyoreDoc,
    )

    retriever.index(corpus_loader)


def playground(indexer, retriever):
    manager = indexer.index_manager
    # a = manager.es.indices.get_alias(name="kasyore")
    # a = manager.es.indices.delete_alias(
    #     index="kasyore_1707417475-9818451", name="kasyore"
    # )
    # print(a)
    index(retriever)


def main(args):
    indexer = load_indexer(args.recreate_index)
    retriever = load_retirever(indexer)

    playground(indexer, retriever)
    return

    if args.index:
        index(retriever)

    if not args.retrieve == "":
        search_fields = [
            "title",
            "body",
        ]
        results = retriever.retrieve(
            [args.retrieve], 100, search_fields=search_fields
        )
        print(json.dumps(results, indent=2))
    else:
        start_api(retriever, port=9999)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--retrieve", default="")
    parser.add_argument("--recreate_index", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    main(args)
