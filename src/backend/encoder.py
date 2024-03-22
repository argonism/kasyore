import abc
from logging import getLogger
from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch
from more_itertools import chunked
from pydantic import BaseModel
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.backend.corpus_loader import CorpusLoader, Doc
from src.backend.indexer import DenseIndexer, VecRecord

logger = getLogger(__name__)


class DenseEncoder(abc.ABC):
    def encode_corpus(self, texts: Iterable[str]) -> Tuple[np.ndarray, List[str]]:
        raise NotImplementedError

    def encode_queries(self, queries: Iterable[str]) -> np.ndarray:
        raise NotImplementedError


class HFSymetricDenseEncoder(DenseEncoder):
    def __init__(
        self, model_path: str, verbose: bool = True, device: str = "cuda:0"
    ) -> None:
        self.device = device
        self.verbose = verbose

        self.tokenizer, self.model = self.load_model(model_path, device=self.device)

    def load_model(
        self, model_path: str, device: str
    ) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)

        model.eval()
        model.to(device)

        return tokenizer, model

    def pooling(
        self, outputs: np.ndarray, attention_mask: np.ndarray, pooling_method: str
    ) -> np.ndarray:
        if pooling_method == "mean":

            def mean_pooling(token_embeddings, mask):
                token_embeddings = token_embeddings.masked_fill(
                    ~mask[..., None].bool(), 0.0
                )
                sentence_embeddings = (
                    token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
                )
                return sentence_embeddings

            return mean_pooling(outputs[0], attention_mask)
        elif pooling_method == "cls":
            return outputs[0][:, 0, :]
        else:
            raise ValueError(f"Pooling method {pooling_method} not supported.")

    def encode(
        self, docs: Iterable[str], pooling: str = "mean", batch_size: int = 16
    ) -> np.ndarray:
        embeddings = []
        docs_iter = tqdm(docs, desc="encoding") if self.verbose else docs
        for i, chunk in enumerate(chunked(docs_iter, batch_size)):
            inputs = self.tokenizer(
                chunk, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

                outputs = self.pooling(
                    outputs[0], inputs["attention_mask"], pooling_method=pooling
                )
                embeddings.append(outputs.detach().cpu().numpy())

        if self.verbose:
            logger.info(f"Encoded {len(embeddings)} documents.")

        return np.concatenate(embeddings)

    def encode_corpus(
        self, docs: Iterable[str], pooling: str = "mean", batch_size: int = 16
    ) -> np.ndarray:
        return self.encode(docs, pooling, batch_size)

    def encode_queries(
        self, queries: Iterable[str], pooling: str = "mean", batch_size: int = 16
    ) -> np.ndarray:
        return self.encode(queries, pooling, batch_size)


class Retriever(abc.ABC):
    def index(self, corpus: CorpusLoader):
        raise NotImplementedError

    def retrieve(self, query: Iterable[str], top_k: int) -> List[Tuple]:
        raise NotImplementedError


def docs_to_texts(self, docs: Iterable[Doc]) -> Tuple[List[str], List[str]]:
    docids = []
    texts = []
    for doc in docs:
        docids.append(doc.doc_id)
        texts.append(doc.text + " " + doc.title)
    return texts, docids


class DenseRetriever(Retriever):
    def __init__(self, encoder: DenseEncoder, vector_indexer: DenseIndexer) -> None:
        self.encoder = encoder
        self.vector_indexer = vector_indexer

    def encode_docs(
        self, texts: List[str], docids: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        return self.encoder.encode_corpus(texts), docids

    def encode_queries(self, queries: Iterable[str]) -> np.ndarray:
        return self.encoder.encode_queries(queries)

    def index(
        self,
        corpus_loader: CorpusLoader,
        doc_preprocesser: Callable[
            [Iterable[BaseModel]], Tuple[List[str], List[str]]
        ] = docs_to_texts,
        batch_size: int = 100,
    ) -> None:
        def yield_doc_vector(
            embs: np.ndarray, docids: List[str], docs_chunk: List[Doc]
        ):
            for emb, docid, doc in zip(embs, docids, docs_chunk):
                yield VecRecord(emb, docid, doc.title, doc.text)

        for docs_chunk in corpus_loader.load(batch_size=batch_size):
            embeddings, docids = self.encode_docs(doc_preprocesser(docs_chunk))
            self.vector_indexer.index(
                yield_doc_vector(embeddings, docids, docs_chunk)
            )

    def retrieve(self, queries: Iterable[str], top_k: int) -> List[Tuple]:
        embeddings = self.encode_queries(queries)
        return self.vector_indexer.query_vectors(queries, embeddings, top_k)
