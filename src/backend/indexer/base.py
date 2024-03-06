import abc
from typing import Annotated, Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, PlainValidator, ValidationInfo


def ndarray_valicate(v: Any, info: ValidationInfo) -> np.ndarray:
    if not isinstance(v, np.ndarray):
        raise TypeError("must be a numpy.ndarray")
    return v


NdArray = Annotated[np.ndarray, PlainValidator(ndarray_valicate)]


class VecRecord(BaseModel):
    doc: BaseModel
    vec: Optional[NdArray] = None


class DenseIndexer(abc.ABC):
    @abc.abstractmethod
    def index(self, records: Iterable[VecRecord]) -> int:
        raise NotImplementedError

    @abc.abstractmethod
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
    ) -> List[Tuple[str, Dict]]:
        raise NotImplementedError
