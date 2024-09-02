from typing import Any

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta


class RerankerPyterrierTransformer(pt.Transformer):
    def __init__(self, model: Any):
        self.model = model

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.result_frame(inp, ['query', 'text'])
        return pt.apply.by_query(self._transform_by_query)(inp)

    def _transform_by_query(self, query_inp: pd.DataFrame) -> pd.DataFrame:
        results = self.model.rank(
            query=query_inp['query'].iloc[0],
            docs=query_inp['text'].tolist(),
        )
        scores = [r.score for r in results]
        res = query_inp.assign(score=scores)
        res = query_inp.sort_values('score', ascending=False)
        return pt.model.add_ranks(res)
