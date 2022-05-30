from torch import Tensor
from torch_sparse import SparseTensor
from typing import NamedTuple, List, Tuple, Optional

class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    sub_y: Tensor
    sub_y_idx: Tensor
    pos: Optional[Tensor]
    adjs_t: List[SparseTensor]
    
    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            sub_y=self.sub_y.to(*args, **kwargs),
            sub_y_idx=self.sub_y_idx.to(*args, **kwargs),
            pos=self.pos.to(*args, **kwargs) if self.pos is not None else None,
            adjs_t=[adj_t.to(*args, **kwargs) for adj_t in self.adjs_t]
        )

class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)

class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)