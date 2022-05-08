from typing import Dict, Union, Tuple, List, Optional, Callable
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor

from data import Adj, EdgeIndex
from utils import unique

class HeterogeneousNeighborSampler(DataLoader):
    def __init__(self, edge_index_dict: Dict[Tuple, Union[Tensor, SparseTensor]], sizes_dict: Dict[Tuple, List[int]], node_idx: Optional[Tensor] = None, num_nodes: Optional[int] = None, return_e_id: bool = True, transform: Callable = None, **kwargs):
        for key, value in edge_index_dict.items():
            edge_index_dict[key] = value.to('cpu')
            
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        
        self.edge_index_dict = edge_index_dict
        self.node_idx = node_idx
        self.num_nodes = num_nodes
        
        layers = len(next(iter(sizes_dict.values())))
        assert all(len(sizes) == layers for sizes in sizes_dict.values())
        self.sizes_dict = sizes_dict
        
        self.return_e_id = return_e_id
        self.transform = transform
        
        edge_index_first = next(iter(edge_index_dict.values()))
        self.is_sparse_tensor = isinstance(edge_index_first, SparseTensor)
        assert all(type(edge_index) == type(edge_index_first) for edge_index in edge_index_dict.values())
        
        self.adjs_t = {}
        self.__val__ = None
        
        if not self.is_sparse_tensor:
            if (num_nodes is None and node_idx is not None and node_idx.dtype == torch.bool):
                num_nodes = node_idx.size(0)
            if (num_nodes is None and node_idx is not None and node_idx.dtype == torch.long):
                num_nodes = max(max([edge_index.max() for edge_index in edge_index_dict.values()]), int(node_idx.max())) + 1
            if num_nodes is None:
                num_nodes = max([edge_index.max() for edge_index in edge_index_dict.values()]) + 1
                
            for relation, edge_index in edge_index_dict.items():
                value = torch.arange(edge_index.size(1)) if return_e_id else None
                self.adjs_t[relation] = SparseTensor(row=edge_index[0], col=edge_index[1], value=value, sparse_sizes=(num_nodes, num_nodes), is_sorted=True)
        else:
            for relation, adj_t in edge_index_dict.items():
                if return_e_id:
                    if self.__val__ is None:
                        self.__val__ = {}
                    self.__val__[relation] = adj_t.storage.value()
                    value = torch.arange(adj_t.nnz())
                    adj_t = adj_t.set_value(value, layout='coo')
                self.adjs_t[relation] = adj_t
        
        for adj_t in self.adjs_t.values():
            adj_t.storage.rowptr()
        
        if node_idx is None:
            node_idx = torch.arange(max([max(rc.max() for rc in adj_t.coo()[:-1]) for adj_t in self.adjs_t.values()])) + 1
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)
                        
        super(HeterogeneousNeighborSampler, self).__init__(node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)
    
    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        
        batch_size: int = len(batch)
        
        adjs = []
        sample_n_id = batch
        n_ids = []
        
        for layer in range(len(next(iter(self.sizes_dict.values())))):
            rows = []
            cols = []
            e_ids = []
            vals = []
            
            n_ids.append(sample_n_id)
            size = sample_n_id.size(0)
            
            for relation in self.edge_index_dict:
                try: # must catch error - exception if cannot sample any node
                    adj_t, n_id = self.adjs_t[relation].sample_adj(sample_n_id, self.sizes_dict[relation][layer], replace=False)
                except:
                    continue

                row, col, value = adj_t.coo()
                
                row += size
                col += size
                size += n_id.size(0)
                
                rows.append(row)
                cols.append(col)
                e_ids.append(value)
                if self.__val__ is not None:
                    vals.append(self.__val__[relation][value])
                    
                n_ids.append(n_id)
            
            row = torch.cat(rows, dim=0)
            col = torch.cat(cols, dim=0)
            e_ids = torch.cat(e_ids, dim=0)
            next_n_id = torch.cat(n_ids, dim=0)
            next_un, next_inv, next_ind = unique(torch.cat([sample_n_id, next_n_id[row], next_n_id[col]], dim=0))
            next_ind_sort = torch.argsort(next_ind, dim=0)
            next_n_id = next_un[next_ind_sort]
            next_ind_sort_sort = torch.argsort(next_ind_sort)
            row = next_ind_sort_sort[next_inv[sample_n_id.size(0):sample_n_id.size(0) + row.size(0)]]
            col = next_ind_sort_sort[next_inv[sample_n_id.size(0) + row.size(0):]]
            
            row_ind = torch.argsort(row)
            row = row[row_ind]
            col = col[row_ind]
            e_ids = e_ids[row_ind]

            adj_t = SparseTensor(row=row, col=col, value=e_ids, sparse_sizes=(sample_n_id.size(0), next_n_id.size(0)), is_sorted=True)
            e_id = e_ids
            size = adj_t.sparse_sizes()[::-1]
            
            if self.__val__ is not None:
                vals = torch.cat(vals, dim=0)
                vals = vals[row_ind]
                adj_t.set_value_(vals, layout='coo')
            
            if self.is_sparse_tensor:
                adjs.append(Adj(adj_t, e_id, size))
            else:
                edge_index = torch.stack([col, row], dim=0)
                adjs.append(EdgeIndex(edge_index, e_id, size))
            
            sample_n_id = next_n_id
            n_ids.clear()
    
        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (batch_size, sample_n_id, adjs)
        out = self.transform(*out) if self.transform is not None else out
        
        return out
    
    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)