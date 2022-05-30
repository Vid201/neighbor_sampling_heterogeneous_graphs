from typing import Dict, Tuple, List, Optional, Union
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from pytorch_lightning import LightningDataModule
from torch_geometric.data import NeighborSampler
from torch_geometric.datasets import HGBDataset
from data import Batch
from heterogeneous_neighbor_sampler import HeterogeneousNeighborSampler
from sklearn.model_selection import train_test_split

class Freebase(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, sizes: Union[List[int], Dict[Tuple, List[int]]], in_memory: bool = False, hns: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes = sizes
        self.in_memory = in_memory
        self.hns = hns

    @property
    def num_features(self) -> int:
        return 8
    
    @property
    def num_classes(self) -> int:
        return 7
    
    @property
    def num_relations(self) -> int:
        return 64
    
    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading and converting dataset...', end=' ', flush=True)

        dataset = HGBDataset(self.data_dir, 'Freebase')[0]

        # renumber nodes
        num_nodes_dict = {
            node_type: dataset[node_type].num_nodes
            for node_type in dataset.node_types 
        }

        node_types = dataset.node_types 

        node_type_all = np.zeros((sum(num_nodes_dict.values()),))
        node_types_start = {}
        node_types = {}
        start = 0
        num_nodes = 0

        counter = 0

        for node_type, node_count in num_nodes_dict.items():
            node_type_all[start:start+node_count] = counter
            node_types_start[node_type] = start
            node_types[counter] = node_type
            counter += 1
            start += node_count
            num_nodes += node_count

        labeled_nodes_idx = (dataset['book'].train_mask).nonzero(as_tuple=True)[0] + node_types_start['book']
        train_idx, val_idx = train_test_split(labeled_nodes_idx, test_size=(0.06 / 0.3), random_state=42)

        self.train_idx = train_idx
        self.train_idx.share_memory_()
        self.val_idx = val_idx
        self.val_idx.share_memory_()
        self.test_idx = (dataset['book'].test_mask).nonzero(as_tuple=True)[0] + node_types_start['book'] # test is not available (only ids of nodes)
        self.test_idx.share_memory_()

        N = sum([dataset[node_type].num_nodes for node_type in dataset.node_types])
        self.num_nodes = N

        x = np.repeat(np.arange(self.num_features), [dataset[node_type].num_nodes for node_type in dataset.node_types])
        self.x = F.one_hot(torch.from_numpy(x), num_classes=self.num_features)

        self.y = dataset['book'].y.double()
        y_random_mask = (self.y.isnan()) | (self.y == -1)
        y_random_labels = torch.randint(0, self.num_classes, (torch.count_nonzero(y_random_mask),))
        self.y[y_random_mask] = y_random_labels.double()

        if self.hns:
            self.edge_index_dict = {}
            edge_type_counter = 0

            for edge_type in dataset.edge_types:
                src, _, dst = edge_type
                row, col = dataset[edge_type].edge_index

                row += node_types_start[src]
                col += node_types_start[dst]

                row = torch.clone(row)
                col = torch.clone(col)

                if src == dst:
                    row_tmp = torch.clone(row)
                    col_tmp = torch.clone(col)
                    row = torch.hstack([row_tmp, col_tmp])
                    col = torch.hstack([col_tmp, row_tmp])
                
                N = max(row.max(), col.max()).item()
                perm = (N * row).add_(col).numpy().argsort()
                perm = torch.from_numpy(perm)
                row = row[perm]
                col = col[perm]
                val = torch.full(perm.size(), edge_type_counter, dtype=torch.int8)
                edge_type_counter += 1

                self.edge_index_dict[tuple([src, dst])] = SparseTensor(row=row, col=col, value=val, sparse_sizes=(self.num_nodes, self.num_nodes), is_sorted=True) 

                if src != dst:
                    row, col = col, row
                    src, dst = dst, src

                    perm = (N * row).add_(col).numpy().argsort()
                    perm = torch.from_numpy(perm)
                    row = row[perm]
                    col = col[perm]
                    val = torch.full(perm.size(), edge_type_counter, dtype=torch.int8)
                    edge_type_counter += 1

                    self.edge_index_dict[tuple([src, dst])] = SparseTensor(row=row, col=col, value=val, sparse_sizes=(self.num_nodes, self.num_nodes), is_sorted=True) 

            # print(self.edge_index_dict)
        else:
            rows = []
            cols = []

            for edge_type in dataset.edge_types:
                src, _, dst = edge_type
                row, col = dataset[edge_type].edge_index

                row += node_types_start[src]
                col += node_types_start[dst]

                row = torch.clone(row)
                col = torch.clone(col)

                if src == dst:
                    row_tmp = torch.clone(row)
                    col_tmp = torch.clone(col)
                    row = torch.hstack([row_tmp, col_tmp])
                    col = torch.hstack([col_tmp, row_tmp])
                
                rows += [row]
                cols += [col]

                if src != dst:
                    rows += [col]
                    cols += [row]
                
            vals = [
                torch.full(x.size(), i, dtype=torch.int8)
                for i, x in enumerate(rows)
            ]

            row = torch.cat(rows, dim=0)
            del rows
            col = torch.cat(cols, dim=0)
            del cols

            N = dataset.num_nodes
            perm = (N * row).add_(col).numpy().argsort()
            perm = torch.from_numpy(perm)
            row = row[perm]
            col = col[perm]

            val = torch.cat(vals, dim=0)[perm]
            del vals

            self.edge_index = SparseTensor(row=row, col=col, value=val, sparse_sizes=(N, N), is_sorted=True)

        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    
    def train_dataloader(self):
        if self.hns:
            return HeterogeneousNeighborSampler(self.edge_index_dict, sizes_dict=self.sizes, node_idx=self.train_idx, return_e_id=False, transform=self.convert_batch, batch_size=self.batch_size, shuffle=True, num_workers=4)
        else:
            return NeighborSampler(self.edge_index, sizes=self.sizes, node_idx=self.train_idx, return_e_id=False, transform=self.convert_batch, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        if self.hns:
            return HeterogeneousNeighborSampler(self.edge_index_dict, sizes_dict=self.sizes, node_idx=self.val_idx, return_e_id=False, transform=self.convert_batch, batch_size=self.batch_size, num_workers=2)
        else:
            return NeighborSampler(self.edge_index, sizes=self.sizes, node_idx=self.val_idx, return_e_id=False, transform=self.convert_batch, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        if self.hns:
            return HeterogeneousNeighborSampler(self.edge_index_dict, sizes_dict=self.sizes, node_idx=self.val_idx, return_e_id=False, transform=self.convert_batch, batch_size=self.batch_size, num_workers=4)
        else:
            return NeighborSampler(self.edge_index, sizes=self.sizes, node_idx=self.val_idx, return_e_id=False, transform=self.convert_batch, batch_size=self.batch_size, num_workers=4)

    def hidden_test_dataloader(self):
        if self.hns:
            return HeterogeneousNeighborSampler(self.edge_index_dict, sizes_dict=self.sizes, node_idx=self.test_idx, return_e_id=False, transform=self.convert_batch, batch_size=self.batch_size, num_workers=4)
        else:
            return NeighborSampler(self.edge_index, sizes=self.sizes, node_idx=self.test_idx, return_e_id=False, transform=self.convert_batch, batch_size=self.batch_size, num_workers=4)

    def convert_batch(self, batch_size, n_id, adjs):
        t = time.perf_counter()
        if self.in_memory:
            x = self.x[n_id].to(torch.float)
        else:
            x = torch.from_numpy(self.x[n_id.numpy()]).to(torch.float)
        y = self.y[n_id[:batch_size]].to(torch.long)

        sub_n_id = torch.LongTensor(list((set(n_id.tolist()) - set(n_id[:batch_size].tolist())) & set(self.train_idx.tolist())))
        sub_n_id = sub_n_id[sub_n_id < self.y.shape[0]]
        sub_y = self.y[sub_n_id].to(torch.long)
        n_id_dict = {y.item():x for x, y in enumerate(n_id)}
        sub_n_id = torch.tensor([n_id_dict[x.item()] for x in sub_n_id], dtype=n_id.dtype)
        
        return Batch(x=x, y=y, sub_y=sub_y, sub_y_idx=sub_n_id, adjs_t=[adj_t for adj_t, _, _ in adjs], pos=None)