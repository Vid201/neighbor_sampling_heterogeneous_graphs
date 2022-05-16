import os.path as osp
from typing import Dict, Tuple, List, Optional
import time
from tqdm import tqdm
import numpy as np
import torch
from torch_sparse import SparseTensor
from pytorch_lightning import LightningDataModule
from ogb.lsc import MAG240MDataset
from utils import get_col_slice, save_col_slice, construct_sinusoid_encoding_table
from data import Batch
from heterogeneous_neighbor_sampler import HeterogeneousNeighborSampler

class MAG240M(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, sizes_dict: Dict[Tuple, List[int]], in_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes_dict = sizes_dict
        self.in_memory = in_memory
        
    @property
    def num_features(self) -> int:
        return 129 # 768 if not reduced
    
    @property
    def num_classes(self) -> int:
        return 153
    
    @property
    def num_relations(self) -> int:
        return 5
    
    def prepare_data(self):
        dataset = MAG240MDataset(self.data_dir)
        
        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        if not osp.exists(path):
            t = time.perf_counter()
            print('Converting adjacency matrix...', end=' ', flush=True)
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = torch.from_numpy(edge_index)
            adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(dataset.num_papers, dataset.num_papers), is_sorted=True)
            torch.save(adj_t.to_symmetric(), path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')
            
        path = f'{dataset.dir}/full_adj_t.pt'
        if not osp.exists(path):
            t = time.perf_counter()
            print('Merging adjacency matrices...', end=' ', flush=True)
            
            row, col, _ = torch.load(f'{dataset.dir}/paper_to_paper_symmetric.pt').coo()
            rows, cols = [row], [col]
            
            edge_index = dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers # authors are indexed from 0, same as papers in the dataset (thus must add dataset.num_papers)
            rows += [row, col] # add both, make it symmetric (author-paper edges)
            cols += [col, row]
            
            edge_index = dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers
            col += dataset.num_papers + dataset.num_authors # institutions are also indexed from 0
            rows += [row, col]
            cols += [col, row]
            
            edge_types = [
                torch.full(x.size(), i, dtype=torch.int8)
                for i, x in enumerate(rows)
            ]
            
            row = torch.cat(rows, dim=0)
            del rows
            col = torch.cat(cols, dim=0)
            del cols
            
            N = (dataset.num_papers + dataset.num_authors + dataset.num_institutions)
            
            perm = (N * row).add_(col).numpy().argsort() # this way permutation sorts the nodes and sparse tensor operations are faster (multiplication with N and addition with col make nodes sorted in correct way - range between consecutive nodes is number of nodes)
            perm = torch.from_numpy(perm)
            row = row[perm]
            col = col[perm]
            
            edge_type = torch.cat(edge_types, dim=0)[perm]
            del edge_types
            
            full_adj_t = SparseTensor(row=row, col=col, value=edge_type, sparse_sizes=(N, N), is_sorted=True)
            
            torch.save(full_adj_t, path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')
        
        # IGNORE: features are already preprocessed with PCA
        
        # path = f'{dataset.dir}/full_feat.npy'
        # done_flag_path = f'{dataset.dir}/full_feat_done.txt'
        # if not osp.exists(done_flag_path):
        #     t = time.perf_counter()
        #     print('Generating full feature matrix...')
            
        #     node_chunk_size = 100000
        #     dim_chunk_size = 64
        #     N = (dataset.num_papers + dataset.num_authors + dataset.num_institutions)
            
        #     paper_feat = dataset.paper_feat
        #     x = np.memmap(path, dtype=np.float16, mode='w+', shape=(N, self.num_features))
            
        #     print('Copying paper features...')
        #     for i in tqdm(range(0, dataset.num_papers, node_chunk_size)):
        #         j = min(i + node_chunk_size, dataset.num_papers)
        #         x[i:j] = paper_feat[i:j]
                
        #     edge_index = dataset.edge_index('author', 'writes', 'paper') # ask if connection strength is actually 16 authors for shared two papers of interest (as it is said in the article)
        #     row, col = torch.from_numpy(edge_index)
        #     adj_t = SparseTensor(row=row, col=col, sparse_sizes=(dataset.num_authors, dataset.num_papers), is_sorted=True) # sorted probably because number of nodes is very big and sparse tensor operations are faster (matmul ...)
            
        #     print('Generating author features...')
        #     for i in tqdm(range(0, self.num_features, dim_chunk_size)):
        #         j = min(i + dim_chunk_size, self.num_features)
        #         inputs = get_col_slice(paper_feat, start_row_idx=0, end_row_idx=dataset.num_papers, start_col_idx=i, end_col_idx=j)
        #         inputs = torch.from_numpy(inputs)
        #         outputs = adj_t.matmul(inputs, reduce='mean').numpy()
        #         del inputs
        #         save_col_slice(x_src=outputs, x_dst=x, start_row_idx=dataset.num_papers, end_row_idx=dataset.num_papers + dataset.num_authors, start_col_idx=i, end_col_idx=j)
        #         del outputs
                
        #     edge_index = dataset.edge_index('author', 'institution')
        #     row, col = torch.from_numpy(edge_index)
        #     adj_t = SparseTensor(row=col, col=row, sparse_sizes=(dataset.num_institutions, dataset.num_authors), is_sorted=False) # no need to sort the sparse tensor - number of institutions is not big
            
        #     print('Generating institution features...')
        #     for i in tqdm(range(0, self.num_features, dim_chunk_size)):
        #         j = min(i + dim_chunk_size, self.num_features)
        #         inputs = get_col_slice(x, start_row_idx=dataset.num_papers, end_row_idx=dataset.num_papers + dataset.num_authors, start_col_idx=i, end_col_idx=j)
        #         inputs = torch.from_numpy(inputs)
        #         outputs = adj_t.matmul(inputs, reduce='mean').numpy()
        #         del inputs
        #         save_col_slice(x_src=outputs, x_dst=x, start_row_idx=dataset.num_papers + dataset.num_authors, end_row_idx=N, start_col_idx=i, end_col_idx=j)
        #         del outputs
            
        #     x.flush()
        #     del x
        #     print(f'Done! [{time.perf_counter() - t:.2f}s]')
            
        #     with open(done_flag_path, 'w') as f:
        #         f.write('done')
                
        path = f'{dataset.dir}/all_feat_year.npy'
        done_flag_path = f'{dataset.dir}/all_feat_year_done.txt'
        if not osp.exists(done_flag_path):
            t = time.perf_counter()
            print('Generating all feature year matrix...')
            
            node_chunk_size = 100000
            N = (dataset.num_papers + dataset.num_authors + dataset.num_institutions)
            
            paper_year = dataset.all_paper_year.reshape((-1, 1))
            x = np.memmap(path, dtype=np.int64, mode='w+', shape=(N,1))
            
            print('Copying paper years...')
            for i in tqdm(range(0, dataset.num_papers, node_chunk_size)):
                j = min(i + node_chunk_size, dataset.num_papers)
                x[i:j] = paper_year[i:j]
            
            edge_index = dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(row=row, col=col, sparse_sizes=(dataset.num_authors, dataset.num_papers), is_sorted=True)
            
            print('Generating author years...')
            inputs = get_col_slice(paper_year, start_row_idx=0, end_row_idx=dataset.num_papers, start_col_idx=0, end_col_idx=1)
            inputs = torch.from_numpy(inputs)
            outputs = adj_t.matmul(inputs, reduce='mean').numpy()
            del inputs
            save_col_slice(x_src=outputs, x_dst=x, start_row_idx=dataset.num_papers, end_row_idx=dataset.num_papers + dataset.num_authors, start_col_idx=0, end_col_idx=1)
            del outputs
            
            edge_index = dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(dataset.num_institutions, dataset.num_authors), is_sorted=False)
            
            print('Generating institution years...')
            inputs = get_col_slice(x, start_row_idx=dataset.num_papers, end_row_idx=dataset.num_papers + dataset.num_authors, start_col_idx=0, end_col_idx=1)
            inputs = torch.from_numpy(inputs)
            outputs = adj_t.matmul(inputs, reduce='mean').numpy()
            del inputs
            save_col_slice(x_src=outputs, x_dst=x, start_row_idx=dataset.num_papers + dataset.num_authors, end_row_idx=N, start_col_idx=0, end_col_idx=1)
            del outputs
            
            x.flush()
            del x
            print(f'Done! [{time.perf_counter() - t:.2f}s]')
            
            with open(done_flag_path, 'w') as f:
                f.write('done') 

    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        dataset = MAG240MDataset(self.data_dir)
        
        self.train_idx = torch.from_numpy(dataset.get_idx_split('train'))
        self.train_idx.share_memory_()
        self.val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
        self.val_idx.share_memory_()
        self.test_idx = torch.from_numpy(dataset.get_idx_split('valid')) # test is not available
        self.test_idx.share_memory_()
        
        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions
        self.num_nodes = N
        
        x = np.memmap(f'{dataset.dir}/full_feat.npy', dtype=np.float16, mode='r', shape=(N, self.num_features))
        
        if self.in_memory:
            self.x = np.empty((N, self.num_features), dtype=np.float16)
            self.x[:] = x
            self.x = torch.from_numpy(self.x).share_memory_()
        else:
            self.x = x
        
        self.y = torch.from_numpy(dataset.all_paper_label)
        y_random_mask = (self.y.isnan()) | (self.y == -1)
        y_random_labels = torch.randint(0, dataset.num_classes, (torch.count_nonzero(y_random_mask),))
        self.y[y_random_mask] = y_random_labels.double()
        
        path = f'{dataset.dir}/full_adj_t.pt'
        self.adj_t = torch.load(path)
        
        # TODO: read metapath2vec embeddings
        
        self.pos = construct_sinusoid_encoding_table(200, self.num_features)
        self.year = np.memmap(f'{dataset.dir}/all_feat_year.npy', dtype=np.int64, mode='r', shape=(N, 1))
        
        self.edge_index_dict = {}
        
        edge_type = self.adj_t.storage.value() == 0
        self.edge_index_dict[('paper', 'paper')] = self.adj_t.masked_select_nnz(edge_type, layout='coo')
        edge_type = self.adj_t.storage.value() == 1
        self.edge_index_dict[('author', 'paper')] = self.adj_t.masked_select_nnz(edge_type, layout='coo')
        edge_type = self.adj_t.storage.value() == 2
        self.edge_index_dict[('paper', 'author')] = self.adj_t.masked_select_nnz(edge_type, layout='coo')
        edge_type = self.adj_t.storage.value() == 3
        self.edge_index_dict[('author', 'institution')] = self.adj_t.masked_select_nnz(edge_type, layout='coo')
        edge_type = self.adj_t.storage.value() == 4
        self.edge_index_dict[('institution', 'author')] = self.adj_t.masked_select_nnz(edge_type, layout='coo')
                
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    
    def train_dataloader(self):
        return HeterogeneousNeighborSampler(self.edge_index_dict, sizes_dict=self.sizes_dict, node_idx=self.train_idx, return_e_id=False, transform=self.convert_batch, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return HeterogeneousNeighborSampler(self.edge_index_dict, sizes_dict=self.sizes_dict, node_idx=self.val_idx, return_e_id=False, transform=self.convert_batch, batch_size=self.batch_size, num_workers=2)
    
    def test_dataloader(self):
        return HeterogeneousNeighborSampler(self.edge_index_dict, sizes_dict=self.sizes_dict, node_idx=self.val_idx, return_e_id=False, transform=self.convert_batch, batch_size=self.batch_size, num_workers=4)
    
    def hidden_test_dataloader(self):
        return HeterogeneousNeighborSampler(self.edge_index_dict, sizes_dict=self.sizes_dict, node_idx=self.test_idx, return_e_id=False, transform=self.convert_batch, batch_size=self.batch_size, num_workers=4)
    
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
        
        pos = 2021 - self.year[n_id]
        pos = torch.squeeze(self.pos[pos])
                
        return Batch(x=x, y=y, sub_y=sub_y, sub_y_idx=sub_n_id, pos=pos, adjs_t=[adj_t for adj_t, _, _ in adjs])