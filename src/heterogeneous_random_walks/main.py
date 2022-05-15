import argparse
import torch
import os.path as osp
import pprint
# from ogb.lsc import MAG240MDataset
import numpy as np
import pickle

from build.random_walks import heterogeneous_random_walk

def test(args):
    pp = pprint.PrettyPrinter(indent=4)

    # papers: 0, 1
    # authors: 2, 3
    # institutions: 4

    node_types = ['paper', 'author', 'institution']
    edge_types = [
        'paper cites paper',
        'paper rev_writes author',
        'author writes paper',
        'author aff_with institution',
        'institution rev_aff_with author'
    ]

    edge_index_dict = {
        'paper cites paper': torch.tensor([[0, 1], [1, 0]]), 
        'paper rev_writes author': torch.tensor([[0, 0, 1], [2, 2, 3]]),
        'author writes paper': torch.tensor([[2, 2, 3],  [0, 0, 1]]),
        'author aff_with institution': torch.tensor([[2], [4]]),
        'institution rev_aff_with author': torch.tensor([[4], [2]]),
    }

    start = torch.tensor([0, 0, 1, 2, 2, 3, 3, 4] * args.walks)
    start_types = ['paper', 'paper', 'paper', 'author', 'author', 'author', 'author', 'institution'] * args.walks

    rowptr_dict = {}
    col_dict = {}

    for relation, edge_index in edge_index_dict.items():
        row, col = edge_index
        num_nodes = max(int(row.max()), int(col.max()), int(start.max())) + 1
        deg = row.new_zeros(num_nodes)
        deg.scatter_add_(0, row, torch.ones_like(row))
        rowptr = row.new_zeros(num_nodes + 1)
        torch.cumsum(deg, 0, out=rowptr[1:])
        rowptr_dict[relation] = rowptr
        col_dict[relation] = col

    random_walks = heterogeneous_random_walk(node_types, edge_types, rowptr_dict, col_dict, start, start_types, args.length)

    pprint.pprint(random_walks)


def convert(row, col):    
    num_nodes = max(int(row.max()), int(col.max())) + 1
    
    perm = torch.argsort(row * num_nodes + col)
    row_sorted, col_sorted = row[perm], col[perm]
    
    deg = row_sorted.new_zeros(num_nodes)
    deg.scatter_add_(0, row_sorted, torch.ones_like(row_sorted))
    rowptr = row_sorted.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])
    
    return rowptr, col_sorted


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heterogeneous random walks.')
    parser.add_argument('--walks', type=int, help='Number of random walks.', default=100000)
    parser.add_argument('--length', type=int, help='Length of random walks.', default=10)
    parser.add_argument('--demo', help='Demo mode.', action='store_true')

    args = parser.parse_args()

    if args.demo:
        test(args)
    else:
        root = '/lfs/rambo/1/vid/mag240m'

        path = '/lfs/rambo/1/vid/mag240m/mag240m_kddcup2021/rowptr_dict.p'
        path2 = '/lfs/rambo/1/vid/mag240m/mag240m_kddcup2021/col_dict.p'
        path3 = '/lfs/rambo/1/vid/mag240m/mag240m_kddcup2021/train_idx.p'

        # if not osp.exists(path):
        #     dataset = MAG240MDataset(root)

        #     num_nodes_dict = {
        #         'paper': dataset.num_papers,
        #         'author': dataset.num_authors,
        #         'institution': dataset.num_institutions
        #     }
        #     node_types = ['paper', 'author', 'institution']

        #     node_type_all = np.zeros((dataset.num_papers + dataset.num_authors + dataset.num_institutions,))
        #     node_types_start = {}
        #     node_types = {}
        #     start = 0
        #     num_nodes = 0

        #     counter = 0

        #     for node_type, node_count in num_nodes_dict.items():
        #         node_type_all[start:start+node_count] = counter
        #         node_types_start[node_type] = start
        #         node_types[counter] = node_type
        #         counter += 1
        #         start += node_count
        #         num_nodes += node_count

        #     if not osp.exists(path):
        #         edge_index_dict = {
        #             ('paper', 'cites', 'paper'): dataset.edge_index('paper', 'paper'),
        #             ('author', 'writes', 'paper'): dataset.edge_index('author', 'paper'),
        #             ('author', 'affiliated_with', 'institution'): dataset.edge_index('author', 'institution')
        #         }
                
        #         rowptr_dict = {}
        #         col_dict = {}

        #         for edge_type, edge_index in edge_index_dict.items():
        #             src, rel, dst = edge_type
                    
        #             row, col = edge_index
        #             row += node_types_start[src]
        #             col += node_types_start[dst]
                    
        #             row = torch.from_numpy(row)
        #             col = torch.from_numpy(col)
                    
        #             if src == dst:
        #                 row_tmp = torch.clone(row)
        #                 col_tmp = torch.clone(col)
        #                 row = torch.hstack([row_tmp, col_tmp])
        #                 col = torch.hstack([col_tmp, row_tmp])
                    
        #             rowptr, coll = convert(row, col)
        #             rowptr_dict[f'{src} {rel} {dst}'] = rowptr
        #             col_dict[f'{src} {rel} {dst}'] = coll
                    
        #             if src != dst:
        #                 rowptr, coll = convert(col, row)
        #                 rowptr_dict[f'{dst} rev_{rel} {src}'] = rowptr
        #                 col_dict[f'{dst} rev_{rel} {src}'] = coll
                    
        #             print(f'{edge_type} finished!', flush=True)
                    
        #         pickle.dump(rowptr_dict, open(path, 'wb'), protocol=4)
        #         pickle.dump(col_dict, open(path2, 'wb'), protocol=4)
                
        #         train_idx = torch.from_numpy(dataset.get_idx_split('train')) + node_types_start['paper']

        #         pickle.dump(train_idx, open(path3, 'wb'), protocol=4)
        # else:

        rowptr_dict = pickle.load(open(path, 'rb'))
        col_dict = pickle.load(open(path2, 'rb'))
        train_idx = pickle.load(open(path3, 'rb'))

        node_types = ['paper', 'author', 'institution']
        edge_types = list(col_dict.keys())
        start = torch.from_numpy(np.random.choice(train_idx, args.walks))
        start_types = ['paper'] * args.walks

        random_walks = heterogeneous_random_walk(node_types, edge_types, rowptr_dict, col_dict, start, start_types, args.length)
        random_walks_nodes, random_walks_edge_types = random_walks

        path4 = '/lfs/rambo/1/vid/mag240m/mag240m_kddcup2021/random_walks_nodes.pt'
        torch.save(random_walks_nodes, path4)
        path5 = '/lfs/rambo/1/vid/mag240m/mag240m_kddcup2021/random_walks_edge_types.pt'
        torch.save(random_walks_edge_types, path5)

        print("Finished!")
