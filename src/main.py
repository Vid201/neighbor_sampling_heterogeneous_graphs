import argparse
import time
import os
import os.path as osp
import glob
import yaml
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from mag240m import MAG240M
from models.runimp import RUNIMP
from ogb.lsc import MAG240MEvaluator
from tqdm import tqdm


def process_sampling_sizes(method, edge_types_per_depth, sampling_budget, sizes):
    calculated_sizes = {}

    for edge_type in sizes.keys():
        calculated_sizes[edge_type] = [0] * len(sizes[edge_type])

    for depth, edge_types in enumerate(edge_types_per_depth):
        ssum = sum(sizes[edge_type][depth] for edge_type in edge_types)
        for edge_type in edge_types:
            calculated_sizes[edge_type][depth] = sizes[edge_type][depth] / ssum

    for edge_type, values in calculated_sizes.items():
        for index, value in enumerate(values):
            calculated_sizes[edge_type][index] = int(value * sampling_budget[index])

    for depth, edge_types in enumerate(edge_types_per_depth):
        ssum = sum(calculated_sizes[edge_type][depth] for edge_type in edge_types)
        for edge_type in edge_types:
            if ssum >= sampling_budget[depth]:
                break
            calculated_sizes[edge_type][depth] += 1
            ssum += 1

    return calculated_sizes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN experiment.')
    parser.add_argument('--mode', type=str, help='Train/test mode.', default='train', choices=['train', 'test'])
    parser.add_argument('--model', type=str, help='GNN model.', default='runimp', choices=['runimp', 'runimp_option1'])
    parser.add_argument('--dataset', type=str, help='Dataset.', default='mag240m', choices=['mag240m'])
    parser.add_argument('--hns', type=str, help='Heterogeneous neighbor sampler parameters.', default='original', choices=['original', 'option1'])

    args = parser.parse_args()

    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    device = f'cuda:{config["common"]["device"]}' if torch.cuda.is_available() else 'cpu'
    seed_everything(config['common']['seed'])

    sizes = {}
    for edge, vals in config[args.dataset]['heterogenous_neighbor_sampler'][args.hns]['sizes'].items():
        sizes[tuple(edge.split('_'))] = vals

    if args.hns != 'original':
        edge_types_per_depth = []
        
        if args.dataset == 'mag240m':
            edge_types_per_depth = [
                [('paper', 'paper'), ('paper', 'author')],
                [('paper', 'paper'),('paper', 'author'), ('author', 'paper'), ('author', 'institution')]
            ]

        sizes = process_sampling_sizes(args.hns, edge_types_per_depth, config[args.dataset]['heterogenous_neighbor_sampler'][args.hns]['sampling_budget'], sizes)

    if args.dataset == 'mag240m':
        datamodule = MAG240M(config[args.dataset]['root'], config[args.model]['batch_size'], sizes, config[args.model]['in_memory'])

    if args.mode == 'train':
        if args.model.startswith('runimp'):
            model = RUNIMP(args.model, datamodule.num_features, datamodule.num_classes, config[args.model]['hidden_channels'], datamodule.num_relations, num_layers=len(sizes[('paper', 'paper')]), dropout=config[args.model]['dropout'])

        print(f'#Params {sum([p.numel() for p in model.parameters()])}')

        checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1)
        
        log_dir = f'../logs/{args.model}'
        if not osp.exists(f'{log_dir}/lightning_logs'):
            os.makedirs(f'{log_dir}/lightning_logs')

        trainer = Trainer(accelerator='gpu', devices=[config['common']['device']], max_epochs=config[args.model]['epochs'], callbacks=[checkpoint_callback], default_root_dir=log_dir)
        t = time.perf_counter()
        trainer.fit(model, datamodule=datamodule)
        
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    else:
        dirs = glob.glob(f'../logs/{args.model}/lightning_logs/*')
        version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in dirs])
        log_dir = f'../logs/{args.model}/lightning_logs/version_{version}'
        print(f'Evaluating saved model in {log_dir}...')
        ckpt = glob.glob(f'{log_dir}/checkpoints/*')[0]

        if not osp.exists(f'{log_dir}/lightning_logs'):
            os.makedirs(f'{log_dir}/lightning_logs')

        trainer = Trainer(accelerator='gpu', devices=[config['common']['device']], max_epochs=1, default_root_dir=log_dir)
        model = RUNIMP.load_from_checkpoint(checkpoint_path=ckpt, hparams_file=f'{log_dir}/hparams.yaml')

        datamodule.batch_size = 16
        datamodule.sizes_dict = {
            x: [160] * len(sizes[x])
            for x in sizes.keys()
        } # no neighbor sampling when doing inference

        trainer.test(model=model, datamodule=datamodule)

        evaluator = MAG240MEvaluator()
        loader = datamodule.hidden_test_dataloader()

        model.eval()
        model.to(device)
        y_preds = []
        for batch in tqdm(loader):
            batch = batch.to(device)
            with torch.no_grad():
                out = model(batch.x, batch.sub_y, batch.sub_y_idx, batch.pos, batch.adjs_t).argmax(dim=-1).cpu()
                y_preds.append(out)
        res = {'y_pred': torch.cat(y_preds, dim=0)}
        print(res)
