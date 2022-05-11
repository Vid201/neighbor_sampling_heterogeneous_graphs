import argparse
import time
import os
import os.path as osp
import yaml
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from mag240m import MAG240M
from models.runimp import RUNIMP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN experiment.')
    parser.add_argument('--mode', type=str, help='Train/test mode.', default='train', choices=['train', 'test'])
    parser.add_argument('--model', type=str, help='GNN model.', default='runimp', choices=['runimp'])
    parser.add_argument('--dataset', type=str, help='Dataset.', default='mag240m', choices=['mag240m'])
    parser.add_argument('--hns', type=str, help='Heterogeneous neighbor sampler parameters.', default='original', choices=['original'])

    args = parser.parse_args()

    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    device = f'cuda:{config["common"]["device"]}' if torch.cuda.is_available() else 'cpu'
    seed_everything(config['common']['seed'])

    sizes = {}
    for edge, vals in config[args.dataset]['heterogenous_neighbor_sampler'][args.hns].items():
        sizes[tuple(edge.split('_'))] = vals

    if args.dataset == 'mag240m':
        datamodule = MAG240M(config[args.dataset]['root'], config[args.model]['batch_size'], sizes, config[args.model]['in_memory'])

    if args.mode == 'train':
        if args.model == 'runimp':
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
