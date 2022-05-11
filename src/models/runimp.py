from typing import List
import torch
from torch import Tensor
from torch.nn import ModuleList, Embedding, Sequential, Linear, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning import LightningModule
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor
from torchmetrics import Accuracy

class RUNIMP(LightningModule):
    def __init__(self, model: str, in_channels: int, out_channels: int, hidden_channels: int, num_relations: int, num_layers: int, heads: int = 4, dropout: float = 0.5, attn_dropout: float = 0.6):
        super().__init__()
        self.save_hyperparameters()
        self.model = model.lower()
        self.num_relations = num_relations
        self.dropout = dropout
        self.attn_dropout = attn_dropout # I think this is the label_rate (masking percentage of labels - used in conv operation)
        self.in_dropout = 0.3
        
        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()
        self.path_attns = ModuleList()
        self.path_norms = ModuleList()
        
        self.label_emb = Embedding(out_channels, in_channels)
        # self.m2v_emb = Linear(64, in_channels)
        
        self.label_mlp = Sequential(
            Linear(2 * in_channels, hidden_channels),
            BatchNorm1d(hidden_channels, momentum=0.9),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
            Linear(hidden_channels, in_channels)
        )
        
        self.in_drop = Dropout(p=self.in_dropout)
        
        self.convs.append(ModuleList([GATConv(in_channels, hidden_channels // heads, heads, dropout=attn_dropout, add_self_loops=False) for _ in range(num_relations)]))
        for _ in range(num_layers - 1):
            self.convs.append(ModuleList([GATConv(hidden_channels, hidden_channels // heads, heads, dropout=attn_dropout, add_self_loops=False) for _ in range(num_relations)]))
    
        self.skips.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.skips.append(Linear(hidden_channels, hidden_channels))
            
        for _ in range(num_layers):
            self.norms.append(ModuleList([BatchNorm1d(hidden_channels, momentum=0.9) for _ in range(num_relations + 1)]))
            
        for _ in range(num_layers):
            self.path_attns.append(Linear(hidden_channels, 1))
            self.path_norms.append(BatchNorm1d(hidden_channels, momentum=0.9))
            
        self.drop = Dropout(p=self.dropout)
        
        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels, momentum=0.9),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
            Linear(hidden_channels, out_channels)
        )
        
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        
    def forward(self, x: Tensor, y: Tensor, y_idx: Tensor, pos: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        # m2v_out = self.in_drop(self.m2v_emb(m2v))
        # x = x + m2v_out
        
        x += pos
        
        if torch.numel(y_idx) > 0:
            y_out = torch.squeeze(self.in_drop(self.label_emb(y.view(-1, 1))))
            x_ind = x[y_idx]
            xy = torch.cat([y_out, x_ind], dim=1)
            xy = self.label_mlp(xy)
            x[y_idx] = xy

        for i, adj_t in enumerate(adjs_t):
            temp_x = []

            x_target = x[:adj_t.size(0)]
            skip_x = self.skips[i](x_target)
            skip_x = self.norms[i][0](skip_x)
            skip_x = F.elu(skip_x)
            temp_x.append(skip_x)
            
            for j in range(self.num_relations):
                edge_type = adj_t.storage.value() == j
                if edge_type.sum() > 0:
                    subadj_t = adj_t.masked_select_nnz(edge_type, layout='coo')
                    if subadj_t.nnz() > 0:
                        feature_x = self.convs[i][j]((x, x_target), subadj_t)
                        feature_x = self.norms[i][j+1](feature_x)
                        feature_x = F.elu(feature_x)
                        temp_x.append(feature_x)
            
            temp_x = torch.stack(temp_x, dim=1)
            temp_x_attn = self.path_attns[i](temp_x)
            temp_x_attn = F.softmax(temp_x_attn, dim=1)
            temp_x_attn = torch.transpose(temp_x_attn, 1, 2)
            x = torch.bmm(temp_x_attn, temp_x)[:, 0]
            x = self.path_norms[i](x)
            x = self.drop(x)
            
        out = self.mlp(x)
        return out
                              
    def training_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.sub_y, batch.sub_y_idx, batch.pos, batch.adjs_t)
        train_loss = F.cross_entropy(y_hat, batch.y)
        self.train_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.x.shape[0])
        self.log('train_loss', train_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.x.shape[0])
        return train_loss
    
    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.sub_y, batch.sub_y_idx, batch.pos, batch.adjs_t)
        val_loss = F.cross_entropy(y_hat, batch.y)
        self.val_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch.x.shape[0])
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch.x.shape[0])
        
    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.sub_y, batch.sub_y_idx, batch.pos, batch.adjs_t)
        self.test_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch.x.shape[0])
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]