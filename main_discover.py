import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.metrics import Accuracy,F1
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from utils.data_v2 import get_datamodule
from utils.nets import MultiHeadBERT
from utils.eval import ClusterMetrics, classifyMetrics
from utils.sinkhorn_knopp import SinkhornKnopp
from utils.util import TSNE_visualization, pca_visualization

import numpy as np
from argparse import ArgumentParser
from datetime import datetime
import os
import random
import pandas as pd


parser = ArgumentParser()
parser.add_argument("--dataset", default="banking", type=str, help="dataset")
parser.add_argument("--data_dir", default="dataset/banking", type=str, help="data directory")
parser.add_argument("--OOD_ratio", default=1.0, type=float, help="softmax temperature")
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--batch_size", default=256, type=int, help="batch size")
parser.add_argument("--num_workers", default=10, type=int, help="number of workers")
parser.add_argument("--arch", default="bert-base-uncased", type=str, help="backbone architecture")
parser.add_argument("--base_lr", default=0.4, type=float, help="learning rate")
parser.add_argument("--min_lr", default=0.01, type=float, help="min learning rate")
parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--weight_decay_opt", default=1.5e-4, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")
parser.add_argument("--hidden_dim", default=2048, type=int, help="hidden dim in proj/pred head")
parser.add_argument("--overcluster_factor", default=1, type=int, help="overclustering factor")
parser.add_argument("--num_heads", default=5, type=int, help="number of heads for clustering")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of hidden layers")
parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
parser.add_argument("--project", default="GID_benchmark", type=str, help="wandb project")
parser.add_argument("--entity", default=None, type=str, help="wandb entity")
parser.add_argument("--offline", default=False, action="store_true", help="disable wandb")
parser.add_argument("--num_labeled_classes", default=80, type=int, help="number of labeled classes")
parser.add_argument("--num_unlabeled_classes", default=20, type=int, help="number of unlab classes")
parser.add_argument("--pretrained", type=str, help="pretrained checkpoint path")
parser.add_argument("--multicrop", default=False, action="store_true", help="activates multicrop")
parser.add_argument("--num_large_crops", default=2, type=int, help="number of large crops")
parser.add_argument("--num_small_crops", default=2, type=int, help="number of small crops")
parser.add_argument("--save_results_path", type=str, default='outputs_v1', help="The path to save results.")
parser.add_argument('--mode', type=str, default="none", help="Random seed for initialization.")
parser.add_argument('--seed', type=int, default=0, help="Random seed for initialization.")
parser.add_argument("--IND_ratio", default=1.0, type=float, help="softmax temperature")
parser.add_argument("--label_ratio", default=1.0, type=float, help="softmax temperature")
parser.add_argument("--download", default=False, action="store_true", help="wether to download")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(10)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Discoverer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters({k: v for (k, v) in kwargs.items() if not callable(v)})

        # build model
        self.model = MultiHeadBERT.from_pretrained(
            self.hparams.arch,
            self.hparams.num_labeled_classes,
            self.hparams.num_unlabeled_classes,
            overcluster_factor=self.hparams.overcluster_factor,
            num_heads=self.hparams.num_heads,
            num_hidden_layers=self.hparams.num_hidden_layers,
        )

        state_dict = torch.load(self.hparams.pretrained, map_location=self.device)
        state_dict = {k: v for k, v in state_dict.items() if ("unlab" not in k)}
        self.model.load_state_dict(state_dict, strict=False)
        print("loading pretrain model:", self.hparams.pretrained)
        self.freeze_parameters(self.model)
        self.best_head = 0

        # Sinkorn-Knopp
        self.sk = SinkhornKnopp(
            num_iters=self.hparams.num_iters_sk, epsilon=self.hparams.epsilon_sk
        )

        self.total_features = torch.empty((0, 768)).to(self.device)
        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        # metrics
        '''
        self.metrics = torch.nn.ModuleList(
            [
                ClusterMetrics(self.hparams.num_heads),
                ClusterMetrics(self.hparams.num_heads),
                Accuracy(),
            ]
        )
        '''
        self.metrics_inc = torch.nn.ModuleList(
            [
                Accuracy(),
                ClusterMetrics(self.hparams.num_heads),
                ClusterMetrics(self.hparams.num_heads),
            ]
        )

        self.metrics_inc_test = torch.nn.ModuleList(
            [
                Accuracy(),
                #classifyMetrics(),
                ClusterMetrics(self.hparams.num_heads),
                ClusterMetrics(self.hparams.num_heads),
            ]
        )

        self.test_results = {}

        # buffer for best head tracking
        self.register_buffer("loss_per_head", torch.zeros(self.hparams.num_heads))

    def freeze_parameters(self,model):
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum_opt,
            weight_decay=self.hparams.weight_decay_opt,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=self.hparams.min_lr,
            eta_min=self.hparams.min_lr,
        )
        return [optimizer], [scheduler]

    def cross_entropy_loss(self, preds, targets):
        preds = F.log_softmax(preds / self.hparams.temperature, dim=-1)
        return -torch.mean(torch.sum(targets * preds, dim=-1))

    def swapped_prediction(self, logits, targets):
        loss = 0
        for view in range(self.hparams.num_large_crops):
            for other_view in np.delete(range(self.hparams.num_crops), view):
                loss += self.cross_entropy_loss(logits[other_view], targets[view])
        return loss / (self.hparams.num_large_crops * (self.hparams.num_crops - 1))

    def on_epoch_start(self):
        self.loss_per_head = torch.zeros_like(self.loss_per_head)

    def unpack_batch(self, batch):
        input_ids, input_mask, segment_ids, label_ids = batch
        mask_lab = label_ids < self.hparams.num_labeled_classes
        return input_ids, input_mask, segment_ids, label_ids, mask_lab

    def training_step(self, batch, _):
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, mask_lab = self.unpack_batch(batch)

        nlc = self.hparams.num_labeled_classes

        # normalize prototypes
        self.model.normalize_prototypes()

        # forward
        outputs = self.model(input_ids, input_mask, segment_ids, mode="discovery")

        # gather outputs
        outputs["logits_lab"] = (
            outputs["logits_lab"].unsqueeze(1).expand(-1, self.hparams.num_heads, -1, -1)
        )

        logits = torch.cat([outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)
        logits_over = torch.cat([outputs["logits_lab"], outputs["logits_unlab_over"]], dim=-1)


        # create targets
        targets_lab = (
            F.one_hot(label_ids[mask_lab], num_classes=self.hparams.num_labeled_classes)
            .float()
            .to(self.device)
        )

        targets = torch.zeros_like(logits)
        targets_over = torch.zeros_like(logits_over)

        # generate pseudo-labels with sinkhorn-knopp and fill unlab targets
        for v in range(self.hparams.num_large_crops):
            for h in range(self.hparams.num_heads):
                targets[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets_over[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets[v, h, ~mask_lab, nlc:] = self.sk(
                    outputs["logits_unlab"][v, h, ~mask_lab]
                ).type_as(targets)
                targets_over[v, h, ~mask_lab, nlc:] = self.sk(
                    outputs["logits_unlab_over"][v, h, ~mask_lab]
                ).type_as(targets)

        # compute swapped prediction loss
        loss_cluster = self.swapped_prediction(logits, targets)
        loss_overcluster = self.swapped_prediction(logits_over, targets_over)

        # total loss
        loss = (loss_cluster + loss_overcluster) / 2

        # update best head tracker
        self.loss_per_head += loss_cluster.clone().detach()

        # log
        results = {
            "loss": loss.detach(),
            "loss_cluster": loss_cluster.mean(),
            "loss_overcluster": loss_overcluster.mean(),
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }

        self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dl_idx):
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        tag = self.trainer.datamodule.dataloader_mapping[dl_idx]

        # forward
        outputs = self.model(input_ids, input_mask, segment_ids, mode="eval")

        if "OOD" in tag:  # use clustering head
            preds = outputs["logits_unlab"]
            preds_inc = torch.cat(
                [
                    outputs["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1),
                    outputs["logits_unlab"],
                ],
                dim=-1,
            )
        elif "IND" in tag:  # use supervised classifier
            preds = outputs["logits_lab"]
            best_head = torch.argmin(self.loss_per_head)
            preds_inc = torch.cat(
                [outputs["logits_lab"], outputs["logits_unlab"][best_head]], dim=-1
            )
        else:
            preds_inc = torch.cat(
                [
                    outputs["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1),
                    outputs["logits_unlab"],
                ],
                dim=-1,
            )
        #preds = preds.max(dim=-1)[1]
        preds_inc = preds_inc.max(dim=-1)[1]

        self.metrics_inc[dl_idx].update(preds_inc, label_ids)   # 用于计算总体ACC指标


    def validation_epoch_end(self, _):
        #results = [m.compute() for m in self.metrics]
        results_inc = [m.compute() for m in self.metrics_inc]
        best_head = results_inc[1]["acc"].index(max(results_inc[1]["acc"]))
        self.best_head = best_head
        val_acc = results_inc[1]["acc"][best_head]
        # log
        val_results = {
            #"val/loss_supervised": loss_supervised,
            "val/acc": val_acc,
        }
        print(results_inc)
        self.log_dict(val_results, on_step=False, on_epoch=True)
        self.log("val_acc", val_acc)
        return val_results

    def test_step(self, batch, batch_idx, dl_idx):
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        tag = self.trainer.datamodule.dataloader_mapping[dl_idx]

        # forward
        outputs = self.model(input_ids, input_mask, segment_ids, mode="eval")

        if "OOD" in tag:  # use clustering head
            preds = outputs["logits_unlab"]
            preds_inc = torch.cat(
                [
                    outputs["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1),
                    outputs["logits_unlab"],
                ],
                dim=-1,
            )
        elif "IND" in tag:  # use supervised classifier
            preds = outputs["logits_lab"]
            best_head = torch.argmin(self.loss_per_head)
            preds_inc = torch.cat(
                [outputs["logits_lab"], outputs["logits_unlab"][best_head]], dim=-1
            )
        else:
            preds_inc = torch.cat(
                [
                    outputs["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1),
                    outputs["logits_unlab"],
                ],
                dim=-1,
            )
        # preds = preds.max(dim=-1)[1]
        preds_inc = preds_inc.max(dim=-1)[1]

        self.metrics_inc_test[dl_idx].update(preds_inc, label_ids)  # 用于计算总体ACC指标

    def test_epoch_end(self, _):
        results_inc = [m.compute() for m in self.metrics_inc_test]
        '''
        IND_f1 = results_inc[0]["f1"][0].cpu().numpy()
        IND_pre = results_inc[0]["pre"][0].cpu().numpy()
        IND_rec = results_inc[0]["rec"][0].cpu().numpy()


        OOD_f1 = results_inc[1]["f1"][self.best_head].cpu().numpy()
        OOD_pre = results_inc[1]["pre"][self.best_head].cpu().numpy()
        OOD_rec = results_inc[1]["rec"][self.best_head].cpu().numpy()

        ALL_f1 = results_inc[2]["f1"][self.best_head].cpu().numpy()
        ALL_pre = results_inc[2]["pre"][self.best_head].cpu().numpy()
        ALL_rec = results_inc[2]["rec"][self.best_head].cpu().numpy()

        self.test_results["IND_F1"] = IND_f1
        self.test_results["OOD_F1"] = OOD_f1
        self.test_results["OOD_pre"] = OOD_pre
        self.test_results["OOD_rec"] = OOD_rec
        self.test_results["ALL_F1"] = ALL_f1
        self.test_results["ALL_pre"] = ALL_pre
        self.test_results["ALL_rec"] = ALL_rec
        '''

        IND_acc = results_inc[0].cpu().numpy()
        OOD_acc = results_inc[1]["acc"][self.best_head].cpu().numpy()
        ALL_acc = results_inc[2]["acc"][self.best_head].cpu().numpy()

        self.test_results["IND_acc"] = IND_acc
        self.test_results["OOD_acc"] = OOD_acc
        self.test_results["ALL_acc"] = ALL_acc
        
        return self.test_results
    '''

    def test_step(self, batch, batch_idx):
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        #tag = self.trainer.datamodule.dataloader_mapping[dl_idx]
        self.total_features = self.total_features.cuda()
        self.total_labels = self.total_labels.cuda()
        # forward
        feats = self.model(input_ids, input_mask, segment_ids, mode="analysis")

        self.total_features = torch.cat((self.total_features, feats))
        self.total_labels = torch.cat((self.total_labels, label_ids))

    def test_epoch_end(self, _):
        feats = self.total_features.cpu().numpy()
        y_true = self.total_labels.cpu().numpy()

        self.tsne_visualization_2(feats, y_true)
    '''
    def tsne_visualization_2(self, x,y):
        #label_list=[0,1,2,3,4,5,6,7,8, 95,96,97,98, 99, 100]   V1
        #label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 135, 136, 137, 138, 139, 140]
        #label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 95, 96, 97, 105, 106, 107]
        label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 105, 96, 97, 145, 106, 107]
        path = "./outputs_v6"
        #TSNE_visualization(x, y, label_list, os.path.join(path, "pca_train_b2.png"+str(args.seed)))
        TSNE_visualization(x, y, label_list, os.path.join(path, "pca_train_all.pdf"))

    def pca_visualization_2(self,x,y):
        label_list=[0,1,2,3,4,5,6,7,8]
        path = "./outputs_v6"
        pca_visualization(x, y, label_list, os.path.join(path, "pca_train.pdf"))
        #pca_visualization(x, predicted, label_list, os.path.join(path, "pca_train_2.png"))


def save_results(args, test_results):
    if not os.path.exists(args.save_results_path):
        os.makedirs(args.save_results_path)

    var = [args.dataset, args.num_labeled_classes, args.num_unlabeled_classes, args.batch_size, args.max_epochs]
    names = ['dataset', 'num_labeled_classes', 'num_unlabeled_classes', 'batch_size', 'max_epochs']
    vars_dict = {k: v for k, v in zip(names, var)}
    results = dict(test_results, **vars_dict)
    keys = list(results.keys())
    values = list(results.values())

    file_name = 'results_GID_v3.csv'
    results_path = os.path.join(args.save_results_path, file_name)

    if not os.path.exists(results_path):
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori, columns=keys)
        df1.to_csv(results_path, index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results, index=[1])
        df1 = df1.append(new, ignore_index=True)
        df1.to_csv(results_path, index=False)
    data_diagram = pd.read_csv(results_path)

    print('test_results', data_diagram)




def main(args):
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dm = get_datamodule(args, "discover")

    root_path = "GID_checkpoints/" + args.dataset + "-" + args.comment + "_split_v" + str(3)
    print(root_path)

    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max',
                                          dirpath=root_path)

    run_name = "-".join(["discover", args.arch, args.dataset, args.comment])
    wandb_logger = pl.loggers.WandbLogger(
        save_dir=args.log_dir,
        name=run_name,
        project=args.project,
        entity=args.entity,
        offline=args.offline,
    )

    model = Discoverer(**args.__dict__)
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, dm)
    #print(trainer.checkpoint_callback.best_model_path)

    #root_path = "/home1/myt2021/Research/OOD/UNO-intent/UNO_checkpoints/OOD_"+str(args.num_unlabeled_classes)+"_seed30"
    #for filename in os.listdir(root_path):
    #    best_path = os.path.join(root_path, "epoch=61-step=1115.ckpt")
    #print(best_path)
    test_model = Discoverer.load_from_checkpoint(checkpoint_path=trainer.checkpoint_callback.best_model_path)
    trainer.test(model=test_model, datamodule=dm)
    save_results(args, test_model.test_results)

if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    if not args.multicrop:
        args.num_small_crops = 0
    args.num_crops = args.num_large_crops + args.num_small_crops

    main(args)
