import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.metrics import Accuracy

from utils.data import get_datamodule
from utils.nets import MultiHeadBERT
from utils.callbacks import PretrainCheckpointCallback

from argparse import ArgumentParser
from datetime import datetime
import random
import os
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from utils.util import *
from analysis import *

parser = ArgumentParser()
parser.add_argument("--dataset", default="banking", type=str, help="dataset")
parser.add_argument("--data_dir", default="dataset/banking", type=str, help="data directory")
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--checkpoint_dir", default="checkpoints_v1", type=str, help="checkpoint dir")
parser.add_argument("--batch_size", default=256, type=int, help="batch size")
parser.add_argument("--num_workers", default=5, type=int, help="number of workers")
parser.add_argument("--arch", default="bert-base-uncased", type=str, help="backbone architecture")
parser.add_argument("--base_lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--weight_decay_opt", default=1.0e-4, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
parser.add_argument("--project", default="GID_benchmark", type=str, help="wandb project")
parser.add_argument("--entity", default="myt517", type=str, help="wandb entity")
parser.add_argument("--offline", default=False, action="store_true", help="disable wandb")
parser.add_argument("--num_labeled_classes", default=80, type=int, help="number of labeled classes")
parser.add_argument("--num_unlabeled_classes", default=20, type=int, help="number of unlab classes")
parser.add_argument("--pretrained", type=str, default=None, help="pretrained checkpoint path")
parser.add_argument("--save_results_path", type=str, default='outputs_v1', help="The path to save results.")
parser.add_argument('--mode', type=str, default="none", help="Random seed for initialization.")
parser.add_argument('--IND_class', type=str, default="none", help="Random seed for initialization.")
parser.add_argument('--OOD_class', type=str, default="none", help="Random seed for initialization.")
parser.add_argument("--IND_ratio", default=1.0, type=float, help="softmax temperature")
parser.add_argument("--label_ratio", default=1.0, type=float, help="softmax temperature")
parser.add_argument("--download", default=False, action="store_true", help="wether to download")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(10)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Pretrainer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters({k: v for (k, v) in kwargs.items() if not callable(v)})

        # build model
        self.model = MultiHeadBERT.from_pretrained(
            self.hparams.arch,
            self.hparams.num_labeled_classes,
            self.hparams.num_unlabeled_classes,
            num_heads=None,
        )
        self.freeze_parameters(self.model)

        if self.hparams.pretrained is not None:
            state_dict = torch.load(self.hparams.pretrained)
            self.model.load_state_dict(state_dict, strict=False)

        # metrics
        self.accuracy = Accuracy()

        self.test_results = 0
        self.t_step = 0

        self.OOD_features = torch.empty((0, 768)).to(self.device)
        self.OOD_labels = torch.empty(0, dtype=torch.long).to(self.device)


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

    def training_step(self, batch, batch_idx):
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        # normalize prototypes
        self.model.normalize_prototypes()

        # forward
        outputs = self.model(input_ids, input_mask, segment_ids, mode="pretrain")

        #print(outputs["logits_lab"].shape)
        #print(label_ids,label_ids.shape)

        # supervised loss
        loss_supervised = F.cross_entropy(outputs["logits_lab"] / self.hparams.temperature, label_ids)

        # log
        results = {
            "loss_supervised": loss_supervised,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True)

        # reweight loss
        return loss_supervised

    def validation_step(self, batch, batch_idx):
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        # forward
        logits = self.model(input_ids, input_mask, segment_ids, mode="eval")["logits_lab"]
        _, preds = logits.max(dim=-1)

        # calculate loss and accuracy
        loss_supervised = F.cross_entropy(logits, label_ids)
        acc = self.accuracy(preds, label_ids)

        # log
        results = {
            "val/loss_supervised": loss_supervised,
            "val/acc": acc,
        }
        self.log_dict(results, on_step=False, on_epoch=True)
        return results

    def test_step(self, batch, batch_idx):
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        batch_num = label_ids.shape[0]

        # forward
        logits = self.model(input_ids, input_mask, segment_ids, mode="eval")["logits_lab"]
        _, preds = logits.max(dim=-1)

        # calculate loss and accuracy
        loss_supervised = F.cross_entropy(logits, label_ids)
        acc = self.accuracy(preds, label_ids)

        # log
        results = {
            "test/loss_supervised": loss_supervised,
            "test/acc": acc,
        }
        print(results)
        self.t_step += batch_num
        self.test_results += results['test/acc'] * batch_num

        #self.log_dict(results, on_step=False, on_epoch=True)
        return results

    '''

    def test_step(self, batch, batch_idx):
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        batch_num = label_ids.shape[0]

        # forward
        feats = self.model(input_ids, input_mask, segment_ids, mode="analysis")
        #print(feats.device)
        self.OOD_features = self.OOD_features.cuda()
        self.OOD_labels = self.OOD_labels.cuda()
        print(self.OOD_features.device)
        self.OOD_features = torch.cat((self.OOD_features, feats))
        self.OOD_labels = torch.cat((self.OOD_labels, label_ids))
    '''

def main(args):
    set_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # build datamodule
    dm = get_datamodule(args, "pretrain")

    # logger
    run_name = "-".join(["pretrain", args.arch, args.dataset, args.comment])
    wandb_logger = pl.loggers.WandbLogger(
        save_dir=args.log_dir,
        name=run_name,
        project=args.project,
        entity=args.entity,
        offline=args.offline,
    )

    model = Pretrainer(**args.__dict__)
    trainer = pl.Trainer.from_argparse_args(
        args, logger=wandb_logger, callbacks=[PretrainCheckpointCallback()]
    )
    trainer.fit(model, dm)

def test(args):
    set_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # build datamodule
    dm = get_datamodule(args, "pretrain")

    test_model = Pretrainer(**args.__dict__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_file = args.checkpoint_dir+"/pretrain-bert-base-uncased-"+args.dataset+"-" + args.comment + ".cp"
    state_dict = torch.load(pretrain_file, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if ("unlab" not in k)}
    test_model.model.load_state_dict(state_dict, strict=False)

    #test_model = Pretrainer.load_from_checkpoint(checkpoint_path="checkpoints/pretrain-bert-base-uncased-clinc-120_30.cp")

    trainer = pl.Trainer(gpus=1, precision=16)
    trainer.test(model=test_model, datamodule=dm)

    t_acc = test_model.test_results/test_model.t_step
    print("testing results:", t_acc)
    test_results = {}
    test_results["pretrain_acc"] = t_acc.cpu().numpy()

    save_results(args, test_results)


def analysis(args):
    set_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # build datamodule
    dm = get_datamodule(args, "analysis")
    label_index, sample_nums = dm.staticstics()
    #dm.staticstics()
    #draw(label_index, sample_nums)
    exit()


    test_model = Pretrainer(**args.__dict__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_file = "checkpoints_cross_datasets/pretrain-bert-base-uncased-cross-" + args.comment + ".cp"
    state_dict = torch.load(pretrain_file, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if ("unlab" not in k)}
    test_model.model.load_state_dict(state_dict, strict=False)

    # test_model = Pretrainer.load_from_checkpoint(checkpoint_path="checkpoints/pretrain-bert-base-uncased-clinc-120_30.cp")

    trainer = pl.Trainer(gpus=1, precision=16)
    trainer.test(model=test_model, datamodule=dm)

    transferability_metrics={}

    # sc值
    x_feats = test_model.OOD_features.cpu().numpy()
    y_label = test_model.OOD_labels.cpu().numpy()
    km = KMeans(n_clusters=15).fit(x_feats)
    y_pred = km.labels_
    score = metrics.silhouette_score(x_feats, y_pred)
    print(score)
    transferability_metrics["SC"] = score

    # 类内距
    min_d, max_d, mean_d = intra_distance(x_feats, y_label, 15)
    print(mean_d)
    transferability_metrics["intra_distance"] = mean_d

    # 类间距
    min_d, max_d, mean_d = inter_distance(x_feats, y_label, 15)
    print(mean_d)
    transferability_metrics["inter_distance"] = mean_d

    save_results(args, transferability_metrics)


def save_results(args, test_results):
    if not os.path.exists(args.save_results_path):
        os.makedirs(args.save_results_path)

    var = [args.dataset, args.num_labeled_classes, args.num_unlabeled_classes, args.batch_size, args.max_epochs]
    names = ['dataset', 'num_labeled_classes', 'num_unlabeled_classes', 'batch_size', 'max_epochs']
    vars_dict = {k: v for k, v in zip(names, var)}
    results = dict(test_results, **vars_dict)
    keys = list(results.keys())
    values = list(results.values())

    file_name = 'results_check_v1.csv'
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



if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    #main(args)
    #test(args)
    analysis(args)