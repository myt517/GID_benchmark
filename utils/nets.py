import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel


class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        super().__init__()
        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        return self.prototypes(x)


class MultiHead(nn.Module):
    def __init__(
        self, input_dim, num_prototypes, num_heads
    ):
        super().__init__()
        self.num_heads = num_heads

        # prototypes
        self.prototypes = torch.nn.ModuleList(
            [Prototypes(input_dim, num_prototypes) for _ in range(num_heads)]
        )

        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        for p in self.prototypes:
            p.normalize_prototypes()

    def forward_head(self, head_idx, feats):
        z = F.normalize(feats, dim=1)
        return self.prototypes[head_idx](z), z

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]




class MultiHeadBERT(BertPreTrainedModel):
    def __init__(
        self,
        config,
        num_labeled,
        num_unlabeled,
        overcluster_factor=3,
        num_heads=4,
        num_hidden_layers=1,
    ):
        super(MultiHeadBERT, self).__init__(config)

        # backbone
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.head_lab = Prototypes(config.hidden_size, num_labeled)


        if num_heads is not None:
            self.head_unlab = MultiHead(
                input_dim=config.hidden_size,
                num_prototypes=num_unlabeled,
                num_heads=num_heads
            )
            self.head_unlab_over = MultiHead(
                input_dim=config.hidden_size,
                num_prototypes=num_unlabeled * overcluster_factor,
                num_heads=num_heads
            )

        self.apply(self.init_bert_weights)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_lab.normalize_prototypes()
        if getattr(self, "head_unlab", False):
            self.head_unlab.normalize_prototypes()
            self.head_unlab_over.normalize_prototypes()

    def forward_heads(self, feats):
        out = {"logits_lab": self.head_lab(F.normalize(feats))}
        if hasattr(self, "head_unlab"):
            logits_unlab, proj_feats_unlab = self.head_unlab(feats)
            logits_unlab_over, proj_feats_unlab_over = self.head_unlab_over(feats)
            out.update(
                {
                    "logits_unlab": logits_unlab,
                    "proj_feats_unlab": proj_feats_unlab,
                    "logits_unlab_over": logits_unlab_over,
                    "proj_feats_unlab_over": proj_feats_unlab_over,
                }
            )
        return out

    def forward(self, input_ids, input_mask, segment_ids, mode="pretrain"):
        if mode == "pretrain":
            encoded_layer_12_emb01, pooled_output_01 = self.bert(input_ids, segment_ids, input_mask,
                                                                 output_all_encoded_layers=False)
            pooled_output_01 = self.dense(encoded_layer_12_emb01.mean(dim=1))
            pooled_output_01 = self.activation(pooled_output_01)
            pooled_output_01 = self.dropout(pooled_output_01)
            out_01 = self.forward_heads(pooled_output_01)
            out_dict_01 = {"feats": pooled_output_01}
            for key in out_01.keys():
                out_dict_01[key] = out_01[key]
            #print(out_dict_01.keys())
            return out_dict_01
        elif mode == "discovery":
            encoded_layer_12_emb01, pooled_output_01 = self.bert(input_ids, segment_ids, input_mask,
                                                                 output_all_encoded_layers=False)
            encoded_layer_12_emb02, pooled_output_02 = self.bert(input_ids, segment_ids, input_mask,
                                                                 output_all_encoded_layers=False)

            pooled_output_01 = self.dense(encoded_layer_12_emb01.mean(dim=1))
            pooled_output_02 = self.dense(encoded_layer_12_emb02.mean(dim=1))

            pooled_output_01 = self.activation(pooled_output_01)
            pooled_output_02 = self.activation(pooled_output_02)

            pooled_output_01 = self.dropout(pooled_output_01)
            pooled_output_02 = self.dropout(pooled_output_02)

            feats = [pooled_output_01, pooled_output_02]
            out = [self.forward_heads(f) for f in feats]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict

        elif mode == "eval":
            encoded_layer_12_emb01, pooled_output_01 = self.bert(input_ids, segment_ids, input_mask,
                                                                 output_all_encoded_layers=False)
            pooled_output_01 = self.dense(encoded_layer_12_emb01.mean(dim=1))
            pooled_output_01 = self.activation(pooled_output_01)
            pooled_output_01 = self.dropout(pooled_output_01)
            out_01 = self.forward_heads(pooled_output_01)
            out_dict_01 = {"feats": pooled_output_01}
            for key in out_01.keys():
                out_dict_01[key] = out_01[key]
            #print(out_dict_01.keys())
            return out_dict_01

        elif mode == "analysis":
            encoded_layer_12_emb01, pooled_output_01 = self.bert(input_ids, segment_ids, input_mask,
                                                                 output_all_encoded_layers=False)
            pooled_output_01 = self.dense(encoded_layer_12_emb01.mean(dim=1))
            pooled_output_01 = self.activation(pooled_output_01)
            pooled_output_01 = self.dropout(pooled_output_01)

            return pooled_output_01




