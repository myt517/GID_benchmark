import torch
import torchvision
import pytorch_lightning as pl

from utils.transforms import get_transforms
from utils.transforms import DiscoverTargetTransform

from torch.utils.data import Dataset
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from pytorch_pretrained_bert.tokenization import BertTokenizer

from utils.util import *

import numpy as np
import os
import csv
from analysis import TFIDF


IND_1=[["credit_cards", "travel", "home", "meta", "utility", "small_talk", "auto_and_commute", "work"],["credit_cards", "travel", "home", "meta", "utility", "small_talk", "kitchen_and_dining", "banking"],["credit_cards", "meta", "utility", "small_talk", "kitchen_and_dining", "banking", "work", "auto_and_commute"]]
OOD_1=[["kitchen_and_dining", "banking"],["work", "auto_and_commute"],["travel","home"]]


IND_2=[["credit_cards", "meta", "utility", "small_talk", "work", "auto_and_commute"],["credit_cards", "meta", "utility", "small_talk", "kitchen_and_dining", "banking"],["travel", "home", "meta", "utility", "small_talk", "kitchen_and_dining"]]
OOD_2=[["travel", "home", "kitchen_and_dining", "banking"],["travel", "home", "auto_and_commute", "work"],["credit_cards", "banking", "auto_and_commute", "work"]]


IND_3=[["credit_cards", "meta", "utility", "small_talk"]]
OOD_3=[["travel", "home", "kitchen_and_dining", "banking","auto_and_commute","work"]]




class OriginSamples(Dataset):
    def __init__(self, train_x, train_y):
        assert len(train_y) == len(train_x)
        self.train_x = train_x
        self.train_y = train_y

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    content_list = examples.train_x
    label_list = examples.train_y

    for i in range(len(content_list)):
        tokens_a = tokenizer.tokenize(content_list[i])

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        #if label_list[i] == "oos":
        #    label_id = 149
        #else:
        label_id = label_map[label_list[i]]

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def get_datamodule(args, mode):
    if mode == "pretrain":
        if args.dataset == "GID-SD":
            return PretrainBankingDataModule(args)
        elif args.dataset == "GID-MD":
            return PretrainClincDataModule(args)
        else:
            return PretrainClincDataModule(args)
    elif mode == "discover":
        if args.dataset == "GID-SD":
            return DiscoverBankingDataModule(args)
        elif args.dataset == "GID-MD":
            return DiscoverClincDataModule(args)
        else:
            return DiscoverClincDataModule(args)
    elif mode == "analysis":
        if args.dataset == "banking":
            return AnalysisBankingDataModule(args)
        else:
            return AnalysisClincDataModule(args)


class PretrainClincDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir + args.dataset + "-" + str(int(args.OOD_ratio))
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes   # IND类别数
        self.num_unlabeled_classes = args.num_unlabeled_classes   # OOD类别数
        self.bert_model = args.arch # BERT backbone用哪一个
        self.max_seq_length = 30  # 数据集最大token长度
        self.IND_class = list(self.get_labels_ind(self.data_dir))
        self.OOD_class = list(self.get_labels_ood(self.data_dir))

        assert len(self.IND_class) == self.num_labeled_classes
        assert len(self.OOD_class) == self.num_unlabeled_classes
        print("the number of IND samples: ", len(self.IND_class))
        print("the number of OOD samples: ", len(self.OOD_class))

        print(self.OOD_class)
        for k in range(len(self.OOD_class)):
            print(self.OOD_class[k])

    def get_labels_ind(self, data_dir):
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train_ind.csv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels

    def get_labels_ood(self, data_dir):
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train_ood.csv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels

    def get_datasets(self, data_dir, quotechar=None):
        with open(data_dir, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i=0
            for line in reader:
                if (i==0):
                    i+=1
                    continue
                line[0] = line[0].strip()
                lines.append(line)
            return lines


    def get_samples(self, labelled_examples):
        content_list, labels_list = [], []
        for example in labelled_examples:
            text = example[0]
            label = example[-1]
            content_list.append(text)
            labels_list.append(label)

        data = OriginSamples(content_list,labels_list)

        return data


    def get_loader(self, labelled_examples, label_list, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)
        features = convert_examples_to_features(labelled_examples, label_list, self.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        #self.label_map = label_map

        if mode == "train":
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "validation":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "test":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_IND_data_dir = os.path.join(self.data_dir, "train_ind.csv")
        train_OOD_data_dir = os.path.join(self.data_dir, "train_ood.csv")
        val_IND_data_dir = os.path.join(self.data_dir, "eval_ind.csv")
        val_OOD_data_dir = os.path.join(self.data_dir, "eval_ood.csv")
        test_IND_data_dir = os.path.join(self.data_dir, "test_ind.csv")
        test_OOD_data_dir = os.path.join(self.data_dir, "test_ood.csv")


        train_IND = self.get_datasets(train_IND_data_dir)
        train_OOD = self.get_datasets(train_OOD_data_dir)
        val_IND = self.get_datasets(val_IND_data_dir)
        val_OOD = self.get_datasets(val_OOD_data_dir)
        test_IND = self.get_datasets(test_IND_data_dir)
        test_OOD = self.get_datasets(test_OOD_data_dir)

        print("the numbers of IND/OOD train samples: ", len(train_IND), len(train_OOD))
        print("the numbers of IND/OOD validation samples: ", len(val_IND), len(val_OOD))
        print("the numbers of IND/OOD test samples: ", len(test_IND), len(test_OOD))

        self.train_IND = self.get_samples(train_IND)
        self.val_IND = self.get_samples(val_IND)
        self.test_IND = self.get_samples(test_IND)


    def train_dataloader(self):
        return self.get_loader(self.train_IND, self.IND_class, mode="train")


    def val_dataloader(self):
        return self.get_loader(self.val_IND, self.IND_class, mode="validation")


    def test_dataloader(self):
        return self.get_loader(self.test_IND, self.IND_class, mode="test")


class DiscoverClincDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir + args.dataset + "-" + str(int(args.OOD_ratio))
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes  # IND类别数
        self.num_unlabeled_classes = args.num_unlabeled_classes  # OOD类别数
        self.bert_model = args.arch  # BERT backbone用哪一个
        self.max_seq_length = 30  # 数据集最大token长度
        self.IND_class = list(self.get_labels_ind(self.data_dir))
        self.OOD_class = list(self.get_labels_ood(self.data_dir))

        self.all_label_list = []
        self.all_label_list.extend(self.IND_class)
        self.all_label_list.extend(self.OOD_class)

        assert len(self.IND_class) == self.num_labeled_classes
        assert len(self.OOD_class) == self.num_unlabeled_classes
        print("the number of IND samples: ", len(self.IND_class))
        print("the number of OOD samples: ", len(self.OOD_class))

        print(self.OOD_class)
        for k in range(len(self.OOD_class)):
            print(self.OOD_class[k])

    def get_labels_ind(self, data_dir):
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train_ind.csv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels

    def get_labels_ood(self, data_dir):
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train_ood.csv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels

    def get_datasets(self, data_dir, quotechar=None):
        with open(data_dir, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i=0
            for line in reader:
                if (i==0):
                    i+=1
                    continue
                line[0] = line[0].strip()

                if line[-1] in self.all_label_list:
                    lines.append(line)
            return lines

    def divide_datasets(self, origin_data):
        labeled_examples, unlabeled_examples = [], []
        for example in origin_data:
            if example[-1] in self.IND_class:
                labeled_examples.append(example)
            elif example[-1] in self.OOD_class:
                unlabeled_examples.append(example)
        return labeled_examples, unlabeled_examples

    def get_samples(self, labelled_examples):
        content_list, labels_list = [], []
        for example in labelled_examples:
            text = example[0]
            label = example[-1]
            #if label in self.all_label_list:
            content_list.append(text)
            labels_list.append(label)

        print(len(content_list), len(labels_list))
        data = OriginSamples(content_list,labels_list)

        return data


    def get_loader(self, labelled_examples, label_list, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)
        features = convert_examples_to_features(labelled_examples, label_list, self.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        #self.label_map = label_map

        if mode == "train":
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "validation":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "test":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_IND_data_dir = os.path.join(self.data_dir, "train_ind.csv")
        train_OOD_data_dir = os.path.join(self.data_dir, "train_ood.csv")
        val_IND_data_dir = os.path.join(self.data_dir, "eval_ind.csv")
        val_OOD_data_dir = os.path.join(self.data_dir, "eval_ood.csv")
        test_IND_data_dir = os.path.join(self.data_dir, "test_ind.csv")
        test_OOD_data_dir = os.path.join(self.data_dir, "test_ood.csv")

        train_IND = self.get_datasets(train_IND_data_dir)
        train_OOD = self.get_datasets(train_OOD_data_dir)
        val_IND = self.get_datasets(val_IND_data_dir)
        val_OOD = self.get_datasets(val_OOD_data_dir)
        test_IND = self.get_datasets(test_IND_data_dir)
        test_OOD = self.get_datasets(test_OOD_data_dir)

        train_set, val_set, test_set = [], [], []
        train_set.extend(train_IND)
        train_set.extend(train_OOD)
        val_set.extend(val_IND)
        val_set.extend(val_OOD)
        test_set.extend(test_IND)
        test_set.extend(test_OOD)

        print("the numbers of all train samples: ", len(train_set))
        print("the numbers of all validation samples: ", len(val_set))
        print("the numbers of all test samples: ", len(test_set))
        print("the numbers of IND/OOD train samples: ", len(train_IND), len(train_OOD))
        print("the numbers of IND/OOD validation samples: ", len(val_IND), len(val_OOD))
        print("the numbers of IND/OOD test samples: ", len(test_IND), len(test_OOD))

        self.train_all = self.get_samples(train_set)

        self.val_IND = self.get_samples(val_IND)
        self.val_OOD = self.get_samples(val_OOD)
        self.val_all = self.get_samples(val_set)

        self.test_IND = self.get_samples(test_IND)
        self.test_OOD = self.get_samples(test_OOD)
        self.test_all = self.get_samples(test_set)


    @property
    def dataloader_mapping(self):
        return {0: "IND", 1: "OOD", 2: "ALL"}

    def train_dataloader(self):
        return self.get_loader(self.train_all, self.all_label_list, mode="train")

    def val_dataloader(self):
        val_IND_loadedr = self.get_loader(self.val_IND, self.IND_class, mode="validation")
        val_OOD_loadedr = self.get_loader(self.val_OOD, self.OOD_class, mode="validation")
        val_loader = self.get_loader(self.val_all, self.all_label_list, mode="validation")
        return [val_IND_loadedr, val_OOD_loadedr, val_loader]

    def test_dataloader(self):
        test_IND_loadedr = self.get_loader(self.test_IND, self.IND_class, mode="validation")
        test_OOD_loadedr = self.get_loader(self.test_OOD, self.OOD_class, mode="validation")
        test_loader = self.get_loader(self.test_all, self.all_label_list, mode="test")
        return [test_IND_loadedr, test_OOD_loadedr, test_loader]



class AnalysisClincDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes  # IND类别数
        self.num_unlabeled_classes = args.num_unlabeled_classes  # OOD类别数
        self.bert_model = args.arch  # BERT backbone用哪一个
        self.max_seq_length = 30  # 数据集最大token长度
        self.all_label_list = self.get_labels(self.data_dir)  # 获取所有类别标签

        self.IND_class = list(
            np.random.choice(np.array(self.all_label_list), self.num_labeled_classes, replace=False))  # IND类别列表
        self.OOD_class = list(set(self.all_label_list).difference(set(self.IND_class)))  # OOD类别列表

        if args.mode == "cross_domain":
            self.IND_class, self.OOD_class = cross_domain_division(
                #IND_domains=["credit_cards", "travel", "home", "auto_and_commute", "work", "meta", "utility", "small_talk"],
                #OOD_domain=["kitchen_and_dining", "banking"]
                #OOD_domain=["travel"]
                IND_domains=IND_3[0],
                OOD_domain=OOD_3[0]
            )

            '''
            train_data_dir = os.path.join(self.data_dir, "train.csv")
            train_set = self.get_datasets(train_data_dir)
            train_IND, train_OOD = self.divide_datasets(train_set)
            self.train_IND = self.get_samples(train_IND)
            contents = self.train_IND.train_x
            domain_prototypes_1 = TFIDF(contents)

            self.IND_class = self.OOD_class
            train_IND, train_OOD = self.divide_datasets(train_set)
            self.train_IND = self.get_samples(train_IND)
            contents = self.train_IND.train_x
            domain_prototypes_2 = TFIDF(contents)

            a_norm = np.linalg.norm(domain_prototypes_1)
            b_norm = np.linalg.norm(domain_prototypes_2)
            cos = np.dot(domain_prototypes_1, domain_prototypes_2) / (a_norm * b_norm)

            print(cos)
            exit()
            '''
            #self.unknown_label_list = list(set(self.all_label_list).difference(set(self.known_label_list)))
        elif args.mode == "noise_ood":
            self.oos_train = get_oos(ratio=0.8)
            print("selected oos num:", len(self.oos_train))
            #exit()
        elif args.mode == "imbalanced":
            self.OOD_list_ranking = get_imbalanced(self.OOD_class)
            print()

        self.all_label_list = []
        self.all_label_list.extend(self.IND_class)
        self.all_label_list.extend(self.OOD_class)
        '''
        self.all_label_list = ['gas', 'new_card', 'next_holiday', 'transactions', 'meal_suggestion', 'pto_request', 'what_is_your_name',
         'change_ai_name', 'schedule_maintenance', 'calendar', 'who_do_you_work_for', 'no', 'text', 'direct_deposit',
         'credit_limit', 'timer', 'calendar_update', 'insurance_change', 'spelling', 'insurance', 'greeting', 'yes',
         'shopping_list', 'change_accent', 'mpg', 'order_checks', 'report_fraud', 'definition', 'rewards_balance',
         'travel_notification', 'reminder_update', 'pto_balance', 'accept_reservations', 'fun_fact', 'calculator',
         'rollover_401k', 'pin_change', 'account_blocked', 'how_busy', 'transfer', 'ingredient_substitution',
         'travel_suggestion', 'distance', 'make_call', 'ingredients_list', 'alarm', 'are_you_a_bot', 'goodbye',
         'smart_home', 'food_last', 'travel_alert', 'who_made_you', 'current_location', 'reset_settings', 'jump_start',
         'lost_luggage', 'maybe', 'find_phone', 'exchange_rate', 'order', 'replacement_card_duration', 'recipe',
         'measurement_conversion', 'play_music', 'income', 'repeat', 'change_speed', 'meaning_of_life', 'plug_type',
         'book_flight', 'do_you_have_pets', 'payday', 'improve_credit_score', 'pto_used', 'carry_on', 'gas_type',
         'spending_history', 'directions', 'credit_score', 'international_fees', 'vaccines', 'what_can_i_ask_you',
         'time', 'restaurant_suggestion', 'traffic', 'restaurant_reviews', 'order_status', 'sync_device', 'uber',
         'schedule_meeting', 'flip_coin', 'tire_pressure', 'thank_you', 'expiration_date', 'what_song',
         'restaurant_reservation', 'car_rental', 'interest_rate', 'cancel', 'tire_change', 'where_are_you_from',
         'bill_due', 'min_payment', 'confirm_reservation', 'damaged_card', 'apr', 'weather', 'todo_list_update',
         'freeze_account', 'card_declined', 'reminder', 'timezone', 'cook_time', 'next_song', 'what_are_your_hobbies',
         'change_language', 'share_location', 'international_visa', 'pto_request_status', 'w2', 'whisper_mode',
         'roll_dice', 'change_volume', 'report_lost_card', 'cancel_reservation', 'nutrition_info', 'tell_joke',
         'application_status', 'redeem_rewards', 'change_user_name', 'pay_bill', 'last_maintenance', 'book_hotel',
         'credit_limit_change', 'taxes', 'shopping_list_update', 'date', 'calories', 'user_name', 'routing',
         'how_old_are_you', 'meeting_schedule', 'oil_change_when', 'bill_balance', 'todo_list', 'balance', 'translate',
         'update_playlist', 'oil_change_how', 'flight_status']

        '''

        assert len(self.IND_class) == self.num_labeled_classes
        assert len(self.OOD_class) == self.num_unlabeled_classes
        print("the number of IND samples: ", len(self.IND_class))
        print("the number of OOD samples: ", len(self.OOD_class))

        print(self.OOD_class)
        for k in range(len(self.OOD_class)):
            print(self.OOD_class[k])

        '''
        这三个，文本用不到

        self.dataset_class = getattr(torchvision.datasets, args.dataset)
        self.transform_train = get_transforms(
            "unsupervised",
            args.dataset,
            multicrop=args.multicrop,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
        )
        self.transform_val = get_transforms("eval", args.dataset)
        '''
    '''
    def staticstics(self):

        IND_domains = {
            "banking": [],
            "credit_cards": [],
            "kitchen_and_dining": [],
            "home": [],
            "work": [],
            "utility": [],
            "travel": [],
            "auto_and_commute": [],
            "small_talk": [],
            "meta": [],
        }

        OOD_domains = {
            "banking": [],
            "credit_cards": [],
            "kitchen_and_dining": [],
            "home": [],
            "work": [],
            "utility": [],
            "travel": [],
            "auto_and_commute": [],
            "small_talk": [],
            "meta": [],
        }

        for label in self.IND_class:
            for k in CLINC_domian_intent.keys():
                if label in CLINC_domian_intent[k]:
                    IND_domains[k].append(label)

        for label in self.OOD_class:
            for k in CLINC_domian_intent.keys():
                if label in CLINC_domian_intent[k]:
                    OOD_domains[k].append(label)

        IND_NUMS, OOD_NUMS = {}, {}
        for k in IND_domains.keys():
            IND_NUMS[k] = len(IND_domains[k])
        for k in OOD_domains.keys():
            OOD_NUMS[k] = len(OOD_domains[k])



        print(IND_NUMS)
        print("------------------------------------------")
        print(OOD_NUMS)
    '''


    def staticstics(self):
        label_index = []
        sample_nums = []
        label_samples_frame = {} # IND
        label_samples_frame_OOD = {}  # OOD


        test = pd.read_csv(os.path.join(self.data_dir, "train.csv"), sep="\t")
        labels = test['label']
        texts = test['text']

        for k in range(len(labels)):
            if labels[k] not in label_samples_frame.keys() and labels[k] in self.IND_class:
                label_samples_frame[labels[k]] = [texts[k]]
            elif labels[k] in label_samples_frame.keys():
                label_samples_frame[labels[k]].append(texts[k])
            else:
                continue

        for k in range(len(labels)):
            if labels[k] not in label_samples_frame_OOD.keys() and labels[k] in self.OOD_class:
                label_samples_frame_OOD[labels[k]] = [texts[k]]
            elif labels[k] in label_samples_frame_OOD.keys():
                label_samples_frame_OOD[labels[k]].append(texts[k])
            else:
                continue

        #print(len(label_samples_frame_OOD.keys()))
        #for label in label_samples_frame_OOD.keys():
        #    print(label_samples_frame_OOD[label][0:10])

        if os.path.isdir("./dataset/datasets/OIR-MD-60") == False:
            os.mkdir("./dataset/datasets/OIR-MD-60")
        self.write_csv(label_samples_frame, "./dataset/datasets/OIR-MD-60/train_ind.csv")
        self.write_csv(label_samples_frame_OOD, "./dataset/datasets/OIR-MD-60/train_ood.csv")

        '''
        for k in range(len(labels)):
            if labels[k] not in label_samples_frame.keys():
                label_samples_frame[labels[k]]  = 1
            else:
                label_samples_frame[labels[k]] += 1

        i=0
        for label in label_samples_frame.keys():
            label_index.append(i)
            sample_nums.append(label_samples_frame[label])
            i+=1
        '''
        return label_index, sample_nums

    def write_csv(self, train_OOD_selected, file_name):
        f = open(file_name, 'w', encoding='utf-8')
        csv_writer = csv.writer(f, delimiter='\t')
        csv_writer.writerow(["text", "label"])

        print(len(train_OOD_selected.keys()))
        for label in train_OOD_selected.keys():
            for line in train_OOD_selected[label]:
                csv_writer.writerow([line, label])
            #label_list = [label]*10
            #csv_writer.writerow([train_OOD_selected[label][0:10],label_list])
            #print(train_OOD_selected[label][0:10], [label]*10)

        #for i in range(len(train_OOD_selected)):
            #csv_writer.writerow([train_OOD_selected[i][0], train_OOD_selected[i][-1]])

        f.close()



    def get_labels(self, data_dir):
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.csv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels

    def get_datasets(self, data_dir, quotechar=None):
        with open(data_dir, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i=0
            for line in reader:
                if (i==0):
                    i+=1
                    continue
                line[0] = line[0].strip()

                if line[-1] in self.all_label_list:
                    lines.append(line)
            return lines

    def divide_datasets(self, origin_data):
        labeled_examples, unlabeled_examples = [], []
        for example in origin_data:
            if example[-1] in self.IND_class:
                labeled_examples.append(example)
            elif example[-1] in self.OOD_class:
                unlabeled_examples.append(example)
        return labeled_examples, unlabeled_examples

    def get_samples(self, labelled_examples):
        content_list, labels_list = [], []
        for example in labelled_examples:
            text = example[0]
            label = example[-1]
            content_list.append(text)
            labels_list.append(label)

        data = OriginSamples(content_list,labels_list)

        return data


    def get_loader(self, labelled_examples, label_list, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)
        features = convert_examples_to_features(labelled_examples, label_list, self.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        #self.label_map = label_map

        if mode == "train":
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "validation":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "test":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader


    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train.csv")
        val_data_dir = os.path.join(self.data_dir, "eval.csv")
        test_data_dir = os.path.join(self.data_dir, "test.csv")

        train_set = self.get_datasets(train_data_dir)
        val_set = self.get_datasets(val_data_dir)
        test_set = self.get_datasets(test_data_dir)

        train_IND, train_OOD = self.divide_datasets(train_set)
        val_IND, val_OOD = self.divide_datasets(val_set)
        test_IND, test_OOD = self.divide_datasets(test_set)

        '''
        train_OOD_selected = self.get_datasets(os.path.join(self.data_dir, "imbalanced_ood.csv"))
        #train_OOD_selected = imbalanced_division(train_OOD, self.OOD_list_ranking)
        train_OOD = train_OOD_selected
        train_set = []
        train_set.extend(train_IND)
        train_set.extend(train_OOD)
        '''
        #train_OOD.extend(self.oos_train)
        #train_set.extend(self.oos_train)

        print("the numbers of all train samples: ", len(train_set))
        print("the numbers of all validation samples: ", len(val_set))
        print("the numbers of all test samples: ", len(test_set))
        print("the numbers of IND/OOD train samples: ", len(train_IND), len(train_OOD))
        print("the numbers of IND/OOD validation samples: ", len(val_IND), len(val_OOD))
        print("the numbers of IND/OOD test samples: ", len(test_IND), len(test_OOD))

        self.train_IND = self.get_samples(train_IND)
        self.train_OOD = self.get_samples(train_OOD)
        self.train_all = self.get_samples(train_set)

        self.val_IND = self.get_samples(val_IND)
        self.val_OOD = self.get_samples(val_OOD)
        self.val_all = self.get_samples(val_set)

        self.test_IND = self.get_samples(test_IND)
        self.test_OOD = self.get_samples(test_OOD)
        self.test_all = self.get_samples(test_set)


    @property
    def dataloader_mapping(self):
        return {0: "IND", 1: "OOD", 2: "ALL"}

    def train_dataloader(self):
        test_IND_loadedr = self.get_loader(self.test_IND, self.IND_class, mode="validation")
        test_OOD_loadedr = self.get_loader(self.test_OOD, self.OOD_class, mode="validation")
        test_loader = self.get_loader(self.test_all, self.all_label_list, mode="test")
        return [test_IND_loadedr, test_OOD_loadedr, test_loader]

    def val_dataloader(self):
        val_IND_loadedr = self.get_loader(self.val_IND, self.IND_class, mode="validation")
        val_OOD_loadedr = self.get_loader(self.val_OOD, self.OOD_class, mode="validation")
        val_loader = self.get_loader(self.val_all, self.all_label_list, mode="validation")
        return [val_IND_loadedr, val_OOD_loadedr, val_loader]

    def test_dataloader(self):
        return self.get_loader(self.train_all, self.all_label_list, mode="train")


class PretrainBankingDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir + args.dataset + "-" + str(int(args.OOD_ratio))
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes   # IND类别数
        self.num_unlabeled_classes = args.num_unlabeled_classes   # OOD类别数
        self.bert_model = args.arch # BERT backbone用哪一个
        self.max_seq_length = 55  # 数据集最大token长度
        self.IND_class = list(self.get_labels_ind(self.data_dir))
        self.OOD_class = list(self.get_labels_ood(self.data_dir))

        assert len(self.IND_class) == self.num_labeled_classes
        assert len(self.OOD_class) == self.num_unlabeled_classes
        print("the number of IND samples: ", len(self.IND_class))
        print("the number of OOD samples: ", len(self.OOD_class))

        print(self.OOD_class)
        for k in range(len(self.OOD_class)):
            print(self.OOD_class[k])

    def get_labels_ind(self, data_dir):
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train_ind.csv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels

    def get_labels_ood(self, data_dir):
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train_ood.csv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels

    def get_datasets(self, data_dir, quotechar=None):
        with open(data_dir, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i=0
            for line in reader:
                if (i==0):
                    i+=1
                    continue
                line[0] = line[0].strip()
                lines.append(line)
            return lines

    def get_samples(self, labelled_examples):
        content_list, labels_list = [], []
        for example in labelled_examples:
            text = example[0]
            label = example[-1]
            content_list.append(text)
            labels_list.append(label)

        data = OriginSamples(content_list,labels_list)

        return data


    def get_loader(self, labelled_examples, label_list, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)
        features = convert_examples_to_features(labelled_examples, label_list, self.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        #self.label_map = label_map

        if mode == "train":
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "validation":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "test":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_IND_data_dir = os.path.join(self.data_dir, "train_ind.csv")
        train_OOD_data_dir = os.path.join(self.data_dir, "train_ood.csv")
        val_IND_data_dir = os.path.join(self.data_dir, "eval_ind.csv")
        val_OOD_data_dir = os.path.join(self.data_dir, "eval_ood.csv")
        test_IND_data_dir = os.path.join(self.data_dir, "test_ind.csv")
        test_OOD_data_dir = os.path.join(self.data_dir, "test_ood.csv")


        train_IND = self.get_datasets(train_IND_data_dir)
        train_OOD = self.get_datasets(train_OOD_data_dir)
        val_IND = self.get_datasets(val_IND_data_dir)
        val_OOD = self.get_datasets(val_OOD_data_dir)
        test_IND = self.get_datasets(test_IND_data_dir)
        test_OOD = self.get_datasets(test_OOD_data_dir)

        print("the numbers of IND/OOD train samples: ", len(train_IND), len(train_OOD))
        print("the numbers of IND/OOD validation samples: ", len(val_IND), len(val_OOD))
        print("the numbers of IND/OOD test samples: ", len(test_IND), len(test_OOD))

        self.train_IND = self.get_samples(train_IND)
        self.val_IND = self.get_samples(val_IND)
        self.test_IND = self.get_samples(test_IND)


    def train_dataloader(self):
        return self.get_loader(self.train_IND, self.IND_class, mode="train")


    def val_dataloader(self):
        return self.get_loader(self.val_IND, self.IND_class, mode="validation")


    def test_dataloader(self):
        return self.get_loader(self.test_IND, self.IND_class, mode="test")


class DiscoverBankingDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir + args.dataset + "-" + str(int(args.OOD_ratio))
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes  # IND类别数
        self.num_unlabeled_classes = args.num_unlabeled_classes  # OOD类别数
        self.bert_model = args.arch  # BERT backbone用哪一个
        self.max_seq_length = 55  # 数据集最大token长度
        self.IND_class = list(self.get_labels_ind(self.data_dir))
        self.OOD_class = list(self.get_labels_ood(self.data_dir))

        self.all_label_list = []
        self.all_label_list.extend(self.IND_class)
        self.all_label_list.extend(self.OOD_class)

        assert len(self.IND_class) == self.num_labeled_classes
        assert len(self.OOD_class) == self.num_unlabeled_classes
        print("the number of IND samples: ", len(self.IND_class))
        print("the number of OOD samples: ", len(self.OOD_class))

        print(self.OOD_class)
        for k in range(len(self.OOD_class)):
            print(self.OOD_class[k])

    def get_labels_ind(self, data_dir):
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train_ind.csv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels

    def get_labels_ood(self, data_dir):
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train_ood.csv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels

    def get_datasets(self, data_dir, quotechar=None):
        with open(data_dir, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i=0
            for line in reader:
                if (i==0):
                    i+=1
                    continue
                line[0] = line[0].strip()
                lines.append(line)
            return lines


    def get_samples(self, labelled_examples):
        content_list, labels_list = [], []
        for example in labelled_examples:
            text = example[0]
            label = example[-1]
            content_list.append(text)
            labels_list.append(label)

        data = OriginSamples(content_list,labels_list)

        return data


    def get_loader(self, labelled_examples, label_list, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)
        features = convert_examples_to_features(labelled_examples, label_list, self.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        #self.label_map = label_map

        if mode == "train":
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "validation":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "test":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader


    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_IND_data_dir = os.path.join(self.data_dir, "train_ind.csv")
        train_OOD_data_dir = os.path.join(self.data_dir, "train_ood.csv")
        val_IND_data_dir = os.path.join(self.data_dir, "eval_ind.csv")
        val_OOD_data_dir = os.path.join(self.data_dir, "eval_ood.csv")
        test_IND_data_dir = os.path.join(self.data_dir, "test_ind.csv")
        test_OOD_data_dir = os.path.join(self.data_dir, "test_ood.csv")

        train_IND = self.get_datasets(train_IND_data_dir)
        train_OOD = self.get_datasets(train_OOD_data_dir)
        val_IND = self.get_datasets(val_IND_data_dir)
        val_OOD = self.get_datasets(val_OOD_data_dir)
        test_IND = self.get_datasets(test_IND_data_dir)
        test_OOD = self.get_datasets(test_OOD_data_dir)

        train_set, val_set, test_set = [], [], []
        train_set.extend(train_IND)
        train_set.extend(train_OOD)
        val_set.extend(val_IND)
        val_set.extend(val_OOD)
        test_set.extend(test_IND)
        test_set.extend(test_OOD)

        print("the numbers of all train samples: ", len(train_set))
        print("the numbers of all validation samples: ", len(val_set))
        print("the numbers of all test samples: ", len(test_set))
        print("the numbers of IND/OOD train samples: ", len(train_IND), len(train_OOD))
        print("the numbers of IND/OOD validation samples: ", len(val_IND), len(val_OOD))
        print("the numbers of IND/OOD test samples: ", len(test_IND), len(test_OOD))

        self.train_all = self.get_samples(train_set)

        self.val_IND = self.get_samples(val_IND)
        self.val_OOD = self.get_samples(val_OOD)
        self.val_all = self.get_samples(val_set)

        self.test_IND = self.get_samples(test_IND)
        self.test_OOD = self.get_samples(test_OOD)
        self.test_all = self.get_samples(test_set)


    @property
    def dataloader_mapping(self):
        return {0: "IND", 1: "OOD", 2: "ALL"}

    def train_dataloader(self):
        return self.get_loader(self.train_all, self.all_label_list, mode="train")

    def val_dataloader(self):
        val_IND_loadedr = self.get_loader(self.val_IND, self.IND_class, mode="validation")
        val_OOD_loadedr = self.get_loader(self.val_OOD, self.OOD_class, mode="validation")
        val_loader = self.get_loader(self.val_all, self.all_label_list, mode="validation")
        return [val_IND_loadedr, val_OOD_loadedr, val_loader]

    def test_dataloader(self):
        test_IND_loadedr = self.get_loader(self.test_IND, self.IND_class, mode="validation")
        test_OOD_loadedr = self.get_loader(self.test_OOD, self.OOD_class, mode="validation")
        test_loader = self.get_loader(self.test_all, self.all_label_list, mode="test")
        return [test_IND_loadedr, test_OOD_loadedr, test_loader]


class AnalysisBankingDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes   # IND类别数
        self.num_unlabeled_classes = args.num_unlabeled_classes   # OOD类别数
        self.bert_model = args.arch # BERT backbone用哪一个
        self.max_seq_length = 55  # 数据集最大token长度
        self.all_label_list = self.get_labels(self.data_dir)  # 获取所有类别标签

        self.IND_class = list(np.random.choice(np.array(self.all_label_list), self.num_labeled_classes, replace=False))  # IND类别列表
        self.OOD_class = list(set(self.all_label_list).difference(set(self.IND_class)))  # OOD类别列表


        assert len(self.IND_class) == self.num_labeled_classes
        assert len(self.OOD_class) == self.num_unlabeled_classes
        print("the number of IND samples: ", len(self.IND_class))
        print("the number of OOD samples: ", len(self.OOD_class))

        print(self.OOD_class)
        for k in range(len(self.OOD_class)):
            print(self.OOD_class[k])

    def staticstics(self):
        label_index = []
        sample_nums = []
        label_samples_frame = {} # IND
        label_samples_frame_OOD = {}  # OOD


        test = pd.read_csv(os.path.join(self.data_dir, "train.csv"), sep="\t")
        labels = test['label']
        texts = test['text']

        for k in range(len(labels)):
            if labels[k] not in label_samples_frame.keys() and labels[k] in self.IND_class:
                label_samples_frame[labels[k]] = [texts[k]]
            elif labels[k] in label_samples_frame.keys():
                label_samples_frame[labels[k]].append(texts[k])
            else:
                continue

        for k in range(len(labels)):
            if labels[k] not in label_samples_frame_OOD.keys() and labels[k] in self.OOD_class:
                label_samples_frame_OOD[labels[k]] = [texts[k]]
            elif labels[k] in label_samples_frame_OOD.keys():
                label_samples_frame_OOD[labels[k]].append(texts[k])
            else:
                continue

        #print(len(label_samples_frame_OOD.keys()))
        #for label in label_samples_frame_OOD.keys():
        #    print(label_samples_frame_OOD[label][0:10])

        if os.path.isdir("./dataset/datasets/OIR-SD-60") == False:
            os.mkdir("./dataset/datasets/OIR-SD-60")
        self.write_csv(label_samples_frame, "./dataset/datasets/OIR-SD-60/train_ind.csv")
        self.write_csv(label_samples_frame_OOD, "./dataset/datasets/OIR-SD-60/train_ood.csv")

        '''
        for k in range(len(labels)):
            if labels[k] not in label_samples_frame.keys():
                label_samples_frame[labels[k]]  = 1
            else:
                label_samples_frame[labels[k]] += 1

        i=0
        for label in label_samples_frame.keys():
            label_index.append(i)
            sample_nums.append(label_samples_frame[label])
            i+=1
        '''
        return label_index, sample_nums

    def write_csv(self, train_OOD_selected, file_name):
        f = open(file_name, 'w', encoding='utf-8')
        csv_writer = csv.writer(f, delimiter='\t')
        csv_writer.writerow(["text", "label"])

        print(len(train_OOD_selected.keys()))
        for label in train_OOD_selected.keys():
            for line in train_OOD_selected[label]:
                csv_writer.writerow([line, label])
            #label_list = [label]*10
            #csv_writer.writerow([train_OOD_selected[label][0:10],label_list])
            #print(train_OOD_selected[label][0:10], [label]*10)

        #for i in range(len(train_OOD_selected)):
            #csv_writer.writerow([train_OOD_selected[i][0], train_OOD_selected[i][-1]])

        f.close()


    def get_labels(self, data_dir):
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.csv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels

    def get_datasets(self, data_dir, quotechar=None):
        with open(data_dir, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i=0
            for line in reader:
                if (i==0):
                    i+=1
                    continue
                line[0] = line[0].strip()
                lines.append(line)
            return lines

    def divide_datasets(self, origin_data):
        labeled_examples, unlabeled_examples = [], []
        for example in origin_data:
            if example[-1] in self.IND_class:
                labeled_examples.append(example)
            elif example[-1] in self.OOD_class:
                unlabeled_examples.append(example)
        return labeled_examples, unlabeled_examples

    def get_samples(self, labelled_examples):
        content_list, labels_list = [], []
        for example in labelled_examples:
            text = example[0]
            label = example[-1]
            content_list.append(text)
            labels_list.append(label)

        data = OriginSamples(content_list,labels_list)

        return data


    def get_loader(self, labelled_examples, label_list, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)
        features = convert_examples_to_features(labelled_examples, label_list, self.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        #self.label_map = label_map

        if mode == "train":
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "validation":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        elif mode == "test":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train.csv")
        val_data_dir = os.path.join(self.data_dir, "eval.csv")
        test_data_dir = os.path.join(self.data_dir, "test.csv")

        train_set = self.get_datasets(train_data_dir)
        val_set = self.get_datasets(val_data_dir)
        test_set = self.get_datasets(test_data_dir)

        train_IND, train_OOD = self.divide_datasets(train_set)
        val_IND, val_OOD = self.divide_datasets(val_set)
        test_IND, test_OOD = self.divide_datasets(test_set)
        print("the numbers of all train samples: ", len(train_set))
        print("the numbers of all validation samples: ", len(val_set))
        print("the numbers of all test samples: ", len(test_set))
        print("the numbers of IND/OOD train samples: ", len(train_IND), len(train_OOD))
        print("the numbers of IND/OOD validation samples: ", len(val_IND), len(val_OOD))
        print("the numbers of IND/OOD test samples: ", len(test_IND), len(test_OOD))

        self.train_IND = self.get_samples(train_IND)
        self.val_IND = self.get_samples(val_IND)
        self.test_IND = self.get_samples(test_IND)


    def train_dataloader(self):
        return self.get_loader(self.train_IND, self.IND_class, mode="train")


    def val_dataloader(self):
        return self.get_loader(self.val_IND, self.IND_class, mode="validation")


    def test_dataloader(self):
        return self.get_loader(self.test_IND, self.IND_class, mode="test")



