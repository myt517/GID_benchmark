import torch
import torchvision
import pytorch_lightning as pl

from utils.transforms import get_transforms
from utils.transforms import DiscoverTargetTransform

from torch.utils.data import Dataset
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from pytorch_pretrained_bert.tokenization import BertTokenizer

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import List
import os
import csv,json
import pandas as pd
import random
matplotlib.use('Agg')

random.seed(0)

CLINC_domian_intent = {
    "banking": ["freeze_account",
"routing",
"pin_change",
"bill_due",
"pay_bill",
"account_blocked",
"interest_rate",
"min_payment",
"bill_balance",
"transfer",
"order_checks",
"balance",
"spending_history",
"transactions",
"report_fraud"],
    "credit_cards": ["replacement_card_duration",
"expiration_date",
"damaged_card",
"improve_credit_score",
"report_lost_card",
"card_declined",
"credit_limit_change",
"apr",
"redeem_rewards",
"credit_limit",
"rewards_balance",
"application_status",
"credit_score",
"new_card",
"international_fees"],
    "kitchen_and_dining": ["food_last",
"confirm_reservation",
"how_busy",
"ingredients_list",
"calories",
"nutrition_info",
"recipe",
"restaurant_reviews",
"restaurant_reservation",
"meal_suggestion",
"restaurant_suggestion",
"cancel_reservation",
"ingredient_substitution",
"cook_time",
"accept_reservations"],
    "home": ["what_song",
"play_music",
"todo_list_update",
"reminder",
"reminder_update",
"calendar_update",
"order_status",
"update_playlist",
"shopping_list",
"calendar",
"next_song",
"order",
"todo_list",
"shopping_list_update",
"smart_home"],
    "work": ["pto_request_status",
"next_holiday",
"insurance_change",
"insurance",
"meeting_schedule",
"payday",
"taxes",
"income",
"rollover_401k",
"pto_balance",
"pto_request",
"w2",
"schedule_meeting",
"direct_deposit",
"pto_used"],
    "utility": ["weather",
"alarm",
"date",
"find_phone",
"share_location",
"timer",
"make_call",
"calculator",
"definition",
"measurement_conversion",
"flip_coin",
"spelling",
"time",
"roll_dice",
"text"],
    "travel": ["plug_type",
"travel_notification",
"translate",
"flight_status",
"international_visa",
"timezone",
"exchange_rate",
"travel_suggestion",
"travel_alert",
"vaccines",
"lost_luggage",
"book_flight",
"book_hotel",
"carry_on",
"car_rental"],
    "auto_and_commute": ["current_location",
"oil_change_when",
"oil_change_how",
"uber",
"traffic",
"tire_pressure",
"schedule_maintenance",
"gas",
"mpg",
"distance",
"directions",
"last_maintenance",
"gas_type",
"tire_change",
"jump_start"],
    "small_talk": ["who_made_you",
"meaning_of_life",
"who_do_you_work_for",
"do_you_have_pets",
"what_are_your_hobbies",
"fun_fact",
"what_is_your_name",
"where_are_you_from",
"goodbye",
"thank_you",
"greeting",
"tell_joke",
"are_you_a_bot",
"how_old_are_you",
"what_can_i_ask_you"],
    "meta": ["change_speed",
"user_name",
"whisper_mode",
"yes",
"change_volume",
"no",
"change_language",
"repeat",
"change_accent",
"cancel",
"sync_device",
"change_user_name",
"change_ai_name",
"reset_settings",
"maybe"],



}

def get_imbalanced(OOD_list):
    OOD_list_ranking = []
    for i in range(4):
        tmp_list = random.sample(OOD_list, 15)
        OOD_list = list(set(OOD_list).difference(set(tmp_list)))
        OOD_list_ranking.append(tmp_list)

    print(OOD_list_ranking)

    #for i in range(12):
    #    for k in range(len(OOD_list_ranking[i])):
    #        print(OOD_list_ranking[i][k])

    return OOD_list_ranking

def imbalanced_division_v2(train_OOD, OOD_list, imbalanced_ratio=6):
    train_OOD_selected = []

    min_num = 120/imbalanced_ratio

    for i in range(len(OOD_list)):
        label = OOD_list[i]
        samples = [example for example in train_OOD if example[-1] == label]
        num_samples = int(min_num * imbalanced_ratio ** (i/60))
        print(num_samples)
        samples_selected = random.sample(samples, num_samples)
        train_OOD_selected.extend(samples_selected)

    file_name = './dataset/clinc/imbalanced_ood'+'_'+str(imbalanced_ratio)+'.tsv'
    write_csv(train_OOD_selected, file_name)
    exit()

    return train_OOD_selected

def imbalanced_division(train_OOD, OOD_list_ranking):
    train_OOD_selected = []

    for i in range(len(OOD_list_ranking)):
        for k in range(len(OOD_list_ranking[i])):
            label = OOD_list_ranking[i][k]
            samples = [example for example in train_OOD if example[-1] == label]
            num_samples = random.randint((i + 1) * 10 - 9, (i + 1) * 10)
            print(num_samples)
            samples_selected = random.sample(samples, num_samples)
            train_OOD_selected.extend(samples_selected)

    file_name = './dataset/clinc/imbalanced_ood.tsv'
    write_csv(train_OOD_selected, file_name)


    return train_OOD_selected

def write_csv(train_OOD_selected, file_name):
    f = open(file_name, 'w', encoding='utf-8')
    csv_writer = csv.writer(f, delimiter='\t')
    csv_writer.writerow(["text", "label"])

    for i in range(len(train_OOD_selected)):
        csv_writer.writerow([train_OOD_selected[i][0], train_OOD_selected[i][-1]])

    f.close()


def get_noise(dataset, ratio=0.1):
    oos_file_path="./dataset/clinc/eval.tsv"
    num_samples = int(7200*ratio)
    print(len(dataset))

    oos_list_selected = random.sample(dataset, num_samples)

    #print(oos_list_selected)


    return oos_list_selected




def get_oos(ratio=0.1):
    oos_file_path="./dataset/clinc/data_oos_plus.json"
    oos_list = []
    with open(oos_file_path, 'r') as f:
        data_frame = json.load(f)
    print(len(data_frame["oos_train"]))
    print(len(data_frame["oos_test"]))
    print(len(data_frame["oos_val"]))
    print(data_frame["oos_train"][0][0])
    print(data_frame["oos_train"][0][1])

    for i in range(len(data_frame["oos_train"])):
        oos_list.append(data_frame["oos_train"][i])

    for i in range(len(data_frame["oos_test"])):
        oos_list.append(data_frame["oos_test"][i])

    for i in range(len(data_frame["oos_val"])):
        oos_list.append(data_frame["oos_val"][i])
    print(len(oos_list))
    num_samples = 360

    oos_list_selected = random.sample(oos_list, num_samples)

    #print(oos_list_selected)


    return oos_list_selected



def cross_domain_division(IND_domains, OOD_domain = None):

    IND_class, OOD_class = [], []
    for d in IND_domains:
        IND_class.extend(CLINC_domian_intent[d])

    if OOD_domain != None:
        for d in OOD_domain:
            OOD_class.extend(CLINC_domian_intent[d])

        return IND_class, OOD_class

    return  IND_class



def intra_distance(X, predicted_y, num_labels):
    cluster_center = []
    for i in range(num_labels):
        X_feats = X[predicted_y == i]
        center_x = np.mean(X_feats, axis=0)
        cluster_center.append(center_x)
    #print(len(cluster_center))

    intra_cluster_distance = []
    for i in range(num_labels):
        X_feats = X[predicted_y == i]
        #dist = np.dot(X_feats, cluster_center[i].T)
        dist = np.sqrt(np.sum(np.square(X_feats - cluster_center[i]), axis=1))
        dist = dist.tolist()
        intra_cluster_distance.append(dist)

    min_list, max_list, mean_list = [], [], []
    for i in range(len(intra_cluster_distance)):
        min_list.append(min(intra_cluster_distance[i]))
        max_list.append(max(intra_cluster_distance[i]))
        mean_list.append(sum(intra_cluster_distance[i])/len(intra_cluster_distance[i]))


    min_d = min(min_list)
    max_d = max(max_list)
    mean_d = sum(mean_list)/len(mean_list)

    return min_d, max_d, mean_d


def inter_distance(X, predicted_y, num_labels):
    cluster_center = []
    for i in range(num_labels):
        X_feats = X[predicted_y == i]
        #print(X_feats)
        center_x = np.mean(X_feats, axis=0)
        cluster_center.append(center_x)
    print(len(cluster_center), cluster_center[0].shape)

    #print(cluster_center)
    #print("______________________________________")

    #knn.fit(cluster_center, labels)
    #y_pred = knn.predict(cluster_center)
    #print(y_pred)

    inter_cluster_distance = []
    for i in range(len(cluster_center)):
        dist_list = []
        for j in range(len(cluster_center)):
            #dist = np.dot(cluster_center[i], cluster_center[j].T)
            dist = np.sqrt(np.sum(np.square(cluster_center[i] - cluster_center[j])))
            dist_list.append(dist)
        inter_cluster_distance.append(dist_list)

    inter_distance = []
    for i in range(len(inter_cluster_distance)):
        inter_cluster_distance[i].sort()
        #print(inter_cluster_distance[i])
        tmp = np.mean(inter_cluster_distance[i][-4:-1])
        inter_distance.append(tmp)

    inter_distance = np.array(inter_distance)
    print(inter_distance.shape)

    # Min
    min_d = np.float(inter_distance.min())
    # Max
    max_d = np.float(inter_distance.max())
    # Mean
    mean_d = np.float(inter_distance.mean())


    return min_d, max_d, mean_d


def TSNE_visualization(X: np.ndarray,
                       y: pd.Series,
                       classes: List[str],
                       save_path: str):
    X_embedded = TSNE(n_components=2).fit_transform(X)

    color_list = ["blueviolet", "green", "blue", "yellow", "purple", "black", "brown", "cyan", "gray", "pink", "orange",
                  "red", "greenyellow", "sandybrown", "deeppink", 'olive', 'm', 'navy']

    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots()
    i = 0
    for _class in classes:
        if _class == "unseen":
            ax.scatter(X_embedded[y == _class, 0], X_embedded[y == _class, 1],
                       label=_class, alpha=0.5, s=20, edgecolors='none', color="gray")
        else:
            ax.scatter(X_embedded[y == _class, 0], X_embedded[y == _class, 1],
                       label=i, alpha=0.5, s=6, edgecolors='none', zorder=15, color=color_list[i])
        i+=1
    ax.grid(True)
    #plt.legend(loc='best')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    fig.subplots_adjust(right=0.8)
    #plt.savefig(save_path, bbox_inches='tight', pad_inches=0, format="pdf")

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, format="pdf")
    #plt.savefig(save_path, format="png")

    print()



def pca_visualization(X: np.ndarray,
                      y: pd.Series,
                      classes: List[str],
                      save_path: str):
    """
    Apply PCA visualization for features.
    """
    print("-----------------------")
    print(X.shape)
    red_features = PCA(n_components=2, svd_solver="full").fit_transform(X)
    print(red_features.shape)
    print(red_features[y == 1, 0])

    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots()
    for _class in classes:
        if _class == "unseen":
            ax.scatter(red_features[y == _class, 0], red_features[y == _class, 1],
                    label=_class, alpha=0.5, s=20, edgecolors='none', color="gray")
        else:
            ax.scatter(red_features[y == _class, 0], red_features[y == _class, 1],
                    label=_class, alpha=0.5, s=20, edgecolors='none', zorder=15)
    ax.legend(loc=2)
    ax.grid(True)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, format="pdf")
    #plt.savefig(save_path, format="png")
