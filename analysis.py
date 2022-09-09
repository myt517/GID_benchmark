import matplotlib.pyplot as plt
import numpy as np
import csv, os
import time

def draw(label_index, sample_nums):
    plt.figure(figsize=(25, 6))
    plt.tick_params(labelsize=12)
    plt.bar(label_index, height=sample_nums, width=0.5, color="blue")

    plt.xlabel("class index", fontsize=16)
    plt.ylabel("the number of samples", fontsize=16)
    plt.ylim([0, 200])
    plt.xlim([-1, 77])
    plt.xticks(label_index, label_index)
    plt.legend(loc='lower right', fontsize=15)

    plt.savefig("./outputs/banking_sta.png", bbox_inches='tight', pad_inches=0.05)
    plt.savefig("./outputs/banking_sta.pdf", bbox_inches='tight', pad_inches=0.025)
    # plt.show()
    plt.close()


def longtail(file_name, quotechar=None):
    label_list = list(get_labels(file_name))
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        i = 0
        for line in reader:
            if (i == 0):
                i += 1
                continue
            line[0] = line[0].strip()
            lines.append(line)
    frame={}
    for k in label_list:
        frame[k] = 0

    for line in lines:
        for k in label_list:
            if line[-1] == k:
                frame[k] +=1

    print(frame)

    return frame

def longtail_graph(frame1, frame2, frame3):
    num_list1, num_list2, num_list3 = [], [], []
    for k in frame1.keys():
        num_list1.append(frame1[k])

    for k in frame2.keys():
        num_list2.append(frame2[k])

    for k in frame3.keys():
        num_list3.append(frame3[k])

    num_list1.sort()
    num_list2.sort()
    num_list3.sort()

    plt.figure(figsize=(8, 6))
    #plt.bar(x=range(len(num_list)), height=num_list)

    plt.plot(range(len(num_list1)), num_list1, linewidth=2, c="#41719c", marker=".", ms=9, label="imbalance ratio = 2")
    plt.plot(range(len(num_list2)), num_list2, linewidth=2, c="#ed7d31", marker=".", ms=9, label="imbalance ratio = 3")
    plt.plot(range(len(num_list3)), num_list3, linewidth=2, c="#70ad47", marker=".", ms=9, label="imbalance ratio = 6")

    plt.yticks([20, 40, 60, 80, 100, 120], fontsize=16)
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], fontsize=16)
    plt.ylim([10, 130])
    plt.xlim([0, 65])

    plt.ylabel("the number of samples per class", fontsize=16)
    plt.xlabel("class index", fontsize=16)
    plt.legend(loc='best', fontsize=18)

    plt.savefig("./outputs/test.png", bbox_inches='tight', pad_inches=0.05)
    plt.savefig("./outputs/test.pdf", bbox_inches='tight', pad_inches=0.025)


def get_labels(data_dir):
     import pandas as pd
     test = pd.read_csv(data_dir, sep="\t")
     labels = np.unique(np.array(test['label']))
     return labels

def Line_chart():
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.tick_params(labelsize=12)

    epoch1 = [1, 0.8, 0.6, 0.4, 0.2]
    model1 = [98.6,	98.37,	98.14,	97.84,	96.11]
    model2 = [98.07, 98.06, 97.81, 96.9, 94.59]
    model3 = [93.13, 95.34, 95.57, 89.53, 65.03]



    # sns.set(style="darkgrid")
    plt.plot(epoch1, model1, linewidth=2, c="#41719c", marker=".", ms=9, label="End-to-End")
    plt.plot(epoch1, model3, linewidth=2, c="#ed7d31", marker=".", ms=9, label="DeepAligned-Mix")
    plt.plot(epoch1, model2, linewidth=2, c="#70ad47", marker=".", ms=9, label="DeepAligned")

    plt.yticks([20, 40, 60, 80, 100], fontsize=16)
    plt.xticks([1, 0.8, 0.6, 0.4, 0.2], fontsize=16)
    plt.ylim([60, 100])
    plt.xlim([1.1, 0.1])

    plt.ylabel("IND F1", fontsize=16)
    plt.xlabel("Ratio of IND sample numbers", fontsize=16)
    plt.legend(loc='best', fontsize=19)

    plt.subplot(1, 2, 2)

    epoch1 = [0.1, 0.3, 0.5, 0.7, 0.9]
    model1 = [80.89, 91.56, 98.67, 96.89, 95.56]
    model2 = [69.27, 84.13,	97.09, 93.24, 90.70]
    model3 = [81.98, 90.88,	98.18, 95.76, 94.93]

    base_model1 = [95.11, 95.11, 95.11, 95.11, 95.11]
    base_model2 = [89.81, 89.81, 89.81, 89.81, 89.81]
    base_model3 = [94.13, 94.13, 94.13, 94.13, 94.13]

    plt.plot(epoch1, model1, linewidth=2, c="#41719c", marker=".", ms=9, label="End-to-End")
    plt.plot(epoch1, model3, linewidth=2, c="#ed7d31", marker=".", ms=9, label="DeepAligned-Mix")
    plt.plot(epoch1, model2, linewidth=2, c="#70ad47", marker=".", ms=9, label="DeepAligned")

    plt.yticks([20, 40, 60, 80, 100], fontsize=16)
    plt.xticks([1, 0.8, 0.6, 0.4, 0.2], fontsize=16)
    plt.ylim([50, 100])
    plt.xlim([1.1, 0.1])

    plt.ylabel("OOD F1", fontsize=16)
    plt.xlabel("Ratio of IND sample numbers", fontsize=16)
    plt.legend(loc='best', fontsize=19)

    plt.savefig("./outputs/test.png", bbox_inches='tight', pad_inches=0.05)
    plt.savefig("./outputs/test.pdf", bbox_inches='tight', pad_inches=0.025)

def bar():
    #xname = ["5%","10%", "15%"]
    xname = ["2", "3", "6"]
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    #UNO = [92.59, 90.17, 84.86]
    plt.tick_params(labelsize=12)

    #IND noise
    '''
    pipeline_deep = [94.44,	91.93,	87.78]
    endtoend_deep = [96.74,	94.44,	96]
    UNO = [96.74,	95.78,	93.56]
    '''
    # OOD imbalance

    pipeline_deep = [86.08,	87.51,	81.52]
    endtoend_deep = [81.87,	75.82,	71.77]
    UNO = [82.07,	76.37,	67.27]

    '''
    pipeline_deep = [90.13, 86.49, 81.95]
    endtoend_deep = [86.67, 85.57, 85.11]
    kmeans = [74.34, 69.4, 68.94]
    UNO = [92.59, 90.17, 84.86]
    '''
    #plt.bar(np.arange(0, 3, 1) - 0.3, height=kmeans, width=0.2, color="mediumturquoise", label="k-means")
    plt.bar(np.arange(0, 3, 1) - 0.2, height=pipeline_deep, width=0.2, color="mediumslateblue", label="DeepAligned")
    plt.bar(np.arange(0, 3, 1), height=endtoend_deep, width=0.2, color="lightcoral", label="DeepAligned-Mix")
    plt.bar(np.arange(0, 3, 1) + 0.2, height=UNO, width=0.2, color="mediumseagreen", label="End-to-End")
    plt.xlabel("imbalance ratio", fontsize=16)
    plt.ylabel("OOD F1", fontsize=16)
    plt.ylim([30, 100])
    plt.xticks(np.arange(0, 3, 1), xname)
    plt.legend(loc='lower right', fontsize=15)

    plt.subplot(1, 2, 2)

    plt.tick_params(labelsize=12)
    '''
    pipeline_deep = [88.33,	89.89,	88.67]
    endtoend_deep = [86.11,	84.44,	84.44]
    UNO = [90.44,	90.11,	87.33]

    '''

    pipeline_deep = [92.76,	93.15,	90.55]
    endtoend_deep = [84.53,	83.87,	75.5]
    UNO = [91.48,	89.09,	84.76]

    '''
    UNO = [95.68, 94.42, 92.57]
    pipeline_deep = [94.67, 92.83, 90.86]
    endtoend_deep = [88.31, 87.61, 85.76]
    kmeans = [87.82, 85.63, 85.3]
    '''
    #plt.bar(np.arange(0, 3, 1) - 0.3, height=kmeans, width=0.2, color="mediumturquoise", label="k-means")
    plt.bar(np.arange(0, 3, 1) - 0.2, height=pipeline_deep, width=0.2, color="mediumslateblue", label="DeepAligned")
    plt.bar(np.arange(0, 3, 1), height=endtoend_deep, width=0.2, color="lightcoral", label="DeepAligned-Mix")
    plt.bar(np.arange(0, 3, 1) + 0.2, height=UNO, width=0.2, color="mediumseagreen", label="End-to-End")
    plt.xlabel("imbalance ratio", fontsize=16)
    plt.ylabel("ALL F1", fontsize=16)
    plt.ylim([70, 100])
    plt.xticks(np.arange(0, 3, 1), xname)
    # plt.axis('off')
    plt.legend(loc='lower right', fontsize=15)

    '''
    plt.subplot(1, 3, 3)

    plt.tick_params(labelsize=12)

    pipeline_deep = [92, 91.11,	88.13]
    endtoend_deep = [89.16,	87.51,	89.24]
    UNO = [94.22,	93.47,	91.07]

    

    pipeline_deep = [92.76,	93.15,	90.55]
    endtoend_deep = [84.53,	83.87,	75.5]
    UNO = [91.48,	89.09,	84.76]

    UNO = [95.68, 94.42, 92.57]
    pipeline_deep = [94.67, 92.83, 90.86]
    endtoend_deep = [88.31, 87.61, 85.76]
    kmeans = [87.82, 85.63, 85.3]

    #plt.bar(np.arange(0, 3, 1) - 0.3, height=kmeans, width=0.2, color="mediumturquoise", label="k-means")
    plt.bar(np.arange(0, 3, 1) - 0.2, height=pipeline_deep, width=0.2, color="mediumslateblue", label="DeepAligned")
    plt.bar(np.arange(0, 3, 1), height=endtoend_deep, width=0.2, color="lightcoral", label="DeepAligned-Mix")
    plt.bar(np.arange(0, 3, 1) + 0.2, height=UNO, width=0.2, color="mediumseagreen", label="End-to-End")
    plt.xlabel("the ratio of IND noise", fontsize=16)
    plt.ylabel("ALL ACC", fontsize=16)
    plt.ylim([70, 100])
    plt.xticks(np.arange(0, 3, 1), xname)
    # plt.axis('off')
    plt.legend(loc='lower right', fontsize=15)

    '''


    plt.savefig("./outputs/test.png", bbox_inches='tight', pad_inches=0.05)
    plt.savefig("./outputs/test.pdf", bbox_inches='tight', pad_inches=0.025)
    #plt.show()
    plt.close()

def TFIDF(corpus):
    # coding:utf-8
    from sklearn.feature_extraction.text import CountVectorizer

    # 语料
    '''
    corpus = [
        'This is the first document.',
        'This is the second second document.',
        'And the third one.',
        'Is this the first document?',
    ]
    '''
    # 将文本中的词语转换为词频矩阵
    vectorizer = CountVectorizer()
    # 计算个词语出现的次数
    X = vectorizer.fit_transform(corpus)
    # 获取词袋中所有文本关键词
    word = vectorizer.get_feature_names()

    # ----------------------------------------------------

    from sklearn.feature_extraction.text import TfidfTransformer

    # 类调用
    transformer = TfidfTransformer()
    # 将词频矩阵X统计成TF-IDF值
    tfidf = transformer.fit_transform(X)
    # 查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
    tfidf_array=tfidf.toarray()
    tfidf_array_protptype = np.mean(tfidf, axis=0)
    print(tfidf_array.shape)
    print(tfidf_array_protptype, tfidf_array_protptype.shape)

    return tfidf_array_protptype

