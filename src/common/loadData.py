import os
from common.reader import JSONLineReader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()


# {0: 0, 1: 0, 2: 1}

def load_all_data(file_in, is_type_contradiction, labels_to_exclude, replace_label):
    jsonl_reader = JSONLineReader()
    datas = jsonl_reader.read(os.getcwd() + file_in)
    df_in = pd.DataFrame(datas)
    # df_in = df_in.iloc[0:10]
    label = 'label'
    if is_type_contradiction:
        label = 'label_contradiction'
    for value in labels_to_exclude:
        df_in = df_in[df_in[label] != value]
    df_in = df_in.reset_index(drop=True)
    df_in[label] = labelencoder.fit_transform(df_in[label])
    if replace_label:
        df_in['label'] = df_in['label'].replace(replace_label)

    l = df_in[label].to_list()
    h = df_in['headline'].to_list()
    p = df_in['body'].to_list()

    list_of_tuples = list(zip(h, p, l))
    return pd.DataFrame(list_of_tuples, columns=['text_a', 'text_b', 'labels'])