import os
from common.score import scorePredict
import pandas as pd
import numpy as np
import wandb
from sklearn.metrics import accuracy_score
from simpletransformers.classification.classification_model import ClassificationModel


def train_predict_model(df_train, df_test, is_predict, use_cuda, wandb_project=None, wandb_config=None):
    labels_test = pd.Series(df_test['labels']).to_numpy()
    labels = list(df_train['labels'].unique())
    labels.sort()

    if wandb_config:
        wandb.init(project=wandb_project, config=wandb_config)
    elif wandb_project:
        wandb.init(project=wandb_project, config=wandb.config)

    model = ClassificationModel('bert', 'dccuchile/bert-base-spanish-wwm-cased',
                                 num_labels=len(labels), use_cuda=use_cuda, args={
                                'learning_rate':2e-5,
                                'num_train_epochs': 3,
                                'reprocess_input_data': True,
                                'overwrite_output_dir': True,
                                'process_count': 10,
                                'train_batch_size': 4,
                                'eval_batch_size': 4,
                                'max_seq_length': 512,
                                'multiprocessing_chunksize': 10,
                                'fp16': True,
                                'fp16_opt_level': 'O1',
                                'tensorboard_dir': 'tensorboard',
                                'wandb_project': wandb_project})


    model.train_model(df_train)

    results = ''
    if is_predict:
        text_a = df_test['text_a']
        text_b = df_test['text_b']
        df_result = pd.concat([text_a, text_b], axis=1)
        value_in = df_result.values.tolist()
        _, model_outputs_test = model.predict(value_in)
    else:
        result, model_outputs_test, wrong_predictions = model.eval_model(df_test, acc=accuracy_score)
        results = result['acc']
    y_predict = np.argmax(model_outputs_test, axis=1)
    print(scorePredict(y_predict, labels_test, labels))
    return results


def predict(df_test, use_cuda, model_dir):
    model = ClassificationModel(model_type='bert', model_name=os.getcwd() + model_dir, use_cuda=use_cuda)
    labels_test = pd.Series(df_test['labels']).to_numpy()
    labels = list(df_test['labels'].unique())
    labels.sort()
    text_a = df_test['text_a']
    text_b = df_test['text_b']
    df_result = pd.concat([text_a, text_b], axis=1)
    value_in = df_result.values.tolist()
    _, model_outputs_test = model.predict(value_in)
    y_predict = np.argmax(model_outputs_test, axis=1)
    print(scorePredict(y_predict, labels_test, labels))