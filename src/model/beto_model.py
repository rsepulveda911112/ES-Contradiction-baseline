from src.common.score import scorePredict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from simpletransformers.classification.classification_model import ClassificationModel

TransformerModel = ClassificationModel


def train_predict_model(df_train, df_test, is_predict, use_cuda):
    labels_test = pd.Series(df_test['labels']).to_numpy()
    labels = list(df_train['labels'].unique())
    labels.sort()
    model = TransformerModel('bert', 'dccuchile/bert-base-spanish-wwm-cased',
                             num_labels=len(labels), use_cuda=use_cuda, args={
                            'learning_rate':2e-5,
                            'num_train_epochs': 3,
                            'reprocess_input_data': True,
                            'overwrite_output_dir': True,
                            'process_count': 10,
                            'train_batch_size': 4,
                            'eval_batch_size': 4,
                            'max_seq_length': 512,
                            'fp16': True,
                            'fp16_opt_level': 'O1'})
    model.train_model(df_train)

    results = ''
    value_predict = ''
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
    value_predict = scorePredict(y_predict, labels_test, labels)

    return value_predict, results