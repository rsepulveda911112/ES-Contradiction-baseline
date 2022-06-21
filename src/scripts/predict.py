import argparse
from common.loadData import load_all_data
from model.beto_model import train_predict_model, predict
from common.score import scorePredict
from sklearn.model_selection import KFold
import random
import pandas as pd


def main(parser):
    args = parser.parse_args()

    is_type_contradiction = args.is_type_contradiction
    use_cuda = args.use_cuda
    is_cross_validation = args.is_cross_validation
    training_set = args.training_set
    test_set = args.test_set
    model_dir_1 = args.model_dir_1
    model_dir_2 = args.model_dir_2
    label_to_exclude = args.label_to_exclude


    label_map = {'unrelated': 2, 'compatible': 0, 'contradictory': 1}
    df_test = load_all_data(test_set, is_type_contradiction, label_to_exclude)


    if model_dir_1 != "":
        y_predict_1 = predict(df_test, use_cuda, model_dir_1)
        df_result = df_test
        df_result['predict'] = y_predict_1
        if model_dir_2 != '':
            df_y_1 = pd.DataFrame(y_predict_1, columns=['predict'])
            df_y_1_0 = df_y_1[df_y_1['predict'] == 0]
            df_y_1_1 = df_y_1[df_y_1['predict'] == 1]

            p_test_1 = df_test.loc[df_y_1_0.index]
            p_test_1['predict'] = df_y_1_0['predict'].values
            p_test_1['predict'] = p_test_1['predict'].replace(0, 3)

            df_test_2 = df_test.loc[df_y_1_1.index]
            y_predict_2 = predict(df_test_2, use_cuda, model_dir_2)
            df_test_2['predict'] = y_predict_2
            df_result = pd.concat([p_test_1, df_test_2], axis=0)

    labels = list(df_test['label'].unique())
    labels.sort()
    print(scorePredict(df_result['predict'].values, df_result['label'].values, labels))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters

    parser.add_argument("--is_type_contradiction",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if you want to train with contradiction type labels.")

    parser.add_argument("--use_cuda",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if cuda is present.")

    parser.add_argument("--is_cross_validation",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if you want to make a cross-validation.")

    parser.add_argument("--training_set",
                        default="/data/ES_Contradiction_train_consolidado.json",
                        type=str,
                        help="This parameter is the relative dir of training set.")

    parser.add_argument("--test_set",
                        default="/data/ES_Contradiction_test_consolidado.json",
                        type=str,
                        help="This parameter is the relative dir of test set.")

    parser.add_argument("--model_dir_1",
                        default="/result/related/outputs/",
                        type=str,
                        help="This parameter is the relative dir of model for predict.")

    parser.add_argument("--model_dir_2",
                        default="/result/stance/outputs/",
                        type=str,
                        help="This parameter is the relative dir of model for predict.")

    parser.add_argument('--label_to_exclude',
                        default=[],
                        nargs='+',
                        help="This parameter should be used if you want to execute experiments with fewer classes.")
    main(parser)
