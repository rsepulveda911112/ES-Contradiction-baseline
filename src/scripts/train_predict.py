import argparse
import json
import os

import yaml
import ast
from common.loadData import load_all_data
from model.beto_model import train_predict_model, predict
from sklearn.model_selection import KFold


def main(parser):
    args = parser.parse_args()

    is_type_contradiction = args.is_type_contradiction
    use_cuda = args.use_cuda
    is_cross_validation = args.is_cross_validation
    training_set = args.training_set
    test_set = args.test_set
    model_dir = args.model_dir
    label_to_exclude = args.label_to_exclude

    with open(os.getcwd() + args.setting_file, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            replace_labels = parsed_yaml['replace_labels'] if 'replace_labels' in parsed_yaml.keys() else None
            wandb_name = parsed_yaml['wandb_name'] if 'wandb_name' in parsed_yaml.keys() else None
            wandb_config = parsed_yaml['wandb_config'] if 'wandb_config' in parsed_yaml.keys() else None
        except yaml.YAMLError as exc:
            print(exc)

    if model_dir == "":
        df_train = load_all_data(training_set, is_type_contradiction, label_to_exclude, replace_labels)
        if is_cross_validation:
            n = 5
            kf = KFold(n_splits=n, random_state=3, shuffle=True)
            results = []
            for train_index, val_index in kf.split(df_train):
                train_df = df_train.iloc[train_index]
                val_df = df_train.iloc[val_index]
                acc = train_predict_model(train_df, val_df, False, True)
                results.append(acc)
            print("results", results)
            print(f"Mean-Precision: {sum(results) / len(results)}")
        else:
            df_test = load_all_data(test_set, is_type_contradiction, label_to_exclude, replace_labels)
            train_predict_model(df_train, df_test, is_cross_validation, use_cuda, wandb_name, wandb_config)
    else:
        df_test = load_all_data(test_set, is_type_contradiction, label_to_exclude)
        predict(df_test, use_cuda, model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters

    parser.add_argument("--is_type_contradiction",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if you want to train with contradiction type labels.")

    parser.add_argument("--use_cuda",
                        default=True,
                        action='store_true',
                        help="This parameter should be True if cuda is present.")

    parser.add_argument("--is_cross_validation",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if you want to make a cross-validation.")

    parser.add_argument("--training_set",
                        default="/data/ES_Contradiction_train_consolidated.json",
                        type=str,
                        help="This parameter is the relative dir of training set.")

    parser.add_argument("--test_set",
                        default="/data/ES_Contradiction_test_consolidated.json",
                        type=str,
                        help="This parameter is the relative dir of test set.")

    parser.add_argument("--model_dir",
                        default="",
                        type=str,
                        help="This parameter is the relative dir of model for predict.")

    parser.add_argument('--label_to_exclude',
                        default=[],
                        nargs='+',
                        help="This parameter should be used if you want to execute experiments with fewer classes.")

    parser.add_argument("--setting_file",
                        default="/setting_train.yaml",
                        type=str,
                        help="This parameter is the relative dir of setting file.")



    main(parser)
