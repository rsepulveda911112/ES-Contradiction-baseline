from model.beto_model import train_predict_model
from common.loadData import load_all_data
from sklearn.model_selection import KFold


if __name__ == '__main__':
    n = 5
    kf = KFold(n_splits=n, random_state=3, shuffle=True)
    label_to_exclude = ['unrelated']
    df_train = load_all_data('/data/ES_Contradiction_train_v1.json', False, label_to_exclude)
    results = []
    result_predic = ''
    for train_index, val_index in kf.split(df_train):
        train_df = df_train.iloc[train_index]
        val_df = df_train.iloc[val_index]
        value_predict, acc = train_predict_model(train_df, val_df, False, True)
        result_predic += value_predict
        results.append(acc)
    print(result_predic)
    print("results", results)
    print(f"Mean-Precision: {sum(results) / len(results)}")