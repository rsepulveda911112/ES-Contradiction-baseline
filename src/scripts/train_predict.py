from src.common.loadData import load_all_data
from src.model.beto_model import train_predict_model

if __name__ == '__main__':
    label_to_exclude = ['unrelated']
    is_type_contradiction = False
    use_cuda = True
    is_predict = True

    df_train = load_all_data('/data/ES_Contradiction_train_v1.json', is_type_contradiction, label_to_exclude)
    df_test = load_all_data('/data/ES_Contradiction_test_v1.json', is_type_contradiction, label_to_exclude)

    value_predict = train_predict_model(df_train, df_test, is_predict, use_cuda)
    print(value_predict)