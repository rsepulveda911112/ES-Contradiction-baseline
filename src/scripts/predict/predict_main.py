from common.loadData import load_all_data
from model.beto_model import predict

if __name__ == '__main__':
    label_to_exclude = ['unrelated']
    is_type_contradiction = False
    use_cuda = True
    dir_model = '/output/'
    df_test = load_all_data('/data/ES_Contradiction_test_v1.json', is_type_contradiction, label_to_exclude)
    predict(df_test, use_cuda, dir_model)