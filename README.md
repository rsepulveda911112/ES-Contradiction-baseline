# ES_Contradiction_baseline

**(2021/03/09) create readme **

### Requirements
* Python 3.6
* Pytorch
* Transformers

### Installation
* Create a Python Environment and activate it:
```bash 
    virtualenv beto --python=python3
    cd ./beto
    source bin/activate
```
* Install the required dependencies. 
You need to have at least version 21.0.1 of pip installed. Next you may install requirements.txt.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Download ES-Contradiction-dataset
Create a folder to store the dataset.

```bash
mkdir data
```
* Download ES-Contradiction-dataset from zenodo and copy in this folder:
https://zenodo.org/record/4586766#.YEelgFNKi-I


### Description of the parameters
These parameters allow configuring the system to train or predict.

|Field|Description|
|---|---|
|label_to_exclude|This parameter should be used if you want to execute experiments with fewer classes.|
|is_type_contradiction|This parameter should be used if you want to train with contradiction type labels.|
|use_cuda|This parameter can be used if cuda is present.|
|is_cross_validation|This parameter should be used if you want to make a cross-validation.|
|training_set|This parameter is the relative dir of training set.|
|test_set|This parameter is the relative dir of test set.|
|model_dir|This parameter is the relative dir of model for predict.|

For example, if you want to predict only Compatible and Contradiction use this configuration:
```bash
--label_to_exclude ['unrelated']
```
For example, if you want to predict only type of contradiction:
```bash
--is_type_contradiction
```

### Train and predict ES-Contradiction-dataset
Execute this command to train and predict on the dataset
```bash
PYTHONPATH=src python src/scripts/train_predict.py --training_set "/data/ES_Contradiction_train_v1.json" --test_set "/data/ES_Contradiction_test_v1.json"
```

### Predict ES-Contradiction-dataset
Execute this command to predict all the dataset
```bash
PYTHONPATH=src python src/scripts/train_predict.py --test_set "/data/ES_Contradiction_test_v1.json" --model_dir "/outputs/checkpoint-x-epoch-y"
```

### Predict ES-Contradiction-dataset (type of contradiction)
Execute this command to predict the type of contradiction on the dataset
```bash
PYTHONPATH=src python src/scripts/train_predict.py --test_set "/data/ES_Contradiction_test_v1.json" --model_dir "/outputs/checkpoint-x-epoch-y" --is_type_contradiction --label_to_exclude "none"
```

### Train and predict with cross-validation using cuda
Execute this command to make cross-validation in training set of ES-Contradiction-dataset
```bash
PYTHONPATH=src python src/scripts/train_predict.py --training_set "/data/ES_Contradiction_train_v1.json" --is_cross_validation --use_cuda
```



### Contacts:
If you have any questions please contact the authors.
  * Robiert Sepúlveda Torres rsepulveda911112@gmail.com  
 
### License:
  * Apache License Version 2.0 

