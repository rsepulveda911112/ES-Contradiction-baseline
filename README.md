# ES_Headline_Contradiction

**(2021/03/09) created readme **
**(2022/01/31) updated readme **

### Requirements
* Python 3.8
* Pytorch
* Transformers

### Installation in your host
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
pip install torch==1.10.2
pip install -r requirements.txt
```

### Installation in docker container

You only need to have docker installed. 
 
Docker images tested:
* pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

You need to grant permission to install_docker.sh file:
```bash
chmod 777 install_docker.sh
```

If you have GPU you will use this command:
```bash
docker run --name name_container -it --net=host --gpus device=device_number -v folder_dir_with_code:/workspace pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel bash -c "./install_docker.sh"
```

If you have not GPU you will use this command:
```bash
docker run --name name_container -it --net=host -v folder_dir_with_code:/workspace pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel bash -c "./install_docker.sh"
```

folder_dir_with_code is this git repository 

### Download ES_Headline_Contradiction dataset
Create a folder to store the dataset.

```bash
mkdir data
```
* Download ES_Headline_Contradiction dataset from zenodo and copy in this folder:
https://zenodo.org/record/4586766#.YEelgFNKi-I


### Description of the parameters
These parameters allow configuring the system to train or predict.

|Field|Description|
|---|---|

|is_type_contradiction|This parameter should be used if you want to train with contradiction type labels.|
|use_cuda|This parameter can be used if cuda is present.|
|is_cross_validation|This parameter should be used if you want to make a cross-validation.|
|training_set|This parameter is the relative dir of training set.|
|test_set|This parameter is the relative dir of test set.|
|model_dir|This parameter is the relative dir of model for predict.|
|label_to_exclude|This parameter should be used if you want to execute experiments with fewer classes.|
|setting_file|This parameter is the relative dir of setting file.|


For example, if you want to predict only Compatible and Contradiction use this configuration:
```bash
--label_to_exclude ['unrelated']
```
For example, if you want to predict only type of contradiction:
```bash
--is_type_contradiction
```

### Train and predict ES_Headline_Contradiction
Execute this command to train and predict on the dataset
```bash
PYTHONPATH=src python src/scripts/train_predict.py --training_set "/data/ES_Contradiction_train_consolidated.json" --test_set "/data/ES_Contradiction_test_consolidated.json"
```

### Predict ES_Headline_Contradiction
Execute this command to predict all the dataset
```bash
PYTHONPATH=src python src/scripts/train_predict.py --test_set "/data/ES_Contradiction_train_consolidated.json" --model_dir "/outputs/checkpoint-x-epoch-y"
```

### Predict ES_Headline_Contradiction (type of contradiction)
Execute this command to predict the type of contradiction on the dataset
```bash
PYTHONPATH=src python src/scripts/train_predict.py --test_set "/data/ES_Contradiction_train_consolidated.json" --model_dir "/outputs/checkpoint-x-epoch-y" --is_type_contradiction --label_to_exclude "none"
```

### Train and predict with cross-validation using cuda
Execute this command to make cross-validation in training set of ES-Contradiction-dataset
```bash
PYTHONPATH=src python src/scripts/train_predict.py --training_set "/data/ES_Contradiction_train_consolidated.json" --is_cross_validation --use_cuda
```

### Train and predict with HeadlineStanceChecker
```bash
PYTHONPATH=src python src/scripts/train_predict.py --setting_file "/setting_train.yaml" --training_set "/data/ES_Contradiction_train_consolidated.json" --test_set "/data/ES_Contradiction_test_consolidated.json" --use_cuda
```
You use the setting file setting_train.yaml. This file allows updating the labels of the dataset to adapt to architecture.
The command above allows training the first stage of HeadlineStanceChecker and bellow 

```bash
PYTHONPATH=src python src/scripts/train_predict.py --training_set "/data/ES_Contradiction_train_consolidated.json" --test_set "/data/ES_Contradiction_test_consolidated.json" --label_to_exclude ['unrelated']
```
### Contacts:
If you have any questions please contact the authors.
  * Robiert Sepúlveda Torres rsepulveda911112@gmail.com  
 
### License:
  * Apache License Version 2.0 

