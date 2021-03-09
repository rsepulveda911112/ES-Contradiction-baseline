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

### Train and predict ES-Contradiction-dataset
Execute this command to train and predict on the dataset
```bash
PYTHONPATH=src python src/scripts/train_predict.py
```

### Train and predict with cross-validation
Execute this command to make cross-validation in training set of ES-Contradiction-dataset
```bash
PYTHONPATH=src python src/scripts/train_predict_cross.py
```

### Description of the parameters
These parameters allow configuring the system to train.

|Field|Description|
|---|---|
|label_to_exclude|This parameter should be used if you want to execute experiments with fewer classes.|
|is_type_contradiction|This parameter should be True if you want to train with contradiction type labels..|
|use_cuda|This parameter should be True if cuda is present.|
|is_predict|This parameter should be True if you want to predict and False if you want evaluate.|


For example, if you want to predict only Compatible and Contradiction use this configuration:
```python
label_to_exclude = ['unrelated']
```

For example, if you want to predict only type of contradiction:
```python
is_type_contradiction = True
label_to_exclude = ['none']
```

### Contacts:
If you have any questions please contact the authors.
  * Robiert Sepúlveda Torres rsepulveda911112@gmail.com  
 
### License:
  * Apache License Version 2.0 

