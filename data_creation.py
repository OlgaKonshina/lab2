import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import opendatasets as od

#od.download(
    #"https://www.kaggle.com/datasets/mexwell/heart-disease-dataset/data?select=heart_statlog_cleveland_hungary_final.csv")
df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')
# разбиваем на тренировочную и валидационную
x_train, x_val, = train_test_split(df, test_size=0.2,random_state=42)
# Путь для сохранения файлов train

#path_train = Path('/home/olga/MLOps/lab2/train/train_data.csv')
#path_train.parent.mkdir(parents=True, exist_ok=True)
# Путь для сохранения данных test
#path_test = Path('/home/olga/MLOps/lab2/test/test_data.csv')
#path_test.parent.mkdir(parents=True, exist_ok=True)

# сохранение данных в csv
x_train.to_csv('train.csv', index=False)
x_val.to_csv('test.csv', index=False)

print('Tренировочные данные: сохранены')
print('Тестовые данные: сохранены')
