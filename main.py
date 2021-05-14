import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

file_path = '/kaggle/input/unimelb/'
train_data = pd.read_csv(file_path + 'unimelb_training.csv')
test_data = pd.read_csv(file_path + 'unimelb_test.csv')
train = train_data.copy()
test = test_data.copy()
train.head(5)

except_features = ['Grant.Application.ID', 'Sponsor.Code',
                   'RFCD.Code.1', 'RFCD.Percentage.1', 'RFCD.Code.2',
                   'RFCD.Percentage.2', 'RFCD.Code.3', 'RFCD.Percentage.3',
                   'RFCD.Code.4', 'RFCD.Percentage.4','RFCD.Code.5',
                   'RFCD.Percentage.5', 'Grant.Category.Code',
                   'Start.date','Unnamed: 251']

for col in range(1,16):
    if 'No..of.Years.in.Uni.at.Time.of.Grant.' + str(col) in list(train.columns):
        except_features.append('No..of.Years.in.Uni.at.Time.of.Grant.' + str(col))

for col in list(train.columns):
    if train[col].isnull().sum() > 3000:
        except_features.append(col)

print(except_features)

train = train.drop(except_features, axis = 1)
test = test.drop(except_features, axis = 1)

for col in list(train.columns):
    if train[col].dtypes == 'object':
        train[col].fillna(train[col].value_counts().idxmax(), inplace = True)
    elif train[col].dtypes == 'float64':
        train[col].fillna(train[col].mean(), inplace = True)

from sklearn.preprocessing import LabelEncoder

for col in list(train.columns):
    if train[col].dtypes == 'object':
        train[col] = LabelEncoder().fit_transform(train[col])

for col in list(test.columns):
    if test[col].dtypes == 'object':
        test[col] = LabelEncoder().fit_transform(test[col])

y = train['Grant.Status']
X = train.drop(['Grant.Status'], axis = 1)

from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3)

from sklearn.metrics import mean_squared_error, mean_absolute_error

def err(y, y_pred):
    return (mean_absolute_error(y, y_pred))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# model = RandomForestClassifier()
model = GradientBoostingClassifier()
model.fit(train_X, train_y)
print(err(test_y, model.predict(test_X)))
