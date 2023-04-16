import numpy as np
import pandas as pd
import sys
import os
import pickle
from sklearn.model_selection import train_test_split
from simple_impute import simple_imputer
from sklearn.model_selection import StratifiedKFold
from numpy import savetxt
from Windows import make_3d_tensor_slices
from sklearn.preprocessing import label_binarize
from LSTM import LSTM_Model
from CONV1D import CONV1D_Model
from sklearn.metrics import roc_auc_score

#print full numpy
np.set_printoptions(threshold=sys.maxsize)


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

MAX_LEN = 240
NUM_CLASSES = 4

output = 'output/'

def categorize_age(age):
    if age > 10 and age <= 30:
        cat = 1
    elif age > 30 and age <= 50:
        cat = 2
    elif age > 50 and age <= 70:
        cat = 3
    else:
        cat = 4
    return cat

def categorize_ethnicity(ethnicity):
    if 'AMERICAN INDIAN' in ethnicity:
        ethnicity = 'AMERICAN INDIAN'
    elif 'ASIAN' in ethnicity:
        ethnicity = 'ASIAN'
    elif 'WHITE' in ethnicity:
        ethnicity = 'WHITE'
    elif 'HISPANIC' in ethnicity:
        ethnicity = 'HISPANIC/LATINO'
    elif 'BLACK' in ethnicity:
        ethnicity = 'BLACK'
    else:
        ethnicity = 'OTHER'
    return ethnicity

# Task specifics
INTERVENTION = 'vent'
DATAFILEVITAL = 'vitals_hourly_data_preprocessed.h5'
# DATAFILEINTERVENTION = '../../preprocess/output/all_hourly_data.h5'
DATAFILEINTERVENTION =  'vitals_hourly_data_preprocessed.h5'
DATAFILEPATIENT = 'vitals_hourly_data.h5'
#Ymort = pd.read_hdf('../../vitals_hourly_data_preprocessed.h5', 'Y')[['in_hospital_mortality']]


# LOAD DATA
X = pd.read_hdf(DATAFILEVITAL,'X') # ADD IN)HOSPITAL_MORTALITY IN preprocess_formal file in vitals_hourly_data_preprocessed.h5
#Y = pd.read_hdf(DATAFILEINTERVENTION,'interventions')
Y = pd.read_hdf(DATAFILEINTERVENTION, 'Y') [['vent', 'los']]
static = pd.read_hdf(DATAFILEPATIENT,'patients_data') # probably I should keep this as the make windows loop is for patients


Y = Y[[INTERVENTION]]

print ('Shape of X : ', X.shape)
print ('Shape of Y : ', Y.shape)
print ('Shape of static : ', static.shape)

print(X.columns.tolist())
#print('X.head', X.head())
X.to_csv(output + 'X.csv')
# Preprocessing data
# train_ids, test_ids = train_test_split(X.reset_index(), test_size=0.2,
        #                                random_state=0, stratify=X['in_hospital_mortality'])
# print('train_ids=', train_ids)
# print(train_ids.columns.tolist())
# split_train_ids, val_ids = train_test_split(train_ids, test_size=0.125,
                       #                 random_state=0, stratify=train_ids['in_hospital_mortality'])
# Imputation and Standardization of Time Series Features
#problem here. Maybe not needed imputation done already in X, in_hospital_mortality added in the patients data, check how add it to X of preprocess fil
#X_clean = simple_imputer(X,train_ids['subject_id'])

print('Y columns: ', Y.columns.tolist())

X = X.sort_index(axis = 0, level = 'icustay_id')
Y = Y.sort_index(axis = 0, level = 'icustay_id')

# Remove the icustays that were less than 48 hours:
indices_to_remove = []
# for i, row in Y.iterrows():
  #   if row['los'] < 48:
    #     indices_to_remove.append(i)

#print('static.head:', static.head(10))
print('static.columns:', static.columns.tolist())
# use gender, first_careunit, age and ethnicity for prediction
static_to_keep = static[['gender', 'admission_age', 'ethnicity', 'first_careunit', 'intime']]
static_to_keep.loc[:, 'intime'] = static_to_keep['intime'].astype('datetime64').apply(lambda x : x.hour)
static_to_keep.loc[:, 'admission_age'] = static_to_keep['admission_age'].apply(categorize_age)
static_to_keep.loc[:, 'ethnicity'] = static_to_keep['ethnicity'].apply(categorize_ethnicity)
static_to_keep = pd.get_dummies(static_to_keep, columns = ['gender', 'admission_age', 'ethnicity', 'first_careunit'])

static_to_keep.to_csv(output + 'static_to_keep.csv')
print('X.shape[0]',X.shape[0])
X_merge = pd.merge(X.reset_index(), static_to_keep.reset_index(), left_on=['icustay_id'],
                   right_on=['icustay_id'])
print('Xmerge.shape[0]',X_merge.shape[0])
X_merge = X_merge.set_index(['subject_id','hadm_id','hours_in'])

# X_merge = X_merge.set_index('icustay_id')


print ('Shape of X after: ', X.shape)
print ('Shape of Y after: ', Y.shape)

def create_x_matrix(x): # extract the first 48 hours for every icustay.
    zeros = np.zeros((MAX_LEN, x.shape[1]-4))
    x = x.values
    x = x[:MAX_LEN, 4:] # the first four columns are for subject_id, icustay_id, hadm_id and hours_in.
    zeros[0:x.shape[0], :] = x
    return zeros

def create_y_matrix(y):
   # y = y['vent'].to_numpy()
    zeros = np.zeros((MAX_LEN, y.shape[1]-4))
    y = y.values
    y = y[:,4:]
    y = y[:MAX_LEN, :]
    zeros[:y.shape[0], :] = y
    return y

X_merge.to_csv(output + 'XMERGE.csv')
Y.to_csv(output + 'Y.csv')
print('####XMERGE#####')
#print(X_merge.reset_index()['subject_id'].unique())
print('####Y#####')
#print(Y.reset_index()['subject_id'].unique())
print('############')
XKEYS = X_merge.reset_index()['subject_id'].unique()
YKEYS = Y.reset_index()['subject_id'].unique()
equal_arrays = (XKEYS== YKEYS)

#print(equal_arrays)

temp3 = []
for element in XKEYS:
    if element not in YKEYS:
        temp3.append(element)

print('XMERGE columns', X_merge.columns.tolist())
x = np.array(list(X_merge.reset_index().groupby(['subject_id', 'icustay_id','hadm_id']).apply(create_x_matrix)))
print('y index', Y.index)
y = np.array(list(Y.reset_index().groupby(['subject_id', 'icustay_id', 'hadm_id']).apply(create_y_matrix))) [:] # [:,:,0]

del X, Y
print ('Before training: Shape of x : ', x.shape)
print ('Before training: Shape of y: ', y.shape)
yd = pd.DataFrame(y)
yd.reset_index().to_csv(output + "yprediction.csv")


#print('yd head', yd.head(100))


lengths = np.array(list(X_merge.reset_index().groupby('subject_id').apply(lambda x: x.shape[0])))
keys = pd.Series(X_merge.reset_index()['subject_id'].unique())

print("X tensor shape: ", x.shape)
print("Y tensor shape: ", y.shape)
print("lengths shape: ", lengths.shape)

print('static', static['subject_id'])

train_ids, test_ids = train_test_split(static.reset_index(), test_size=0.2,
                                       random_state=0, stratify=static['mort_hosp'])
#print('train_ids', train_ids)

split_train_ids, val_ids = train_test_split(train_ids, test_size=0.125,
                                            random_state=0, stratify=train_ids['mort_hosp'])

#print('split_train_ids', split_train_ids)
print('val_ids', val_ids)

print('train_ids[subject_id]', train_ids['subject_id'])
print('test_ids[subject_id]', test_ids['subject_id'])
print('split_train_ids[subject_id]', split_train_ids['subject_id'])
print('val_ids[subject_id]', val_ids['subject_id'])
# MAKE TENSORS
train_indices = np.where(keys.isin(train_ids['subject_id']))[0]
print("train_indices", train_indices)
test_indices = np.where(keys.isin(test_ids['subject_id']))[0]
print("test_indices", test_indices)
split_train_indices = np.where(keys.isin(split_train_ids['subject_id']))[0]
print("split_train_indices", split_train_indices)
val_indices = np.where(keys.isin(val_ids['subject_id']))[0]
print("val_indices ", val_indices)
X_train = x[split_train_indices]
print("Training size: ", X_train.shape[0])
Y_train = y[split_train_indices]
X_test = x[test_indices]
Y_test = y[test_indices]
X_val = x[val_indices]
Y_val = y[val_indices]
lengths_train = lengths[split_train_indices]
lengths_val = lengths[val_indices]
lengths_test = lengths[test_indices]

#print('XTRAIN row  0:', x[1 , ...])
print("Training size: ", X_train.shape[0])
print("Validation size: ", X_val.shape[0])
print("Test size: ", X_test.shape[0])


print("Training X_train ndim: ", X_train.ndim)
print("Training X_train shape: ", X_train.shape)
print("Training y_train ndim: ", Y_train.ndim)
print("Training y_train shape: ", Y_train.shape)
#skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

#for train_index, test_index in skf.split(x,y):
  #  print("TRAIN:", train_index, "TEST:", test_index)

     # Split in training and test set, and split the training further in training and validation sets
  #  x_train, x_val, y_train, y_val = train_test_split(x[train_index], y[train_index], test_size=0.125, random_state=0, stratify=y[train_index])
   # x_test, y_test = x[test_index], y[test_index]


x_train, y_train = make_3d_tensor_slices(X_train, Y_train, lengths_train)
x_val, y_val = make_3d_tensor_slices(X_val, Y_val, lengths_val)
x_test, y_test = make_3d_tensor_slices(X_test, Y_test, lengths_test)

y_train_classes = label_binarize(y_train, classes=range(NUM_CLASSES))
if (y_val.shape[0] != 0):
    y_val_classes = label_binarize(y_val, classes=range(NUM_CLASSES))
y_test_classes = label_binarize(y_test, classes=range(NUM_CLASSES))

print('x_train.shape = ', x_train.shape)
 # LSTM_Model(x_train, y_train, x_test, x_val)
CONV1D_Model(x_train, y_train, y_train_classes, x_val, y_val_classes, x_test, y_test_classes)

