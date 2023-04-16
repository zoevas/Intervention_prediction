import numpy as np
import pandas as pd
import os
import json

csvfolder = 'csv'

HDF_DIR = '../Prediction/Mechanical_Ventilation/'

events_data = pd.read_hdf(os.path.join(HDF_DIR, 'vitals_hourly_data.h5'), 'X')
events_data = events_data.reset_index()
print('events_data (X): ', events_data.shape)

patients_data = pd.read_hdf(os.path.join(HDF_DIR, 'vitals_hourly_data.h5'), 'patients_data')
print('patients_data: ', patients_data.shape)

outcomes = pd.read_hdf(os.path.join(HDF_DIR, 'vitals_hourly_data.h5'), 'Y')
print('outcomes (Y): ', outcomes.shape)

# Load the config file that contains information about continuous/categorical variables:
config = json.load(open('resources/discretizer_config.json', 'r'))
is_categorical = config['is_categorical_channel']

# Get categorical variables:
categorical_var = []
continuous_var = []
for key, value in is_categorical.items():
    if value:
        categorical_var.append(key)
    else:
        continuous_var.append(key)
print('Categorical: ', categorical_var[1:])
print('Continuous: ', continuous_var)
categorical_var = categorical_var[1:]


# Map variables to the same metric:
UNIT_CONVERSIONS = [
    ('weight',                   'oz',  None,             lambda x: x/16.*0.45359237),
    ('weight',                   'lbs', None,             lambda x: x*0.45359237),
    ('fraction inspired oxygen', None,  lambda x: x > 1,  lambda x: x/100.),
    ('oxygen saturation',        None,  lambda x: x <= 1, lambda x: x*100.),
    ('temperature',              'f',   lambda x: x > 79, lambda x: (x - 32) * 5./9),
    ('height',                   'in',  None,             lambda x: x*2.54),
]

print('before unit conversions')
variable_names = events_data['LEVEL1'].str
variable_units = events_data['valueuom'].str
for name, unit, check, convert_function in UNIT_CONVERSIONS:
    indices_variable = variable_names.contains(name, case=False, na=False)
    needs_conversion_filter_indices = indices_variable & False
    if unit is not None:
        needs_conversion_filter_indices |= variable_names.contains(unit, case=False, na=False) | variable_units.contains(unit, case=False, na=False)
    if check is not None:
        needs_conversion_filter_indices |= check(events_data['value'])
    idx = indices_variable & needs_conversion_filter_indices
    events_data.loc[idx, 'value'] = convert_function(events_data['value'][idx])

print('after unit conversions')
# Detect and remove outliers. For this, they use two different outlier ranges:
# 1) for each variable, they have an upper and lower threshold for detecting unusable outliers.
#    If the outlier falls outside of these threshold, it is treated as missing.
# 2) they also have a physiologically valid range of measurements. If the non-outlier falls outside this range,
     # it is replaced with the nearest valid value.

variable_ranges = pd.read_csv('resources/variable_ranges.csv', index_col = None)
variable_ranges['LEVEL2'] = variable_ranges['LEVEL2'].str.lower()
variable_ranges = variable_ranges.set_index('LEVEL2')

variables_all = events_data['LEVEL2']
non_null_variables = ~events_data.value.isnull()
variables = set(variables_all)
range_names = set(variable_ranges.index.values)
range_names = [i.lower() for i in range_names]

print('before outlier conversion')

for var_name in variables:
    var_name_lower = var_name.lower()

    if var_name_lower in range_names:
        out_low, out_high, val_low, val_high = [
            variable_ranges.loc[var_name_lower, x] for x in ('OUTLIER LOW', 'OUTLIER HIGH', 'VALID LOW', 'VALID HIGH')
        ]

        # First find the indices of the variables that we need to check for outliers:
        indices_variable = non_null_variables & (variables_all == var_name)

        # Check for low outliers and if they are not extreme, replace them with the imputation value:
        outlier_low_indices = (events_data.value < out_low)
        low_not_outliers = ~outlier_low_indices & (events_data.value < val_low)
        valid_low_indices = indices_variable & low_not_outliers
        events_data.loc[valid_low_indices, 'value'] = val_low

        # Check for high outliers and if they are not extreme, replace them with the imputation value:
        outlier_high_indices = (events_data.value > out_high)
        high_not_outliers = ~outlier_high_indices & (events_data.value > val_high)
        valid_high_indices = indices_variable & high_not_outliers
        events_data.loc[valid_high_indices, 'value'] = val_high

        # Treat values that are outside the outlier boundaries as missing:
        outlier_indices = indices_variable & (outlier_low_indices | outlier_high_indices)
        events_data.loc[outlier_indices, 'value'] = np.nan

print('after outlier conversion')

events_data = events_data.drop(columns=['valueuom', 'dbsource', 'linksto', 'category', 'unitname'])
events_data.to_csv(csvfolder+ '/events_data_copy_beforeagg.csv')

events_data = events_data.set_index(['icustay_id', 'itemid', 'label', 'LEVEL1', 'LEVEL2'])
events_data = events_data.groupby(['icustay_id', 'subject_id', 'hadm_id', 'LEVEL2', 'hours_in'])
events_data = events_data.agg(['mean', 'std', 'count'])
#events_data = events_data.agg(['mean', 'std'])
events_data.columns = events_data.columns.droplevel(0)
events_data.columns.names = ['Aggregation Function']
events_data = events_data.unstack(level = 'LEVEL2')
events_data.columns = events_data.columns.reorder_levels(order=['LEVEL2', 'Aggregation Function'])
events_data.to_csv(csvfolder+ '/events_data_copy_afteragg.csv')

print('aggregation')
# Make sure we have a row for every hour:
missing_hours_fill = pd.DataFrame([[i, x] for i, y in patients_data['max_hours'].iteritems() for x in range(y+1)],
                                 columns=[patients_data.index.names[0], 'hours_in'])
missing_hours_fill['tmp'] = np.NaN

fill_df = patients_data.reset_index()[['subject_id', 'hadm_id', 'icustay_id']].join(
     missing_hours_fill.set_index('icustay_id'), on='icustay_id')
fill_df.set_index(['icustay_id', 'subject_id', 'hadm_id', 'hours_in'], inplace=True)

events_data = events_data.reindex(fill_df.index)
events_data = events_data.sort_index(axis = 0).sort_index(axis = 1)

idx = pd.IndexSlice
events_data.loc[:, idx[:, 'count']] = events_data.loc[:, idx[:, 'count']].fillna(0)

# Save this version of the data as a .csv file, so we can apply different imputation methods in another notebook:
idx = pd.IndexSlice
timeseries_data = events_data.loc[:, idx[:, 'mean']]
timeseries_data = timeseries_data.droplevel('Aggregation Function', axis = 1)
timeseries_data = timeseries_data.reset_index()
timeseries_data.to_csv(csvfolder+ '/mimic_timeseries_data_not_imputed.csv')

idx = pd.IndexSlice
timeseries_data = events_data.loc[:, idx[:, ['mean', 'count']]]

# Get the mean across hours for each variable and each patient:
icustay_means = timeseries_data.loc[:, idx[:, 'mean']].groupby(['subject_id', 'hadm_id', 'icustay_id']).mean()
# Get the global mean for each variable:
global_means = timeseries_data.loc[:, idx[:, 'mean']].mean(axis = 0)

# Forward fill the nan time series, or otherwise fill in the patient's mean or global mean:
timeseries_data.loc[:, idx[:, 'mean']] = timeseries_data.loc[:, idx[:, 'mean']].groupby(
    ['subject_id', 'hadm_id', 'icustay_id']).fillna(method='ffill').groupby(
    ['subject_id', 'hadm_id', 'icustay_id']).fillna(icustay_means).fillna(global_means)

# Create a mask that indicates if the variable is present:
timeseries_data.loc[:, idx[:, 'count']] = (events_data.loc[:, idx[:, 'count']] > 0).astype(float)
timeseries_data.rename(columns={'count': 'mask'}, level='Aggregation Function', inplace=True)

# Add a variable that indicates the time since the last measurement to the dataframe:
is_absent = (1 - timeseries_data.loc[:, idx[:, 'mask']])
hours_of_absence = is_absent.cumsum()
time_since_measured = hours_of_absence - hours_of_absence[is_absent==0].fillna(method='ffill')
time_since_measured.rename(columns={'mask': 'time_since_measured'}, level='Aggregation Function', inplace=True)
timeseries_data = pd.concat((timeseries_data, time_since_measured), axis = 1)
timeseries_data.loc[:, idx[:, 'time_since_measured']] = timeseries_data.loc[:, idx[:, 'time_since_measured']].fillna(100)
timeseries_data.sort_index(axis=1, inplace=True)

print('before standarization')
# STANDARIZATION OF CONTINUOUS DATA

# Minmax standardization:
def minmax(x):
    mins = x.min()
    maxes = x.max()
    x_std = (x - mins) / (maxes - mins)
    return x_std

def std_time_since_measurement(x):
    idx = pd.IndexSlice
    x = np.where(x==100, 0, x)
    means = x.mean()
    stds = x.std() + 0.0001
    x_std = (x - means)/stds
    return x_std

timeseries_data.loc[:, idx[continuous_var, 'mean']] = timeseries_data.loc[:, idx[continuous_var, 'mean']].apply(lambda x: minmax(x))
timeseries_data.loc[:, idx[:, 'time_since_measured']] = timeseries_data.loc[:, idx[:, 'time_since_measured']].apply(lambda x: std_time_since_measurement(x))

# Check this file before one hot encoding categorical variables
timeseries_data.to_csv(csvfolder+ '/timeseries_data_before_onehotencoding.csv')

#  One-hot encoding categorical variables
# First we need to round the categorical variables to the nearest category:
categorical_data = timeseries_data.loc[:, idx[categorical_var, 'mean']].copy(deep=True)
categorical_data = categorical_data.round()
# The get_dummies() function is used to convert categorical variable into dummy/indicator variables.
one_hot = pd.get_dummies(categorical_data, columns=categorical_var)
np.set_printoptions(threshold=np.inf)
one_hot.to_csv('csv/one_hot.csv')
print('categorical_var: ', categorical_var)
# Clean up the columns that we do not need and add the dummy encodings:
for c in categorical_var:
    if c in timeseries_data.columns:
        timeseries_data.drop(c, axis = 1, inplace=True)
        print('c=', c)
timeseries_data.columns = timeseries_data.columns.droplevel(-1)
timeseries_data = pd.merge(timeseries_data.reset_index(), one_hot.reset_index(), how='inner', left_on=['subject_id', 'icustay_id', 'hadm_id', 'hours_in'],
                           right_on=['subject_id', 'icustay_id', 'hadm_id', 'hours_in'])
timeseries_data = timeseries_data.set_index(['subject_id', 'icustay_id', 'hadm_id', 'hours_in'])

print('timeseries before rename')
timeseries_data.to_csv(csvfolder+ '/timeseries_data_before_rename.csv')
# First get the number of nan values per variable:
print(outcomes.isna().sum())

# We will replace them with zero:
outcomes = outcomes.fillna(0)

# Save all pre-processed data
# Rename the columns and save the results:
s = timeseries_data.columns.to_series()
timeseries_data.to_csv('s.csv')
timeseries_data.columns = s + s.groupby(s).cumcount().astype(str).replace({'0':''})
timeseries_data.to_csv(csvfolder+ '/timeseries_data.csv')
print('preprocessed data saved')

timeseries_data.to_hdf(os.path.join(HDF_DIR, 'vitals_hourly_data_preprocessed.h5'), 'X')
print('vitals hourly data preprocessed X')
outcomes.to_hdf(os.path.join(HDF_DIR, 'vitals_hourly_data_preprocessed.h5'), 'Y')
print('vitals hourly data preprocessed Y')
print('END')
