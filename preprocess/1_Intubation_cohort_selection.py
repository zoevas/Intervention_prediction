import psycopg2
import numpy as np
import pandas as pd
import os
import sqlalchemy
from sqlalchemy.sql import text
from mimic_querier import *
from datapackage_io_util import (
    load_datapackage_schema,
    load_sanitized_df_from_csv,
    save_sanitized_df_to_csv,
    sanitize_df,
)
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def continuous_outcome_processing(out_data, data, icustay_timediff):
    """
    Args
    ----
    out_data : pd.DataFrame
        index=None
        Contains subset of icustay_id corresp to specific sessions where outcome observed.
    data : pd.DataFrame
        index=icustay_id
        Contains full population of static demographic data
    Returns
    -------
    out_data : pd.DataFrame
    """
    out_data['intime'] = out_data['icustay_id'].map(data['intime'].to_dict())
    out_data['outtime'] = out_data['icustay_id'].map(data['outtime'].to_dict())
    out_data['max_hours'] = out_data['icustay_id'].map(icustay_timediff)
    out_data['starttime'] = pd.to_datetime(out_data['starttime']) - pd.to_datetime(out_data['intime'])
    out_data['starttime'] = out_data.starttime.apply(lambda x: x.days*24 + x.seconds//3600)
    out_data['endtime'] = pd.to_datetime(out_data['endtime']) - pd.to_datetime(out_data['intime'])
    out_data['endtime'] = out_data.endtime.apply(lambda x: x.days*24 + x.seconds//3600)
    out_data = out_data.groupby(['icustay_id'])

    return out_data


def add_outcome_indicators(out_gb):
    print('out_gb[subject_id]', out_gb['subject_id'])
    print('out_gb[subject_id] empty', out_gb.empty)

    subject_id = out_gb['subject_id'].unique()[0]
    hadm_id = out_gb['hadm_id'].unique()[0]
    icustay_id = out_gb['icustay_id'].unique()[0]
    max_hrs = out_gb['max_hours'].unique()[0]
    on_hrs = set()

    for index, row in out_gb.iterrows():
        print('index = ', index)
        print('row = ', row)
        on_hrs.update(range(row['starttime'], row['endtime'] + 1))

    off_hrs = set(range(max_hrs + 1)) - on_hrs
    on_vals = [0]*len(off_hrs) + [1]*len(on_hrs)
    print('onvals = ', on_vals)
    hours = list(off_hrs) + list(on_hrs)
    return pd.DataFrame({'subject_id': subject_id, 'hadm_id':hadm_id,
                        'hours_in':hours, 'on':on_vals}) #icustay_id': icustay_id})

def add_blank_indicators(out_gb):
    subject_id = out_gb['subject_id'].unique()[0]
    hadm_id = out_gb['hadm_id'].unique()[0]
    #icustay_id = out_gb['icustay_id'].unique()[0]
    max_hrs = out_gb['max_hours'].unique()[0]

    hrs = range(max_hrs + 1)
    vals = list([0]*len(hrs))
    return pd.DataFrame({'subject_id': subject_id, 'hadm_id':hadm_id,
                        'hours_in':hrs, 'on':vals})#'icustay_id': icustay_id,

ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']
ITEM_COLS = ['itemid', 'label', 'LEVEL1', 'LEVEL2']

HDF_DIR = '../Prediction/Mechanical_Ventilation/'


outcsv = 'output/'

sqluser = 'zvasilei'
password='postgres'
dbname = 'mimic'
schema_name = 'public, mimic, mimiciii;'

url = 'postgresql+psycopg2://zvasilei:postgres@localhost:5432/mimic'
engine = sqlalchemy.create_engine(url, connect_args={'options': '-csearch_path={}'.format('mimiciii')})

con = psycopg2.connect(host="localhost", database=dbname, user=sqluser, password=password)
cur = con.cursor()
cur.execute('SET search_path to ' + schema_name)

query_args = {'dbname': dbname}
query_args['user']= 'zvasilei'
query_args['password']='postgres'
query_args['host'] = 'localhost'
querier = MIMIC_Querier(query_args=query_args, schema_name=schema_name)

query = """ select * from mimiciii.d_items d where lower(d.category) like \'%intubation%\';"""

with engine.connect().execution_options(autocommit=True) as conn:
        query = conn.execute(text(query))
        patients_data = pd.DataFrame(query.fetchall())

patients_data.to_csv(outcsv + 'intubation.csv')
#Intubation itemid = 224385
query2 = """select * from mimiciii.procedureevents_mv p where p.itemid=224385"""

with engine.connect().execution_options(autocommit=True) as conn:
        query2 = conn.execute(text(query2))
        procedure_data = pd.DataFrame(query2.fetchall())

procedure_data.to_csv(outcsv + 'procedure_data.csv')


query3 = """select * from mimiciii.procedureevents_mv  where lower(ordercategoryname) like \'%ventilation%\';"""

with engine.connect().execution_options(autocommit=True) as conn:
        query3= conn.execute(text(query3))
        procedure_data = pd.DataFrame(query3.fetchall())

procedure_data.to_csv(outcsv + 'procedureVentilationData.csv')


query4 = """
select *
from mimiciii.chartevents ch
where ch.itemid in (720,223848,223849,467,445,448,449,450,1340,1486,1600,224687,
639,654,681,682,684,224685,224684,224686,
218,436,535,444,459,224697,224696,224746,224747,
221,1,1211,1655,2000,226873,224738,224419,224750,227187,543,
543,5865,5866,224707,224709,224705,224706,
60,437,505,506,686,220339,224700,3459,501,502,503,224702,223,667,668,670,671,672,224701) LIMIT 10;""";


with engine.connect().execution_options(autocommit=True) as conn:
        query4= conn.execute(text(query4))
        procedure_data = pd.DataFrame(query4.fetchall())



procedure_data.to_csv(outcsv + "MechVent.csv")




# cohort selection
min_age = 15
limit_population = 200 # if we want to run the query for a small number of patients (for debugging)
if limit_population > 0:
    limit = 'LIMIT ' + str(limit_population)
else:
    limit = ''


query = """
with patient_and_icustay_details as (
    SELECT distinct
        p.gender, p.dob, p.dod, s.*, a.admittime, a.dischtime, a.deathtime, a.ethnicity, a.diagnosis,
        DENSE_RANK() OVER (PARTITION BY a.subject_id ORDER BY a.admittime) AS hospstay_seq,
        DENSE_RANK() OVER (PARTITION BY s.hadm_id ORDER BY s.intime) AS icustay_seq,
        DATE_PART('year', s.intime) - DATE_PART('year', p.dob) as admission_age,
        DATE_PART('day', s.outtime - s.intime) as los_icu,
        CASE when a.deathtime between a.admittime and a.dischtime THEN 1 ELSE 0 END AS mort_hosp
    FROM patients p
        INNER JOIN icustays s ON p.subject_id = s.subject_id
        INNER JOIN admissions a ON s.hadm_id = a.hadm_id 
    WHERE s.first_careunit NOT like 'NICU'
        and s.hadm_id is not null and s.icustay_id is not null
        and (s.outtime >= (s.intime + interval '12 hours'))
        and (s.outtime <= (s.intime + interval '240 hours'))
    ORDER BY s.subject_id 
)
SELECT * 
FROM patient_and_icustay_details 
WHERE hospstay_seq = 1
    and icustay_seq = 1
    and admission_age >=  """ + str(min_age) + """
    and los_icu >= 0.5
""" + str(limit)
patients_data = pd.read_sql_query('SET search_path to ' + schema_name + query, con)

with engine.connect().execution_options(autocommit=True) as conn:
        query = conn.execute(text(query))
        patients_data = pd.DataFrame(query.fetchall())

#print("patients_data = " , patients_data)

patients_data.to_csv(outcsv + 'patients.csv')
# Extraction of vital data and mapping to variables
variables_to_keep = ('Capillary refill rate', 'Diastolic blood pressure', 'Fraction inspired oxygen',
                     'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total',
                     'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure',
                  'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH')



# the file is used from here
var_map = pd.read_csv('../resources/itemid_to_variable_map.csv')
index = pd.Index(var_map)
print("var Index. 1.\n",index)


icu_ids_to_keep = patients_data['icustay_id']
icu_ids_to_keep = tuple(set([str(i) for i in icu_ids_to_keep]))
print('icu_ids_to_keep', icu_ids_to_keep)
#print("icu_ids_to_keep", icu_ids_to_keep)
subjects_to_keep = patients_data['subject_id']
subjects_to_keep = tuple(set([str(i) for i in subjects_to_keep]))
print('subjects_to_keep', subjects_to_keep)

hadms_to_keep = patients_data['hadm_id']
hadms_to_keep = tuple(set([str(i) for i in hadms_to_keep]))

labitems_to_keep = []
chartitems_to_keep = []
for i in range(var_map.shape[0]):
    if var_map['LEVEL2'][i] in variables_to_keep:
        if var_map['LINKSTO'][i] == 'chartevents':
            chartitems_to_keep.append(var_map['ITEMID'][i])
        elif var_map['LINKSTO'][i] == 'labevents':
            labitems_to_keep.append(var_map['ITEMID'][i])

all_to_keep = chartitems_to_keep + labitems_to_keep
var_map = var_map[var_map.ITEMID.isin(all_to_keep)]
chartitems_to_keep = tuple(set([str(i) for i in chartitems_to_keep]))
labitems_to_keep = tuple(set([str(i) for i in labitems_to_keep]))

#print('chartitems_to_keep: ', chartitems_to_keep)
#print('labitems_to_keep: ', labitems_to_keep)


events_query = """
SELECT c.subject_id, i.hadm_id, c.icustay_id, c.charttime, c.itemid, c.value, c.valueuom
FROM icustays i
INNER JOIN chartevents c ON i.icustay_id = c.icustay_id
where c.icustay_id in """ + str(icu_ids_to_keep) + """
  and c.itemid in """ + str(chartitems_to_keep) + """
  and c.charttime between intime and outtime
  and c.error is distinct from 1
  and c.valuenum is not null
UNION ALL
SELECT distinct i.subject_id, i.hadm_id, i.icustay_id, l.charttime, l.itemid, l.value, l.valueuom
FROM icustays i
INNER JOIN labevents l ON i.hadm_id = l.hadm_id
where i.icustay_id in """ + str(icu_ids_to_keep) + """
  and l.itemid in """ + str(labitems_to_keep) + """
  and l.charttime between (intime - interval '6' hour) and outtime
  and l.valuenum > 0 -- lab values cannot be 0 and cannot be negative
"""
print('event query', events_query)

with engine.connect().execution_options(autocommit=True) as conn:
        events_query = conn.execute(text(events_query))
        events_data = pd.DataFrame(events_query.fetchall())

events_data.to_csv(outcsv + 'events.csv')


itemids = tuple(set(events_data.itemid.astype(str)))
print ("itemids: " , itemids)
query_d_items = \
        """
        SELECT itemid, label, dbsource, linksto, category, unitname
        FROM d_items
        WHERE itemid in """ + str(itemids)

with engine.connect().execution_options(autocommit=True) as conn:
        query_d_items = conn.execute(text(query_d_items))
        d_output = pd.DataFrame(query_d_items.fetchall())

d_output.to_csv(outcsv + 'd_output.csv')


# Remove the text from the categorical (Glasgow coma scale) variables so we can make them numeric:
replacement_dictionary = {'4 Spontaneously': '4', '3 To speech': '3', '2 To pain': '2', '1 No Response': '1',
                         '5 Oriented': '5', '1.0 ET/Trach': '1', '4 Confused': '4', '2 Incomp sounds': '2',
                         '3 Inapprop words': '3', 'Spontaneously': '4', 'To Speech': '3', 'None': '1', 'To Pain': '2',
                         '6 Obeys Commands': '6', '5 Localizes Pain': '5', '4 Flex-withdraws': '4', '2 Abnorm extensn': '2',
                         '3 Abnorm flexion': '3', 'No Response-ETT': '1', 'Oriented': '5', 'Confused': '4',
                         'No Response': '1', 'Incomprehensible sounds': '2', 'Inappropriate Words': '3',
                         'Obeys Commands': '6', 'No response': '1', 'Localizes Pain': '5', 'Flex-withdraws': '4',
                         'Abnormal extension': '2', 'Abnormal flexion': '3', 'Abnormal Flexion': '3',
                          'Abnormal Extension': '2'}
for key, value in replacement_dictionary.items():
    events_data['value'] = events_data['value'].replace(key, value)

events_data.to_csv(outcsv + 'eventsModified.csv')

# Change data types and set indices:
events_data['value'] = pd.to_numeric(events_data['value']) #, 'coerce')
events_data = events_data.astype({k: int for k in ['subject_id', 'hadm_id', 'icustay_id']})
patients_data = patients_data.reset_index().set_index('icustay_id')
var_map = var_map[['LEVEL2', 'ITEMID', 'LEVEL1']].rename(
    {'LEVEL2': 'LEVEL2', 'LEVEL1': 'LEVEL1', 'ITEMID': 'itemid'}, axis=1).set_index('itemid')


to_hours = lambda x: max(0, x.days*24 + x.seconds // 3600)
events_data = events_data.set_index('icustay_id').join(patients_data[['intime']])
events_data['hours_in'] = (events_data['charttime'] - events_data['intime']).apply(to_hours)
events_data = events_data.drop(columns=['charttime', 'intime'])

# Join with d_output query and group variables:
events_data = events_data.set_index('itemid', append=True)
events_data = events_data.join(var_map, how = 'outer')
d_output = d_output.set_index('itemid')
events_data = events_data.join(d_output)
events_data = events_data.set_index(['label', 'LEVEL1', 'LEVEL2'], append=True)
patients_data['max_hours'] = (patients_data['outtime'] - patients_data['intime']).apply(to_hours)

# Save results:
np.save('subjects.npy', patients_data['subject_id'])
np.save('times_hours.npy', patients_data['max_hours'])
patients_data.to_hdf(os.path.join(HDF_DIR, 'vitals_hourly_data.h5'), 'patients_data')
events_data.to_csv(outcsv + 'events_data.csv')
events_data.to_hdf(os.path.join(HDF_DIR, 'vitals_hourly_data.h5'), 'X')


vent_query = """
    select i.subject_id, i.hadm_id, v.icustay_id, v.ventnum, v.starttime, v.endtime
    FROM icustay_detail i
    INNER JOIN ventilation_durations v ON i.icustay_id = v.icustay_id
    where v.icustay_id in""" + str(icu_ids_to_keep) +  """ 
    and v.starttime between intime and outtime
    and v.endtime between intime and outtime; """

print('vent_query = ', vent_query)

exclusion_criteria_template_vars = dict(icuids=','.join(icu_ids_to_keep))

vent_out = pd.read_sql_query(vent_query, con)

vent_data= continuous_outcome_processing(vent_out, patients_data, patients_data['max_hours'])

vent_data = vent_data.apply(add_outcome_indicators)
vent_data.to_csv(outcsv + 'ventBefore.csv')
vent_data.rename(columns = {'on':'vent'}, inplace=True)
vent_data = vent_data.reset_index()

# Get the patients without the intervention in there too so that we
ids_with = vent_data['icustay_id']
ids_with = set(map(int, ids_with))
ids_all = set(map(int, icu_ids_to_keep))
print('ids_all', ids_all)
ids_without = (ids_all - ids_with) # 286338 ----> 67
#ids_without = map(int, ids_without)
print('ids_without', ids_without)

# Create a new fake dataframe with blanks on all vent entries
out_data = events_data.copy(deep=True)
out_data.to_csv(outcsv + 'out_data_initial.csv')
out_data = out_data.reset_index()
out_data = out_data.set_index('icustay_id')
out_data = out_data.iloc[out_data.index.isin(ids_without)]
out_data.to_csv(outcsv + 'out_data_iloc.csv')
out_data = out_data.reset_index()
out_data = out_data[['subject_id', 'hadm_id', 'icustay_id']]
out_data['max_hours'] = out_data['icustay_id'].map(patients_data['max_hours'])

# Create all 0 column for vent
out_data = out_data.groupby('icustay_id')
out_data = out_data.apply(add_blank_indicators)
out_data.rename(columns = {'on':'vent'}, inplace=True)
out_data = out_data.reset_index()
# Concatenate all the data vertically
Y = pd.concat([vent_data[['subject_id', 'hadm_id', 'icustay_id', 'hours_in', 'vent']],
                   out_data[['subject_id', 'hadm_id', 'icustay_id', 'hours_in', 'vent']]],
                  axis=0)

#https://github.com/MLforHealth/MIMIC_Extract/blob/master/mimic_direct_extract.py#L447
vent_data.to_csv(outcsv + 'ventOut.csv')
out_data.to_csv(outcsv + 'out_data.csv')

# Start merging all other interventions
table_names = [
      #  'adenosine_durations',
      #  'dobutamine_durations',
      #  'dopamine_durations',
      #  'epinephrine_durations',
      #  'isuprel_durations',
      #  'milrinone_durations',
        'norepinephrine_durations',
      #  'phenylephrine_durations',
      #  'vasopressin_durations',
        'vasopressor_durations'
    ]
# column_names = ['vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel',
                 #    'milrinone', 'norepinephrine', 'phenylephrine', 'vasopressin']
column_names = ['vaso', 'norepinephrine']


for t, c in zip(table_names, column_names):
        # TOTAL VASOPRESSOR DATA
        query = """
        select i.subject_id, i.hadm_id, v.icustay_id, v.vasonum, v.starttime, v.endtime
        FROM icustay_detail i
        INNER JOIN """+ t + """ v ON i.icustay_id = v.icustay_id
        where v.icustay_id in """ + str(icu_ids_to_keep) +  """
        and v.starttime between intime and outtime
        and v.endtime between intime and outtime;
        """
        #new_data = querier.query(query_string=query, extra_template_vars=dict(table=t))
        print('query = ' + query)
        new_data = pd.read_sql_query(query, con)
        new_data.to_csv(outcsv + 'new_data_'+t)
        new_data = continuous_outcome_processing(new_data, patients_data, patients_data['max_hours'])
        new_data = new_data.apply(add_outcome_indicators)
        new_data.rename(columns={'on': c}, inplace=True)
        new_data = new_data.reset_index()
        # c may not be in Y if we are only extracting a subset of the population, in which c was never
        # performed.
        if not c in new_data:
            print("Column ", c, " not in data.")
            continue

        Y = Y.merge(
            new_data[['subject_id', 'hadm_id', 'icustay_id', 'hours_in', c]],
            on=['subject_id', 'hadm_id', 'icustay_id', 'hours_in'],
            how='left'
        )

        # Sort the values
        Y.fillna(0, inplace=True)
        Y[c] = Y[c].astype(int)
        #Y = Y.sort_values(['subject_id', 'icustay_id', 'hours_in']) #.merge(df3,on='name')
        Y = Y.reset_index(drop=True)
        print('Extracted ' + c + ' from ' + t)


print('nivdurations')
#tasks=["colloid_bolus", "crystalloid_bolus", "nivdurations"]
tasks=["nivdurations"]
query = """
            select i.subject_id, i.hadm_id, v.icustay_id, v.starttime, v.endtime
            FROM icustay_detail i
            INNER JOIN norepinephrine_durations v ON i.icustay_id = v.icustay_id
            where v.icustay_id in """ + str(icu_ids_to_keep) +  """
            and v.starttime between intime and outtime
            and v.endtime between intime and outtime;
            """

        #new_data = querier.query(query_string=query, extra_template_vars=dict(table=task))
new_data = pd.read_sql_query(query, con)

new_data = continuous_outcome_processing(new_data, patients_data, patients_data['max_hours'])
new_data = new_data.apply(add_outcome_indicators)
new_data.rename(columns = {'on':'nivdurations'}, inplace=True)
new_data = new_data.reset_index()
Y = Y.merge(
    new_data[['subject_id', 'hadm_id', 'icustay_id', 'hours_in', "nivdurations"]],
     on=['subject_id', 'hadm_id', 'icustay_id', 'hours_in'],
     how='left'
)
Y.fillna(0, inplace=True)
Y["nivdurations"] = Y["nivdurations"].astype(int)
Y = Y.reset_index(drop=True)

Y = Y.filter(items=['subject_id', 'hadm_id', 'icustay_id', 'hours_in', 'vent'] + column_names + tasks)
Y.subject_id = Y.subject_id.astype(int)
Y.icustay_id = Y.icustay_id.astype(int)
Y.hours_in = Y.hours_in.astype(int)
Y.vent = Y.vent.astype(int)
Y.vaso = Y.vaso.astype(int)
y_id_cols = ID_COLS + ['hours_in']
Y = Y.sort_values(y_id_cols)

Y.set_index(y_id_cols, inplace=True)
print('Shape of patients_data : ', patients_data.shape)
print('Shape of X : ', events_data.shape)
print('Shape of Y : ', Y.shape)

outcome_schema = load_datapackage_schema(os.path.join('../resources/', 'outcome_data_spec.json'))
outPath = './output'
outcome_hd5_filename = 'outcomes_hourly_data.h5'
dynamic_hd5_filt_filename = 'all_hourly_data.h5'

# Turn back into columns
df = Y.reset_index()
df = sanitize_df(df, outcome_schema)
csv_fpath = os.path.join(outPath, 'outcomes_hourly_data.csv')
save_sanitized_df_to_csv(csv_fpath, df, outcome_schema)

Y = df

col_names  = list(df.columns.values)
col_names = col_names[3:]
with open(os.path.join(outPath, 'outcomes_colnames.txt'), 'w') as f:
   f.write('\n'.join(col_names))


#X = save_numerics()
#RETURN NUMERICS HERE CREATE A FUNCTION TO BE CALLED FOR THE VITAL SEPARATELY
#if X is not None: print("Numerics", X.shape, X.index.names, X.columns.names)
#if X is None: print("SKIPPED vitals_hourly_data")
#else:         print("LOADED vitals_hourly_data")

if Y is not None: print("Outcomes", Y.shape, Y.index.names, Y.columns.names, Y.columns)


print('patient_data[los]=',  patients_data['los'].head(10))
patients_data['los'].to_csv(outcsv + 'PATIENT_LOS.csv')
#Y['los'] = patients_data['los'] * 24.0
Y.to_csv(outcsv + 'Y_LOS.csv')
#X.to_hdf(os.path.join(outPath, dynamic_hd5_filt_filename), 'vitals_labs')
#Y.to_hdf(os.path.join(outPath, dynamic_hd5_filt_filename), 'interventions')
#Y.to_hdf('../vitals_hourly_data.h5', 'Y')
print('Y columns', Y.columns.tolist())
print('END')

# Extract length of stay and in-hospital mortality
outcomes = pd.DataFrame(index=patients_data.index)
# In hospital mortality: patient has died after the admittime to hospital and before the outtime:
mortality = patients_data.dod.notnull() & ((patients_data.admittime <= patients_data.dod) & (patients_data.outtime >= patients_data.dod))
mortality = mortality | (patients_data.deathtime.notnull() & ((patients_data.admittime <= patients_data.deathtime) &
                                                             (patients_data.dischtime >= patients_data.deathtime)))
patients_data['in_hospital_mortality'] = mortality.astype(int)
shared_idx = events_data.index
shared_sub = list(events_data.index.get_level_values('icustay_id').unique())

patients_data = patients_data[patients_data.index.get_level_values('icustay_id').isin(set(shared_sub))]
patients_data.to_hdf(os.path.join(HDF_DIR, 'vitals_hourly_data.h5'), 'patients_data')
patients_data.to_csv(outcsv + 'patient.csv')

print('patient_data[los]=',  patients_data['los'].head(10))
print('patient_data columns', patients_data.columns.tolist())


print('Y index: ', Y.index)
print('FINAL Shape of patients_data : ', patients_data.shape)
print('FINAL Shape of X : ', events_data.shape)
print('FINAL Shape of Y : ', Y.shape)

# Length of stay (in hours):
Y['los'] = patients_data['los'] * 24.0

Y.to_hdf(os.path.join(HDF_DIR, 'vitals_hourly_data.h5'), 'Y')
Y.to_csv(outcsv + 'YINTUBATION.csv')
