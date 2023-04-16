# Intervention_prediction
## About
This repo contains code related with intervention prediction and specifically for Mechanical Ventilation using the [MIMIC III Clinical database](https://physionet.org/content/mimiciii/1.4/). Specifically, it contains:
- Cohort Selection
- Preprocessing of data
- Prediction of Intervention prediction
- UI for giving the new input to the prediction model

The prediction can be easily tailored to other predictions by changing the target variable to be other type of intervention such as red blood transfusion, vassopressor usage, non-invasive ventilation. The main changes will be to the cohort selection query for retrieving the corresponding target variable.

## Step-by-step instructions
### Setting up the MIMIC III dataset on a postgres database
At least 47GB should be available on the disk.
1. Given that the dataset is downloaded, follow the steps from this [buildmimic](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/postgres) in order to load the data into a postgres database.

2. Create specific tables by running some scripts from [MIT-LCP concepts](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/concepts) so as to have the knowledge that can be used as a target variable if a patient has been ventilated or not

Go to the concepts_postgres folder, run the postgres-functions.sql and postgres-make-concepts.sql scripts, in that order.
```
zvasilei@xxx:~/Downloads/mimicClinicalVaso/mimicClinical/mimic-code/mimic-iii/concepts$ DBCONNEXTRA="user=<username> password=<pwd>" bash postgres_make_concepts.sh
```

## Description
### Cohort Selection (```preprocess/1_Intubation_cohort_selection.py```)
Patients who are younger than 15 years old are excluded. Only first ICU admission that took at least one day and less than 10 days are retrieved.
The final cohort is 34472.
The patients, icu stays and admission tables are joined in order to compute patient's age, and other information.
From the charted events and lab events, 17 different variables are retrieved. 

### Preprocessing of the dataset (```preprocess/2_Intubation_preprocess.py```)
For the preprocessing, we have been based on the preprocessing pipeline of this paper:

[MIMIC-Extract: A Data Extraction, Preprocessing, and
Representation Pipeline for MIMIC-III](https://arxiv.org/pdf/1907.08322.pdf)

Code: https://github.com/MLforHealth/MIMIC_Extract

In the cohort selection part, we have extracted patients, events, and outcomes (if the patient has been mechanically ventilated or not).
Here, the data are pre-processed:
- Map event variables to the same metric
- Detect and remove or replace outliers in the event data
- Reorganize the events data so that we have a column for every variable and a row for every hour and every ICU stay
- Impute time series data: if the variable was measured for the subject, but only after the current hour, we replace the NaN value with the average for this specific subject/ICU stay combination. If the variable was never measured, we replace with the global average.
- Standarize the continuous event variables using a min max scaler
- One-hot encode the categorical event variables
For the outliers, and the unit conversion, we have used formal checks based on the [aforementioned paper](https://github.com/MLforHealth/MIMIC_Extract).

### Prediction (```Prediction/Mechanical_Ventilation/Mechanical_Ventilation_Prediction.py```)

For the mechanical ventilation prediction, our approach has been founded on this method: 
[Clinical Intervention Prediction and Understanding with Deep Neural Networks
](https://www.semanticscholar.org/paper/Clinical-Intervention-Prediction-and-Understanding-Suresh-Hunt/5dba3ab85f106874178e1e2d52fc4247afed912e)

### Running the demo on a docker container (```dockerization/main.py```)
Run the [deployment.sh](https://github.com/zoevas/Intervention_prediction/blob/main/dockerization/deployment.sh)
Hit localhost:8080. You can give your input and get the prediction on the UI.
