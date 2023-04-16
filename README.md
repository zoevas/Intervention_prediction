# Intervention_prediction
## About
This repo contains code related with intervention prediction and specifically for Mechanical Ventilation using the [MIMIC III Clinical database](https://physionet.org/content/mimiciii/1.4/). Specifically, it contains:
- Cohort Selection
- Preprocessing of data
- Prediction of Intervention prediction
- UI for giving the new input to the prediction model

## Step-by-step instructions
### Setting up the MIMIC III dataset on a postgres database
At least 47GB should be available on the disk.
1. Given that the dataset is downloaded, follow the steps from this [buildmimic](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/postgres) in order to load the data into a postgres database.

2. Create specific tables by running some scripts from [MIT-LCP concepts](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/concepts) so as to have the knowledge that can be used as a target variable if a patient has been ventilated or not

Go to the concepts_postgres folder, run the postgres-functions.sql and postgres-make-concepts.sql scripts, in that order.
```
zvasilei@xxx:~/Downloads/mimicClinicalVaso/mimicClinical/mimic-code/mimic-iii/concepts$ DBCONNEXTRA="user=<username> password=<pwd>" bash postgres_make_concepts.sh
```

## Background
### Cohort Selection (```preprocess/1_Intubation_cohort_selection.py```)


### Preprocessing of the dataset (```preprocess/2_Intubation_preprocess.py```)
For the preprocessing, we have been based on the preprocessing pipeline of this paper:

[MIMIC-Extract: A Data Extraction, Preprocessing, and
Representation Pipeline for MIMIC-III](https://arxiv.org/pdf/1907.08322.pdf)

Code: https://github.com/MLforHealth/MIMIC_Extract

### Prediction (```Prediction/Mechanical_Ventilation/Mechanical_Ventilation_Prediction.py```)

For the mechanical ventilation prediction, our approach has been founded on this method: 
[Clinical Intervention Prediction and Understanding with Deep Neural Networks
](https://www.semanticscholar.org/paper/Clinical-Intervention-Prediction-and-Understanding-Suresh-Hunt/5dba3ab85f106874178e1e2d52fc4247afed912e)

