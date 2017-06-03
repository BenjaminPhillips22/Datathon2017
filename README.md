

### Melbourne 2017 Datathon

Here is a quick guide to help people get started in the kaggle competition.

http://rpubs.com/benjaminphillips22/GettingStarted2017Datathon

Here is my submission for the insight report. You can find the code that produced the plots here on github.

http://rpubs.com/benjaminphillips22/273476
 
The data processing stage. How I extracted features from the data set.
This is in the some_functions file and executed in the get_data file.

Here is a list of the features I generated. This feature space was cut down during the modelling process as not every feature added value.

	['Patient_ID', 'Diabetes_in_2016', 'Diabetes_length_time',
       'Osteoporosis_length_time', 'Depression_length_time',
       'Epilepsy_length_time', 'Lipids_length_time',
       'Hypertension_length_time', 'Heart Failure_length_time',
       'Anti-Coagulant_length_time', 'Immunology_length_time',
       'Urology_length_time',
       'Chronic Obstructive Pulmonary Disease (COPD)_length_time',
       'Diabetes_time_since_2016', 'Osteoporosis_time_since_2016',
       'Depression_time_since_2016', 'Epilepsy_time_since_2016',
       'Lipids_time_since_2016', 'Hypertension_time_since_2016',
       'Heart Failure_time_since_2016', 'Anti-Coagulant_time_since_2016',
       'Immunology_time_since_2016', 'Urology_time_since_2016',
       'Chronic Obstructive Pulmonary Disease (COPD)_time_since_2016',
       'Diabetes_never_on', 'Osteoporosis_never_on', 'Depression_never_on',
       'Epilepsy_never_on', 'Lipids_never_on', 'Hypertension_never_on',
       'Heart Failure_never_on', 'Anti-Coagulant_never_on',
       'Immunology_never_on', 'Urology_never_on',
       'Chronic Obstructive Pulmonary Disease (COPD)_never_on',
       'PARACETAMOL_count', 'PERINDOPRIL_count', 'ESOMEPRAZOLE_count',
       'PANTOPRAZOLE_count', 'RAMIPRIL_count', 'AMLODIPINE_count',
       'OMEPRAZOLE_count', 'CLOPIDOGREL_count', 'LERCANIDIPINE_count',
       'RABEPRAZOLE_count', 'SALBUTAMOL CFC FREE_count', 'unknown_count',
       'time_since_first', 'time_since_last', 'F', 'M', 'U',
       'Inner Regional Australia', 'Major Cities of Australia', 'remote',
       'ses_code'] 
 

 
 Ben Phillips