import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt





def extract_features(df, df_missing, ChronicIllness_LookUp, Drug_LookUp, Patients_LookUp, remoteness):
    
    #append missing data
    df = df.append(df_missing, ignore_index=True)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # These are dates, so convert them from string to datetime
    df.Prescription_Week = pd.to_datetime(df.Prescription_Week)
    df.Dispense_Week = pd.to_datetime(df.Dispense_Week)
   
    # This merge (outer) connects chronicIllness to our data by mergeing the two data sets
    df_merge = pd.merge(left=ChronicIllness_LookUp[['ChronicIllness','MasterProductID']], right=df, how='outer', left_on='MasterProductID', right_on='Drug_ID')
    df_merge.ChronicIllness = df_merge.ChronicIllness.fillna(value = 'unknown')
    
    # get our 'Diabetes_in_2016' variable
    df_merge['in_2016'] = (df_merge.Dispense_Week>=pd.to_datetime("2016-01-01"))
    df_merge['Diabetes_in_2016'] = (df_merge.ChronicIllness=='Diabetes') & df_merge['in_2016']

    # prepare for output now, this is Patiend_ID and target label
    df_bigs = df_merge[['Patient_ID','Diabetes_in_2016']].groupby(['Patient_ID'], as_index=False).agg({'Diabetes_in_2016':np.any})
    df_bigs.Diabetes_in_2016 = df_bigs.Diabetes_in_2016.astype(int)
    
    # now to remove transactions after 2016
    df_merge = df_merge.ix[~df_merge.in_2016, ]
    
    # add a GenericIngreintName column (use this to update some ChronicIllness unknowns.)
    df_merge = pd.merge(left=Drug_LookUp[['GenericIngredientName','MasterProductID']], right=df_merge, how='right', left_on='MasterProductID', right_on='Drug_ID')
    
    
    # defin these for later
    ingredients = ['PARACETAMOL','PERINDOPRIL','ESOMEPRAZOLE','PANTOPRAZOLE','RAMIPRIL','AMLODIPINE','OMEPRAZOLE',
               'CLOPIDOGREL','LERCANIDIPINE','RABEPRAZOLE','SALBUTAMOL CFC FREE','unknown']

    chronics = ['Diabetes', 'Osteoporosis', 'Depression', 'Epilepsy', 'Lipids','Hypertension', 'Heart Failure',
            'Anti-Coagulant', 'Immunology','Urology', 'Chronic Obstructive Pulmonary Disease (COPD)']

    
    # replace half of the 'unknown's with the top 10 (or 11 or 12) top ingredients for the unknowns
    update_chronicIllnes_mask = (df_merge.ChronicIllness=='unknown') & (df_merge.GenericIngredientName.isin(ingredients))
    df_merge.ix[update_chronicIllnes_mask,'ChronicIllness'] = df_merge.ix[update_chronicIllnes_mask,'GenericIngredientName']

    # reduce df_merge
    df_merge = df_merge[['ChronicIllness', 'Patient_ID', 'Drug_ID','Dispense_Week']]
    
    # now to the ChronicIllness pivot
    aggregations = {
        'Dispense_Week': { # work on the "Dispense_Week" column
            'min_date': 'min',  # get the min, and call this result 'min_date'
            'max_date': 'max', # 
        }
    }
    df_chronic = df_merge.groupby(['Patient_ID','ChronicIllness'], as_index=False).agg(aggregations)
    df_chronic.columns = np.concatenate([df_chronic.columns.get_level_values(0)[range(0,2)],df_chronic.columns.get_level_values(1)[range(2,4)]])
    
    # make some new variables
    df_chronic['time_on_drug'] = (df_chronic.max_date -  df_chronic.min_date ).apply(lambda s : (s.days/30)+1 )
    df_chronic['time_since_2016'] = (pd.to_datetime("2016-01-02") - df_chronic.max_date).apply(lambda s : s.days/30)
    # now add the 'never_on_drug' variable; 0 for on it, 1 for not on it, add 1 later
    df_chronic['never_on_drug'] = 0
    
    # first one (time on drug)
    df_chronic_t1 = df_chronic[['Patient_ID','ChronicIllness','time_on_drug']].pivot(index = 'Patient_ID', columns = 'ChronicIllness', values = 'time_on_drug').reset_index().fillna(value=0)
    c = df_chronic_t1.Patient_ID
    df_chronic_t1 = df_chronic_t1[chronics]
    df_chronic_t1.columns = df_chronic_t1.columns + '_length_time'
    df_chronic_t1['Patient_ID'] = c

    
    # second (time time since 2016 )
    df_chronic_t2 = df_chronic[['Patient_ID','ChronicIllness','time_since_2016']].pivot(index = 'Patient_ID', columns = 'ChronicIllness', values = 'time_since_2016').reset_index().fillna(value=0)
    c = df_chronic_t2.Patient_ID
    df_chronic_t2 = df_chronic_t2[chronics]
    df_chronic_t2.columns = df_chronic_t2.columns + '_time_since_2016'
    df_chronic_t2['Patient_ID'] = c


    
    # third ('never_on_drug' variable; 0 for on it, 1 for not on it, add 1 now)
    df_chronic_t3 = df_chronic[['Patient_ID','ChronicIllness','never_on_drug']].pivot(index = 'Patient_ID', columns = 'ChronicIllness', values = 'never_on_drug').reset_index().fillna(value=1)
    c = df_chronic_t3.Patient_ID
    df_chronic_t3 = df_chronic_t3[chronics]
    df_chronic_t3.columns = df_chronic_t3.columns + '_never_on'
    df_chronic_t3['Patient_ID'] = c

    
    # still got to get the count for ingredients 
    df_ingredients = df_merge.groupby(['Patient_ID','ChronicIllness'], as_index=False).count()
    df_ingredients.rename(columns={'Drug_ID':'count'}, inplace=True)
    df_ingredients.drop(['Dispense_Week'], inplace = True, axis=1)
    df_ingredients = df_ingredients.pivot(index = 'Patient_ID', columns = 'ChronicIllness', values = 'count').reset_index().fillna(value = 0)
    c = df_ingredients.Patient_ID
    df_ingredients = df_ingredients[ingredients]
    df_ingredients.columns = df_ingredients.columns + '_count'
    df_ingredients['Patient_ID'] = c

    
    # get time_since_first and time_since_last
    aggregations = {
        'Dispense_Week': { # work on the "Dispense_Week" column
            'min_date': 'min',  # get the min, and call this result 'min_date'
            'max_date': 'max', # 
        }
    }
    df_first_last = df_merge.groupby(['Patient_ID'], as_index=False).agg(aggregations)
    df_first_last.columns = ['Patient_ID','min_date','max_date']
    df_first_last['time_since_first'] = (pd.to_datetime("2016-01-02") - df_first_last.min_date).apply(lambda s : s.days/30)
    df_first_last['time_since_last'] = (pd.to_datetime("2016-01-02") - df_first_last.max_date).apply(lambda s : s.days/30)
    df_first_last.drop(['min_date','max_date'], axis = 1, inplace = True)

    
    
    # gender (one hot encode), ses, remote index (one hot encode).
    # remoteness.region_type.value_counts()
    # Major Cities of Australia    995
    # Outer Regional Australia     681
    # Inner Regional Australia     673
    # Remote Australia             189
    # Very Remote Australia        124

    # combine Outer Regional, Remote and Very Remote
    group1 = ['Outer Regional Australia','Remote Australia','Very Remote Australia']
    remo = remoteness.copy()
    remo.ix[remo.region_type.isin(group1), 'region_type'] = "remote"

    # replace ses na with average for that region_type
    remo.ses_code = pd.to_numeric(remo.ses_code, errors= 'coerce')
    av_ses_city = round(remo.ix[remo.region_type=='Major Cities of Australia', 'ses_code'].mean())
    av_ses_inner_reg = round(remo.ix[remo.region_type=='Inner Regional Australia', 'ses_code'].mean())
    av_ses_remote = round(remo.ix[remo.region_type=='remote', 'ses_code'].mean())
    remo.ix[remo.region_type=='Major Cities of Australia', 'ses_code'] = remo.ix[remo.region_type=='Major Cities of Australia', 'ses_code'].fillna(av_ses_city)
    remo.ix[remo.region_type=='Inner Regional Australia', 'ses_code'] = remo.ix[remo.region_type=='Inner Regional Australia', 'ses_code'].fillna(av_ses_inner_reg)
    remo.ix[remo.region_type=='remote', 'ses_code'] = remo.ix[remo.region_type=='remote', 'ses_code'].fillna(av_ses_remote)

    #pivot
    remo.drop(['percentage'], inplace=True, axis=1)
    remo['ones'] = 1
    atemp = remo.pivot(index = 'postcode', columns = 'region_type', values = 'ones').reset_index().fillna(value=0)
    atemp = pd.merge(left = atemp, right = remo[['postcode','ses_code']], how = 'right', on='postcode')

    # now add these to Patients_LookUp
    p_lookup = Patients_LookUp.copy()
    p_lookup.columns = ['Patient_ID', 'gender', 'year_of_birth', 'postcode']
    p_lookup.drop('year_of_birth',axis=1, inplace=True)
    p_lookup = pd.concat([p_lookup.drop('gender',axis=1),p_lookup.gender.str.get_dummies()],axis=1)
    # not every postcode as a remote or ses, put to zero, may not be best choice...
    p_lookup = pd.merge(left = p_lookup, right = atemp, how = 'left', on = 'postcode').fillna(value=0).drop('postcode', axis = 1)

    
    # from df_merge, get the Patient_ID and their outcome
    df_bigs = pd.merge(left=df_bigs, right=df_chronic_t1, how='inner', on='Patient_ID')
    df_bigs = pd.merge(left=df_bigs, right=df_chronic_t2, how='inner', on='Patient_ID')
    df_bigs = pd.merge(left=df_bigs, right=df_chronic_t3, how='inner', on='Patient_ID')
    df_bigs = pd.merge(left=df_bigs, right=df_ingredients, how='inner', on='Patient_ID')
    df_bigs = pd.merge(left=df_bigs, right=df_first_last, how='inner', on='Patient_ID')
    df_bigs = pd.merge(left=df_bigs, right=p_lookup, how='left', on='Patient_ID').fillna(value=0)

    
    # and done :)

    
    return df_bigs







# make a function for extracting data from the kaggle data # keep the 'Diabetes_in_2016' label even though it should be 2015
def extract_features_from_kaggle(df, df_missing, ChronicIllness_LookUp, Drug_LookUp, Patients_LookUp, remoteness):
    
    #append missing data
    df = df.append(df_missing, ignore_index=True)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # These are dates, so convert them from string to datetime
    df.Prescription_Week = pd.to_datetime(df.Prescription_Week)
    df.Dispense_Week = pd.to_datetime(df.Dispense_Week)
   
    # This merge (outer) connects chronicIllness to our data by mergeing the two data sets
    df_merge = pd.merge(left=ChronicIllness_LookUp[['ChronicIllness','MasterProductID']], right=df, how='outer', left_on='MasterProductID', right_on='Drug_ID')
    df_merge.ChronicIllness = df_merge.ChronicIllness.fillna(value = 'unknown')
    
    # get our 'Diabetes_in_2016' variable
    df_merge['in_2016'] = (df_merge.Dispense_Week>=pd.to_datetime("2015-01-01"))
    df_merge['Diabetes_in_2016'] = (df_merge.ChronicIllness=='Diabetes') & df_merge['in_2016']

    # prepare for output now, this is Patiend_ID and target label
    df_bigs = df_merge[['Patient_ID','Diabetes_in_2016']].groupby(['Patient_ID'], as_index=False).agg({'Diabetes_in_2016':np.any})
    df_bigs.Diabetes_in_2016 = df_bigs.Diabetes_in_2016.astype(int)
    
    # now to remove transactions after 2016
    df_merge = df_merge.ix[~df_merge.in_2016, ]
    
    # add a GenericIngreintName column (use this to update some ChronicIllness unknowns.)
    df_merge = pd.merge(left=Drug_LookUp[['GenericIngredientName','MasterProductID']], right=df_merge, how='right', left_on='MasterProductID', right_on='Drug_ID')
    
    
    # defin these for later
    ingredients = ['PARACETAMOL','PERINDOPRIL','ESOMEPRAZOLE','PANTOPRAZOLE','RAMIPRIL','AMLODIPINE','OMEPRAZOLE',
               'CLOPIDOGREL','LERCANIDIPINE','RABEPRAZOLE','SALBUTAMOL CFC FREE','unknown']

    chronics = ['Diabetes', 'Osteoporosis', 'Depression', 'Epilepsy', 'Lipids','Hypertension', 'Heart Failure',
            'Anti-Coagulant', 'Immunology','Urology', 'Chronic Obstructive Pulmonary Disease (COPD)']

    
    # replace half of the 'unknown's with the top 10 (or 11 or 12) top ingredients for the unknowns
    update_chronicIllnes_mask = (df_merge.ChronicIllness=='unknown') & (df_merge.GenericIngredientName.isin(ingredients))
    df_merge.ix[update_chronicIllnes_mask,'ChronicIllness'] = df_merge.ix[update_chronicIllnes_mask,'GenericIngredientName']

    # reduce df_merge
    df_merge = df_merge[['ChronicIllness', 'Patient_ID', 'Drug_ID','Dispense_Week']]
    
    # now to the ChronicIllness pivot
    aggregations = {
        'Dispense_Week': { # work on the "Dispense_Week" column
            'min_date': 'min',  # get the min, and call this result 'min_date'
            'max_date': 'max', # 
        }
    }
    df_chronic = df_merge.groupby(['Patient_ID','ChronicIllness'], as_index=False).agg(aggregations)
    df_chronic.columns = np.concatenate([df_chronic.columns.get_level_values(0)[range(0,2)],df_chronic.columns.get_level_values(1)[range(2,4)]])
    
    # make some new variables
    df_chronic['time_on_drug'] = (df_chronic.max_date -  df_chronic.min_date ).apply(lambda s : (s.days/30)+1 )
    df_chronic['time_since_2016'] = (pd.to_datetime("2015-01-02") - df_chronic.max_date).apply(lambda s : s.days/30)
    # now add the 'never_on_drug' variable; 0 for on it, 1 for not on it, add 1 later
    df_chronic['never_on_drug'] = 0
    
    # first one (time on drug)
    df_chronic_t1 = df_chronic[['Patient_ID','ChronicIllness','time_on_drug']].pivot(index = 'Patient_ID', columns = 'ChronicIllness', values = 'time_on_drug').reset_index().fillna(value=0)
    c = df_chronic_t1.Patient_ID
    df_chronic_t1 = df_chronic_t1[chronics]
    df_chronic_t1.columns = df_chronic_t1.columns + '_length_time'
    df_chronic_t1['Patient_ID'] = c

    
    # second (time time since 2016 )
    df_chronic_t2 = df_chronic[['Patient_ID','ChronicIllness','time_since_2016']].pivot(index = 'Patient_ID', columns = 'ChronicIllness', values = 'time_since_2016').reset_index().fillna(value=0)
    c = df_chronic_t2.Patient_ID
    df_chronic_t2 = df_chronic_t2[chronics]
    df_chronic_t2.columns = df_chronic_t2.columns + '_time_since_2016'
    df_chronic_t2['Patient_ID'] = c


    
    # third ('never_on_drug' variable; 0 for on it, 1 for not on it, add 1 now)
    df_chronic_t3 = df_chronic[['Patient_ID','ChronicIllness','never_on_drug']].pivot(index = 'Patient_ID', columns = 'ChronicIllness', values = 'never_on_drug').reset_index().fillna(value=1)
    c = df_chronic_t3.Patient_ID
    df_chronic_t3 = df_chronic_t3[chronics]
    df_chronic_t3.columns = df_chronic_t3.columns + '_never_on'
    df_chronic_t3['Patient_ID'] = c

    
    # still got to get the count for ingredients 
    df_ingredients = df_merge.groupby(['Patient_ID','ChronicIllness'], as_index=False).count()
    df_ingredients.rename(columns={'Drug_ID':'count'}, inplace=True)
    df_ingredients.drop(['Dispense_Week'], inplace = True, axis=1)
    df_ingredients = df_ingredients.pivot(index = 'Patient_ID', columns = 'ChronicIllness', values = 'count').reset_index().fillna(value = 0)
    c = df_ingredients.Patient_ID
    df_ingredients = df_ingredients[ingredients]
    df_ingredients.columns = df_ingredients.columns + '_count'
    df_ingredients['Patient_ID'] = c

    
    # get time_since_first and time_since_last
    aggregations = {
        'Dispense_Week': { # work on the "Dispense_Week" column
            'min_date': 'min',  # get the min, and call this result 'min_date'
            'max_date': 'max', # 
        }
    }
    df_first_last = df_merge.groupby(['Patient_ID'], as_index=False).agg(aggregations)
    df_first_last.columns = ['Patient_ID','min_date','max_date']
    df_first_last['time_since_first'] = (pd.to_datetime("2015-01-02") - df_first_last.min_date).apply(lambda s : s.days/30)
    df_first_last['time_since_last'] = (pd.to_datetime("2015-01-02") - df_first_last.max_date).apply(lambda s : s.days/30)
    df_first_last.drop(['min_date','max_date'], axis = 1, inplace = True)

    
    
    # gender (one hot encode), ses, remote index (one hot encode).
    # remoteness.region_type.value_counts()
    # Major Cities of Australia    995
    # Outer Regional Australia     681
    # Inner Regional Australia     673
    # Remote Australia             189
    # Very Remote Australia        124

    # combine Outer Regional, Remote and Very Remote
    group1 = ['Outer Regional Australia','Remote Australia','Very Remote Australia']
    remo = remoteness.copy()
    remo.ix[remo.region_type.isin(group1), 'region_type'] = "remote"

    # replace ses na with average for that region_type
    remo.ses_code = pd.to_numeric(remo.ses_code, errors= 'coerce')
    av_ses_city = round(remo.ix[remo.region_type=='Major Cities of Australia', 'ses_code'].mean())
    av_ses_inner_reg = round(remo.ix[remo.region_type=='Inner Regional Australia', 'ses_code'].mean())
    av_ses_remote = round(remo.ix[remo.region_type=='remote', 'ses_code'].mean())
    remo.ix[remo.region_type=='Major Cities of Australia', 'ses_code'] = remo.ix[remo.region_type=='Major Cities of Australia', 'ses_code'].fillna(av_ses_city)
    remo.ix[remo.region_type=='Inner Regional Australia', 'ses_code'] = remo.ix[remo.region_type=='Inner Regional Australia', 'ses_code'].fillna(av_ses_inner_reg)
    remo.ix[remo.region_type=='remote', 'ses_code'] = remo.ix[remo.region_type=='remote', 'ses_code'].fillna(av_ses_remote)

    #pivot
    remo.drop(['percentage'], inplace=True, axis=1)
    remo['ones'] = 1
    atemp = remo.pivot(index = 'postcode', columns = 'region_type', values = 'ones').reset_index().fillna(value=0)
    atemp = pd.merge(left = atemp, right = remo[['postcode','ses_code']], how = 'right', on='postcode')

    # now add these to Patients_LookUp
    p_lookup = Patients_LookUp.copy()
    p_lookup.columns = ['Patient_ID', 'gender', 'year_of_birth', 'postcode']
    p_lookup.drop('year_of_birth',axis=1, inplace=True)
    p_lookup = pd.concat([p_lookup.drop('gender',axis=1),p_lookup.gender.str.get_dummies()],axis=1)
    # not every postcode as a remote or ses, put to zero, may not be best choice...
    p_lookup = pd.merge(left = p_lookup, right = atemp, how = 'left', on = 'postcode').fillna(value=0).drop('postcode', axis = 1)

    
    # from df_merge, get the Patient_ID and their outcome
    df_bigs = pd.merge(left=df_bigs, right=df_chronic_t1, how='inner', on='Patient_ID')
    df_bigs = pd.merge(left=df_bigs, right=df_chronic_t2, how='inner', on='Patient_ID')
    df_bigs = pd.merge(left=df_bigs, right=df_chronic_t3, how='inner', on='Patient_ID')
    df_bigs = pd.merge(left=df_bigs, right=df_ingredients, how='inner', on='Patient_ID')
    df_bigs = pd.merge(left=df_bigs, right=df_first_last, how='inner', on='Patient_ID')
    df_bigs = pd.merge(left=df_bigs, right=p_lookup, how='left', on='Patient_ID').fillna(value=0)

    
    # and done :)

    
    return df_bigs






# from stack overflow
# http://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.ix[perm[:train_end]]
    validate = df.ix[perm[train_end:validate_end]]
    test = df.ix[perm[validate_end:]]
    return train, validate, test