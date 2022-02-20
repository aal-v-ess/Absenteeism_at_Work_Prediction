#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
import pickle

class absenteeism_model_fs():
    
    def __init__(self, model_file, scaler_file):
        with open('model_fs', 'rb') as model_file, open('scaler_fs', 'rb') as scaler_file:
            self.reg_fs = pickle.load(model_file)
            self.scaler_fs = pickle.load(scaler_file)
            self.data = None

    def load_and_clean_data(self, data_file):

        df = pd.read_csv(data_file, sep=';')
        self.df_with_predictions = df.copy()
        df.drop(['ID'], axis=1, inplace=True)
        df['Absenteeism time in Hours'] = 'NaN'
        reason_columns = pd.get_dummies(df['Reason for absence'], drop_first=True)
        df.drop(columns=['Reason for absence'], inplace=True)

        reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1) # diseases
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1) # Injuries
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1) # Pregnancy related
        reason_type_4 = reason_columns.loc[:, 22:28].max(axis=1) # routine stuff

        df = pd.concat([df, reason_type_1, reason_type_3, reason_type_3, reason_type_4], axis=1)

        df.rename(columns={0:'Reason_1', 1:'Reason_2', 2:'Reason_3', 3:'Reason_4'}, inplace=True)
        
        cols = ['Reason_1', 'Reason_2', 'Reason_3','Reason_4', 'Month of absence', 'Day of the week', 'Seasons', 
                'Transportation expense', 'Distance from Residence to Work','Service time', 'Age', 
                'Work load Average/day ', 'Hit target', 'Disciplinary failure', 'Education', 'Son', 'Social drinker',
                'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index', 'Absenteeism time in hours']
        df = df[cols]
        
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})

        targets = np.where(df['Absenteeism time in hours'] > df['Absenteeism time in hours'].median(), 1, 0)

        df.drop(['Absenteeism time in hours'], axis=1, inplace=True)

        fs_feature_selector = SequentialFeatureSelector(LogisticRegression(n_jobs=-1),
                k_features=10,
                forward=True,
                verbose=2,
                scoring='roc_auc',
                cv=4)
        
        fs_features = fs_feature_selector.fit(df.iloc[:,:-1], targets)
        fs_filtered_features = df.iloc[:,:-1].columns[list(fs_features.k_feature_idx_)]
        df = df[fs_filtered_features]
        
        self.preprocessed_data = df.copy()
        
        self.data = self.scaler_fs.transform(df)
        
    def predicted_probability(self):
        if (self.data is not None):
            pred = self.reg_fs.predic_proba(self.data)[:,1]
            return pred
        
    def predicted_output_category(self):
        if(self.data is not None):
            pred_outputs = self.reg_fs.predict(self.data)
            return pred_outputs
        
    def predicted_outputs(self):
        if(self.data is not None):
            self.preprocessed_data['Probability'] = self.reg_fs.predict_proba(self.data)[:,1]
            self.preprocessed_data['Prediction'] = self.reg_fs.predict(self.data)
            return self.preprocessed_data
        
        
        
    
    
    
    
    

