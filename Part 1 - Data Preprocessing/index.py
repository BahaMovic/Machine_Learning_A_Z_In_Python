
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Data.csv')
indep = dataset.iloc[:,:-1].values
dep = dataset.iloc[:,3].values
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(indep[:,1:3])
indep[:,1:3] =  imputer.transform(indep[:,1:3])

labelEncoder = LabelEncoder()
labelEncoder = labelEncoder.fit(indep[:,0])
indep[:,0] = labelEncoder.transform(indep[:,0])

oneHotEncoder =  OneHotEncoder(categorical_features=[0])
indep = oneHotEncoder.fit_transform(indep).toarray()

dep = labelEncoder.fit_transform(dep)
#dep = oneHotEncoder.fit_transform(dep)
#print(dep)

indep_traning , indep_test , dep_traning , dep_test = train_test_split(indep,dep,test_size=0.2,random_state =10)

indep_SC = StandardScaler()
indep_test2 = indep_test
indep_traning = indep_SC.fit_transform(indep_traning)
indep_test = indep_SC.transform(indep_test2)


