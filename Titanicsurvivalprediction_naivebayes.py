##import the Libraries
import pandas as pd
import numpy as np

##Load he dataset & Summarize
dataset=pd.read_csv("titanicsurvival.csv")

# print(dataset.shape)
# print(dataset.head(6))

##Mapping data  to binary values
gender_set = set(dataset['Gender']) #get the unigue values 
# print(gender_set) # output -{'male', 'female'}
dataset['Gender'] = dataset['Gender'].map({'female':0, 'male':1}).astype(int)
# print(dataset.head(6))


##Segregating the dataset into X & Y
# X=dataset.iloc[:,:-1] #rows,Column
# Y=dataset.iloc[:,-1] #iloc -index location 
X=dataset.drop('Survived',axis="columns")
Y=dataset['Survived'] # dataset.Survived
# print(X.head(6))
# print(Y)

##Finding & Removing NA values from our Features X
X.columns[X.isna().any()]
print(X.isna().any())
X.Age=X.Age.fillna(X.Age.mean()) 
print("Test again to Ckeck ang NA  Values")
print(X.isna().any())

##Splitting Dataset to train and test
from sklearn.model_selection import train_test_split

X_train,X_test,y_train ,y_test =train_test_split(X,Y,test_size=0.2,random_state=0)

#Model Training 
from sklearn.naive_bayes import GaussianNB
model_NB=GaussianNB()
model_NB.fit(X_train,y_train)

#Prediction for all test data  
y_pred=model_NB.predict(X_test)
# print(np.column_stack((y_pred,y_test)))


#Accuracy of our model 
from sklearn.metrics import accuracy_score
print("Accuracy of the model:{}%".format(accuracy_score(y_test,y_pred)*100))


#Predicting , wheather Person Survived or Not 
Pclass=int(input("Enter Person's Pclass Number:")) 
Gender=int(input("Enter Person's Gender 0-female 1-male(0 or 1):"))
Age=int(input("Enter Person's Age:"))
Fare=float(input("ENter Person's Fare:"))

person=[[Pclass,Gender,Age,Fare]]
result=model_NB.predict(person)

print(result)

if result==1:
    print("Reson might be Survived.")
else:
    print("Person might not be Survived")
