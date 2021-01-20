# Load libraries
import pandas as pd
import numpy as np
from numpy import set_printoptions
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
#scikitlearn libraries
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import manifold

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#-----------------------------------------------------------------------------------
#read data
data=pd.read_csv('DataB.csv') 
print(data.shape)
#print(data.head)

### Summarize the Dataset
## shape
#print(data.shape)
## head
#print(data.head(20))
## descriptions
#print(data.describe())
## class distribution
#print(data.groupby('gnd').size())

#deleting first coloumn
cols = [0]
data.drop(data.columns[cols],axis=1,inplace=True)
#print(data)
# separating last coloumn for future plotting 
targets = data['gnd'].astype('category') # targets.unique()

#drop the classification column
df1 = data.drop(labels=['gnd'], axis=1)

#--------------------------------------------------------------------------------------
#L.L.E. dim reduction
locline = manifold.LocallyLinearEmbedding(n_neighbors=5, n_components=4)
locline.fit(df1)
manifold_4Db = locline.transform(df1)
manifold_4D2 = pd.DataFrame(manifold_4Db, columns=['Component 1', 'Component 2','Component 3', 'Component 4'])
manifold_4D2= manifold_4D2.join(targets)
#print (manifold_4D2)
palette = sns.color_palette("bright",5)  #Choosing color
sns.scatterplot(x="Component 1", y="Component 2", data= manifold_4D2, palette=palette, hue="gnd",legend='full')
plt.savefig("Locally linear embedding.png")
plt.close()

##---------------------------------------------------------------------------------------
#isomap dim reduction
iso = manifold.Isomap(n_neighbors=5, n_components=4)
iso.fit(df1)
manifold_4Da = iso.transform(df1)
manifold_4D1 = pd.DataFrame(manifold_4Da, columns=['Component 1', 'Component 2','Component 3', 'Component 4'])
manifold_4D1= manifold_4D1.join(targets)
#print (manifold_4D1)
palette = sns.color_palette("bright",5)  #Choosing color
sns_plot1= sns.scatterplot(x="Component 1", y="Component 2", data= manifold_4D1, palette=palette, hue="gnd",legend='full')
plt.savefig("isomap.png")
plt.close()

#------------------------------------------------------------------------------------
#  PCA dim reduction
array = df1.values
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(array)
df_pca = pd.DataFrame(data = principalComponents, columns=['Component 1', 'Component 2','Component 3', 'Component 4'])
df_pca= df_pca.join(targets)
palette = sns.color_palette("bright",5)  #Choosing color
sns_plot1= sns.scatterplot(x="Component 1", y="Component 2", data= df_pca, palette=palette, hue="gnd",legend='full')
plt.savefig("PCA.png")
plt.close()


#---------------------------------------------------------------------------------
#LDA dim reduction
arrayX = df1.values
arrayY= targets.values
lda = LinearDiscriminantAnalysis(n_components=4)
ldacomponents= lda.fit_transform(arrayX,arrayY)
df_lda = pd.DataFrame(data = ldacomponents , columns=['Component 1', 'Component 2','Component 3', 'Component 4'])
df_lda= df_lda.join(targets)
palette = sns.color_palette("bright",5)  #Choosing color
sns_plot1= sns.scatterplot(x="Component 1", y="Component 2", data=df_lda , palette=palette, hue="gnd",legend='full')
plt.savefig("LDA.png")
plt.close()
##---------------------

###------------------------------------------------------------------------
#Naive Baytes classification after LLE feature-reduction
# descriptions
print(manifold_4D2.describe())
array = manifold_4D2.values
# separate array into input and output components
X = array[:,0:4]
Y = array[:,4]
Y=Y.astype('int') #Added due to Unknown label type
#Normalize the data
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
## summarize transformed data
#set_printoptions(precision=3)
#print(normalizedX)
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(normalizedX, Y, test_size=0.30)
print(X_train.shape, y_train.shape)
#Create a Gaussian NB Classifier
model = GaussianNB()
# Train the model using the training sets
model.fit(X_train,y_train)
#Predict Output
y_pred= model.predict(X_test) 
# Model Accuracy, how often is the classifier correct?
print("LLE-reduced Accuracy:",metrics.accuracy_score(y_test, y_pred))
target_names = ['0', '1', '2','3','4']
print('classification report...\n')
print(classification_report(y_test, y_pred, target_names=target_names))
print('confusion matrix...\n')
print(confusion_matrix(y_test, y_pred))

print('Repeated Random Test-Train Splits....\n')
n_splits = 10
test_size = 0.30
seed = 7
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
results = cross_val_score(model, normalizedX, Y, cv=kfold)
print("LLE-reduced Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
print('....')
###-------------------------------------------------------------------------------------------

####----------------------------------------------------------------------
#Naive Baytes classifier after isomap feature-reduction
# descriptions
print(manifold_4D1.describe())
array = manifold_4D1.values
# separate array into input and output components
X = array[:,0:4]
Y = array[:,4]
Y=Y.astype('int') #Added due to Unknown label type
#Normalize the data
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
## summarize transformed data
#set_printoptions(precision=3)
#print(normalizedX)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(normalizedX, Y, test_size=0.30)
#print(X_train.shape, y_train.shape)
#Create a Gaussian NB Classifier
model = GaussianNB()
# Train the model using the training sets
model.fit(X_train,y_train)
#Predict Output
y_pred= model.predict(X_test) 
# Model Accuracy, how often is the classifier correct?
print("ISOMAP-reduced Accuracy:",metrics.accuracy_score(y_test, y_pred))
target_names = ['0', '1', '2','3','4']
print('classification report...\n')
print(classification_report(y_test, y_pred, target_names=target_names))
print('confusion matrix...\n')
print(confusion_matrix(y_test, y_pred))

print('Repeated Random Test-Train Splits....\n')
n_splits = 10
test_size = 0.30
seed = 7
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
results = cross_val_score(model, normalizedX, Y, cv=kfold)
print("ISOMAP-redcued Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
print('....')
###---------------------------------------------------------------------------------------
#
#Naive Baytes classifier after PCA feature-reduction
# descriptions
print(df_pca.describe())
array = df_pca.values
# separate array into input and output components
X = array[:,0:4]
Y = array[:,4]
Y=Y.astype('int') #Added due to Unknown label type
#Normalize the data
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
## summarize transformed data
#set_printoptions(precision=3)
#print(normalizedX)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(normalizedX, Y, test_size=0.30)
#print(X_train.shape, y_train.shape)
#Create a Gaussian NB Classifier
model = GaussianNB()
# Train the model using the training sets
model.fit(X_train,y_train)
#Predict Output
y_pred= model.predict(X_test) 
# Model Accuracy, how often is the classifier correct?
print("PCA-reduced Accuracy:",metrics.accuracy_score(y_test, y_pred))
target_names = ['0', '1', '2','3','4']
print('classification report...\n')
print(classification_report(y_test, y_pred, target_names=target_names))
print('confusion matrix...\n')
print(confusion_matrix(y_test, y_pred))

print('Repeated Random Test-Train Splits...\n')
n_splits = 10
test_size = 0.30
seed = 7
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
results = cross_val_score(model, normalizedX, Y, cv=kfold)
print("PCA-redcued Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
print('....')
#----------------------------------------------------------------------------------------

#Naive Baytes classifier after LDA feature-reduction
# descriptions
print(df_lda.describe())
array = df_lda.values
# separate array into input and output components
X = array[:,0:4]
Y = array[:,4]
Y=Y.astype('int') #Added due to Unknown label type
#Normalize the data
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
## summarize transformed data
#set_printoptions(precision=3)
#print(normalizedX)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(normalizedX, Y, test_size=0.30)
#print(X_train.shape, y_train.shape)
#Create a Gaussian NB Classifier
model = GaussianNB()
# Train the model using the training sets
model.fit(X_train,y_train)
#Predict Output
y_pred= model.predict(X_test) 
# Model Accuracy, how often is the classifier correct?
print("LDA-reduced Accuracy:",metrics.accuracy_score(y_test, y_pred))
target_names = ['0', '1', '2','3','4']
print('classification report...\n')
print(classification_report(y_test, y_pred, target_names=target_names))
print('confusion matrix...')
print(confusion_matrix(y_test, y_pred))


print('Repeated Random Test-Train Splits...]\n')
n_splits = 10
test_size = 0.30
seed = 7
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
results = cross_val_score(model, normalizedX, Y, cv=kfold)
print("LDA-reduced Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
