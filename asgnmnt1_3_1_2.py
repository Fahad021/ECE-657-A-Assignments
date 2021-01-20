# Load libraries
import pandas as pd
import numpy as np
from numpy import set_printoptions
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn import manifold


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

#putting only digit-3 values in data2
data2= data.iloc[1237:1635]
data2= data2.reset_index()
cols = [0]
data2.drop(data2.columns[cols],axis=1,inplace=True)
#print(df3)

# separating last coloumn for future plotting 
targets = data2['gnd'].astype('category') # targets.unique()
#from data-2 and putting all features into df1
df1 = data2.drop(labels=['gnd'], axis=1)

#--------------------------------------------------------------------------------------
##L.L.E.
locline = manifold.LocallyLinearEmbedding(n_neighbors=5, n_components=4)
locline.fit(df1)
manifold_4Db = locline.transform(df1)
manifold_4D2 = pd.DataFrame(manifold_4Db, columns=['Component 1', 'Component 2','Component 3', 'Component 4'])
manifold_4D2= manifold_4D2.join(targets)
#print (manifold_4D2)
#palette = sns.color_palette("bright",5)  #Choosing color
#sns.scatterplot(x="Component 1", y="Component 2", data= manifold_4D2, palette=palette, hue="gnd",legend='full')
sns.scatterplot(x="Component 1", y="Component 2", data= manifold_4D2, hue="gnd",legend='full')
plt.savefig("Locally linear embedding.png")
plt.close()


##---------------------------------------------------------------------------------------
#isomap
iso = manifold.Isomap(n_neighbors=5, n_components=4)
iso.fit(df1)
manifold_4Da = iso.transform(df1)
manifold_4D1 = pd.DataFrame(manifold_4Da, columns=['Component 1', 'Component 2','Component 3', 'Component 4'])
manifold_4D1= manifold_4D1.join(targets)
#print (manifold_4D1)
#palette = sns.color_palette("bright",5)  #Choosing color
#sns_plot1= sns.scatterplot(x="Component 1", y="Component 2", data= manifold_4D1, palette=palette, hue="gnd",legend='full')
sns.scatterplot(x="Component 1", y="Component 2", data= manifold_4D1, hue="gnd",legend='full')
plt.savefig("isomap.png")
plt.close()


