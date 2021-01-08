

# -*- coding: utf-8 -*-
"""riceleafsvm1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11DuKjqwI_Uu3KukjRkAaLZm3DPIC1nc7
"""
'''
# Commented out IPython magic to ensure Python compatibility.
#!pip install ipython-autotime
# %load_ext autotime

#!pip install bing-image-downloader

from bing_image_downloader import downloader
downloader.download("blast rice leaf", limit=30, output_dir='images',adult_filter_off=True)

from bing_image_downloader import downloader
downloader.download("blight rice leaf", limit=30, output_dir='images',adult_filter_off=True)

from bing_image_downloader import downloader
downloader.download("brownspot rice leaf", limit=30, output_dir='images',adult_filter_off=True)

'''

"""Preprocessing
1. resizing
2. flattening
"""



import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import json

target=[]
images=[]
flat_data=[]

DATADIR='D:/Python/Python Programs/Classification/multiclass image classification from net/images1'
CATEGORIES=[ "blast rice leaf", "rice bacterial leaf blight", "rice brown spot", "rice false smut","rice sheath blight" ]

for category in CATEGORIES:
  class_num=CATEGORIES.index(category) #label encoding the values
  path=os.path.join(DATADIR, category) #Create path to all images
  print(path)
  for img in os.listdir(path):
    img_array=imread(os.path.join(path,img))
    #print(img_array.shape)
    #plt.imshow(img_array)
    img_resized=resize(img_array,(150,150,3)) #Normalizes the value from 0 to 1
    flat_data.append(img_resized.flatten())
    images.append(img_resized)
    target.append(class_num)

flat_data=np.array(flat_data)
images=np.array(images)
target=np.array(target)

#np.save('c:\\images.npy',images)
#np.save('target.npy',target)

#flat_data
#len(flat_data[0])             #150*150*3

#target

unique,count=np.unique(target, return_counts=True)

plt.bar(CATEGORIES,count)

"""#Split Data into Training and Testing"""

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(flat_data, target, test_size=0.05, random_state=109)

param_grid = [
              { 'C':[1, 10, 100, 1000], 'kernel':['linear','poly']},
              { 'C':[2, 20, 200, 2000], 'gamma':[0.001, 0.0001, .00001], 'kernel':['rbf']}           
]

clf = GridSearchCV(svm.SVC(probability=True, gamma='scale'), param_grid)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print(accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

"""#Save model using pickle library"""

import pickle
#pickle.dump(clf, open('img_model.p','wb'))
file = open('img_model.p', 'wb')
pickle.dump(clf, file)
file.close()


#model=pickle.load(open('img_model.p','rb'))
file = open('img_model.p', 'rb')
model = pickle.load(file)
file.close()

# Testing a brand new image
flat_data=[]
url=input('Enter your url')
img = imread(url)
img_resized=resize(img, (150,150,3))
flat_data.append(img_resized.flatten())
flat_data=np.array(flat_data)
print(img.shape)
plt.imshow(img_resized)
y_out=model.predict(flat_data)
y_out=CATEGORIES[y_out[0]]
print(f'PREDICTED OUTPUT:{y_out}')
a=json.dumps(y_out)
