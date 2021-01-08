# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:37:23 2020

@author: anany
"""


import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image



st.set_option("deprecation.showfileUploaderEncoding",False)
st.title("Image Classifier using Machine Learning")
st.text('Upload the Image')

model=pickle.load(open("img_model.p","rb"))

upload_file=st.file_uploader("Choose an image", type="jpg")
if upload_file is not None:
  img=Image.open(upload_file)
 # img_resized=resize(img, (150,150,3))
  st.image(img, caption='Uploaded Image')
  
  if (st.button('PREDICT')):
    CATEGORIES=["blast rice leaf", "rice bacterial leaf blight", "rice brown spot", "rice false smut","rice sheath blight"]
    st.write('Result....')
    flat_data=[]
    img=np.array(img)
    img_resized=resize(img, (150,150,3))
    flat_data.append(img_resized.flatten())
    flat_data=np.array(flat_data)
    y_out=model.predict(flat_data)
    y_out=CATEGORIES[y_out[0]]
    st.title(f'PREDICTED OUTPUT:{y_out}')
    q=model.predict_proba(flat_data)
    for index, item in enumerate(CATEGORIES):
      st.write(f'{item}:{q[0][index]*100}%')
