# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 22:37:24 2021

@author: bakayoko
"""
#WE TRY TO CREATE A FRONT END UI USING FLASGGER AND SWAGGER
from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger#it will automatically generate a front end API


"""the first step when we use flak"""

app = Flask(__name__)#instanciate the flask object 
Swagger(app)
pickle_in=open("classifier.pkl","rb")#open the ML model
classifier=pickle.load(pickle_in)


@app.route('/foobarbaby')#il s'agit d'un declarateur pour que tout ceci fonctionne sur flask
def welcome():
    return'welcome class'

@app.route('/predict',methods=["GET"])
def predict():
    
    #we just create the fields of our web app there
    """Banks note authentification
    This is using docstring for specifications.
    ---
    parameters:
        - name: variance
          type: number
          in: query
          required: true
          
        - name: skewness
          type: number
          in: query
          required:true
        
        - name: curtosis
          type: number
          in: query
          required:true
        
        - name: entropy
          type: number
          in: query
          required:true
    responses:
        200:
            description:The outputs values
    """
    variance=request.args.get('variance	')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    pred=classifier.predict([[variance,skewness,curtosis,entropy]])
    return'the value predicted is'+str(pred)
    
    
@app.route('/predict_file',methods=["POST"])#we use post method because we have a lot of data
def predict_file():
    
    
    """ Bannks note authentification
    This is using docstring for specifications
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
         
    responses:
        200:
            description:The outputs values
    """
        
    
    test=pd.read_csv(request.files.get("file"))
    pred=classifier.predict([[test]])
    return'the value predicted is'+str(list(pred))
    
    

if __name__ == '__main__':
    app.run()
    