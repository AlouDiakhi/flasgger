from flask import Flask,request
import pandas as pd
import numpy as np
import pickle

"""the first stzp when we use flak"""

app = Flask(__name__)#instanciate the flask object 

pickle_in=open("classifier.pkl","rb")#open the ML model
classifier=pickle.load(pickle_in)


@app.route('/foobarbaby')#il s'agit d'un declarateur pour que tout ceci fonctionne sur flask
def welcome():
    return'welcome class'

@app.route('/predict')
def predict():
    variance=request.args.get('variance	')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    pred=classifier.predict([[variance,skewness,curtosis,entropy]])
    return'the value predicted is'+str(pred)
    
    
@app.route('/predict_file',methods=["POST"])#we use post method because we have a lot of data
def predict_file():
     test=pd.read_csv(request.files.get("file"))
     pred=classifier.predict([[test]])
     return'the value predicted is'+str(list(pred))
    
    

if __name__ == '__main__':
    app.run()
    
    
    
    
    
    
    
    
    
    