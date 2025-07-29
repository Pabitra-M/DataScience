import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## Importing the model
ridgereg_model=pickle.load(open("models/ridgereg.pkl","rb"))
standerd_scaler=pickle.load(open("models/scaler.pkl","rb"))





@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET' , 'POST'])
def predict():
    if request.method == 'POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Class = float(request.form.get('Class'))
        Region=float(request.form.get('Region'))


    

        data=np.array([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Class,Region]])
        new_data=standerd_scaler.transform(data)
        output=ridgereg_model.predict(new_data)
        return render_template('home.html',result=output[0])
    else :
        return render_template('home.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0')