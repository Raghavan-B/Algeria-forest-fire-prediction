from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

#importing ridge regressor and standard scaler pickle
ridge_model = pickle.load(open("models/ridge_model.pkl","rb"))
standardscaler = pickle.load(open("models/standard_scaler.pkl","rb"))


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods = ["GET","POST"])
def predict_datapoint():
    if request.method == "POST":
        ##Getting all the input values
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        WS = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        data_scaled = standardscaler.transform([[Temperature,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = np.round(ridge_model.predict(data_scaled),2)

        #Displaying the results
        return render_template("home.html",result = result[0])    
    else:    
       return render_template("home.html")


if __name__ == "__main__":
    app.debug = True
    app.run()

