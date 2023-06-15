#This prediction is intended to import the model from the saved model rf_model.sav and try to predict the output that should be shown on the user screen.
import joblib

def predict(data):
    pipe = joblib.load("rf_model.sav")
    return pipe.predict(data)