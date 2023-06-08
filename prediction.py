import joblib

def predict(data):
    pipe = joblib.load("rf_model.sav")
    return pipe.predict(data)