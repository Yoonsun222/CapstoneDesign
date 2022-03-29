from django.shortcuts import render
import joblib
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
import numpy as np
import json
import sys
import os

# Load the model from the static folder

loaded_class_model = joblib.load(open('C:/Users/profi/Desktop/2022캡디/capstone_design/ml_model/ml_class_model.pkl', 'rb'))


# Create your views here.
@api_view(['GET'])
def index(request):
    return_data = {
        "error_code" : "0",
        "info" : "success",
    }
    return Response(return_data)

@api_view(["POST"])
def predict_patient_status(request):
    try:
        patient_json_info = request.data
        patient_info = np.array(list(patient_json_info.values()))

        patient_status = loaded_class_model.predict([patient_info])
        model_confidence_score=  np.max(loaded_class_model.predict_proba([patient_info]))
        
        model_prediction = {
            'info': 'success',
            'patient_status': patient_status[0],
            'model_confidence_proba': float("{:.2f}".format(model_confidence_score*100))
        }

    except ValueError as ve:
        model_prediction = {
            'error_code' : '-1',
            "info": str(ve)
        }

    return Response(model_prediction)