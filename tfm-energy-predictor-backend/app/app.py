import io
import json
import sys
import traceback

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from flask import Request
from flask_sqlalchemy import SQLAlchemy
from starlette.responses import JSONResponse
from werkzeug.exceptions import NotFound

from app.config import Config
from app.dtos_objects import DateRange, EmissionsPrediction
from app.services import hugging_face_req
from .database import engine, Base

if sys.version_info < (3, 7):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
else:
    sys.stdout.reconfigure(encoding='utf-8')

"""
Initialize the app
"""

config = Config()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:8888"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
db = SQLAlchemy()
Base.metadata.create_all(bind=engine)


def generate_data():
    hours = [f"{hour:02d}h" for hour in range(24)]
    real = np.random.rand(24) * 100
    predicted = np.random.rand(24) * 100
    df = pd.DataFrame({
        "time": hours,
        "real": real,
        "predicted": predicted
    })
    return df


def build_response(status_code, message, data=None):
    content = {'status': status_code, 'message': message, 'content': {}}
    if data is not None:
        if data['status'] == 200:
            content['content'] = data['content']
        else:
            content['status'] = data['status']
            content['message'] = 'Error predicting...'
    return JSONResponse(status_code=status_code, content=content)


def do_req_hugging_face(params, req: str):
    result = hugging_face_req(params, req)
    return build_response(200, 'OK', data=result)


@app.get("/api/config")
def get_config():
    return {
        "SQLALCHEMY_DATABASE_URI": config.SQLALCHEMY_DATABASE_URI,
        "SQLALCHEMY_TRACK_MODIFICATIONS": config.SQLALCHEMY_TRACK_MODIFICATIONS
    }


@app.get("/api")
def default_url():
    return build_response(200, 'OK')


@app.post("/api/emissions")
def predict_emissions(date_range: DateRange):
    """
    Predict emissions
    :param date_range: The date range to predict emissions
    :return: The emissions prediction
    """
    return do_req_hugging_face(date_range, 'emissions')


@app.post("/api/simu-emissions")
def simulate_emissions(input_parameters: EmissionsPrediction):
    """
    Simulate emissions for the given date range and the given predictor columns
    :param input_parameters: The date range and energy variables to predict emissions
    :return: The emissions prediction
    """
    return do_req_hugging_face(input_parameters, 'simu-emissions')


@app.post("/api/get-predictor-cols-emissions")
def get_predictor_emissions(date_range: DateRange):
    """
    Retrieve the predictor columns values for the given date range
    :param date_range: The date range to retrieve the predictor columns values
    :return: The predictor columns values for the given date range
    """
    return do_req_hugging_face(date_range, 'get-predictor-cols-emissions')


@app.post("/api/generation")
def predict_generation(date_range: DateRange):
    """
    Predict emissions
    :param date_range: The date range to predict emissions
    :return: The emissions prediction
    """
    return do_req_hugging_face(date_range, 'generation')


@app.post("/api/demand")
def predict_demand(date_range: DateRange):
    """
    Predict emissions
    :param date_range: The date range to predict emissions
    :return: The emissions prediction
    """
    return do_req_hugging_face(date_range, 'demand')


@app.exception_handler(404)
async def http_exception_handler(request: Request, exc: HTTPException):
    return build_response(404, 'Not found')


@app.exception_handler(Exception)
async def general_exception(request: Request, e: Exception):
    """
    Error handler for all exceptions
    :param request: The request
    :param e: The exception
    :return: The error message and status code
    """
    if e is NotFound:
        status_code = 404
        message = 'Not found'
    else:
        status_code = 500
        message = 'Internal server error'
    print(e)
    print(traceback.format_exc())
    return build_response(status_code, message + ' ' + str(e))
