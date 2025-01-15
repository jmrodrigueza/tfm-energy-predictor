import json
import traceback

import pandas as pd
from fastapi import FastAPI, HTTPException
from flask import Request
from starlette.responses import JSONResponse
from werkzeug.exceptions import NotFound

from predictor import Predictor
from dtos_objects import DateRange, EmissionsPrediction, datetime_encoder, EmissionsPredicted

import sys
import io

if sys.version_info < (3, 7):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
else:
    sys.stdout.reconfigure(encoding='utf-8')

"""
Initialize the app
"""
pred_emi = Predictor('Transformer_emi_model')
# Load the parquet data only first time
pred_emi.load_parquet_data()
pred_emi.append_data_energy_days_before()
pred_emi.print_parquet_data()
pred_emi.load_model()

pred_sim_emi = Predictor('ANN_simu_emi_model')
pred_sim_emi.append_data_energy_days_before()
pred_sim_emi.load_model()

pred_gen = Predictor('Transformer_gen_model', win_size=1, bsize=64)
pred_gen.append_data_energy_days_before()
pred_gen.load_model()

pred_dem = Predictor('Transformer_dem_model', win_size=1, bsize=64)
pred_dem.append_data_energy_days_before()
pred_dem.load_model()

app = FastAPI()


def generic_predictor(predictor: str, input_parameters):
    """
    Predict emissions
    :param predictor: The predictor to use
    :param input_parameters: The input parameters date range or emissions prediction
    :return: The emissions prediction
    """
    json_response = None
    if isinstance(input_parameters, DateRange):
        date_range: DateRange = input_parameters
        if date_range.start_date > date_range.end_date:
            json_response = build_response(400, 'Start date must be before or equal end date')
        else:
            if predictor == 'emissions':
                json_response = build_response(200, 'OK',
                                               in_content=pred_emi.predict(date_range.start_date, date_range.end_date))
            elif predictor == 'get-predictor-cols-emissions':
                json_response = build_response(200, 'OK',
                                               in_content=pred_sim_emi.get_predictor_columns(date_range.start_date,
                                                                                             date_range.end_date))
            elif predictor == 'generation':
                json_response = build_response(200, 'OK',
                                               in_content=pred_gen.predict(date_range.start_date, date_range.end_date))
            elif predictor == 'demand':
                json_response = build_response(200, 'OK',
                                               in_content=pred_dem.predict(date_range.start_date, date_range.end_date))
    elif isinstance(input_parameters, EmissionsPrediction) and predictor == 'simu-emissions':
        json_response = build_response(200, 'OK', in_content=pred_sim_emi.predict_emissions(input_parameters))
    if json_response is None:
        json_response = build_response(400, 'Invalid input parameters')
    return json_response


def build_response(status_code, message, in_content=None):
    content = {'status': status_code, 'message': message, 'content': {}}
    if in_content is not None:
        if isinstance(in_content, tuple):
            df, mape = in_content
            json_content = json.loads(df.to_json(orient='columns', date_format='iso'))
        elif isinstance(in_content, pd.DataFrame):
            df = in_content
            json_content = json.loads(df.to_json(orient='columns', date_format='iso'))
            mape = []
        elif isinstance(in_content, list) and (all([isinstance(item, EmissionsPrediction) for item in in_content]) or\
                                               all([isinstance(item, EmissionsPredicted) for item in in_content])):
            json_content = json.loads(json.dumps([item.dict() for item in in_content], default=datetime_encoder))
            mape = None
        else:
            json_content = json.loads(json.dumps([]))
            mape = []

        content['content'] = json_content
        if mape is not None:
            content['content']['MAPE'] = mape
    return JSONResponse(status_code=status_code, content=content)


@app.get("/")
def default_url():
    return build_response(200, 'OK')


@app.post("/emissions")
def predict_emissions(date_range: DateRange):
    """
    Predict emissions
    :param date_range: The date range to predict emissions
    :return: The emissions prediction
    """
    return generic_predictor('emissions', date_range)


@app.post("/simu-emissions")
def simulate_emissions(input_parameters: EmissionsPrediction):
    """
    Simulate emissions for the given date range and the given predictor columns
    :param input_parameters: The date range and energy variables to predict emissions
    :return: The emissions prediction
    """
    return generic_predictor('simu-emissions', input_parameters)


@app.post("/get-predictor-cols-emissions")
def get_predictor_cols_emissions(date_range: DateRange):
    """
    Retrieve the predictor columns values for the given date range
    :param date_range: The date range to predict emissions
    :return: The emissions prediction
    """
    return generic_predictor('get-predictor-cols-emissions', date_range)


@app.post("/generation")
def predict_generation(date_range: DateRange):
    """
    Predict emissions
    :param date_range: The date range to predict emissions
    :return: The emissions prediction
    """
    return generic_predictor('generation', date_range)


@app.post("/demand")
def predict_demand(date_range: DateRange):
    """
    Predict emissions
    :param date_range: The date range to predict emissions
    :return: The emissions prediction
    """
    return generic_predictor('demand', date_range)


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
