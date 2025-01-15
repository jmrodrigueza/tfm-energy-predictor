from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field


class DateRange(BaseModel):
    start_date: datetime = Field(..., description="Start date with format YYYY-MM-DDT00:00:00Z")
    end_date: datetime = Field(..., description="Fecha de fin en formato YYYY-MM-DDT00:00:00Z")


class EmissionsPrediction(BaseModel):
    date_instant: datetime = Field(..., description="Instant date with format YYYY-MM-DDT00:00:00Z")
    Carbon_gen: float = Field(..., description="Carbon generation")
    Ciclo_combinado_gen: float = Field(..., description="Combined cycle generation")
    Motores_diesel_gen: float = Field(..., description="Diesel engine generation")
    Turbina_de_gas_gen: float = Field(..., description="Gas turbine generation")
    Turbina_de_vapor_gen: float = Field(..., description="Steam turbine generation")
    Cogeneracion_y_residuos_gen: float = Field(..., description="Cogeneration and waste generation")


class EmissionsPredicted(BaseModel):
    date_instant: datetime = Field(..., description="Instant date with format YYYY-MM-DDT00:00:00Z")
    Carbon_emi_pred: float = Field(..., description="Carbon emissions")
    Ciclo_combinado_emi_pred: float = Field(..., description="Combined cycle emissions")
    Motores_diesel_emi_pred: float = Field(..., description="Diesel engine emissions")
    Turbina_de_gas_emi_pred: float = Field(..., description="Gas turbine emissions")
    Turbina_de_vapor_emi_pred: float = Field(..., description="Steam turbine emissions")
    Cogeneracion_y_residuos_emi_pred: float = Field(..., description="Cogeneration and waste emissions")



def datetime_encoder(obj):
    """
    Custom encoder for datetime objects
    :param obj: The object to encode
    :return: The encoded object
    """
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
