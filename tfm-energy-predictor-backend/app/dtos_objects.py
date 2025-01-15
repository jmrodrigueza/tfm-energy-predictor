from datetime import datetime
from typing import Any

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


