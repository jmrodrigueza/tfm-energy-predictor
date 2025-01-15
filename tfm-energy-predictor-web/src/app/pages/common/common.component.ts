import { Injectable } from '@angular/core';

export enum RetrieveDataEnum {
  ALL,
  GENERATION,
  DEMAND,
  EMISSIONS
}

export interface DateRange {
    start_date: string;
    end_date: string;
}

export interface EmissionsPrediction {
  date_instant: string;
  Carbon_gen: number;
  Ciclo_combinado_gen: number;
  Motores_diesel_gen: number;
  Turbina_de_gas_gen: number;
  Turbina_de_vapor_gen: number;
  Cogeneracion_y_residuos_gen: number;
}

export interface EmissionsPredicted {
  date_instant: string;
  Carbon_emi_pred: number;
  Ciclo_combinado_emi_pred: number;
  Motores_diesel_emi_pred: number;
  Turbina_de_gas_emi_pred: number;
  Turbina_de_vapor_emi_pred: number;
  Cogeneracion_y_residuos_emi_pred: number;
}


@Injectable({
  providedIn: 'root',
})
export class CommonService {
  static minDate: string = '2023-09-18';
  static maxDate: string = '2024-09-17';
  static defaultDateTime: string = '2023-09-18T00:00:00.000';

  static localIsoStringToDate(localDate: Date): string {
    const localISOString =
      localDate.getFullYear() +
      '-' +
      String(localDate.getMonth() + 1).padStart(2, '0') +
      '-' +
      String(localDate.getDate()).padStart(2, '0') +
      'T' +
      String(localDate.getHours()).padStart(2, '0') +
      ':' +
      String(localDate.getMinutes()).padStart(2, '0') +
      ':' +
      String(localDate.getSeconds()).padStart(2, '0') +
      '.' +
      String(localDate.getMilliseconds()).padStart(3, '0');
    return localISOString;
  }

  static getDateRangeFromSelectedDate(selectedDate: Date): DateRange {
    const start_date = new Date(selectedDate);
    const end_date = new Date(selectedDate);
    end_date.setTime(end_date.getTime() + 24 * 60 * 60 * 1000 - 1);
    return {
      start_date: CommonService.localIsoStringToDate(start_date),
      end_date: CommonService.localIsoStringToDate(end_date),
    };
  }

  static adaptDatetimeArray(datetimes: Map<string, string>): Map<string, string> {
    datetimes.forEach((value, key) => {
      const date = new String(value);
      datetimes.set(key, date.slice(0, 10) + ' ' + date.slice(11, 16));
    });
    return datetimes;
  }
}
