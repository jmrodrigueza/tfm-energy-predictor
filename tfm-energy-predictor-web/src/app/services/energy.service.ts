import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import { DateRange, EmissionsPrediction } from '../pages/common/common.component';



@Injectable({  providedIn: 'root'})
export class EnergyService {
  private readonly baseUrl = environment.apiBaseUrl;

  constructor(private http: HttpClient) {}

  getEnergyDemmandData(dateRange: DateRange): Observable<any> {
    return this.http.post(`${this.baseUrl}/demand`, dateRange).pipe(
      catchError(error => {
        console.error('Error in the API:', error);
        return of([]);
      })
    );
  }

  getEnergyGenerationData(dateRange: DateRange): Observable<any> {
    return this.http.post(`${this.baseUrl}/generation`, dateRange).pipe(
      catchError(error => {
        console.error('Error in the API:', error);
        return of([]);
      })
    );
  }

  getEnergyEmissionData(dateRange: DateRange): Observable<any> {
    return this.http.post(`${this.baseUrl}/emissions`, dateRange).pipe(
      catchError(error => {
        console.error('Error in the API:', error);
        return of([]);
      })
    );
  }

  getPredictorColsEmissions(dateRange: DateRange): Observable<any> {
    return this.http.post(`${this.baseUrl}/get-predictor-cols-emissions`, dateRange).pipe(
      catchError(error => {
        console.error('Error in the API:', error);
        return of([]);
      })
    );
  }

  getEmissionsSimulation(input_params: EmissionsPrediction): Observable<any> {
    return this.http.post(`${this.baseUrl}/simu-emissions`, input_params).pipe(
      catchError(error => {
        console.error('Error in the API:', error);
        return of([]);
      })
    );
  }
}