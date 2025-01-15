import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { TranslateModule, TranslateService } from '@ngx-translate/core';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatTabsModule } from '@angular/material/tabs';
import { CommonModule } from '@angular/common';
import { FlexLayoutModule } from '@angular/flex-layout';
import { MatTimepickerModule } from '@angular/material/timepicker';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { provideNativeDateAdapter } from '@angular/material/core';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatIconModule } from '@angular/material/icon';
import { CommonService, EmissionsPredicted, EmissionsPrediction } from '../common/common.component';
import { coerceNumberProperty } from '@angular/cdk/coercion';
import { MatSliderModule } from '@angular/material/slider';
import { EnergyService } from '../../services/energy.service';
import { MatCardModule } from '@angular/material/card';
import { MatDividerModule } from '@angular/material/divider';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import { NotificationService } from '../../services/notification.service';

@Component({
  selector: 'app-energy-simulation',
  providers: [provideNativeDateAdapter()],
  imports: [
    CommonModule,
    FormsModule,
    TranslateModule,
    MatButtonModule,
    MatInputModule,
    MatToolbarModule,
    MatTabsModule,
    FlexLayoutModule,
    MatInputModule,
    MatTimepickerModule,
    MatDatepickerModule,
    MatTooltipModule,
    MatIconModule,
    MatSliderModule,
    MatCardModule,
    MatDividerModule,
    MatProgressBarModule
  ],
  templateUrl: './energy-simulation.component.html',
  styleUrl: './energy-simulation.component.scss',
  standalone: true
})
export class EnergySimulationComponent implements OnInit {
  minDate: string = CommonService.minDate;
  maxDate: string = CommonService.maxDate;
  minSliderValue: number = 0;
  maxSliderValue: number = 6000;
  step: number = 5;
  thumbLabel: boolean = true;
  selectedDate: Date;
  energyDataToSimulate: EmissionsPrediction;
  energyDataRealValue: EmissionsPrediction;
  energyDataSimulatedValue: EmissionsPredicted;
  thickInterval: number = 100;
  showProgressBar: boolean = false;
  dataNoAvailable: boolean = false;

  constructor(
    private translate: TranslateService,
    private energyService: EnergyService,
    private cdr: ChangeDetectorRef,
    private notificationService: NotificationService) {
    this.selectedDate = new Date(CommonService.defaultDateTime);
    this.energyDataToSimulate = {
      date_instant: CommonService.localIsoStringToDate(this.selectedDate),
      Carbon_gen: 0,
      Ciclo_combinado_gen: 0,
      Motores_diesel_gen: 0,
      Turbina_de_gas_gen: 0,
      Turbina_de_vapor_gen: 0,
      Cogeneracion_y_residuos_gen: 0
    };
    this.energyDataRealValue = {
      date_instant: CommonService.localIsoStringToDate(this.selectedDate),
      Carbon_gen: 0,
      Ciclo_combinado_gen: 0,
      Motores_diesel_gen: 0,
      Turbina_de_gas_gen: 0,
      Turbina_de_vapor_gen: 0,
      Cogeneracion_y_residuos_gen: 0
    };
    
    this.energyDataSimulatedValue = {
      date_instant: CommonService.localIsoStringToDate(this.selectedDate),
      Carbon_emi_pred: 0,
      Ciclo_combinado_emi_pred: 0,
      Motores_diesel_emi_pred: 0,
      Turbina_de_gas_emi_pred: 0,
      Turbina_de_vapor_emi_pred: 0,
      Cogeneracion_y_residuos_emi_pred: 0
    };
  }
  ngOnInit(): void {
  }

  ngAfterViewInit() {
    this.energyService.getPredictorColsEmissions(CommonService.getDateRangeFromSelectedDate(this.selectedDate)).subscribe(
      (data: any) => {
        // if data contains 'content' key
        if (data.content) {
          this.energyDataToSimulate = { ...data.content[0] };
          this.energyDataRealValue = { ...data.content[0] };
        }
      }
    );
  }

  get tickInterval(): number | 'auto' {
    return 0;
  }

  set tickInterval(value) {
    this.thickInterval = coerceNumberProperty(value);
  }

  formatThumbLabel(value: number): string {
    if (value >= 10) {
      return Math.round(value / 10) + 'd';
    }
    return value.toString();
  }

  getElementsInEnergyDataToSimulate(): Array<string> {
    return Object.keys(this.energyDataToSimulate).filter(key => key.includes('_gen'));
  }

  resetToRealValues() {
    this.energyDataToSimulate.Carbon_gen = this.energyDataRealValue.Carbon_gen;
    this.energyDataToSimulate.Ciclo_combinado_gen = this.energyDataRealValue.Ciclo_combinado_gen;
    this.energyDataToSimulate.Motores_diesel_gen = this.energyDataRealValue.Motores_diesel_gen;
    this.energyDataToSimulate.Turbina_de_gas_gen = this.energyDataRealValue.Turbina_de_gas_gen;
    this.energyDataToSimulate.Turbina_de_vapor_gen = this.energyDataRealValue.Turbina_de_vapor_gen;
    this.energyDataToSimulate.Cogeneracion_y_residuos_gen = this.energyDataRealValue.Cogeneracion_y_residuos_gen;
  }

  protected setDataNoAvailable() {
    this.dataNoAvailable = true;
    this.notificationService.showNotification(this.translate.instant('DATA_NOT_AVAILABLE'), 'Close', false);
    this.cdr.detectChanges();
  }

  protected getErrorCallback(error: any) {
    this.showProgressBar = false;
    this.setDataNoAvailable();
  }

  simulateEnergyData() {
    this.showProgressBar = true;
    this.dataNoAvailable = false;
    this.energyService.getEmissionsSimulation(this.energyDataToSimulate).subscribe({
        next: (data: any) => {
          if (data.content && data.status == 200) {
            this.energyDataSimulatedValue = { ...data.content[0] };
          }
          if (!(data && data.content) && data.status !== 200) {
            this.setDataNoAvailable();
          }
          this.showProgressBar = false;
        },
        error: (error) => this.getErrorCallback(error)
    });
  }
}
