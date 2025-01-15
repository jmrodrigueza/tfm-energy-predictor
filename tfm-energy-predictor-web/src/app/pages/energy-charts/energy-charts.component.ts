import { Component, ChangeDetectionStrategy, OnInit, ViewChild, ChangeDetectorRef } from '@angular/core';
import { ChartConfiguration, ChartOptions } from 'chart.js';
import { FormsModule } from '@angular/forms';
import { TranslateModule, TranslateService } from '@ngx-translate/core';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatTabsModule } from '@angular/material/tabs';
import { MatTabGroup } from '@angular/material/tabs';
import { MatCardModule } from '@angular/material/card';
import { CommonModule } from '@angular/common';
import { BaseChartDirective } from 'ng2-charts';
import { FlexLayoutModule } from '@angular/flex-layout';
import { MatIconModule } from '@angular/material/icon';
import { Chart, registerables} from 'chart.js';
import { EnergyService } from '../../services/energy.service';
import { CommonService, RetrieveDataEnum } from '../common/common.component';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { provideNativeDateAdapter } from '@angular/material/core';
import { MatTimepickerModule } from '@angular/material/timepicker';
import { MatDividerModule } from '@angular/material/divider';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { NotificationService } from '../../services/notification.service';

Chart.register(...registerables);



@Component({
  selector: 'app-energy-charts',
  templateUrl: './energy-charts.component.html',
  styleUrls: ['./energy-charts.component.scss'],
  standalone: true,
  providers: [provideNativeDateAdapter()],
  imports: [
    CommonModule,
    FormsModule,
    TranslateModule,
    MatButtonModule,
    MatInputModule,
    MatDatepickerModule,
    MatTimepickerModule,
    MatToolbarModule,
    MatTabsModule,
    MatTabGroup,
    MatCardModule,
    BaseChartDirective,
    FlexLayoutModule,
    MatIconModule,
    MatDividerModule,
    MatTooltipModule,
    MatProgressSpinnerModule
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class EnergyChartsComponent implements OnInit {
  minDate: string = CommonService.minDate;
  maxDate: string = CommonService.maxDate;
  // The selected date
  selectedDate: Date;
  showProgressSpinner: boolean = false;
  dataNoAvailable: boolean = false;

  // General configuration for the line charts
  public lineChartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        position: 'top',
      },
      tooltip: {
        enabled: true,
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Time (Hours)',
        },
      },
      y: {
        min: 0,
        title: {
          display: true,
          text: 'Value',
        },
      },
    },
  };

  // Demand: configuration and data
  public demandChartData: ChartConfiguration<'line'>['data'] = {
    labels: Array.from([]),
    datasets: [],
  };

  // Production: configuration and data
  public productionChartData: ChartConfiguration<'line'>['data'] = {
    labels: Array.from([]),
    datasets: [],
  };

  // Emissions: configuration and data
  public emissionsChartData: ChartConfiguration<'line'>['data'] = {
    labels: Array.from([]),
    datasets: [],
  };

  @ViewChild('demandChart') demandChart!: BaseChartDirective;
  @ViewChild('productionChart') productionChart!: BaseChartDirective;
  @ViewChild('emissionsChart') emissionsChart!: BaseChartDirective;
  @ViewChild('enertyTabs') enertyTabs!: MatTabGroup;

  mapeMetrics: Map<string, number> = new Map([]);
  mapeMetricsDemand: Map<string, number> = new Map([]);
  mapeMetricsProduction: Map<string, number> = new Map([]);
  mapeMetricsEmissions: Map<string, number> = new Map([]);

  constructor(
    private energyService: EnergyService,
    private cdr: ChangeDetectorRef,
    private translate: TranslateService,
    private notificationService: NotificationService) { 
    this.selectedDate = new Date(CommonService.defaultDateTime);
  }

  ngOnInit() {
  }

  getDatasetsFromMap(datasetMap: Map<string, string>, key: string) {
    return {
      data: Array.from(datasetMap.values()),
      label: this.translate.instant(key),
      borderColor: this.getRandomColor(key),
      backgroundColor: this.getRandomColor(key, 0.2),
      fill: false
    };
  }

  getRandomColor(key: string, alpha: number = 1): string {
    let base = 0;
    let range = 255;
    if (key.endsWith('_real')) {
      base = 127;
      range =  128;
    } else {
      base = 0;
      range = 127;
    }
    const r = Math.floor(base + Math.random() * range);
    const g = Math.floor(base + Math.random() * range);
    const b = Math.floor(base + Math.random() * range);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  updateChartDataCallback(data: any, genericChart: BaseChartDirective): Array<any> {
    let dataAndMape: Array<any> = [];
    let genericChartNew: any;
    let mapeMetric: Map<string, number> = new Map([]);

    if (data && data.content && genericChart) {
      let datetimes: Map<string, string> = new Map();
      let datasetsMap: Array<any> = [];

      for (const key in data.content) {
        if (key === 'datetime') {
          datetimes = new Map(Object.entries(data.content[key]));
          datetimes.forEach((value, key) => {
            const date = new String(value);
            datetimes.set(key, date.slice(0, 10) + ' ' + date.slice(11, 16));
          });
        } else if (key === 'MAPE') {
          mapeMetric = new Map(Object.entries(data.content[key]));
        } else {
          datasetsMap.push(this.getDatasetsFromMap(new Map(Object.entries(data.content[key])), key));
        }
        this.cdr.detectChanges();
      }
      genericChartNew = {
        ...genericChart,
        labels: Array.from(datetimes.values()),
        datasets: datasetsMap,
      };
    }
    dataAndMape.push(genericChartNew);
    dataAndMape.push(mapeMetric);
    return dataAndMape;
  }

  protected setDataNoAvailable() {
    this.dataNoAvailable = true;
    this.notificationService.showNotification(this.translate.instant('DATA_NOT_AVAILABLE'), 'Close', false);
    this.cdr.detectChanges();
  }

  protected getErrorCallback(error: any) {
    this.showProgressSpinner = false;
    this.setDataNoAvailable();
  }

  updateData(dataToRetrieve: RetrieveDataEnum = RetrieveDataEnum.ALL) {
    let dataAndMape: Array<any> = [];
    if (dataToRetrieve === RetrieveDataEnum.ALL || dataToRetrieve === RetrieveDataEnum.DEMAND) {
      this.showProgressSpinner = true;
      this.dataNoAvailable = false;
      this.energyService.getEnergyDemmandData(
        CommonService.getDateRangeFromSelectedDate(this.selectedDate)
      ).subscribe({
        next: (data) => {
          if (data.content && data.status == 200) {
            dataAndMape = this.updateChartDataCallback(data, this.demandChart);
            this.demandChartData = dataAndMape[0];
            this.mapeMetricsDemand = dataAndMape[1];
            if (this.enertyTabs.selectedIndex === 0) {
              this.mapeMetrics = this.mapeMetricsDemand;
            }
          }
          if (!(data && data.content) || data.status !== 200) {
            this.setDataNoAvailable();
          }
          this.showProgressSpinner = false;
        },
        error: (error) => this.getErrorCallback(error)
      });
    }
    if(dataToRetrieve === RetrieveDataEnum.ALL || dataToRetrieve === RetrieveDataEnum.GENERATION) {
      this.energyService.getEnergyGenerationData(
        CommonService.getDateRangeFromSelectedDate(this.selectedDate)).subscribe({
          next: (data) => {
            if (data.content && data.status == 200) {
              dataAndMape = this.updateChartDataCallback(data, this.productionChart);
              this.productionChartData = dataAndMape[0];
              this.mapeMetricsProduction = dataAndMape[1];
              if (this.enertyTabs.selectedIndex === 1) {
                this.mapeMetrics = this.mapeMetricsProduction;
              }
            }
          },
          error: (error) => this.getErrorCallback(error)
        });
    }
    if(dataToRetrieve === RetrieveDataEnum.ALL || dataToRetrieve === RetrieveDataEnum.EMISSIONS) {
      this.energyService.getEnergyEmissionData(
        CommonService.getDateRangeFromSelectedDate(this.selectedDate)).subscribe({
          next: (data) => {
            if (data.content && data.status == 200) {
              dataAndMape = this.updateChartDataCallback(data, this.emissionsChart);
              this.emissionsChartData = dataAndMape[0];
              this.mapeMetricsEmissions = dataAndMape[1];
              if (this.enertyTabs.selectedIndex === 2) {
                this.mapeMetrics = this.mapeMetricsEmissions;
              }
            }
          },
          error: (error) => this.getErrorCallback(error)
        });
    }
  }

  onTabChanged(index: number): void {
    if (index === 0) {
      this.mapeMetrics = this.mapeMetricsDemand;
    } else if (index === 1) {
      this.mapeMetrics = this.mapeMetricsProduction;
    } else if (index === 2) {
      this.mapeMetrics = this.mapeMetricsEmissions;
    }
  }

  ngAfterViewInit() {
    const allCharts = [RetrieveDataEnum.ALL];
    for (const itemChart of allCharts) {
      this.updateData(itemChart);
    }
  }

  refreshData() {
    this.updateData();
  }
}
