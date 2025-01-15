import { ChangeDetectorRef, Component, OnInit, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { TranslateService, TranslateModule } from '@ngx-translate/core';
import { MatTableDataSource, MatTableModule } from '@angular/material/table';
import { MatPaginatorModule } from '@angular/material/paginator';
import { MatSort, MatSortModule } from '@angular/material/sort';
import { MatInputModule } from '@angular/material/input';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MatTimepickerModule } from '@angular/material/timepicker';
import { EnergyService } from '../../services/energy.service';
import { CommonService, RetrieveDataEnum } from '../common/common.component';
import { MatIconModule } from '@angular/material/icon';
import { provideNativeDateAdapter } from '@angular/material/core';
import { saveAs } from 'file-saver';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { NotificationService } from '../../services/notification.service';


interface EnergyData {
  type: string;
  variable: string;
  datetime: string;
  value: number;
}

@Component({
  selector: 'app-energy-table',
  templateUrl: './energy-tables.component.html',
  styleUrls: ['./energy-tables.component.scss'],
  standalone: true,
  providers: [provideNativeDateAdapter()],
  imports: [
    CommonModule,
    TranslateModule,
    MatButtonModule,
    MatTableModule,
    MatPaginatorModule,
    MatDatepickerModule,
    MatTimepickerModule,
    MatSortModule,
    MatInputModule,
    FormsModule,
    MatToolbarModule,
    MatIconModule,
    MatSortModule,
    MatTooltipModule,
    MatProgressSpinnerModule
  ]
})
export class EnergyTablesComponent implements OnInit {
  minDate: string = CommonService.minDate;
  maxDate: string = CommonService.maxDate;
  selectedDate: Date;
  showProgressSpinner = false;
  dataNoAvailable: boolean = false;


  displayedColumns: string[] = ['type', 'variable', 'datetime', 'value'];

  energyData = new MatTableDataSource<EnergyData>([]);
  @ViewChild(MatSort) sort!: MatSort;
  
  retrieveDataEnumNames = new Map<number, string>(
    Object.entries(RetrieveDataEnum)
        .filter(([key, value]) => typeof value === "number")
        .map(([key, value]) => [value as number, key])
  );

  constructor(
    private translate: TranslateService,
    private energyService: EnergyService,
    private cdr: ChangeDetectorRef,
    private notificationService: NotificationService) {
    this.translate.setDefaultLang('es');
    this.selectedDate = new Date(CommonService.defaultDateTime);
  }

  ngOnInit(): void {
  }

  ngAfterViewInit() {
    const allCharts = [RetrieveDataEnum.ALL];
    for (const itemChart of allCharts) {
      this.updateData(itemChart);
    }
    this.energyData.sort = this.sort;
  }

  switchLanguage(lang: string) {
    this.translate.use(lang);
  }

  applyFilter(filterValue: string) {
    this.energyData.filter = filterValue.trim().toLowerCase();
  }

  protected   setDataNoAvailable() {
    this.dataNoAvailable = true;
    this.notificationService.showNotification(this.translate.instant('DATA_NOT_AVAILABLE'), 'Close', false);
    this.cdr.detectChanges();
  }

  protected getErrorCallback(error: any) {
    this.showProgressSpinner = false;
    this.setDataNoAvailable();
  }

  updateChartDataCallback(data: any, type: RetrieveDataEnum): Array<EnergyData> {
    let datasetsMap: Array<any> = [];

    if (data && data.content) {
      let datetimeMap: Map<string, string> = CommonService.adaptDatetimeArray(new Map(Object.entries(data.content['datetime'])));
      Object.keys(data.content).forEach((key) => {
        if (key === "datetime" || key.includes("MAPE")) return;
        const valueMap = new Map(Object.entries(data.content[key]));
        Array.from(valueMap.keys()).forEach((index) => {
            datasetsMap.push({
                type: this.retrieveDataEnumNames.get(type),
                variable: key,
                datetime: datetimeMap.get(index),
                value: valueMap.get(index)
            });
        });
      });
    }

    return datasetsMap;
  }

  updateData(dataToRetrieve: RetrieveDataEnum = RetrieveDataEnum.ALL) {
    this.energyData.data = [];
    if (dataToRetrieve === RetrieveDataEnum.ALL || dataToRetrieve === RetrieveDataEnum.DEMAND) {
      this.showProgressSpinner = true;
      this.dataNoAvailable = false;
      this.energyService.getEnergyDemmandData(
        CommonService.getDateRangeFromSelectedDate(this.selectedDate)
      ).subscribe({
        next: (data) => {
          if (data.content && data.status === 200) {
            for (const item of this.updateChartDataCallback(data, RetrieveDataEnum.DEMAND)) {
              this.energyData.data = [...this.energyData.data, item];
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
        CommonService.getDateRangeFromSelectedDate(this.selectedDate)).subscribe((data) => {
          if (data.content && data.status === 200) {
            for (const item of this.updateChartDataCallback(data, RetrieveDataEnum.GENERATION)) {
              this.energyData.data = [...this.energyData.data, item];
            }
          }
        });
    }
    if(dataToRetrieve === RetrieveDataEnum.ALL || dataToRetrieve === RetrieveDataEnum.EMISSIONS) {
      this.energyService.getEnergyEmissionData(
        CommonService.getDateRangeFromSelectedDate(this.selectedDate)).subscribe((data) => {                
          if (data.content && data.status === 200) {
            for (const item of this.updateChartDataCallback(data, RetrieveDataEnum.EMISSIONS)) {
              this.energyData.data = [...this.energyData.data, item];
            }
          }
      });
    }
  }

  refreshData() {
    this.updateData();
  }

  exportToCSV(): void {
    const csvRows: string[] = [];
    const headers = this.displayedColumns.join(',');
    csvRows.push(headers);

    this.energyData.data.forEach(row => {
      const values = this.displayedColumns.map(col => row[col as keyof typeof row]);
      csvRows.push(values.join(','));
    });

    const csvContent = csvRows.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    saveAs(blob, 'data.csv');
  }
}
