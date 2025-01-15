import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EnergyChartsComponent } from './energy-charts.component';

describe('EnergyChartsComponent', () => {
  let component: EnergyChartsComponent;
  let fixture: ComponentFixture<EnergyChartsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EnergyChartsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(EnergyChartsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
