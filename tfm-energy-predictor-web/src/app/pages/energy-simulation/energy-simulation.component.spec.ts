import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EnergySimulationComponent } from './energy-simulation.component';

describe('EnergySimulationComponent', () => {
  let component: EnergySimulationComponent;
  let fixture: ComponentFixture<EnergySimulationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EnergySimulationComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(EnergySimulationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
