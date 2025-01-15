import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EnergyTablesComponent } from './energy-tables.component';

describe('EnergyTablesComponent', () => {
  let component: EnergyTablesComponent;
  let fixture: ComponentFixture<EnergyTablesComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EnergyTablesComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(EnergyTablesComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
