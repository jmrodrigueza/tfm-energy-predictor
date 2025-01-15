import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: 'charts', pathMatch: 'full' },
  {
    path: 'charts',
    loadComponent: () =>
      import('./pages/energy-charts/energy-charts.component').then((m) => m.EnergyChartsComponent),
  },
  {
    path: 'tables',
    loadComponent: () =>
      import('./pages/energy-tables/energy-tables.component').then((m) => m.EnergyTablesComponent),
  },
  {
    path: 'simulation',
    loadComponent: () =>
      import('./pages/energy-simulation/energy-simulation.component').then((m) => m.EnergySimulationComponent),
  },
  {
    path: 'help',
    loadComponent: () =>
      import('./pages/help-page/help-page.component').then((m) => m.HelpPageComponent),
  },
];
