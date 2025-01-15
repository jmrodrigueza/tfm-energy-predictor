import { ChangeDetectorRef, Component, LOCALE_ID } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatListModule } from '@angular/material/list'; 
import { MatMenuModule } from '@angular/material/menu';
import { RouterModule } from '@angular/router';
import { TranslateService, TranslateModule } from '@ngx-translate/core';
import { MatIconModule } from '@angular/material/icon';
import { FlexLayoutModule } from '@angular/flex-layout';
import { MAT_DATE_LOCALE } from '@angular/material/core';
import { registerLocaleData } from '@angular/common';
import localeEs from '@angular/common/locales/es';

registerLocaleData(localeEs, 'es-ES');

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
  standalone: true,
  imports: [
    RouterModule,
    TranslateModule,
    MatButtonModule,
    MatInputModule,
    MatToolbarModule,
    MatListModule,
    MatMenuModule,
    MatIconModule,
    FlexLayoutModule
  ],
  providers: [{ provide: MAT_DATE_LOCALE, useValue: 'es-ES' }, { provide: LOCALE_ID, useValue: 'es-ES' }],
})
export class AppComponent {
  currentLanguage: string;

  constructor(private translate: TranslateService, private cdr: ChangeDetectorRef) {
    this.currentLanguage = 'es';
    this.translate.setDefaultLang(this.currentLanguage);
  }

  switchLanguage(lang: string) {
    this.translate.use(lang);
    this.currentLanguage = lang;
    this.cdr.detectChanges();
  }
}
