import { NgIf } from '@angular/common';
import { Component, HostListener } from '@angular/core';
import { FlexLayoutModule } from '@angular/flex-layout';
import { MatButton } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatDividerModule } from '@angular/material/divider';
import { MatIconModule } from '@angular/material/icon';
import { TranslateModule } from '@ngx-translate/core';

@Component({
  selector: 'app-help-page',
  standalone: true,
  imports: [
    FlexLayoutModule,
    TranslateModule,
    MatIconModule,
    MatButton,
    NgIf,
    MatCardModule,
    MatDividerModule
  ],
  templateUrl: './help-page.component.html',
  styleUrl: './help-page.component.scss'
})
export class HelpPageComponent {
  windowScrolled: boolean = false;
  @HostListener('window:scroll', [])

  protected onWindowScroll() {
    this.windowScrolled = window.scrollY > 50;
  }

  protected scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}
