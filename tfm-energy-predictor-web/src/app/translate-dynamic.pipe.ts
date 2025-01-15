import { Pipe, PipeTransform } from '@angular/core';
import { TranslateService } from '@ngx-translate/core';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

@Pipe({
    name: 'translateDynamic',
    standalone: true,
    pure: false
})
export class TranslateDynamicPipe implements PipeTransform {
  constructor(private translate: TranslateService) {}

  transform(key: string): Observable<string> {
    return this.translate.stream(key).pipe(
      map((value) => value || '')
    );
  }
}
