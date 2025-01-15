import { Injectable } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';

@Injectable({
  providedIn: 'root'
})
export class NotificationService {
  
    constructor(private snackBar: MatSnackBar) { }

    showNotification(message: string, action: string = 'Close', success: boolean = true) {
        this.snackBar.open(message, action, {
            duration: 20000,
            horizontalPosition: 'right',
            verticalPosition: 'bottom',
            panelClass: success ? ['success-snackbar'] : ['error-snackbar']
        });
    }
}