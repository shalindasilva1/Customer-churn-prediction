import { Component, OnInit, SecurityContext } from '@angular/core';
import { FileUploadService } from '../file-upload.service';
import {DomSanitizer, SafeUrl} from '@angular/platform-browser';
import { empty } from 'rxjs';

declare var require: any;
const FileSaver = require('file-saver');

@Component({
  selector: 'app-file-upload',
  templateUrl: './file-upload.component.html',
  styleUrls: ['./file-upload.component.css'],
})
export class FileUploadComponent implements OnInit {
  // Variable to store shortLink from api response
  shortLink = '';
  loading = false; // Flag variable
  file: File | any; // Variable to store file
  // Inject service
  constructor(private fileUploadService: FileUploadService,
              private sanitizer: DomSanitizer) {}

  ngOnInit(): void {}

  // On file Select

  onChange(event: any): void {
    this.file = event.target.files[0];
  }

  // OnClick of button Upload
  onUpload(): void {
    this.loading = !this.loading;
    console.log(this.file);
    this.fileUploadService.upload(this.file).subscribe((event: any) => {
      if (typeof event === 'object') {
        // Short link via api response
        this.shortLink = event.link.replace('/C:\Users\Shalinda\source\repos\shalindasilva1\ML-Project\Web\web-app\src\/gi', '');

        this.loading = false; // Flag variable
      }
    });
  }

  DownloadLink(url: string): void{
    FileSaver.saveAs(url, 'Output.csv');
  }

  Sanitize(url: string): string {
    const result = this.sanitizer.sanitize(SecurityContext.RESOURCE_URL, this.sanitizer.bypassSecurityTrustResourceUrl(url));
    if (result == null) { return ''; }
    return result;
  }
}
