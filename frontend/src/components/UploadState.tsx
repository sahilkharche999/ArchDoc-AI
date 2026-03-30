import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Upload, FileText, Check, } from "lucide-react";
import { Input } from "./ui/input";
import { Document, pdfjs } from "react-pdf";

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface UploadStateProps {
  onStartProcessing: (jobId: string, filePath: string) => void;
}

export function UploadState({ onStartProcessing }: UploadStateProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [startPage, setStartPage] = useState<string>("");
  const [endPage, setEndPage] = useState<string>("");
  const [numPages, setNumPages] = useState<number>();
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type === "application/pdf") {
      setFile(droppedFile);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
  setNumPages(numPages); 
}

  const uploadFile = async () => {
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);
  formData.append("start_page", startPage);
  formData.append("end_page", endPage);

  const response = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/upload`, {
    method: "POST",
    body: formData,
  });

  const data = await response.json();

  return data;
};
  
  return (
    <div className="flex items-center justify-center min-h-screen p-8">
      <Card className="w-full max-w-2xl">
        <CardHeader>
          <CardTitle className="text-2xl text-center">
            Upload Engineering Drawings
          </CardTitle>
          <p className="text-center text-muted-foreground mt-2">
            Upload your PDF drawings to begin the estimation process
          </p>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Drag and Drop Area */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
              isDragging
                ? "border-accent bg-accent/5"
                : "border-border hover:border-accent/50"
            }`}
          >
            <input
              type="file"
              id="file-upload"
              accept=".pdf"
              onChange={handleFileInput}
              className="hidden"
            />
            <label
              htmlFor="file-upload"
              className="cursor-pointer flex flex-col items-center"
            >
              {file && (
  <div style={{ display: "none" }}>
    <Document
      file={file}
      onLoadSuccess={onDocumentLoadSuccess}
      onLoadError={(err) => console.error("PDF load error:", err)}
    />
  </div>
)}
              {file ? (
                <>
                  <FileText className="w-16 h-16 text-accent mb-4" />
                  <p className="text-foreground mb-2">{file.name}</p>
                  <p className="text-sm text-muted-foreground">
                    Click to change file
                  </p>
                </>
              ) : (
                <>
                  <Upload className="w-16 h-16 text-muted-foreground mb-4" />
                  <p className="text-foreground mb-2">
                    Drag and drop your PDF here
                  </p>
                  <p className="text-sm text-muted-foreground mb-4">
                    or click to browse
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Supports: PDF files up to 50MB
                  </p>
                </>
              )}
            </label>
          </div>



          {/* Processing Checklist */}
          {numPages && (
  <p className="text-sm text-muted-foreground">
    Total Pages: 1 → {numPages}
  </p>
)}
          <div className="bg-muted/50 rounded-lg p-4">
            <p className="text-sm mb-3 text-muted-foreground">
              ENTER THE STARTING AND ENDING STRUCTURAL (S) PAGES NUMBERS:
            </p>
            <div className="space-y-2">
              <Input placeholder="Starting Page" value={startPage} onChange={(e) => setStartPage(e.target.value)}/>
              <Input placeholder="Ending Page" value={endPage} onChange={(e) => setEndPage(e.target.value)}/>
            </div>
          </div>

                    {/* Action Button */}
          <Button
            onClick={async () => {
            const result = await uploadFile();
            if (result) {
              console.log(result)
              onStartProcessing(result.job_id, result.file_path);
            }
            }}
            disabled={!file || !startPage || !endPage}
            className="w-full bg-accent hover:bg-accent/90 text-accent-foreground"
            size="lg"
          >
            Run Estimation
          </Button>

          {/* Info */}
          <div className="text-center">
            <p className="text-xs text-muted-foreground">
              Typical processing time: 30-60 minutes for standard drawings
            </p>
          </div>

        </CardContent>
      </Card>
    </div>
  );
}

