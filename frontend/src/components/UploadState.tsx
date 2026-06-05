import {useState,useEffect} from "react";
import {Card, CardContent, CardHeader, CardTitle} from "./ui/card";
import {Button} from "./ui/button";
import {FileText, Upload,} from "lucide-react";
import {Input} from "./ui/input";
import {Document, pdfjs,Page} from "react-pdf";
import { useAuth } from "../app/context/AuthContext";
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url
).toString();

interface UploadStateProps {
    onStartProcessing: (jobId: string, filePath: string) => void;
}

export function UploadState({onStartProcessing}: UploadStateProps) {

    const [showPreview, setShowPreview] = useState(false);
    const [needsRotation, setNeedsRotation] = useState<boolean | null>(null);
    const [pageRotations, setPageRotations] = useState<{[key: number]: number}>({});
    const [currentPage, setCurrentPage] = useState(1);
    const [isFixing, setIsFixing] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [startPage, setStartPage] = useState<string>("");
    const [endPage, setEndPage] = useState<string>("");
    const [numPages, setNumPages] = useState<number>();
    const [isUploading, setIsUploading] = useState(false);
    const [fileUrl, setFileUrl] = useState<string | null>(null);
    const [pdfVersion, setPdfVersion] = useState(0);
    const [sheetPrefix, setSheetPrefix] = useState<string>("");
    const { logout, token } = useAuth();
    const [pdfPageRotations, setPdfPageRotations] = useState<{[key: number]: number}>({});
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
            setFileUrl(URL.createObjectURL(droppedFile));
            setShowPreview(true);
            setNeedsRotation(null);
        }
    };

    const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if (selectedFile) {
            setFile(selectedFile);
            setFileUrl(URL.createObjectURL(selectedFile));
            setShowPreview(true);
            setNeedsRotation(null);
        }
    };

    function onDocumentLoadSuccess({numPages}: { numPages: number }) {
        setNumPages(numPages);
    }

    const uploadFile = async () => {
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);
        formData.append("start_page", startPage);
        formData.append("end_page", endPage);
        formData.append("sheet_prefix", sheetPrefix);


        const response = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/upload`, {
            method: "POST",
            body: formData,
            headers: { "Authorization": `Bearer ${token}` }
        });

        const data = await response.json();

        return data;
    };

    const rotatePage = (direction: "left" | "right") => {
  setPageRotations(prev => {
    const current = prev[currentPage - 1] || 0;
    const newRotation =
      direction === "right"
        ? (current + 90) % 360
        : (current - 90 + 360) % 360;

    return {
      ...prev,
      [currentPage - 1]: newRotation,
    };
  });
};

    async function handleApplyRotation() {
  if (!file) return;
  
  try {
    setIsFixing(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("rotation_map", JSON.stringify(pageRotations));
    console.log("Sending rotation map:", pageRotations);

    const response = await fetch(
      `${import.meta.env.VITE_API_URL}/api/v1/pdf/fix`,
      {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}` },
        body: formData,
      }
    );

    if (!response.ok) {
      throw new Error("Failed to fix PDF");
    }

    const blob = await response.blob();

    const newFile = new File([blob], file.name, {
      type: "application/pdf",
    });

    const newUrl = URL.createObjectURL(newFile);
    const oldUrl = fileUrl;

    setFile(newFile);
    setFileUrl(newUrl);
    setPdfVersion(prev => prev + 1);
    if (oldUrl) {
        setTimeout(() => URL.revokeObjectURL(oldUrl), 1000);
    }
    // setPageRotations({});
   setNeedsRotation(false);
    setCurrentPage(1);

  } catch (err) {
    console.error("Fix failed:", err);
  } finally {
    setIsFixing(false);
  }
}

    useEffect(() => {
  return () => {
    if (fileUrl) URL.revokeObjectURL(fileUrl);
  };
}, [fileUrl]);

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
                           
                            {file ? (
                                <>
                                    <FileText className="w-16 h-16 text-accent mb-4"/>
                                    <p className="text-foreground mb-2">{file.name}</p>
                                    <p className="text-sm text-muted-foreground">
                                        Click to change file
                                    </p>
                                </>
                            ) : (
                                <>
                                    <Upload className="w-16 h-16 text-muted-foreground mb-4"/>
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
                    {isFixing && (
  <p className="text-sm text-muted-foreground">
    Fixing orientation... please wait
  </p>
)}

                    {showPreview && file && (
                        <div className="flex flex-col items-center gap-4">
                            
                            <Document  key={pdfVersion}file={fileUrl}
                              onLoadSuccess={(data) => {
                                    console.log("PDF LOADED SUCCESS:", data);
                                    onDocumentLoadSuccess(data);
                                }}
                                onLoadError={(err) => {
                                    console.error(" PDF LOAD ERROR:", err);
                                }}
                               loading={<p>Loading preview...</p>}>
                            <Page
                            pageNumber={currentPage}
                            width={400}
                             onLoadSuccess={(page) => {
                                // page.rotate is the built-in PDF rotation (0, 90, 180, 270)
                                setPdfPageRotations(prev => ({
                                    ...prev,
                                    [currentPage]: page.rotate || 0
                                }));
                            }}
                            rotate={(pageRotations[currentPage - 1] || 0) + (pdfPageRotations[currentPage] || 0)}
                            />
                            </Document>
                            <div className="flex gap-4 items-center">
                            <Button
                                disabled={currentPage <= 1}
                                onClick={() => setCurrentPage(p => p - 1)}
                            >
                                Prev
                            </Button>

                            <span>Page {currentPage} / {numPages}</span>

                            <Button
                                disabled={currentPage >= (numPages || 1)}
                                onClick={() => setCurrentPage(p => p + 1)}
                            >
                                Next
                            </Button>
                            </div>

                          
                            {needsRotation === null && (
                            <div className="flex gap-4">
                                <Button onClick={() => setNeedsRotation(false)}>
                                Looks Good 
                                </Button>
                                <Button onClick={() => setNeedsRotation(true)}>
                                Needs Rotation 
                                </Button>
                            </div>
                            )}

                            {needsRotation && (
                            <div className="flex gap-4">
                                <Button onClick={() => rotatePage("right")}>
                                Rotate Right
                                </Button>
                                <Button onClick={() => rotatePage("left")}>
                                Rotate Left
                                </Button>
                               <Button onClick={handleApplyRotation} disabled={isFixing}>
                                {isFixing ? "Fixing..." : "Apply Fix"}
                                </Button>
                            </div>
                            )}
                        </div>
                        )}

                    


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
                            <Input placeholder="Starting Page" value={startPage}
                                   onChange={(e) => setStartPage(e.target.value)}/>
                            <Input placeholder="Ending Page" value={endPage}
                                   onChange={(e) => setEndPage(e.target.value)}/>
                        </div>
                    </div>

                     {/* SETTINGS  */}
                    <Input 
                        placeholder="Sheet prefix to strip (optional, e.g. FA31137)" 
                        value={sheetPrefix}
                        onChange={(e) => setSheetPrefix(e.target.value)}
                    />
                    <p className="text-xs text-muted-foreground">
                        If sheet numbers in title block include a project prefix (e.g. FA31137-ST-DT-0001), 
                        enter the prefix here so callout references match correctly.
                    </p>

                    {/* Action Button */}
                    <Button
                        onClick={
                            async () => {
                            if (isUploading) return; 
                            try {
                                setIsUploading(true); 
                                const result = await uploadFile();
                                if (result) {
                                    onStartProcessing(result.job_id, result.file_path);
                                }else{
                                    setIsUploading(false);
                                }
                            }
                            catch (err) {
                                console.error("Upload failed:", err);
                                setIsUploading(false);
                            }
                            }}
                        disabled={!file || !startPage || !endPage || isUploading ||   needsRotation === null || needsRotation === true}
                        className="w-full bg-accent hover:bg-accent/90 text-accent-foreground"
                        size="lg"
                        style={{cursor:'pointer'}}
                    >
                       {isUploading ? "Uploading..." : "Run Estimation"}
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

