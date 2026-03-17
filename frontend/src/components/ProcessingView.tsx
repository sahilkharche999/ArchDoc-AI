import { useEffect, useState, useMemo } from "react";
import { Check, Loader2 } from "lucide-react";
import { Card, CardContent } from "./ui/card";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface ProcessingViewProps {
  jobId: string;
  filePath: string;
  onComplete: (result: any) => void;
}

const steps = [
  { label: "Pages Classified", delay: 1000 },
  { label: "Floor Plans Identified", delay: 2000 },
  { label: "Structural Symbols Extracted", delay: 3000 },
  { label: "Bill of Materials Generated", delay: 4000 },
];

export function ProcessingView({ jobId, filePath, onComplete, }: ProcessingViewProps) {
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  const [loadingResult, setLoadingResult] = useState(false);
  const [numPages, setNumPages] = useState<number>();
  const [pageNumber, setPageNumber] = useState(1);
  const pdfFileObject = useMemo(() => ({ url: `${import.meta.env.VITE_API_URL}/api/v1/${filePath}` }), [filePath]);
  function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
    setNumPages(numPages);
  }
  useEffect(() => {

    console.log("ProcessingView mounted");
    console.log("filePath:", filePath);

    const url = `${import.meta.env.VITE_API_URL}/api/v1/jobs/stream?job_id=${jobId}&file_path=${encodeURIComponent(filePath)}`;
    console.log("Connecting to:", url);

    const eventSource = new EventSource(url);

    eventSource.onopen = () => {
      console.log("Stream connected");
    };

    eventSource.onerror = (err) => {
      console.error("Stream error:", err);
    };

    eventSource.onmessage = (event) => {

      const data = JSON.parse(event.data);

      console.log("Agent finished:", data.node);

      const nodeMap: Record<string, number> = {
        classify: 0,
        process_text: 1,
        process_plans: 2,
        agent_4_merger: 3
      };

      const stepIndex = nodeMap[data.node];

      if (stepIndex !== undefined) {
        setCompletedSteps(prev => [...prev, stepIndex]);
      }


      if (data.node === "agent_4_merger") {

        eventSource.close();

        setLoadingResult(true);

        fetch(`${import.meta.env.VITE_API_URL}/api/v1/jobs/${jobId}/result`)
          .then(res => res.json())
          .then(result => {

            console.log("BOM Result:", result);

            // Pass data to dashboard
            onComplete(result);

          })
          .catch(err => {
            console.error("Failed to fetch result", err);
          })
          .finally(() => {
            setLoadingResult(false);
          });
      }

    };

    return () => eventSource.close();

  }, [filePath, jobId]);

  return (
    <div className="grid grid-cols-2 gap-6 h-full p-6">
      {loadingResult && (
        <div className="absolute inset-0 bg-background/80 flex items-center justify-center z-50">
          <div className="text-center">
            <Loader2 className="w-12 h-12 animate-spin mx-auto mb-4 text-accent" />
            <p className="text-lg font-medium">Fetching final results...</p>
          </div>
        </div>
      )}
      {/* PDF Preview */}
      <Card className="h-full">
        <CardContent className="p-4 h-full flex flex-col">

          <div className="flex-1 overflow-auto flex justify-center">
            <Document
              file={pdfFileObject}
              onLoadSuccess={onDocumentLoadSuccess}
              onLoadError={(err) => console.error("PDF load error:", err)}
            >
              <Page pageNumber={pageNumber} width={500} />
            </Document>
          </div>
          assets/586e916c-5628-45da-81ca-889ed5cf1e66_structural.pdf
          <div className="flex justify-between items-center mt-4">
            <button
              onClick={() => setPageNumber(p => Math.max(p - 1, 1))}
              disabled={pageNumber <= 1}
            >
              Previous
            </button>

            <p>
              Page {pageNumber} of {numPages}
            </p>

            <button
              onClick={() =>
                setPageNumber(p => (numPages ? Math.min(p + 1, numPages) : p))
              }
              disabled={pageNumber === numPages}
            >
              Next
            </button>
          </div>

        </CardContent>
      </Card>
      {/* Processing Progress */}
      <div className="flex flex-col justify-center">
        <Card>
          <CardContent className="p-6">
            <h2 className="text-xl mb-6">Processing Drawings</h2>
            <div className="space-y-4">
              {steps.map((step, index) => (
                <ProcessingStep
                  key={index}
                  label={step.label}
                  isCompleted={completedSteps.includes(index)}
                  isActive={
                    completedSteps.length === index &&
                    !completedSteps.includes(index)
                  }
                />
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

interface ProcessingStepProps {
  label: string;
  isCompleted: boolean;
  isActive: boolean;
}

function ProcessingStep({
  label,
  isCompleted,
  isActive,
}: ProcessingStepProps) {
  return (
    <div className="flex items-center gap-3">
      <div
        className={`w-6 h-6 rounded-full flex items-center justify-center transition-all ${isCompleted
          ? "bg-emerald-500 text-white"
          : isActive
            ? "bg-accent text-accent-foreground"
            : "bg-muted text-muted-foreground"
          }`}
      >
        {isCompleted ? (
          <Check className="w-4 h-4" />
        ) : isActive ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <div className="w-2 h-2 rounded-full bg-muted-foreground/40" />
        )}
      </div>
      <span
        className={`text-sm transition-colors ${isCompleted ? "text-foreground" : "text-muted-foreground"
          }`}
      >
        {label}
      </span>
    </div>
  );
}
