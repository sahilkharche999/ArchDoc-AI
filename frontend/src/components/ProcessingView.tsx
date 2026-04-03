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
  { label: "Pages Classified" },              // classify
  { label: "Text Rules Processed" },          // process_text
  { label: "Floor Plans Identified" },        // process_plans
  { label: "Details Extracted" },             // process_details
  { label: "Bill of Materials Generated" }    // agent_4_merger
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
  if (!jobId) return;

  const eventSource = new EventSource(
    `${import.meta.env.VITE_API_URL}/api/v1/jobs/${jobId}/stream`
  );

  const nodeMap: Record<string, number> = {
    classify: 0,
    process_text: 1,
    process_plans: 2,
    process_details: 3,
    agent_4_merger: 4
  };

  eventSource.onmessage = async (event) => {
    const data = JSON.parse(event.data);

    const step = data.step?.toLowerCase().trim();
    const status = data.status?.toLowerCase();

    const stepIndex = nodeMap[step];
    console.log(`Here is the current step: ${stepIndex}`)
    if (stepIndex !== undefined) {
      const completed = [];
      for (let i = 0; i <= stepIndex; i++) {
        completed.push(i);
      }
      setCompletedSteps(completed);
    }


    if (status === "completed") {
      eventSource.close();

      setLoadingResult(true);

      const res = await fetch(
        `${import.meta.env.VITE_API_URL}/api/v1/jobs/${jobId}/result`
      );

      const result = await res.json();

      onComplete(result);
    }
  };

  eventSource.onerror = (err) => {
    console.error("SSE error:", err);
    eventSource.close();
  };

  return () => {
    eventSource.close();
  };
}, [jobId]);

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
