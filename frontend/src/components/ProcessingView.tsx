import {useEffect, useMemo, useState,useRef} from "react";
import {Check, Loader2} from "lucide-react";
import {Card, CardContent} from "./ui/card";
import {Document, Page, pdfjs} from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface ProcessingViewProps {
    jobId: string;
    filePath: string;
    onComplete: (result: any) => void;
    onFailed?:()=>void;
}

const steps = [
    {label: "Pages Classified"},              // classify
    // {label: "Text Rules Processed"},          // process_text
    {label: "Floor Plans Identified"},        // process_plans
    {label: "Details Extracted"},             // process_details
    {label: "Bill of Materials Generated"}    // agent_4_merger
];

export function ProcessingView({jobId, filePath, onComplete, onFailed}: ProcessingViewProps) {
    const [completedSteps, setCompletedSteps] = useState<number[]>([]);
    const [loadingResult, setLoadingResult] = useState(false);
    const [numPages, setNumPages] = useState<number>();
    const [pageNumber, setPageNumber] = useState(1);
    const [isPdfReady, setIsPdfReady] = useState(false);
    const [hitlData, setHitlData] = useState<any>(null);
    const [showHITL, setShowHITL] = useState(false);
    const [pageMapEdits, setPageMapEdits] = useState<Record<string, string>>({});
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const imgRef = useRef<HTMLImageElement>(null);
    const drawingRef = useRef<{active:boolean, startX:number, startY:number}>({active:false,startX:0,startY:0});
    const [bboxes, setBboxes] = useState<{x:number,y:number,width:number,height:number}[]>([]);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [deletedMineruIndices, setDeletedMineruIndices] = useState<Set<number>>(new Set());
    const [hitlProgress, setHitlProgress] = useState<{current: number, total: number, remaining: number} | null>(null);
    const eventSourceRef = useRef<EventSource | null>(null);
    const [activeStep, setActiveStep] = useState<number>(0);
    const [hasFailed, setHasFailed] = useState(false);


    
    function onDocumentLoadSuccess({numPages}: { numPages: number }) {
        setNumPages(numPages);
    }

    function connectSSE() {
    if (eventSourceRef.current) eventSourceRef.current.close();
    const es = new EventSource(`${import.meta.env.VITE_API_URL}/api/v1/jobs/${jobId}/stream`);
    const nodeMap: Record<string, number> = {
            classify: 0,
            // process_text: 1,
            process_plans: 1,
            process_details: 2,
            agent_4_merger: 3
        };
    eventSourceRef.current = es;
    es.onmessage = async (event) => { 
        console.log(" RAW SSE EVENT ", event.data);
            let data;
            try {
                data = JSON.parse(event.data);
            } catch (e) {
                console.error("JSON parse failed", event.data);
                return;
            }
            console.log(" PARSED SSE ", data);
            if (data.status === "failed") {
                es.close();
                setHasFailed(true);
                onFailed?.();
                return;
            }
            if (data.step === "hitl_review") {
                console.log(" HITL EVENT FULL ", data);

                if (!data.data) {
                    console.error(" INVALID HITL DATA", data);
                    return;
                }
                if (data.data.type === "classify_review") {
                    setPageMapEdits(data.data.page_map);
                }

                setHitlData(data.data);
                setShowHITL(true);
                setHitlData(data.data);
                if (data.data.total_hitl_pages !== undefined) {
                    setHitlProgress({
                        current: data.data.current_hitl_index,
                        total: data.data.total_hitl_pages,
                        remaining: data.data.remaining_after_this
                    });
                }
                return;
            }
            if (!data.step) return;
            const step = data.step?.toLowerCase().trim();
            const status = data.status?.toLowerCase();

            console.log(" STEP UPDATE", step, status);

            

            if (!data.step) {
                setCompletedSteps([]); 
                return;
            }
            const stepIndex = nodeMap[step];
            if (stepIndex !== undefined) {
                
                if (status === "completed" && step !== "agent_4_merger") {
                    // explicit completed for a mid-step — mark it and all before as done
                    const completed = [];
                    for (let i = 0; i <= stepIndex; i++)completed.push(i);
                    setCompletedSteps(completed);
                    setActiveStep(stepIndex + 1);
                } else {
                    const completed = [];
                    for (let i = 0; i < stepIndex; i++) completed.push(i);
                
                    setCompletedSteps(completed);
                    setActiveStep(stepIndex);
                }
            }

            if (status === "completed" && step === "agent_4_merger"){
                es.close();

                setLoadingResult(true);

                const res = await fetch(
                    `${import.meta.env.VITE_API_URL}/api/v1/jobs/${jobId}/result`
                );

                const result = await res.json();

                onComplete(result);
            }
     };
    es.onerror = (err) => {
        console.error("SSE error:", err);
        // setHasFailed(true);
        // es.close();
        // onFailed?.();
    }

    return () => {
            es.close();
        };
    }

    function normalizePath(path: string) {
  return path
    .replace(/^\/+/, "")      // remove leading /
    .replace(/^data\//, "");  // remove data/
}
    
    const pdfFileObject = useMemo(() => ({url: `${import.meta.env.VITE_API_URL}/api/v1/${normalizePath(filePath)}`}), [filePath]);
    
  useEffect(() => {
    if (hitlData) {
        console.log(" HITL DATA UPDATED ->", hitlData);
    }
}, [hitlData]);

  useEffect(() => {
    redrawCanvas();
  }, [bboxes,hitlData, deletedMineruIndices]);

  useEffect(() => {
    if (!jobId) return;
    const cleanup = connectSSE();
    return cleanup;
}, [jobId]);

  useEffect(() => {
  if (!filePath) return;
  console.log("filePath ->", filePath);
  const checkFile = async () => {
    try {
        const cleanPath = normalizePath(filePath)
        const url = `${import.meta.env.VITE_API_URL}/api/v1/${cleanPath}`
      const res = await fetch(url,{ method: "HEAD" } 
);

      if (res.ok) {
        setIsPdfReady(true);
      } else {
        setTimeout(checkFile, 1000);
      }
    } catch {
      setTimeout(checkFile, 1000);
    }
  };

  checkFile();
}, [filePath]);


    const imageNaturalW = hitlData?.image_width  ?? 0;
    const imageNaturalH = hitlData?.image_height ?? 0;

    function toImageCoords(displayX: number, displayY: number) {
            const canvas = canvasRef.current!;
            const rect   = canvas.getBoundingClientRect();
            const scaleX = imageNaturalW / rect.width;
            const scaleY = imageNaturalH / rect.height;
            return { x: displayX * scaleX, y: displayY * scaleY };
        }

    function getRelativePos(e: React.MouseEvent<HTMLCanvasElement>) {
        const rect = canvasRef.current!.getBoundingClientRect();
        return { x: e.clientX - rect.left, y: e.clientY - rect.top }; // display px
    }
    function redrawCanvas(extraRect?: {x:number,y:number,width:number,height:number} | null) {
        const canvas = canvasRef.current;
        const img    = imgRef.current;
        if (!canvas || !img) return;
        const ctx = canvas.getContext("2d")!;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // green = auto-detected bboxes from backend (already in image-pixel space)
        if (hitlData?.bboxes?.length) {
            for (let idx = 0; idx < hitlData.bboxes.length; idx++) {
                if (deletedMineruIndices.has(idx)) continue;  // skip deleted
                const b = hitlData.bboxes[idx];
                const canvas = canvasRef.current!;
                const displayScale = canvas.width / (canvas.getBoundingClientRect().width || canvas.width);
                const btnSize = Math.max(30, 20 * displayScale);

                ctx.strokeStyle = "#22c55e";
                ctx.lineWidth   = 3;
                ctx.fillStyle   = "rgba(34,197,94,0.1)";
                ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
                ctx.fillRect  (b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

                // × button on green box
                const btnX = b.x1 + (b.x2 - b.x1) - btnSize;
                const btnY = b.y1;
                ctx.fillStyle = "#22c55e";
                ctx.fillRect(btnX, btnY, btnSize, btnSize);
                ctx.fillStyle = "white";
                ctx.font = `bold ${btnSize * 0.75}px sans-serif`;
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("×", btnX + btnSize / 2, btnY + btnSize / 2);
            }
        }
        const displayScale = canvas.width / (canvas.getBoundingClientRect().width || canvas.width);
        const btnSize = Math.max(30, 20 * displayScale);  // at least 30px in image space

        // red = user-drawn boxes (stored in image-pixel space)
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth   = 2;
        ctx.fillStyle   = "rgba(239,68,68,0.1)";
        for (const box of bboxes) {
            ctx.strokeRect(box.x, box.y, box.width, box.height);
            ctx.fillRect  (box.x, box.y, box.width, box.height);

            const btnX = box.x + box.width - btnSize;
            const btnY = box.y;

            ctx.fillStyle = "#ef4444";
            ctx.fillRect(btnX, btnY, btnSize, btnSize);
            ctx.fillStyle = "white";
            ctx.font = `bold ${btnSize * 0.75}px sans-serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("X", btnX + btnSize / 2, btnY + btnSize / 2);

            ctx.strokeStyle = "#ef4444";
            ctx.fillStyle = "rgba(239,68,68,0.1)";
        }
        if (extraRect) {
            ctx.strokeRect(extraRect.x, extraRect.y, extraRect.width, extraRect.height);
            ctx.fillRect  (extraRect.x, extraRect.y, extraRect.width, extraRect.height);
        }
    }

    function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
        const pos = getRelativePos(e);
        drawingRef.current = { active: true, startX: pos.x, startY: pos.y };
    }

    function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
        if (!drawingRef.current.active) return;
        const pos = getRelativePos(e);
        const { startX, startY } = drawingRef.current;
        const a = toImageCoords(startX, startY);
        const b = toImageCoords(pos.x,  pos.y);
        redrawCanvas({
            x:      Math.min(a.x, b.x),
            y:      Math.min(a.y, b.y),
            width:  Math.abs(b.x - a.x),
            height: Math.abs(b.y - a.y),
        });
    }

    function handleMouseUp(e: React.MouseEvent<HTMLCanvasElement>) {
        if (!drawingRef.current.active) return;
        const pos = getRelativePos(e);
        const { startX, startY } = drawingRef.current;
        drawingRef.current = { active: false, startX: 0, startY: 0 };
        const a = toImageCoords(startX, startY);
        const b = toImageCoords(pos.x,  pos.y);
        const rect = {
            x:      Math.min(a.x, b.x),
            y:      Math.min(a.y, b.y),
            width:  Math.abs(b.x - a.x),
            height: Math.abs(b.y - a.y),
        };
        if (rect.width > 5 && rect.height > 5) {
            setBboxes(prev => [...prev, rect]);
        }
    }

    function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    // Only handle clicks, not drag-ends (drawingRef is already reset by mouseUp)
    const pos = getRelativePos(e);
    // Convert display pos back to image coords
    const imgPos = toImageCoords(pos.x, pos.y);

    const canvas = canvasRef.current!;
    const displayScale = canvas.width / (canvas.getBoundingClientRect().width || canvas.width);
    const btnSize = Math.max(30, 20 * displayScale);  

    if (hitlData?.bboxes?.length) {
        for (let idx = 0; idx < hitlData.bboxes.length; idx++) {
            if (deletedMineruIndices.has(idx)) continue;
            const b = hitlData.bboxes[idx];
            const btnX = b.x1 + (b.x2 - b.x1) - btnSize;
            const btnY = b.y1;
            if (imgPos.x >= btnX && imgPos.x <= btnX + btnSize &&
                imgPos.y >= btnY && imgPos.y <= btnY + btnSize) {
                setDeletedMineruIndices(prev => new Set([...prev, idx]));
                return;
            }
        }
    }

    const indexToDelete = bboxes.findIndex(box => {
        const btnX = box.x + box.width - btnSize;
        const btnY = box.y;
        return (
            imgPos.x >= btnX && imgPos.x <= btnX + btnSize &&
            imgPos.y >= btnY && imgPos.y <= btnY + btnSize
        );
    });

    if (indexToDelete !== -1) {
        setBboxes(prev => prev.filter((_, i) => i !== indexToDelete));
    }
}

    function handleImageLoad() {
        const canvas = canvasRef.current;
        const img    = imgRef.current;
        if (!canvas || !img) return;
        // use the dimensions from the backend payload — not img.naturalWidth
        // because the annotated PNG served by FastAPI IS the same size
        canvas.width  = imageNaturalW || img.naturalWidth;
        canvas.height = imageNaturalH || img.naturalHeight;
        redrawCanvas();
    }

    // convert image-pixel bboxes → {x1,y1,x2,y2} integers — exact format nodes.py expects
    function toBackendBboxes() {
        return bboxes.map(b => ({
            x1: Math.round(b.x),
            y1: Math.round(b.y),
            x2: Math.round(b.x + b.width),
            y2: Math.round(b.y + b.height),
        }));
    }

    async function handleApprove() {
        setIsSubmitting(true);
        try {
            // merge original auto-detected + user-drawn
            const original = (hitlData?.bboxes ?? []) as {x1:number,y1:number,x2:number,y2:number}[];
            const survivingMineruBoxes = (hitlData?.bboxes ?? []).filter((_, idx) => !deletedMineruIndices.has(idx));
            await fetch(`${import.meta.env.VITE_API_URL}/api/v1/jobs/${jobId}/hitl`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                corrected_bboxes: [...survivingMineruBoxes.map(b => ({x1:b.x1,y1:b.y1,x2:b.x2,y2:b.y2})), ...toBackendBboxes()],
                deleted_mineru_bboxes: Array.from(deletedMineruIndices).map(idx => hitlData.bboxes[idx])
            }),
                
            });
            setShowHITL(false);
            setHitlData(null);
            setBboxes([]);
            connectSSE(); 
        } finally {
            setIsSubmitting(false);
            setDeletedMineruIndices(new Set());
        }
    }
 
    return (
        
        <div className="grid grid-cols-2 gap-6 h-full p-6">
          {showHITL && hitlData && (
            hitlData.type === "classify_review" ? (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
            <div className="bg-white p-4 rounded shadow-lg max-w-xl w-full mx-4">
                <h3 className="text-base font-medium mb-1">Verify Page Classification</h3>
                <p className="text-sm text-gray-500 mb-3">
                    Review the detected page types and correct any mistakes before processing begins.
                </p>
                <div className="overflow-auto max-h-[60vh] border rounded divide-y">
                    {Object.entries(pageMapEdits)
                        .sort(([a], [b]) => Number(a) - Number(b))
                        .map(([pageNum, pageType]) => (
                            <div key={pageNum} className="flex items-center justify-between px-4 py-2">
                                <span className="text-sm font-medium">Page {Number(pageNum) + 1}</span>
                                <select
                                    value={pageType}
                                    onChange={(e) => setPageMapEdits(prev => ({
                                        ...prev,
                                        [pageNum]: e.target.value
                                    }))}
                                    className="text-sm border rounded px-2 py-1"
                                >
                                    <option value="floor">floor</option>
                                    <option value="section">section</option>
                                    <option value="text">text</option>
                                    <option value="ignore">ignore</option>
                                </select>
                            </div>
                        ))}
                </div>
                <div className="flex justify-end mt-4">
                    <button
                        disabled={isSubmitting}
                        onClick={async () => {
                            setIsSubmitting(true);
                            await fetch(`${import.meta.env.VITE_API_URL}/api/v1/jobs/${jobId}/hitl`, {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ corrected_page_map: pageMapEdits })
                            });
                            setIsSubmitting(false);
                            setShowHITL(false);
                            setHitlData(null);
                            setPageMapEdits({});
                            connectSSE();
                        }}
                        className="px-4 py-2 text-sm bg-emerald-600 text-white rounded hover:bg-emerald-700 disabled:opacity-50 flex items-center gap-2"
                    >
                        {isSubmitting && <Loader2 className="w-4 h-4 animate-spin" />}
                        Confirm Classification
                    </button>
                </div>
            </div>
        </div>
    ):
            <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
                <div className="bg-white p-4 rounded shadow-lg max-w-3xl w-full mx-4">
                    <h3 className="text-base font-medium mb-1">Mark missing components</h3>
                    {hitlProgress && (
                        <p className="text-sm text-gray-500">
                          Page {hitlProgress.current} of {hitlProgress.total} — {hitlProgress.remaining} remaining
                        </p>
                    )}
                    <p className="text-sm text-gray-500 mb-3">
                        Click and drag to draw boxes. Submit when done or skip to continue.
                    </p>

                    <div className="relative overflow-auto max-h-[60vh] border rounded">
                        {/* hidden img — source for canvas drawImage */}
                        <img
                            ref={imgRef}
                            src={`${import.meta.env.VITE_API_URL}${hitlData.image_path}`}
                            alt="HITL source"
                            className="hidden"
                            onLoad={handleImageLoad}
                            crossOrigin="anonymous"
                        />
                        <canvas
                            ref={canvasRef}
                            className="block max-w-full"
                            style={{ cursor: "crosshair" }}
                            onMouseDown={handleMouseDown}
                            onMouseMove={handleMouseMove}
                            onMouseUp={handleMouseUp}
                            onClick={handleCanvasClick}
                        />
                    </div>

                    <div className="flex items-center justify-between mt-4">
                        <span className="text-sm text-gray-500">
                            {bboxes.length} box{bboxes.length !== 1 ? "es" : ""} drawn
                            {bboxes.length > 0 && (
                                <>
                                <button
                                        className="ml-3 text-blue-500 underline text-sm"
                                        onClick={() => setBboxes(prev => prev.slice(0, -1))}
                                    >
                                        Undo
                                    </button>
                             
                                <button
                                    className="ml-3 text-red-500 underline text-sm"
                                    onClick={() => { setBboxes([]); redrawCanvas(); }}
                                >
                                   
                                    Clear all
                                </button>
                                    </>
                            )}
                        </span>
                        <div className="flex gap-3">
                            <button
                                disabled={isSubmitting}
                                onClick={async () => {
                                    setIsSubmitting(true);
                                    await fetch(`${import.meta.env.VITE_API_URL}/api/v1/jobs/${jobId}/hitl`, {
                                        method: "POST",
                                        headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({ 
                                            corrected_bboxes: [],
                                            deleted_mineru_bboxes: []
                                        })
        
                                    });
                                    setIsSubmitting(false);
                                    setShowHITL(false);
                                    setHitlData(null);
                                    setBboxes([]);
                                    setDeletedMineruIndices(new Set())
                                    connectSSE(); 
                                }}
                                className="px-4 py-2 text-sm border rounded hover:bg-gray-50 disabled:opacity-50"
                            >
                                Skip
                            </button>
                            <button
                                onClick={handleApprove}
                                disabled={isSubmitting}
                                className="px-4 py-2 text-sm bg-emerald-600 text-white rounded hover:bg-emerald-700 disabled:opacity-50 flex items-center gap-2"
                            >
                                {isSubmitting && <Loader2 className="w-4 h-4 animate-spin" />}
                                Submit
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        )}
            {loadingResult && (
                <div className="absolute inset-0 bg-background/80 flex items-center justify-center z-50">
                    <div className="text-center">
                        <Loader2 className="w-12 h-12 animate-spin mx-auto mb-4 text-accent"/>
                        <p className="text-lg font-medium">Fetching final results...</p>
                    </div>
                </div>
            )}
            {/* PDF Preview */}
            <Card className="h-full">
                <CardContent className="p-4 h-full flex flex-col">

                    <div className="flex-1 overflow-auto flex justify-center">
                       {!isPdfReady ? (
                              <div className="flex items-center justify-center h-full">
                                <Loader2 className="w-10 h-10 animate-spin" />
                                <p>Loading PDF...</p>
                              </div>
                            ) : (
                              <Document
                                file={pdfFileObject}
                                onLoadSuccess={onDocumentLoadSuccess}
                                onLoadError={(err) => console.error("PDF load error:", err)}
                              >
                                <Page pageNumber={pageNumber} width={500} />
                              </Document>
                                )
                        }
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
                                    isActive={index === activeStep && !completedSteps.includes(index)}
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
                    <Check className="w-4 h-4"/>
                ) : isActive ? (
                    <Loader2 className="w-4 h-4 animate-spin"/>
                ) : (
                    <div className="w-2 h-2 rounded-full bg-muted-foreground/40"/>
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
