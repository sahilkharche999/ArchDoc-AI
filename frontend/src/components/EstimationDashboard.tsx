import {useEffect, useMemo, useRef, useState} from "react";
import {Card, CardContent} from "./ui/card";
import { useSearchParams } from "react-router-dom";
import {Tabs, TabsContent, TabsList, TabsTrigger} from "./ui/tabs";
import {BOMTable} from "./BOMTable";
import {TraceabilityView} from "./TraceabilityView";
import {DashboardHeader} from "./DashboardHeader";
import {SummaryTab} from "./SummaryTab";
import {DollarSign, Package, Weight,Download} from "lucide-react";
import {BOMItem} from "../types/bom"
import {Document, Page, pdfjs} from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";
import * as XLSX from 'xlsx';
import { useAuth } from "../app/context/AuthContext";


pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;


interface EstimationDashboardProps {
    projectId?: string | null;
    bomData: BOMItem[];
    unreferencedDetails:any[];
    message?: string;
}


function formatDate(dateString: string) {
    const date = new Date(dateString)

    return date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric"
    })
}

export function EstimationDashboard({projectId, bomData,unreferencedDetails = [],message}: EstimationDashboardProps) {
    const [pricePerLb, setPricePerLb] = useState(0);
    const [fabricationMarkup, setFabricationMarkup] = useState(0);
    const [galvanizing, setGalvanizing] = useState(false);
    const [searchParams, setSearchParams] = useSearchParams();
    const [projectMeta, setProjectMeta] = useState<any>(null);
    const [editableBom, setEditableBom] = useState<BOMItem[]>(bomData);
    const [numPages, setNumPages] = useState<number>();
    const [pageNumber, setPageNumber] = useState(1);
    const containerRef = useRef<HTMLDivElement | null>(null);
    const [scale, setScale] = useState(1);
    const [zoom, setZoom] = useState(0.4);
    const { logout, token } = useAuth();
    const activeTab = searchParams.get("tab") || "summary";
    useEffect(() => {
        if (!projectId) return;
        fetch(`${import.meta.env.VITE_API_URL}/api/v1/projects/${projectId}`,{ headers: { "Authorization": `Bearer ${token}` }})
            .then(res => res.json())
            .then(data => {
                console.log("projectMeta:", data);
                setProjectMeta(data);
            });
    }, [projectId]);

    useEffect(() => {
        setEditableBom(bomData);
    }, [bomData]);

    const pdfFileObject = useMemo(() => {
        if (!projectMeta?.file_path) return null;
        return {
            url: `${import.meta.env.VITE_API_URL}/api/v1/${projectMeta.file_path}`
        };
    }, [projectMeta]);

    function onDocumentLoadSuccess({numPages}: { numPages: number }) {
        setNumPages(numPages);

        if (containerRef.current) {
            const containerWidth = containerRef.current.offsetWidth;

            const basePdfWidth = 800;

            const newScale = containerWidth / basePdfWidth;

            setScale(newScale);
        }
    }
    async function handleSaveAndExport() {
    // Save to server
    if (projectId) {
        await fetch(`${import.meta.env.VITE_API_URL}/api/v1/jobs/${projectId}/bom`, {
            method: "PATCH",
            headers: {"Content-Type": "application/json" ,"Authorization": `Bearer ${token}`},
            body: JSON.stringify({ bom: editableBom })
        });
    }

    // Export to Excel
    const rows = editableBom.map(item => {
        const totalWeight = item.total_linear_feet * item.quantity * item.lb_per_ft;
        const totalCost = totalWeight * item.charge_per_lb;
        return {
            "Description": item.description,
            "Material Size": item.material_size,
            "Quantity": item.quantity,
            "Total Linear Ft": item.total_linear_feet,
            "Lb/Ft": item.lb_per_ft,
            "Total Weight (lbs)": parseFloat(totalWeight.toFixed(2)),
            "Price/Lb": item.charge_per_lb,
            "Total Cost ($)": parseFloat(totalCost.toFixed(2)),
            "Total Bolts": item.total_bolts,
            "Total Holes": item.total_holes,
            "Weld Inches": item.total_weld_inches,
            "Source Sheet": item.source_sheet,
            "Source Symbol": item.source_symbol,
            "Logic Trace": item.logic_trace,
        };
    });

    const ws = XLSX.utils.json_to_sheet(rows);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "BOM");
    const fileName = `DAX_BOM_${projectMeta?.name || projectId || "export"}.xlsx`;
    XLSX.writeFile(wb, fileName);
}


    // Calculate totals
    const totalLinearFeet = editableBom.reduce(
        (sum, item) => sum + item.total_linear_feet * item.quantity,
        0
    );
    const totalWeight = editableBom.reduce(
        (sum, item) =>
            sum + item.total_linear_feet * item.quantity * item.lb_per_ft,
        0
    );

    const totalPieces = editableBom.reduce(
        (sum, item) => sum + item.quantity,
        0
    );

    const baseCost = editableBom.reduce(
        (sum, item) =>
            sum +
            item.total_linear_feet *
            item.quantity *
            item.lb_per_ft *
            item.charge_per_lb,
        0
    );

    const fabricationCost = baseCost * (fabricationMarkup / 100);
    const galvanizingCost = galvanizing ? totalWeight * 0.45 : 0;

    const totalCost = baseCost + fabricationCost + galvanizingCost;
    return (
        <div className="flex flex-col h-full">
            <DashboardHeader
                projectName={projectMeta?.name}
                projectDate={
                    projectMeta?.date
                        ? formatDate(projectMeta.date)
                        : ""
                }
            />
            <div className="flex-1 overflow-auto p-6 space-y-6">
                {/* Summary Cards */}
                <div className="grid grid-cols-4 gap-4">
                    <SummaryCard
                        title="Total Weight"
                        value={totalWeight.toFixed(0)}
                        icon={<Weight className="w-5 h-5"/>}
                        suffix="lbs"
                    />
                    <SummaryCard
                        title="Total Pieces"
                        value={totalPieces.toString()}
                        icon={<Package className="w-5 h-5"/>}
                    />
                    <SummaryCard
                        title="Estimated Total Cost"
                        value={`$${totalCost.toFixed(2)}`}
                        icon={<DollarSign className="w-5 h-5"/>}
                        highlight
                    />
                </div>

                {/* Tabs */}
                <Tabs value={activeTab} onValueChange={(val) => setSearchParams({ tab: val })} className="w-full">
                    <TabsList className="grid w-full max-w-2xl grid-cols-4">
                        <TabsTrigger value="summary">Summary</TabsTrigger>
                        <TabsTrigger value="bom">Detailed BOM</TabsTrigger>
                        <TabsTrigger value="trace">Traceability</TabsTrigger>
                        <TabsTrigger value="untracked">Untracked</TabsTrigger>
                    </TabsList>

                    <TabsContent value="summary" className="space-y-4">
                        {message && (
                            <div className="rounded-lg border border-yellow-300 bg-yellow-50 px-4 py-3">
                                <p className="text-sm text-yellow-800">{message}</p>
                                {unreferencedDetails.length > 0 && (
                                    <p className="text-xs text-yellow-700 mt-1">
                                        {unreferencedDetails.length} section detail(s) were found — see the Untracked tab.
                                    </p>
                                )}
                            </div>
                        )}
                        {editableBom.length > 0 ? (
                            <SummaryTab
                                bomData={editableBom}
                                baseCost={baseCost}
                                fabricationCost={fabricationCost}
                                galvanizingCost={galvanizingCost}
                                fabricationMarkup={fabricationMarkup}
                                pricePerLb={pricePerLb}
                            />
                        ) : (
                            <Card>
                                <CardContent className="p-6 text-center text-sm text-muted-foreground">
                                    No materials to summarize for this document.
                                </CardContent>
                            </Card>
                        )}
                    </TabsContent>

                    <TabsContent value="bom">
                        <div className="flex justify-end mb-3">
                            <button
                                onClick={handleSaveAndExport}
                                className="flex items-center gap-2 px-4 py-2 bg-accent text-accent-foreground rounded-lg text-sm hover:bg-accent/90 cursor-pointer"
                            >
                                <Download className="w-4 h-4" />
                                Save & Export Excel
                            </button>
                        </div>
                        <BOMTable bomData={editableBom} setEditableBom={setEditableBom} pricePerLb={pricePerLb}/>
                    </TabsContent>


                    <TabsContent value="trace">
                        <TraceabilityView bomData={editableBom}/>
                    </TabsContent>

                    <TabsContent value="untracked">
                        <Card>
                            <CardContent className="p-4">
                                <p className="text-sm text-muted-foreground mb-4">
                                    Section details extracted by Agent 3 but not referenced in the floor plan by Agent 4. Verify these manually.
                                </p>
                                {unreferencedDetails.length === 0 ? (
                                    <p className="text-sm text-green-600">All details were referenced — no untracked items.</p>
                                ) : (
                                    <table className="w-full text-sm">
                                        <thead>
                                            <tr className="border-b">
                                                <th className="text-left py-2">Detail ID</th>
                                                <th className="text-left py-2">Title</th>
                                                <th className="text-left py-2">Materials</th>
                                                <th className="text-left py-2">Sheet</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {unreferencedDetails
                                            .filter(detail => {
                                                const bom = typeof detail.BOM === "string"
                                                    ? JSON.parse(detail.BOM)
                                                    : detail.BOM || [];
                                                return bom.length > 0;
                                            }).map((detail, idx) => {
                                                const bom = typeof detail.BOM === "string" 
                                                    ? JSON.parse(detail.BOM) 
                                                    : detail.BOM || [];
                                                return (
                                                    <tr key={idx} className="border-b hover:bg-muted/50">
                                                        <td className="py-2 font-mono text-orange-600">{detail.ID}</td>
                                                        <td className="py-2">{detail.Title || "—"}</td>
                                                        <td className="py-2">
                                                            {bom.map((m: any, i: number) => (
                                                                <span key={i} className="inline-block bg-orange-100 text-orange-800 text-xs px-2 py-0.5 rounded mr-1 mb-1">
                                                                    {m.item_name}
                                                                </span>
                                                            ))}
                                                        </td>
                                                        <td className="py-2 text-muted-foreground">{detail.Sheet || "—"}</td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                )}
                            </CardContent>
                        </Card>
                    </TabsContent>
                </Tabs>
                {/* PDF Preview */}
                <Card className="h-full">
                    <CardContent className="p-4 h-full flex flex-col">
                        <div className="flex justify-between items-center mb-2">

                            {/* LEFT: Zoom Controls */}
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={() => setZoom(z => Math.max(0.2, z - 0.2))}
                                    className="px-2 py-1 border rounded"
                                >
                                    -
                                </button>

                                <span className="text-sm">
      {(zoom * 100).toFixed(0)}%
    </span>

                                <button
                                    onClick={() => setZoom(z => Math.min(3, z + 0.2))}
                                    className="px-2 py-1 border rounded"
                                >
                                    +
                                </button>
                            </div>

                            {/* RIGHT: Reset */}
                            <button
                                onClick={() => setZoom(1)}
                                className="text-sm text-muted-foreground"
                            >
                                Reset
                            </button>

                        </div>

                        <div ref={containerRef} className="flex-1 overflow-auto flex justify-start">
                            {pdfFileObject && <Document
                                file={pdfFileObject}
                                onLoadSuccess={onDocumentLoadSuccess}
                                onLoadError={(err) => console.error("PDF load error:", err)}
                            >
                                <Page
                                    pageNumber={pageNumber}
                                    scale={scale * zoom}
                                    devicePixelRatio={window.devicePixelRatio || 1}
                                />
                            </Document>}
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
            </div>
        </div>
    );
}

interface SummaryCardProps {
    title: string;
    value: string;
    icon: React.ReactNode;
    suffix?: string;
    highlight?: boolean;
}

function SummaryCard({
                         title,
                         value,
                         icon,
                         suffix,
                         highlight,
                     }: SummaryCardProps) {
    return (
        <Card className={`${highlight ? "border-accent" : ""} transition-all hover:shadow-md`}>
            <CardContent className="p-6">
                <div className="flex items-center justify-between mb-2">
                    <p className="text-sm text-muted-foreground">{title}</p>
                    <div
                        className={`${highlight ? "text-accent" : "text-muted-foreground"
                        }`}
                    >
                        {icon}
                    </div>
                </div>
                <div className="flex items-baseline gap-1">
                    <p className={`text-2xl ${highlight ? "text-accent" : ""}`}>
                        {value}
                    </p>
                    {suffix && <span className="text-sm text-muted-foreground">{suffix}</span>}
                </div>
            </CardContent>
        </Card>
    );
}