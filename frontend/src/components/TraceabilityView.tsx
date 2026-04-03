import React, {useState} from "react";
import {Card, CardContent, CardHeader, CardTitle} from "./ui/card";
import {Table, TableBody, TableCell, TableHead, TableHeader, TableRow,} from "./ui/table";
import {BOMItem} from "../types/bom"
import {ChevronDown, ChevronRight, Info} from "lucide-react";
import {Badge} from "./ui/badge";

interface TraceabilityViewProps {
    bomData: BOMItem[];
}

export function TraceabilityView({bomData}: TraceabilityViewProps) {
    const [expandedRow, setExpandedRow] = useState<number | null>(null);

    const toggleRow = (index: number) => {
        setExpandedRow(prev => (prev === index ? null : index));
    };

    return (
        <Card>
            <CardHeader>
                <CardTitle>Estimation Traceability</CardTitle>
                <p className="text-sm text-muted-foreground">
                    View the logic and source data behind each material calculation
                </p>
            </CardHeader>
            <CardContent>
                <div className="rounded-md border">
                    <Table>
                        <TableHeader>
                            <TableRow className="bg-muted/50">
                                <TableHead className="w-12"></TableHead>
                                <TableHead>Material Size</TableHead>
                                <TableHead className="text-right">Quantity</TableHead>
                                <TableHead className="text-right">Linear Feet</TableHead>
                                <TableHead>Source</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {bomData.map((item, index) => {
                                const isExpanded = expandedRow === index;
                                return (
                                    <React.Fragment key={item.id}>
                                        <TableRow
                                            className="hover:bg-muted/30 cursor-pointer"
                                            onClick={() => toggleRow(index)}
                                        >
                                            <TableCell>
                                                {isExpanded ? (
                                                    <ChevronDown className="w-4 h-4 text-muted-foreground"/>
                                                ) : (
                                                    <ChevronRight className="w-4 h-4 text-muted-foreground"/>
                                                )}
                                            </TableCell>
                                            <TableCell>{item.material_size}</TableCell>
                                            <TableCell className="text-right">
                                                {item.quantity}
                                            </TableCell>
                                            <TableCell className="text-right">
                                                {item.total_linear_feet.toFixed(2)}
                                            </TableCell>
                                            <TableCell>
                                                <Badge variant="outline" className="text-xs">
                                                    <Info className="w-3 h-3 mr-1"/>
                                                    View Details
                                                </Badge>
                                            </TableCell>
                                        </TableRow>
                                        {isExpanded && (
                                            <TableRow key={`${item.id}-details`}>
                                                <TableCell colSpan={5} className="bg-muted/20 p-0">
                                                    <div className="p-4 space-y-3">
                                                        <div>
                                                            <h4 className="text-sm mb-2">
                                                                Extraction Logic
                                                            </h4>
                                                            <div
                                                                className="bg-background rounded p-3 border border-border">
                                                                <p className="text-sm text-foreground">
                                                                    {item.logic_trace}
                                                                </p>
                                                            </div>
                                                        </div>
                                                        <div className="grid grid-cols-3 gap-4">

                                                            <div>
                                                                <p className="text-xs text-muted-foreground mb-1">
                                                                    Source Drawing
                                                                </p>
                                                                <p className="text-sm">{item.source_drawing}</p>
                                                            </div>

                                                        </div>
                                                    </div>
                                                </TableCell>
                                            </TableRow>
                                        )}
                                    </React.Fragment>
                                );
                            })}
                        </TableBody>
                    </Table>
                </div>
            </CardContent>
        </Card>
    );
}