import { useState , useEffect} from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { BOMTable } from "./BOMTable";
import { TraceabilityView } from "./TraceabilityView";
import { DashboardHeader } from "./DashboardHeader";
import { SummaryTab } from "./SummaryTab";
import { Weight, Ruler, Package, DollarSign } from "lucide-react";
import { BOMItem } from "../types/bom"

interface EstimationDashboardProps {
  projectId?: string;
  bomData: BOMItem[];
}


function formatDate(dateString: string) {
  const date = new Date(dateString)

  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric"
  })
}
export function EstimationDashboard({ projectId = "1", bomData  }: EstimationDashboardProps) {
  const [pricePerLb, setPricePerLb] = useState(0);
  const [fabricationMarkup, setFabricationMarkup] = useState(0);
  const [galvanizing, setGalvanizing] = useState(false);
  const [projectMeta, setProjectMeta] = useState<any>(null);
  const [editableBom, setEditableBom] = useState<BOMItem[]>(bomData);
  useEffect(() => {
  
  fetch(`${import.meta.env.VITE_API_URL}/api/v1/projects/`)
    .then(res => res.json())
    .then(data => {
      const project = data.projects.find((p:any) => p.job_id === projectId);
      console.log(project);
      setProjectMeta(project);
    });

}, [projectId]);
useEffect(() => {
  setEditableBom(bomData);
}, [bomData]);
  // Get current project's BOM data


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
  projectDate={formatDate(projectMeta?.date)}
/>
      <div className="flex-1 overflow-auto p-6 space-y-6">
        {/* Summary Cards */}
        <div className="grid grid-cols-4 gap-4">
          <SummaryCard
            title="Total Weight"
            value={totalWeight.toFixed(0)}
            icon={<Weight className="w-5 h-5" />}
            suffix="lbs"
          />
          <SummaryCard
            title="Total Pieces"
            value={totalPieces.toString()}
            icon={<Package className="w-5 h-5" />}
          />
          <SummaryCard
            title="Estimated Total Cost"
            value={`$${totalCost.toFixed(2)}`}
            icon={<DollarSign className="w-5 h-5" />}
            highlight
          />
        </div>

        {/* Tabs */}
        <Tabs defaultValue="summary" className="w-full">
          <TabsList className="grid w-full max-w-2xl grid-cols-3">
            <TabsTrigger value="summary">Summary</TabsTrigger>
            <TabsTrigger value="bom">Detailed BOM</TabsTrigger>
            <TabsTrigger value="trace">Traceability</TabsTrigger>
          </TabsList>

          <TabsContent value="summary" className="space-y-4">
            <SummaryTab
              bomData={editableBom}
              baseCost={baseCost}
              fabricationCost={fabricationCost}
              galvanizingCost={galvanizingCost}
              fabricationMarkup={fabricationMarkup}
              pricePerLb={pricePerLb}
            />
          </TabsContent>

          <TabsContent value="bom">
            <BOMTable   bomData={editableBom} setEditableBom={setEditableBom} pricePerLb={pricePerLb} />
          </TabsContent>

        

          <TabsContent value="trace">
            <TraceabilityView bomData={editableBom} />
          </TabsContent>
        </Tabs>
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