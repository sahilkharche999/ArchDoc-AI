export interface BOMItem {
    id: string
    description: string;
    material_size: string;
    quantity: number;
    total_linear_feet: number;
    lb_per_ft: number;
    charge_per_lb: number;
    total_cost: number;
    total_weight_lbs: number;
    total_bolts: number;
    total_holes: number;
    total_weld_inches: number;
    logic_trace: string;
    source_drawing: string;
}

export interface BOMTableProps {
    bomData: BOMItem[];
    setEditableBom: React.Dispatch<React.SetStateAction<BOMItem[]>>;
    pricePerLb: number;
}