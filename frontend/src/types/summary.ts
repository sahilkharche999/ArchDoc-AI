import { BOMItem } from "./bom";
export interface SummaryTabProps {
  bomData: BOMItem[];
  baseCost: number;
  fabricationCost: number;
  galvanizingCost: number;
  fabricationMarkup: number;
  pricePerLb: number;
}