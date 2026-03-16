import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";
import { BOMItem, BOMTableProps } from "../types/bom"

export function BOMTable({ bomData, setEditableBom, pricePerLb }: BOMTableProps) {

function handleChange(index: number, field: keyof BOMItem, value: number) {
  const updated = [...bomData];

  updated[index] = {
    ...updated[index],
    [field]: value,
  };

  setEditableBom(updated);
}
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Detailed Bill of Materials</CardTitle>
            <p className="text-sm text-muted-foreground mt-1">
              Comprehensive breakdown of all structural steel elements
            </p>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow className="bg-muted/50">
                <TableHead>Material Size</TableHead>
                <TableHead className="text-right">Quantity</TableHead>
                <TableHead className="text-right">Total Linear Ft</TableHead>
                <TableHead className="text-right">Lb/Ft</TableHead>
                <TableHead className="text-right">Total Weight (lbs)</TableHead>
                <TableHead className="text-right">Price/Lb</TableHead>
                <TableHead className="text-right">Total Cost</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {bomData.map((item,index) => {
                const totalWeight =
                  item.total_linear_feet * item.quantity * item.lb_per_ft;
                const totalCost = totalWeight * item.charge_per_lb;

                return (
                  <TableRow key={item.id} className="hover:bg-muted/30">
                    <TableCell>
                      <span>{item.material_size}</span>
                    </TableCell>
                    <TableCell className="text-right">
                      <input
                        type="number"
                        value={item.quantity}
                        onChange={(e) =>
                          handleChange(index, "quantity", Number(e.target.value))
                        }
                        className="w-20 text-right border rounded px-1"
                      />
                    </TableCell>
                    <TableCell className="text-right">
                      <input
                      type="number"
                      value={item.total_linear_feet}
                      onChange={(e) =>
                        handleChange(index, "total_linear_feet", Number(e.target.value))
                      }
                      className="w-24 text-right border rounded px-1"
                     />
                    </TableCell>
                    <TableCell className="text-right text-muted-foreground">
                      {item.lb_per_ft.toFixed(2)}
                    </TableCell>
                    <TableCell className="text-right">
                      {totalWeight.toFixed(2)}
                    </TableCell>
                    <TableCell className="text-right text-muted-foreground">
                      <input
                      type="number"
                      value={item.charge_per_lb}
                      step="0.01"
                      onChange={(e) =>
                        handleChange(index, "charge_per_lb", Number(e.target.value))
                      }
                      className="w-24 text-right border rounded px-1"
                    />
                    </TableCell>
                    <TableCell className="text-right">
                      ${totalCost.toFixed(2)}
                    </TableCell>
                  </TableRow>
                );
              })}
              {/* Total Row */}
              <TableRow className="bg-accent/10 font-medium">
                <TableCell colSpan={1}>Total</TableCell>
                <TableCell className="text-right">
                {bomData.reduce((sum, item) => sum + item.quantity, 0)}
                </TableCell>
                <TableCell className="text-right">
                  {bomData
                    .reduce(
                      (sum, item) =>
                        sum + item.total_linear_feet * item.quantity,
                      0
                    )
                    .toFixed(2)}
                </TableCell>
                <TableCell></TableCell>
                <TableCell className="text-right">
                  {bomData
                    .reduce(
                      (sum, item) =>
                        sum +
                        item.total_linear_feet * item.quantity * item.lb_per_ft,
                      0
                    )
                    .toFixed(2)}
                </TableCell>
                <TableCell></TableCell>
                <TableCell className="text-right text-accent">
                  $
                  {bomData
                    .reduce(
                      (sum, item) =>
                        sum +
                        item.total_linear_feet *
                          item.quantity *
                          item.lb_per_ft *
                          item.charge_per_lb,
                      0
                    )
                    .toFixed(2)}
                </TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}