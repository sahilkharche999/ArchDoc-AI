import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { SummaryTabProps } from "../types/summary"

export function SummaryTab({
  baseCost,
  fabricationCost,
  galvanizingCost,
  fabricationMarkup,
  pricePerLb,
  bomData,
}: SummaryTabProps) {
  // Prepare data for material weight chart
  const materialWeightData = bomData.map((item) => ({
    name: item.material_size,
    weight: item.total_linear_feet * item.quantity * item.lb_per_ft,
    cost: item.total_linear_feet * item.quantity * item.lb_per_ft * pricePerLb,
  }));

  // Color palette
  const colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6"];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        {/* Material Breakdown */}
        <Card>
          <CardHeader>
            <CardTitle>Material Breakdown</CardTitle>
            <p className="text-sm text-muted-foreground">Weight distribution by material type</p>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={materialWeightData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 11 }}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    border: '1px solid #d1d5db',
                    borderRadius: '0.5rem'
                  }}
                  formatter={(value: number) => `${value.toFixed(0)} lbs`}
                />
                <Bar dataKey="weight" radius={[4, 4, 0, 0]}>
                  {materialWeightData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Cost Breakdown */}
        <Card>
          <CardHeader>
            <CardTitle>Cost Breakdown</CardTitle>
            <p className="text-sm text-muted-foreground">Total project cost analysis</p>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm">Base Material Cost</span>
                  <span className="text-sm">${baseCost.toFixed(2)}</span>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all"
                    style={{ width: '100%' }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm">Fabrication ({fabricationMarkup}%)</span>
                  <span className="text-sm">${fabricationCost.toFixed(2)}</span>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div
                    className="bg-emerald-500 h-2 rounded-full transition-all"
                    style={{ width: `${(fabricationCost / baseCost) * 100}%` }}
                  />
                </div>
              </div>

              {galvanizingCost > 0 && (
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm">Galvanizing</span>
                    <span className="text-sm">${galvanizingCost.toFixed(2)}</span>
                  </div>
                  <div className="w-full bg-muted rounded-full h-2">
                    <div
                      className="bg-amber-500 h-2 rounded-full transition-all"
                      style={{ width: `${(galvanizingCost / baseCost) * 100}%` }}
                    />
                  </div>
                </div>
              )}

              <div className="pt-4 border-t">
                <div className="flex justify-between">
                  <span>Total Estimated Cost</span>
                  <span className="text-accent">
                    ${(baseCost + fabricationCost + galvanizingCost).toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

    </div>
  );
}