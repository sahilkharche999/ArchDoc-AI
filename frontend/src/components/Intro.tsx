import { Card, CardContent } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Sparkles } from "lucide-react";

export function Intro({ onStart }: { onStart: () => void }) {
  return (
    <div className="flex items-center justify-center h-screen w-full bg-gradient-to-br from-background to-muted/40">
      
      <Card className="max-w-3xl w-full shadow-xl border">
        <CardContent className="p-10 text-center space-y-8">

          {/* Icon */}
          <div className="flex justify-center">
            <div className="p-4 rounded-full bg-primary/10">
              <Sparkles className="w-8 h-8 text-primary" />
            </div>
          </div>

          {/* Title */}
          <h1 className="text-4xl font-bold">
           DAX Estimation AI
          </h1>

          <p className="text-muted-foreground text-lg">
            Smart material estimation for your structural drawings.
          </p>

          {/* CTA BUTTON 🔥 */}
          <Button size="lg" onClick={onStart}>
            + Start New Estimation
          </Button>

          {/* Instructions */}
          <div className="text-left bg-muted/50 p-6 rounded-xl space-y-3 text-sm text-muted-foreground">
            <p className="font-semibold text-foreground">
              How it works:
            </p>

            <ol className="list-decimal ml-5 space-y-2">
              <li>Click on <b>New Estimation</b></li>
              <li>Upload your PDF drawing</li>
              <li>Enter start & end page numbers</li>
              <li>Run estimation</li>
              <li>View material breakdown</li>
            </ol>
          </div>

        </CardContent>
      </Card>
    </div>
  );
}