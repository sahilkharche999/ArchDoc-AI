interface DashboardHeaderProps {
  projectName?: string;
  projectDate?: string;
}

export function DashboardHeader({
  projectName,
  projectDate,
}: DashboardHeaderProps) {
  return (
    <div className="border-b bg-card px-6 py-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl">{projectName}</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Created on {projectDate}
          </p>
        </div>
      </div>
    </div>
  );
}
