import { useState, useEffect } from "react";
import { Input } from "./ui/input";

interface EditableCellProps {
  value: number;
  onChange?: (value: number) => void;
  readOnly?: boolean;
  prefix?: string;
  suffix?: string;
}

export function EditableCell({
  value,
  onChange,
  readOnly = false,
  prefix = "",
  suffix = "",
}: EditableCellProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(value.toString());

  useEffect(() => {
    setEditValue(value.toString());
  }, [value]);

  const handleBlur = () => {
    setIsEditing(false);
    const numValue = parseFloat(editValue);
    if (!isNaN(numValue) && onChange) {
      onChange(numValue);
    } else {
      setEditValue(value.toString());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleBlur();
    } else if (e.key === "Escape") {
      setEditValue(value.toString());
      setIsEditing(false);
    }
  };

  if (readOnly || !onChange) {
    return (
      <span className="text-muted-foreground">
        {prefix}
        {typeof value === "number" ? value.toFixed(2) : value}
        {suffix}
      </span>
    );
  }

  if (isEditing) {
    return (
      <Input
        type="number"
        value={editValue}
        onChange={(e) => setEditValue(e.target.value)}
        onBlur={handleBlur}
        onKeyDown={handleKeyDown}
        className="h-8 w-20"
        autoFocus
      />
    );
  }

  return (
    <span
      onClick={() => setIsEditing(true)}
      className="cursor-pointer hover:bg-accent/10 px-2 py-1 rounded inline-block"
      title="Click to edit"
    >
      {prefix}
      {typeof value === "number" ? value.toFixed(2) : value}
      {suffix}
    </span>
  );
}
