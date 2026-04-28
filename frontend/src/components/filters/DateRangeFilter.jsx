import { useEffect, useMemo, useRef, useState } from "react";
import { DayPicker } from "react-day-picker";
import "react-day-picker/style.css";

function parseIsoDate(value) {
  if (!value) return undefined;
  return new Date(`${value}T00:00:00`);
}

function toLocalIsoDate(value) {
  if (!value) return "";
  const year = value.getFullYear();
  const month = String(value.getMonth() + 1).padStart(2, "0");
  const day = String(value.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function rangeLabel(startDate, endDate) {
  if (!startDate && !endDate) return "Seleccionar fechas";
  if (startDate && !endDate) return `${startDate} a ...`;
  return `${startDate} a ${endDate}`;
}

export function DateRangeFilter({ startDate, endDate, onChange }) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef(null);

  const selected = useMemo(
    () => ({
      from: parseIsoDate(startDate),
      to: parseIsoDate(endDate)
    }),
    [startDate, endDate]
  );

  const handleSelect = (range) => {
    const from = toLocalIsoDate(range?.from);
    const to = toLocalIsoDate(range?.to);
    if (!from || !to) {
      return;
    }

    onChange({ startDate: from, endDate: to });
    if (from !== to) {
      setIsOpen(false);
    }
  };

  useEffect(() => {
    if (!isOpen) return undefined;

    const onPointerDown = (event) => {
      if (!containerRef.current?.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", onPointerDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
    };
  }, [isOpen]);

  return (
    <div className="date-range-filter" ref={containerRef}>
      <button type="button" className="date-range-trigger" onClick={() => setIsOpen((prev) => !prev)}>
        {rangeLabel(startDate, endDate)}
      </button>
      {isOpen ? (
        <div className="date-range-popover">
          <DayPicker mode="range" min={1} selected={selected} onSelect={handleSelect} numberOfMonths={1} />
        </div>
      ) : null}
    </div>
  );
}
