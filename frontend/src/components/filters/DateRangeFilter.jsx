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
  const [draftRange, setDraftRange] = useState({ from: undefined, to: undefined });
  const containerRef = useRef(null);

  const selected = useMemo(() => {
    if (isOpen) {
      return draftRange;
    }

    return {
      from: parseIsoDate(startDate),
      to: parseIsoDate(endDate)
    };
  }, [draftRange, endDate, isOpen, startDate]);

  const handleSelect = (range) => {
    setDraftRange({ from: range?.from, to: range?.to });
  };

  const handleApply = () => {
    const from = toLocalIsoDate(draftRange.from);
    if (!from) return;
    const to = toLocalIsoDate(draftRange.to || draftRange.from);
    onChange({ startDate: from, endDate: to });
    setIsOpen(false);
  };

  const handleCancel = () => {
    setIsOpen(false);
  };

  const handleClear = () => {
    setDraftRange({ from: undefined, to: undefined });
  };

  const handleToggle = () => {
    setIsOpen((prev) => {
      const next = !prev;
      if (next) {
        setDraftRange({ from: undefined, to: undefined });
      }
      return next;
    });
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
      <button type="button" className="date-range-trigger" onClick={handleToggle}>
        {rangeLabel(startDate, endDate)}
      </button>
      {isOpen ? (
        <div className="date-range-popover">
          <DayPicker mode="range" min={1} selected={selected} onSelect={handleSelect} numberOfMonths={1} />
          <div className="date-range-actions">
            <button type="button" className="date-range-btn secondary" onClick={handleClear}>
              Limpiar
            </button>
            <button type="button" className="date-range-btn secondary" onClick={handleCancel}>
              Cancelar
            </button>
            <button type="button" className="date-range-btn" onClick={handleApply}>
              Aplicar
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
