function toIsoDate(dateValue) {
  const year = dateValue.getFullYear();
  const month = String(dateValue.getMonth() + 1).padStart(2, "0");
  const day = String(dateValue.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

export function defaultLastDays(days = 7) {
  const end = new Date();
  const start = new Date();
  start.setDate(end.getDate() - Math.max(0, days - 1));
  return {
    startDate: toIsoDate(start),
    endDate: toIsoDate(end)
  };
}

export function normalizeDateRange(startDate, endDate) {
  if (!startDate && !endDate) return { startDate: "", endDate: "" };

  let safeStart = startDate || endDate;
  let safeEnd = endDate || startDate;

  if (safeStart > safeEnd) {
    const temp = safeStart;
    safeStart = safeEnd;
    safeEnd = temp;
  }

  return { startDate: safeStart, endDate: safeEnd };
}

export function rangeWindowHours(startDate, endDate) {
  if (!startDate || !endDate) return 24;
  const from = new Date(`${startDate}T00:00:00`);
  const to = new Date(`${endDate}T23:59:59`);
  const diffMs = Math.max(0, to.getTime() - from.getTime());
  return Math.max(1, Math.round(diffMs / 3600000));
}
