export function toQuery(params = {}) {
  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([key, val]) => {
    if (val !== undefined && val !== null && `${val}`.trim() !== "") {
      searchParams.set(key, String(val));
    }
  });
  return searchParams.toString();
}
