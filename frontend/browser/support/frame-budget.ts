/** Compare the app with the same browser's idle cadence, allowing one isolated scheduling pause. */
export function frameBudget(deltas: number[], idle: number[]) {
  const quantile = (values: number[], q: number) => {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))];
  };
  if (deltas.length < 30 || idle.length < 30)
    throw Error("incomplete frame recording");
  const cadence = quantile(idle, 0.5);
  const p95Limit = Math.max(40, quantile(idle, 0.95) + cadence);
  const stallMs = Math.max(120, cadence * 6);
  const stalls = deltas.filter((d) => d > stallMs).length;
  const idleRate = idle.filter((d) => d > stallMs).length / idle.length;
  return {
    p95: quantile(deltas, 0.95),
    p95Limit,
    stalls,
    stallLimit: Math.ceil(idleRate * deltas.length) + 1,
    stallMs,
  };
}
