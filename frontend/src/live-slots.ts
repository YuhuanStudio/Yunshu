/** Keep live requests in their lane while neighbours arrive or finish. Vacant lanes are reused. */
export function liveSlots<T extends { id: string }>(
  previous: readonly (T | null)[],
  rows: readonly T[],
): (T | null)[] {
  const remaining = new Map(rows.map((row) => [row.id, row]));
  const slots = previous.map((row) => {
    const current = row ? remaining.get(row.id) : undefined;
    if (current) remaining.delete(current.id);
    return current ?? null;
  });
  for (const row of remaining.values()) {
    const vacant = slots.indexOf(null);
    if (vacant < 0) slots.push(row);
    else slots[vacant] = row;
  }
  return slots;
}
