/**
 * Pace for the MEMBERSHIP and ORDER of a live list (which rows are in the table and where). A row that
 * arrives at the top pushes every row below it down by its own height: one arrival is a clear event, but
 * a burst of them is the table jumping on every update. So the table takes a change of membership or
 * order at most once per `gapMs`: a change that finds the table calm is shown at once (a request that
 * arrives after a quiet spell appears when it arrives), a change that comes too soon waits for the end of
 * the interval and shows with everything that arrived meanwhile, in ONE step. The data of rows already
 * in the table is not paced (their numbers update as usual). Time is passed in, so tests drive it.
 */
export class RowPacer {
  private applied: string[] | null = null;
  private last = -Infinity;
  readonly gapMs: number;
  constructor(gapMs = 1500) {
    this.gapMs = gapMs;
  }

  private same(ids: readonly string[]): boolean {
    const a = this.applied;
    return !!a && a.length === ids.length && a.every((id, i) => id === ids[i]);
  }

  /**
   * The ids to draw now for the newest `ids`. `jump` shows them at once (another filter, another sort:
   * the reader asked for it) and restarts the interval.
   */
  offer(ids: readonly string[], now: number, jump = false): readonly string[] {
    if (
      this.applied === null ||
      jump ||
      (!this.same(ids) && now - this.last >= this.gapMs)
    ) {
      this.applied = [...ids];
      this.last = now;
    }
    return this.applied;
  }

  /** Milliseconds until the newest offered ids may be drawn, or null when nothing is waiting. */
  dueIn(ids: readonly string[], now: number): number | null {
    return this.same(ids) ? null : Math.max(0, this.last + this.gapMs - now);
  }
}
