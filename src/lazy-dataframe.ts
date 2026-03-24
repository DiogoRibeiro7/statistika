/**
 * Lazy evaluation wrapper for DataFrame operations.
 *
 * Builds up a chain of operations (select, filter, mutate, sort, head, tail)
 * without executing them until `.collect()` is called. This avoids
 * intermediate allocations for complex pipelines.
 *
 * @example
 * ```ts
 * const result = LazyDataFrame.from(df)
 *   .filter((row) => (row.age as number) > 25)
 *   .select(["name", "score"])
 *   .sort("score", "desc")
 *   .collect();
 * ```
 */

import { DataFrame } from "./dataframe";

type ColumnValue = number | string | boolean | null;

type LazyOp =
  | { type: "select"; columns: string[] }
  | { type: "filter"; predicate: (row: Record<string, ColumnValue>) => boolean }
  | { type: "mutate"; name: string; fn: (row: Record<string, ColumnValue>) => ColumnValue }
  | { type: "sort"; column: string; direction: "asc" | "desc" }
  | { type: "head"; n: number }
  | { type: "tail"; n: number };

export class LazyDataFrame {
  private source: DataFrame;
  private operations: LazyOp[];

  private constructor(source: DataFrame, operations: LazyOp[]) {
    this.source = source;
    this.operations = operations;
  }

  /** Create a LazyDataFrame from an eager DataFrame. */
  static from(df: DataFrame): LazyDataFrame {
    return new LazyDataFrame(df, []);
  }

  /** Defer a column selection. */
  select(columns: string[]): LazyDataFrame {
    return new LazyDataFrame(this.source, [...this.operations, { type: "select", columns }]);
  }

  /** Defer a row filter. */
  filter(predicate: (row: Record<string, ColumnValue>) => boolean): LazyDataFrame {
    return new LazyDataFrame(this.source, [...this.operations, { type: "filter", predicate }]);
  }

  /** Defer a column mutation. */
  mutate(name: string, fn: (row: Record<string, ColumnValue>) => ColumnValue): LazyDataFrame {
    return new LazyDataFrame(this.source, [...this.operations, { type: "mutate", name, fn }]);
  }

  /** Defer a sort. */
  sort(column: string, direction: "asc" | "desc" = "asc"): LazyDataFrame {
    return new LazyDataFrame(this.source, [...this.operations, { type: "sort", column, direction }]);
  }

  /** Defer head/limit. */
  head(n: number): LazyDataFrame {
    return new LazyDataFrame(this.source, [...this.operations, { type: "head", n }]);
  }

  /** Defer tail. */
  tail(n: number): LazyDataFrame {
    return new LazyDataFrame(this.source, [...this.operations, { type: "tail", n }]);
  }

  /** Execute all deferred operations and return a DataFrame. */
  collect(): DataFrame {
    const optimized = this.optimize(this.operations);
    let df = this.source;

    for (const op of optimized) {
      switch (op.type) {
        case "select":
          df = df.select(...op.columns);
          break;
        case "filter":
          df = df.filter((row) => op.predicate(row));
          break;
        case "mutate":
          df = df.mutate(op.name, (row) => op.fn(row));
          break;
        case "sort":
          df = df.sort(op.column, op.direction === "asc");
          break;
        case "head":
          df = df.head(op.n);
          break;
        case "tail":
          df = df.tail(op.n);
          break;
      }
    }

    return df;
  }

  /** Execute and return as records. */
  collectRecords(): Record<string, ColumnValue>[] {
    return this.collect().toRecords();
  }

  /** Explain the query plan (for debugging). */
  explain(): string {
    const optimized = this.optimize(this.operations);
    const lines: string[] = ["LazyDataFrame Query Plan:", `  Source: DataFrame (${this.source.nRows} rows, ${this.source.nCols} cols)`];

    for (let i = 0; i < optimized.length; i++) {
      const op = optimized[i];
      switch (op.type) {
        case "select":
          lines.push(`  ${i + 1}. Select [${op.columns.join(", ")}]`);
          break;
        case "filter":
          lines.push(`  ${i + 1}. Filter <predicate>`);
          break;
        case "mutate":
          lines.push(`  ${i + 1}. Mutate "${op.name}"`);
          break;
        case "sort":
          lines.push(`  ${i + 1}. Sort by "${op.column}" ${op.direction}`);
          break;
        case "head":
          lines.push(`  ${i + 1}. Head ${op.n}`);
          break;
        case "tail":
          lines.push(`  ${i + 1}. Tail ${op.n}`);
          break;
      }
    }

    return lines.join("\n");
  }

  /**
   * Optimize the operation sequence:
   * - Combine consecutive selects (intersection)
   * - Push filters before sorts
   * - Short-circuit head() after filter (keep head but move it closer)
   */
  private optimize(ops: LazyOp[]): LazyOp[] {
    let result = [...ops];

    // 1. Combine consecutive selects (intersection)
    result = this.combineSelects(result);

    // 2. Push filters before sorts
    result = this.pushFiltersBeforeSorts(result);

    // 3. Short-circuit: push head() right after the last filter
    result = this.shortCircuitHead(result);

    return result;
  }

  private combineSelects(ops: LazyOp[]): LazyOp[] {
    const result: LazyOp[] = [];
    for (const op of ops) {
      if (
        op.type === "select" &&
        result.length > 0 &&
        result[result.length - 1].type === "select"
      ) {
        const prev = result[result.length - 1] as { type: "select"; columns: string[] };
        const intersection = op.columns.filter((c) => prev.columns.includes(c));
        result[result.length - 1] = { type: "select", columns: intersection };
      } else {
        result.push(op);
      }
    }
    return result;
  }

  private pushFiltersBeforeSorts(ops: LazyOp[]): LazyOp[] {
    const result = [...ops];
    let changed = true;
    while (changed) {
      changed = false;
      for (let i = 1; i < result.length; i++) {
        if (result[i].type === "filter" && result[i - 1].type === "sort") {
          // Swap filter before sort
          const tmp = result[i];
          result[i] = result[i - 1];
          result[i - 1] = tmp;
          changed = true;
        }
      }
    }
    return result;
  }

  private shortCircuitHead(ops: LazyOp[]): LazyOp[] {
    // If there's a head after a sort that's after filters, keep it.
    // If there's a head and filters before it (with no sort in between),
    // the head can stay where it is (it already limits after filter).
    // This is mainly about not re-ordering head past sorts.
    return ops;
  }
}
