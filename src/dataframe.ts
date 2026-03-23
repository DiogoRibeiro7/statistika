/**
 * Column-oriented DataFrame for statistical data manipulation.
 *
 * - **Column storage** with typed access (number, string, boolean).
 * - **Selection & filtering** — select, filter, sort, head, tail, slice.
 * - **Group-by aggregation** — sum, mean, count, min, max, custom.
 * - **Joins** — inner, left, right, full outer on shared keys.
 * - **Mutation** — addColumn, renameColumn, dropColumn, mutate.
 * - **I/O** — fromCSV, toCSV, fromJSON, toJSON, fromRecords, toRecords.
 */

type ColumnValue = number | string | boolean | null;
type Column = ColumnValue[];
type AggFn = (values: ColumnValue[]) => ColumnValue;

export class DataFrame {
  private _columns: Map<string, Column>;
  private _nRows: number;

  private constructor(columns: Map<string, Column>, nRows: number) {
    this._columns = columns;
    this._nRows = nRows;
  }

  // ── Factories ───────────────────────────────────────────────────────────

  /**
   * Create a DataFrame from a column map.
   * All columns must have the same length.
   */
  static create(columns: Record<string, Column>): DataFrame {
    const map = new Map<string, Column>();
    let nRows = -1;
    for (const [name, col] of Object.entries(columns)) {
      if (nRows === -1) nRows = col.length;
      else if (col.length !== nRows) {
        throw new Error(`Column "${name}" has ${col.length} rows, expected ${nRows}`);
      }
      map.set(name, [...col]);
    }
    return new DataFrame(map, Math.max(0, nRows));
  }

  /**
   * Create a DataFrame from an array of row objects.
   */
  static fromRecords(records: Record<string, ColumnValue>[]): DataFrame {
    if (records.length === 0) return new DataFrame(new Map(), 0);
    const keys = Object.keys(records[0]);
    const columns: Record<string, Column> = {};
    for (const k of keys) columns[k] = [];
    for (const row of records) {
      for (const k of keys) {
        columns[k].push(row[k] ?? null);
      }
    }
    return DataFrame.create(columns);
  }

  /**
   * Parse CSV string into a DataFrame.
   * First line is treated as header. Values are auto-typed.
   */
  static fromCSV(csv: string): DataFrame {
    const lines = csv.trim().split("\n");
    if (lines.length === 0) return new DataFrame(new Map(), 0);

    const headers = parseCsvLine(lines[0]);
    const columns: Record<string, Column> = {};
    for (const h of headers) columns[h] = [];

    for (let i = 1; i < lines.length; i++) {
      const vals = parseCsvLine(lines[i]);
      for (let j = 0; j < headers.length; j++) {
        columns[headers[j]].push(autoType(vals[j] ?? ""));
      }
    }
    return DataFrame.create(columns);
  }

  /**
   * Parse JSON array of objects into a DataFrame.
   */
  static fromJSON(json: string): DataFrame {
    const records = JSON.parse(json) as Record<string, ColumnValue>[];
    return DataFrame.fromRecords(records);
  }

  // ── Properties ──────────────────────────────────────────────────────────

  /** Number of rows. */
  get nRows(): number {
    return this._nRows;
  }

  /** Number of columns. */
  get nCols(): number {
    return this._columns.size;
  }

  /** Column names in insertion order. */
  get columnNames(): string[] {
    return [...this._columns.keys()];
  }

  /** Get a column by name. Returns a copy. */
  column(name: string): Column {
    const col = this._columns.get(name);
    if (!col) throw new Error(`Column "${name}" not found`);
    return [...col];
  }

  /** Get a column as number[]. Throws if any value is not a number. */
  numericColumn(name: string): number[] {
    const col = this.column(name);
    for (let i = 0; i < col.length; i++) {
      if (typeof col[i] !== "number") {
        throw new Error(`Column "${name}" row ${i} is not a number`);
      }
    }
    return col as number[];
  }

  // ── Selection & Filtering ───────────────────────────────────────────────

  /** Select specific columns. */
  select(...names: string[]): DataFrame {
    const cols: Record<string, Column> = {};
    for (const name of names) {
      cols[name] = this.column(name);
    }
    return DataFrame.create(cols);
  }

  /** Filter rows by a predicate on row objects. */
  filter(predicate: (row: Record<string, ColumnValue>, i: number) => boolean): DataFrame {
    const cols: Record<string, Column> = {};
    for (const name of this.columnNames) cols[name] = [];

    for (let i = 0; i < this._nRows; i++) {
      const row = this._getRow(i);
      if (predicate(row, i)) {
        for (const name of this.columnNames) {
          cols[name].push(this._columns.get(name)![i]);
        }
      }
    }
    return DataFrame.create(cols);
  }

  /** Sort by a column. */
  sort(column: string, ascending = true): DataFrame {
    const col = this._columns.get(column);
    if (!col) throw new Error(`Column "${column}" not found`);

    const indices = Array.from({ length: this._nRows }, (_, i) => i);
    indices.sort((a, b) => {
      const va = col[a];
      const vb = col[b];
      if (va === vb) return 0;
      if (va === null) return 1;
      if (vb === null) return -1;
      const cmp = va < vb ? -1 : 1;
      return ascending ? cmp : -cmp;
    });

    return this._reindex(indices);
  }

  /** First n rows. */
  head(n = 5): DataFrame {
    return this.slice(0, Math.min(n, this._nRows));
  }

  /** Last n rows. */
  tail(n = 5): DataFrame {
    const start = Math.max(0, this._nRows - n);
    return this.slice(start, this._nRows);
  }

  /** Slice by row indices [start, end). */
  slice(start: number, end: number): DataFrame {
    const cols: Record<string, Column> = {};
    for (const [name, col] of this._columns) {
      cols[name] = col.slice(start, end);
    }
    return DataFrame.create(cols);
  }

  // ── Mutation ────────────────────────────────────────────────────────────

  /** Add or replace a column. Returns a new DataFrame. */
  addColumn(name: string, values: Column): DataFrame {
    if (values.length !== this._nRows) {
      throw new Error(`Column length ${values.length} doesn't match ${this._nRows} rows`);
    }
    const cols: Record<string, Column> = {};
    for (const [n, c] of this._columns) cols[n] = [...c];
    cols[name] = [...values];
    return DataFrame.create(cols);
  }

  /** Create a new column from a row-level function. */
  mutate(name: string, fn: (row: Record<string, ColumnValue>, i: number) => ColumnValue): DataFrame {
    const values: Column = [];
    for (let i = 0; i < this._nRows; i++) {
      values.push(fn(this._getRow(i), i));
    }
    return this.addColumn(name, values);
  }

  /** Drop a column. Returns a new DataFrame. */
  dropColumn(name: string): DataFrame {
    const cols: Record<string, Column> = {};
    for (const [n, c] of this._columns) {
      if (n !== name) cols[n] = [...c];
    }
    return DataFrame.create(cols);
  }

  /** Rename a column. Returns a new DataFrame. */
  renameColumn(oldName: string, newName: string): DataFrame {
    if (!this._columns.has(oldName)) throw new Error(`Column "${oldName}" not found`);
    const cols: Record<string, Column> = {};
    for (const [n, c] of this._columns) {
      cols[n === oldName ? newName : n] = [...c];
    }
    return DataFrame.create(cols);
  }

  // ── Group-by Aggregation ────────────────────────────────────────────────

  /**
   * Group by one or more columns and aggregate.
   *
   * @param groupCols  Column names to group by.
   * @param aggs  Map of output column name → { column, fn } where fn is
   *   "sum" | "mean" | "count" | "min" | "max" or a custom function.
   */
  groupBy(
    groupCols: string[],
    aggs: Record<string, { column: string; fn: string | AggFn }>,
  ): DataFrame {
    // Build group keys
    const groups = new Map<string, number[]>();
    for (let i = 0; i < this._nRows; i++) {
      const key = groupCols.map((c) => String(this._columns.get(c)![i])).join("\0");
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(i);
    }

    // Build result columns
    const resultCols: Record<string, Column> = {};
    for (const c of groupCols) resultCols[c] = [];
    for (const name of Object.keys(aggs)) resultCols[name] = [];

    for (const [, indices] of groups) {
      // Group key columns
      for (const c of groupCols) {
        resultCols[c].push(this._columns.get(c)![indices[0]]);
      }
      // Aggregations
      for (const [name, { column, fn }] of Object.entries(aggs)) {
        const col = this._columns.get(column);
        if (!col) throw new Error(`Column "${column}" not found`);
        const values = indices.map((i) => col[i]);
        resultCols[name].push(applyAgg(values, fn));
      }
    }

    return DataFrame.create(resultCols);
  }

  // ── Joins ───────────────────────────────────────────────────────────────

  /**
   * Join this DataFrame with another on shared key column(s).
   *
   * @param other  Right DataFrame.
   * @param on  Column name(s) to join on.
   * @param how  Join type: "inner" | "left" | "right" | "outer".
   */
  join(
    other: DataFrame,
    on: string | string[],
    how: "inner" | "left" | "right" | "outer" = "inner",
  ): DataFrame {
    const keys = typeof on === "string" ? [on] : on;

    // Build index for right table
    const rightIndex = new Map<string, number[]>();
    for (let i = 0; i < other._nRows; i++) {
      const key = keys.map((k) => String(other._columns.get(k)![i])).join("\0");
      if (!rightIndex.has(key)) rightIndex.set(key, []);
      rightIndex.get(key)!.push(i);
    }

    // Determine output columns
    const leftCols = this.columnNames;
    const rightOnlyCols = other.columnNames.filter((c) => !keys.includes(c));
    // Handle name collisions
    const rightColMap: Record<string, string> = {};
    for (const c of rightOnlyCols) {
      rightColMap[c] = leftCols.includes(c) ? `${c}_right` : c;
    }

    const outCols: Record<string, Column> = {};
    for (const c of leftCols) outCols[c] = [];
    for (const c of rightOnlyCols) outCols[rightColMap[c]] = [];

    const usedRight = new Set<number>();

    // Left side scan
    for (let i = 0; i < this._nRows; i++) {
      const key = keys.map((k) => String(this._columns.get(k)![i])).join("\0");
      const matches = rightIndex.get(key);

      if (matches && matches.length > 0) {
        for (const j of matches) {
          usedRight.add(j);
          for (const c of leftCols) outCols[c].push(this._columns.get(c)![i]);
          for (const c of rightOnlyCols) outCols[rightColMap[c]].push(other._columns.get(c)![j]);
        }
      } else if (how === "left" || how === "outer") {
        for (const c of leftCols) outCols[c].push(this._columns.get(c)![i]);
        for (const c of rightOnlyCols) outCols[rightColMap[c]].push(null);
      }
    }

    // Right-only rows (for right/outer joins)
    if (how === "right" || how === "outer") {
      for (let j = 0; j < other._nRows; j++) {
        if (usedRight.has(j)) continue;
        for (const c of leftCols) {
          if (keys.includes(c)) {
            outCols[c].push(other._columns.get(c)![j]);
          } else {
            outCols[c].push(null);
          }
        }
        for (const c of rightOnlyCols) {
          outCols[rightColMap[c]].push(other._columns.get(c)![j]);
        }
      }
    }

    return DataFrame.create(outCols);
  }

  // ── I/O ─────────────────────────────────────────────────────────────────

  /** Convert to array of row objects. */
  toRecords(): Record<string, ColumnValue>[] {
    const records: Record<string, ColumnValue>[] = [];
    for (let i = 0; i < this._nRows; i++) {
      records.push(this._getRow(i));
    }
    return records;
  }

  /** Serialize to CSV string. */
  toCSV(): string {
    const names = this.columnNames;
    const lines = [names.map(escapeCsv).join(",")];
    for (let i = 0; i < this._nRows; i++) {
      const row = names.map((n) => {
        const v = this._columns.get(n)![i];
        return v === null ? "" : escapeCsv(String(v));
      });
      lines.push(row.join(","));
    }
    return lines.join("\n");
  }

  /** Serialize to JSON string (array of objects). */
  toJSON(): string {
    return JSON.stringify(this.toRecords());
  }

  /** Summary of the DataFrame shape. */
  describe(): { nRows: number; nCols: number; columns: string[] } {
    return { nRows: this._nRows, nCols: this.nCols, columns: this.columnNames };
  }

  // ── Internal ────────────────────────────────────────────────────────────

  private _getRow(i: number): Record<string, ColumnValue> {
    const row: Record<string, ColumnValue> = {};
    for (const [name, col] of this._columns) {
      row[name] = col[i];
    }
    return row;
  }

  private _reindex(indices: number[]): DataFrame {
    const cols: Record<string, Column> = {};
    for (const [name, col] of this._columns) {
      cols[name] = indices.map((i) => col[i]);
    }
    return DataFrame.create(cols);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────

function applyAgg(values: ColumnValue[], fn: string | AggFn): ColumnValue {
  if (typeof fn === "function") return fn(values);
  const nums = values.filter((v): v is number => typeof v === "number");
  switch (fn) {
    case "count":
      return values.length;
    case "sum":
      return nums.reduce((a, b) => a + b, 0);
    case "mean":
      return nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
    case "min":
      return nums.length > 0 ? Math.min(...nums) : null;
    case "max":
      return nums.length > 0 ? Math.max(...nums) : null;
    default:
      throw new Error(`Unknown aggregation function: ${fn}`);
  }
}

function parseCsvLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (i + 1 < line.length && line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        current += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      result.push(current.trim());
      current = "";
    } else {
      current += ch;
    }
  }
  result.push(current.trim());
  return result;
}

function escapeCsv(value: string): string {
  if (value.includes(",") || value.includes('"') || value.includes("\n")) {
    return '"' + value.replace(/"/g, '""') + '"';
  }
  return value;
}

function autoType(value: string): ColumnValue {
  if (value === "" || value === "null" || value === "NA") return null;
  if (value === "true") return true;
  if (value === "false") return false;
  const num = Number(value);
  if (!isNaN(num) && value.trim() !== "") return num;
  return value;
}
