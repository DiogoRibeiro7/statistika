import { mean, median } from "./utils/descriptive";

/** A dataset that may contain null/undefined values representing missing data. */
export type MaybeDataset = (number | null | undefined)[];

/**
 * Result of a missing data analysis.
 */
export interface MissingDataSummary {
  totalValues: number;
  missingCount: number;
  missingProportion: number;
  completeCount: number;
  missingIndices: number[];
}

/**
 * Analyze missing data patterns in a dataset.
 */
export function analyzeMissing(data: MaybeDataset): MissingDataSummary {
  const missingIndices: number[] = [];
  for (let i = 0; i < data.length; i++) {
    if (data[i] == null || Number.isNaN(data[i] as number)) {
      missingIndices.push(i);
    }
  }
  return {
    totalValues: data.length,
    missingCount: missingIndices.length,
    missingProportion: data.length > 0 ? missingIndices.length / data.length : 0,
    completeCount: data.length - missingIndices.length,
    missingIndices,
  };
}

/**
 * Listwise deletion (complete case analysis).
 *
 * Removes any observation that has missing values. For multivariate data,
 * removes entire rows if any column is missing.
 *
 * @param columns - Array of datasets (columns), all of equal length
 * @returns Filtered columns with only complete rows
 */
export function listwiseDeletion(...columns: MaybeDataset[]): number[][] {
  if (columns.length === 0) throw new Error("Must provide at least one column");
  const n = columns[0].length;
  for (const col of columns) {
    if (col.length !== n) throw new Error("All columns must have the same length");
  }

  const result: number[][] = columns.map(() => []);

  for (let i = 0; i < n; i++) {
    const complete = columns.every(
      (col) => col[i] != null && !Number.isNaN(col[i] as number),
    );
    if (complete) {
      for (let j = 0; j < columns.length; j++) {
        result[j].push(columns[j][i] as number);
      }
    }
  }

  return result;
}

/**
 * Pairwise deletion.
 *
 * For computing pairwise statistics (e.g., correlations), uses all
 * available observations for each pair of variables. Returns indices
 * of complete pairs.
 *
 * @param a - First dataset
 * @param b - Second dataset
 * @returns Object with filtered arrays and the indices used
 */
export function pairwiseDeletion(
  a: MaybeDataset,
  b: MaybeDataset,
): { a: number[]; b: number[]; indices: number[] } {
  if (a.length !== b.length) {
    throw new Error("Both datasets must have the same length");
  }

  const resultA: number[] = [];
  const resultB: number[] = [];
  const indices: number[] = [];

  for (let i = 0; i < a.length; i++) {
    if (
      a[i] != null && !Number.isNaN(a[i] as number) &&
      b[i] != null && !Number.isNaN(b[i] as number)
    ) {
      resultA.push(a[i] as number);
      resultB.push(b[i] as number);
      indices.push(i);
    }
  }

  return { a: resultA, b: resultB, indices };
}

/**
 * Mean imputation.
 *
 * Replaces missing values with the mean of the observed values.
 * Simple but can distort variance and correlations.
 *
 * @param data - Dataset with possible missing values
 */
export function meanImputation(data: MaybeDataset): number[] {
  const complete = data.filter(
    (v): v is number => v != null && !Number.isNaN(v),
  );
  if (complete.length === 0) throw new Error("No observed values to compute mean");
  const m = mean(complete);
  return data.map((v) => (v != null && !Number.isNaN(v) ? v : m));
}

/**
 * Median imputation.
 *
 * Replaces missing values with the median of the observed values.
 * More robust to outliers than mean imputation.
 *
 * @param data - Dataset with possible missing values
 */
export function medianImputation(data: MaybeDataset): number[] {
  const complete = data.filter(
    (v): v is number => v != null && !Number.isNaN(v),
  );
  if (complete.length === 0) throw new Error("No observed values to compute median");
  const med = median(complete);
  return data.map((v) => (v != null && !Number.isNaN(v) ? v : med));
}

/**
 * Mode imputation.
 *
 * Replaces missing values with the most frequent observed value.
 * Suitable for categorical/discrete data.
 *
 * @param data - Dataset with possible missing values
 */
export function modeImputation(data: MaybeDataset): number[] {
  const complete = data.filter(
    (v): v is number => v != null && !Number.isNaN(v),
  );
  if (complete.length === 0) throw new Error("No observed values to compute mode");

  const counts = new Map<number, number>();
  for (const v of complete) {
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }

  let mode = complete[0];
  let maxCount = 0;
  for (const [val, count] of counts) {
    if (count > maxCount) {
      maxCount = count;
      mode = val;
    }
  }

  return data.map((v) => (v != null && !Number.isNaN(v) ? v : mode));
}

/**
 * Linear interpolation for missing values.
 *
 * Fills missing values by linearly interpolating between the nearest
 * observed neighbors. Best for ordered/time-series data.
 *
 * @param data - Dataset with possible missing values
 */
export function linearInterpolation(data: MaybeDataset): number[] {
  if (data.length === 0) return [];

  const result = [...data] as (number | null | undefined)[];

  // Find first and last non-missing for edge extrapolation
  let firstIdx = -1;
  let lastIdx = -1;
  for (let i = 0; i < data.length; i++) {
    if (data[i] != null && !Number.isNaN(data[i] as number)) {
      if (firstIdx === -1) firstIdx = i;
      lastIdx = i;
    }
  }

  if (firstIdx === -1) throw new Error("No observed values for interpolation");

  // Forward-fill leading missing values
  for (let i = 0; i < firstIdx; i++) {
    result[i] = data[firstIdx] as number;
  }
  // Backward-fill trailing missing values
  for (let i = lastIdx + 1; i < data.length; i++) {
    result[i] = data[lastIdx] as number;
  }

  // Interpolate interior missing values
  let prevIdx = firstIdx;
  for (let i = firstIdx + 1; i <= lastIdx; i++) {
    if (data[i] != null && !Number.isNaN(data[i] as number)) {
      // Fill any gaps between prevIdx and i
      if (i - prevIdx > 1) {
        const prevVal = data[prevIdx] as number;
        const nextVal = data[i] as number;
        for (let j = prevIdx + 1; j < i; j++) {
          const t = (j - prevIdx) / (i - prevIdx);
          result[j] = prevVal + t * (nextVal - prevVal);
        }
      }
      prevIdx = i;
    }
  }

  return result as number[];
}

/**
 * Forward-fill (last observation carried forward).
 *
 * Fills missing values with the most recent observed value.
 *
 * @param data - Dataset with possible missing values
 */
export function forwardFill(data: MaybeDataset): number[] {
  if (data.length === 0) return [];

  const result = new Array<number>(data.length);
  let lastObserved: number | null = null;

  for (let i = 0; i < data.length; i++) {
    if (data[i] != null && !Number.isNaN(data[i] as number)) {
      lastObserved = data[i] as number;
    }
    if (lastObserved !== null) {
      result[i] = lastObserved;
    } else {
      // Leading missing values — fill later with backward pass
      result[i] = NaN;
    }
  }

  // Fill any leading NaNs with the first observed value
  if (lastObserved !== null) {
    const firstObserved = data.find(
      (v): v is number => v != null && !Number.isNaN(v),
    )!;
    for (let i = 0; i < result.length; i++) {
      if (Number.isNaN(result[i])) result[i] = firstObserved;
      else break;
    }
  }

  return result;
}
