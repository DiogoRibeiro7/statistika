import { Dataset, DescriptiveStats } from "../types";

export function mean(data: Dataset): number {
  if (data.length === 0) throw new Error("Dataset must not be empty");
  return data.reduce((sum, v) => sum + v, 0) / data.length;
}

export function median(data: Dataset): number {
  if (data.length === 0) throw new Error("Dataset must not be empty");
  const sorted = [...data].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function variance(data: Dataset, sample = true): number {
  if (data.length < 2) throw new Error("Dataset must have at least 2 elements");
  const m = mean(data);
  const sumSq = data.reduce((acc, v) => acc + (v - m) ** 2, 0);
  return sumSq / (sample ? data.length - 1 : data.length);
}

export function stdDev(data: Dataset, sample = true): number {
  return Math.sqrt(variance(data, sample));
}

export function describe(data: Dataset): DescriptiveStats {
  return {
    count: data.length,
    mean: mean(data),
    median: median(data),
    variance: variance(data),
    stdDev: stdDev(data),
    min: Math.min(...data),
    max: Math.max(...data),
  };
}
