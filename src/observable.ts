/**
 * Observable/streaming statistics integration.
 *
 * Provides operators and adapters for computing running statistics
 * over data streams. Works with any Observable-compatible library
 * (RxJS, zen-observable, etc.) via a minimal Observable interface.
 *
 * No RxJS dependency required — uses a minimal Observable interface.
 *
 * @example
 * ```ts
 * import { from } from 'rxjs';
 * import { streamingStats, streamingCorrelation } from 'statistika/observable';
 *
 * const data$ = from([10, 20, 30, 40, 50]);
 * const stats$ = streamingStats(data$);
 * stats$.subscribe(s => console.log(s.mean, s.variance));
 * ```
 */

import { OnlineStats, OnlineCovariance } from "./streaming";

/** Minimal Observable interface (compatible with RxJS, zen-observable, etc.) */
export interface ObservableLike<T> {
  subscribe(observer: {
    next?: (value: T) => void;
    error?: (err: unknown) => void;
    complete?: () => void;
  }): { unsubscribe(): void };
}

/** Snapshot of running statistics */
export interface StatsSnapshot {
  count: number;
  mean: number;
  variance: number;
  stdDev: number;
  min: number;
  max: number;
}

/** Snapshot of running correlation */
export interface CorrelationSnapshot {
  count: number;
  correlation: number;
  covarianceXY: number;
  meanX: number;
  meanY: number;
}

/**
 * Create a simple ObservableLike from a subscriber function.
 */
function createObservable<T>(
  subscribeFn: (observer: {
    next: (value: T) => void;
    error: (err: unknown) => void;
    complete: () => void;
  }) => void,
): ObservableLike<T> {
  return {
    subscribe(observer) {
      let unsubscribed = false;
      const safeObserver = {
        next(value: T) {
          if (!unsubscribed && observer.next) observer.next(value);
        },
        error(err: unknown) {
          if (!unsubscribed && observer.error) observer.error(err);
        },
        complete() {
          if (!unsubscribed && observer.complete) observer.complete();
        },
      };
      try {
        subscribeFn(safeObserver);
      } catch (err) {
        safeObserver.error(err);
      }
      return {
        unsubscribe() {
          unsubscribed = true;
        },
      };
    },
  };
}

/**
 * Transform a stream of numbers into a stream of running statistics.
 * Each emitted value is a snapshot of mean, variance, stdDev, min, max.
 */
export function streamingStats(
  source: ObservableLike<number>,
): ObservableLike<StatsSnapshot> {
  return createObservable<StatsSnapshot>((observer) => {
    const stats = new OnlineStats();
    source.subscribe({
      next(value) {
        try {
          stats.push(value);
          const snapshot: StatsSnapshot = {
            count: stats.count,
            mean: stats.mean,
            variance: stats.count >= 2 ? stats.variance : 0,
            stdDev: stats.count >= 2 ? stats.stdDev : 0,
            min: stats.min,
            max: stats.max,
          };
          observer.next(snapshot);
        } catch (err) {
          observer.error(err);
        }
      },
      error(err) {
        observer.error(err);
      },
      complete() {
        observer.complete();
      },
    });
  });
}

/**
 * Transform a stream of [x, y] pairs into running correlation snapshots.
 */
export function streamingCorrelation(
  source: ObservableLike<[number, number]>,
): ObservableLike<CorrelationSnapshot> {
  return createObservable<CorrelationSnapshot>((observer) => {
    const cov = new OnlineCovariance();
    source.subscribe({
      next(value) {
        try {
          cov.push(value[0], value[1]);
          const snapshot: CorrelationSnapshot = {
            count: cov.count,
            correlation: cov.count >= 2 ? cov.correlation : 0,
            covarianceXY: cov.count >= 2 ? cov.covariance : 0,
            meanX: cov.meanX,
            meanY: cov.meanY,
          };
          observer.next(snapshot);
        } catch (err) {
          observer.error(err);
        }
      },
      error(err) {
        observer.error(err);
      },
      complete() {
        observer.complete();
      },
    });
  });
}

/**
 * Windowed statistics — compute stats over a sliding window.
 * Maintains a buffer of the last `windowSize` values and recomputes
 * stats from the window on each new value.
 */
export function windowedStats(
  source: ObservableLike<number>,
  windowSize: number,
): ObservableLike<StatsSnapshot> {
  if (!Number.isInteger(windowSize) || windowSize < 1) {
    throw new Error("windowSize must be a positive integer");
  }

  return createObservable<StatsSnapshot>((observer) => {
    const buffer: number[] = [];

    source.subscribe({
      next(value) {
        try {
          buffer.push(value);
          if (buffer.length > windowSize) {
            buffer.shift();
          }

          const stats = new OnlineStats();
          stats.pushAll(buffer);

          const snapshot: StatsSnapshot = {
            count: stats.count,
            mean: stats.mean,
            variance: stats.count >= 2 ? stats.variance : 0,
            stdDev: stats.count >= 2 ? stats.stdDev : 0,
            min: stats.min,
            max: stats.max,
          };
          observer.next(snapshot);
        } catch (err) {
          observer.error(err);
        }
      },
      error(err) {
        observer.error(err);
      },
      complete() {
        observer.complete();
      },
    });
  });
}

/**
 * Batch statistics — accumulate values and emit stats every N items.
 */
export function batchStats(
  source: ObservableLike<number>,
  batchSize: number,
): ObservableLike<StatsSnapshot> {
  if (!Number.isInteger(batchSize) || batchSize < 1) {
    throw new Error("batchSize must be a positive integer");
  }

  return createObservable<StatsSnapshot>((observer) => {
    let batch: number[] = [];

    source.subscribe({
      next(value) {
        try {
          batch.push(value);
          if (batch.length === batchSize) {
            const stats = new OnlineStats();
            stats.pushAll(batch);

            const snapshot: StatsSnapshot = {
              count: stats.count,
              mean: stats.mean,
              variance: stats.count >= 2 ? stats.variance : 0,
              stdDev: stats.count >= 2 ? stats.stdDev : 0,
              min: stats.min,
              max: stats.max,
            };
            observer.next(snapshot);
            batch = [];
          }
        } catch (err) {
          observer.error(err);
        }
      },
      error(err) {
        observer.error(err);
      },
      complete() {
        // Emit remaining partial batch if any
        if (batch.length > 0) {
          try {
            const stats = new OnlineStats();
            stats.pushAll(batch);
            const snapshot: StatsSnapshot = {
              count: stats.count,
              mean: stats.mean,
              variance: stats.count >= 2 ? stats.variance : 0,
              stdDev: stats.count >= 2 ? stats.stdDev : 0,
              min: stats.min,
              max: stats.max,
            };
            observer.next(snapshot);
          } catch (err) {
            observer.error(err);
          }
        }
        observer.complete();
      },
    });
  });
}

/**
 * Change detection — emit when the mean shifts significantly.
 * Uses a simple CUSUM-like approach. After a warmup period, tracks
 * cumulative deviations from the baseline mean. When the cumulative
 * sum exceeds the threshold (in units of standard deviations), a
 * change point is emitted and the baseline resets.
 */
export function changeDetection(
  source: ObservableLike<number>,
  options?: {
    threshold?: number;
    warmup?: number;
  },
): ObservableLike<{ index: number; oldMean: number; newMean: number }> {
  const threshold = options?.threshold ?? 3.0;
  const warmup = options?.warmup ?? 30;

  return createObservable<{ index: number; oldMean: number; newMean: number }>(
    (observer) => {
      const baselineStats = new OnlineStats();
      const currentStats = new OnlineStats();
      let index = 0;
      let inWarmup = true;
      let baselineMean = 0;
      let baselineStdDev = 1;
      let cusumPos = 0;
      let cusumNeg = 0;

      source.subscribe({
        next(value) {
          try {
            if (inWarmup) {
              baselineStats.push(value);
              index++;
              if (baselineStats.count >= warmup) {
                inWarmup = false;
                baselineMean = baselineStats.mean;
                baselineStdDev =
                  baselineStats.count >= 2 ? baselineStats.stdDev : 1;
                if (baselineStdDev === 0) baselineStdDev = 1;
                cusumPos = 0;
                cusumNeg = 0;
                currentStats.reset();
              }
              return;
            }

            currentStats.push(value);
            const z = (value - baselineMean) / baselineStdDev;
            cusumPos = Math.max(0, cusumPos + z - 0.5);
            cusumNeg = Math.max(0, cusumNeg - z - 0.5);

            index++;

            if (cusumPos > threshold || cusumNeg > threshold) {
              const oldMean = baselineMean;
              const newMean = currentStats.mean;
              observer.next({ index, oldMean, newMean });

              // Reset baseline
              baselineMean = currentStats.mean;
              baselineStdDev =
                currentStats.count >= 2 ? currentStats.stdDev : baselineStdDev;
              if (baselineStdDev === 0) baselineStdDev = 1;
              cusumPos = 0;
              cusumNeg = 0;
              currentStats.reset();
            }
          } catch (err) {
            observer.error(err);
          }
        },
        error(err) {
          observer.error(err);
        },
        complete() {
          observer.complete();
        },
      });
    },
  );
}
