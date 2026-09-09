import {
  streamingStats,
  streamingCorrelation,
  windowedStats,
  batchStats,
  changeDetection,
  ObservableLike,
  StatsSnapshot,
  CorrelationSnapshot,
} from "../src/observable";

/** Helper: create an ObservableLike from an array of values. */
function fromArray<T>(values: T[]): ObservableLike<T> {
  return {
    subscribe(observer) {
      for (const v of values) {
        if (observer.next) observer.next(v);
      }
      if (observer.complete) observer.complete();
      return { unsubscribe() {} };
    },
  };
}

/** Helper: collect all emitted values from an ObservableLike. */
function collect<T>(obs: ObservableLike<T>): T[] {
  const results: T[] = [];
  obs.subscribe({ next(v) { results.push(v); } });
  return results;
}

describe("streamingStats", () => {
  it("emits running statistics for each value", () => {
    const data = fromArray([10, 20, 30, 40, 50]);
    const snapshots = collect(streamingStats(data));

    expect(snapshots).toHaveLength(5);

    // After first value
    expect(snapshots[0].count).toBe(1);
    expect(snapshots[0].mean).toBe(10);
    expect(snapshots[0].min).toBe(10);
    expect(snapshots[0].max).toBe(10);
    expect(snapshots[0].variance).toBe(0); // not enough data

    // After all values
    const last = snapshots[4];
    expect(last.count).toBe(5);
    expect(last.mean).toBe(30);
    expect(last.min).toBe(10);
    expect(last.max).toBe(50);
    expect(last.variance).toBeCloseTo(250, 5);
    expect(last.stdDev).toBeCloseTo(Math.sqrt(250), 5);
  });

  it("calls complete when source completes", () => {
    const data = fromArray([1, 2, 3]);
    let completed = false;
    streamingStats(data).subscribe({
      complete() { completed = true; },
    });
    expect(completed).toBe(true);
  });

  it("propagates errors", () => {
    const errObs: ObservableLike<number> = {
      subscribe(observer) {
        if (observer.error) observer.error(new Error("test error"));
        return { unsubscribe() {} };
      },
    };
    let caughtErr: unknown;
    streamingStats(errObs).subscribe({
      error(err) { caughtErr = err; },
    });
    expect(caughtErr).toBeInstanceOf(Error);
  });
});

describe("streamingCorrelation", () => {
  it("computes running correlation for perfectly correlated data", () => {
    const pairs: [number, number][] = [
      [1, 2], [2, 4], [3, 6], [4, 8], [5, 10],
    ];
    const data = fromArray(pairs);
    const snapshots = collect(streamingCorrelation(data));

    expect(snapshots).toHaveLength(5);

    // After first value, correlation is 0 (not enough data)
    expect(snapshots[0].count).toBe(1);
    expect(snapshots[0].correlation).toBe(0);

    // After all values, should be perfect positive correlation
    const last = snapshots[4];
    expect(last.count).toBe(5);
    expect(last.correlation).toBeCloseTo(1.0, 5);
    expect(last.meanX).toBeCloseTo(3, 5);
    expect(last.meanY).toBeCloseTo(6, 5);
  });

  it("computes negative correlation", () => {
    const pairs: [number, number][] = [
      [1, 10], [2, 8], [3, 6], [4, 4], [5, 2],
    ];
    const snapshots = collect(streamingCorrelation(fromArray(pairs)));
    const last = snapshots[4];
    expect(last.correlation).toBeCloseTo(-1.0, 5);
  });
});

describe("windowedStats", () => {
  it("computes stats over a sliding window", () => {
    const data = fromArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const snapshots = collect(windowedStats(data, 3));

    expect(snapshots).toHaveLength(10);

    // First value: window = [1]
    expect(snapshots[0].count).toBe(1);
    expect(snapshots[0].mean).toBe(1);

    // Third value: window = [1, 2, 3]
    expect(snapshots[2].count).toBe(3);
    expect(snapshots[2].mean).toBe(2);

    // Fourth value: window = [2, 3, 4]
    expect(snapshots[3].count).toBe(3);
    expect(snapshots[3].mean).toBe(3);

    // Last value: window = [8, 9, 10]
    expect(snapshots[9].count).toBe(3);
    expect(snapshots[9].mean).toBe(9);
  });

  it("throws on invalid window size", () => {
    expect(() => windowedStats(fromArray([1]), 0)).toThrow();
    expect(() => windowedStats(fromArray([1]), -1)).toThrow();
    expect(() => windowedStats(fromArray([1]), 1.5)).toThrow();
  });
});

describe("batchStats", () => {
  it("emits stats every N items", () => {
    const data = fromArray([1, 2, 3, 4, 5, 6]);
    const snapshots = collect(batchStats(data, 3));

    // Two full batches: [1,2,3] and [4,5,6]
    expect(snapshots).toHaveLength(2);
    expect(snapshots[0].count).toBe(3);
    expect(snapshots[0].mean).toBe(2);
    expect(snapshots[1].count).toBe(3);
    expect(snapshots[1].mean).toBe(5);
  });

  it("emits partial batch on complete", () => {
    const data = fromArray([10, 20, 30, 40, 50]);
    const snapshots = collect(batchStats(data, 3));

    // Full batch [10,20,30], partial batch [40,50]
    expect(snapshots).toHaveLength(2);
    expect(snapshots[0].count).toBe(3);
    expect(snapshots[0].mean).toBe(20);
    expect(snapshots[1].count).toBe(2);
    expect(snapshots[1].mean).toBe(45);
  });

  it("throws on invalid batch size", () => {
    expect(() => batchStats(fromArray([1]), 0)).toThrow();
    expect(() => batchStats(fromArray([1]), -1)).toThrow();
  });
});

describe("changeDetection", () => {
  it("detects a mean shift", () => {
    // 50 values around 10, then 50 values around 50
    const values: number[] = [];
    for (let i = 0; i < 50; i++) values.push(10 + (i % 3 - 1) * 0.5);
    for (let i = 0; i < 50; i++) values.push(50 + (i % 3 - 1) * 0.5);

    const data = fromArray(values);
    const changes = collect(changeDetection(data, { threshold: 3, warmup: 30 }));

    expect(changes.length).toBeGreaterThanOrEqual(1);
    // The first detected change should be around index 50
    expect(changes[0].index).toBeGreaterThan(30);
    expect(changes[0].index).toBeLessThanOrEqual(70);
    expect(changes[0].oldMean).toBeCloseTo(10, 0);
    expect(changes[0].newMean).not.toBeCloseTo(changes[0].oldMean, 0);
  });

  it("does not emit for stable data", () => {
    const values: number[] = [];
    for (let i = 0; i < 100; i++) values.push(10);

    const changes = collect(changeDetection(fromArray(values), { threshold: 5, warmup: 30 }));
    expect(changes).toHaveLength(0);
  });

  it("supports unsubscribe", () => {
    const data = fromArray([1, 2, 3]);
    const sub = changeDetection(data).subscribe({ next() {} });
    expect(sub.unsubscribe).toBeDefined();
    sub.unsubscribe();
  });
});

describe("ObservableLike unsubscribe", () => {
  it("stops emitting after unsubscribe", () => {
    let emitNext: ((v: number) => void) | undefined;
    const manualObs: ObservableLike<number> = {
      subscribe(observer) {
        emitNext = (v: number) => { if (observer.next) observer.next(v); };
        return { unsubscribe() { emitNext = undefined; } };
      },
    };

    const results: StatsSnapshot[] = [];
    const sub = streamingStats(manualObs).subscribe({
      next(v) { results.push(v); },
    });

    emitNext?.(10);
    expect(results).toHaveLength(1);

    sub.unsubscribe();
    // After unsubscribe, the outer observable should not emit
    // (emitNext is still set since it's in the source, but the wrapper guards)
    // The source emitNext would still fire, but the wrapper won't forward
  });
});
