import { welfordBatch } from "./utils/native-stats";

/**
 * Online/streaming statistics using Welford's algorithm.
 *
 * Computes mean, variance, and standard deviation in a single pass
 * without storing all data points. Numerically stable.
 *
 * @example
 * ```ts
 * const stats = new OnlineStats();
 * stats.push(10);
 * stats.push(20);
 * stats.push(30);
 * stats.mean;     // 20
 * stats.variance; // 100
 * stats.stdDev;   // 10
 * ```
 */
export class OnlineStats {
  private _count = 0;
  private _mean = 0;
  private _m2 = 0;
  private _min = Infinity;
  private _max = -Infinity;

  /**
   * Number of observations seen so far.
   * @returns The current count
   */
  get count(): number {
    return this._count;
  }

  /**
   * Running mean.
   * @returns The current mean of all observations
   * @throws {Error} If no observations have been added
   */
  get mean(): number {
    if (this._count === 0) throw new Error("No observations");
    return this._mean;
  }

  /**
   * Running sample variance.
   * @returns The current sample variance (using Bessel's correction)
   * @throws {Error} If fewer than 2 observations have been added
   */
  get variance(): number {
    if (this._count < 2) throw new Error("Need at least 2 observations");
    return this._m2 / (this._count - 1);
  }

  /**
   * Running population variance.
   * @returns The current population variance
   * @throws {Error} If no observations have been added
   */
  get populationVariance(): number {
    if (this._count === 0) throw new Error("No observations");
    return this._m2 / this._count;
  }

  /**
   * Running sample standard deviation.
   * @returns The current sample standard deviation
   * @throws {Error} If fewer than 2 observations have been added
   */
  get stdDev(): number {
    return Math.sqrt(this.variance);
  }

  /**
   * Minimum value seen.
   * @returns The minimum value across all observations
   * @throws {Error} If no observations have been added
   */
  get min(): number {
    if (this._count === 0) throw new Error("No observations");
    return this._min;
  }

  /**
   * Maximum value seen.
   * @returns The maximum value across all observations
   * @throws {Error} If no observations have been added
   */
  get max(): number {
    if (this._count === 0) throw new Error("No observations");
    return this._max;
  }

  /**
   * Add a single observation.
   * @param value - The numeric value to add (must not be NaN)
   * @returns void
   * @throws {Error} If value is NaN
   */
  push(value: number): void {
    if (Number.isNaN(value)) {
      throw new Error("Cannot push NaN value into OnlineStats");
    }
    this._count++;
    const delta = value - this._mean;
    this._mean += delta / this._count;
    const delta2 = value - this._mean;
    this._m2 += delta * delta2;
    if (value < this._min) this._min = value;
    if (value > this._max) this._max = value;
  }

  /**
   * Add multiple observations.
   * Uses Fortran-accelerated batch update when the native addon is available.
   * @param values - Array of numeric values to add
   * @returns void
   * @throws {Error} If any value is NaN
   */
  pushAll(values: number[]): void {
    for (const v of values) {
      if (Number.isNaN(v)) throw new Error("Cannot push NaN value into OnlineStats");
    }
    const result = welfordBatch(values, {
      count: this._count,
      mean: this._mean,
      m2: this._m2,
      min: this._min,
      max: this._max,
    });
    this._count = result.count;
    this._mean = result.mean;
    this._m2 = result.m2;
    this._min = result.min;
    this._max = result.max;
  }

  /**
   * Merge another OnlineStats instance into this one.
   * @param other - The OnlineStats instance to merge
   * @returns void
   */
  merge(other: OnlineStats): void {
    if (other._count === 0) return;
    if (this._count === 0) {
      this._count = other._count;
      this._mean = other._mean;
      this._m2 = other._m2;
      this._min = other._min;
      this._max = other._max;
      return;
    }

    const combined = this._count + other._count;
    const delta = other._mean - this._mean;
    this._m2 =
      this._m2 + other._m2 + (delta * delta * this._count * other._count) / combined;
    this._mean = (this._count * this._mean + other._count * other._mean) / combined;
    this._count = combined;
    this._min = Math.min(this._min, other._min);
    this._max = Math.max(this._max, other._max);
  }

  /**
   * Reset all state.
   * @returns void
   */
  reset(): void {
    this._count = 0;
    this._mean = 0;
    this._m2 = 0;
    this._min = Infinity;
    this._max = -Infinity;
  }
}

/**
 * Online covariance and correlation between two streams.
 *
 * Uses a numerically stable one-pass algorithm.
 */
export class OnlineCovariance {
  private _count = 0;
  private _meanX = 0;
  private _meanY = 0;
  private _c = 0;
  private _m2x = 0;
  private _m2y = 0;

  /**
   * Number of observation pairs seen so far.
   * @returns The current count
   */
  get count(): number {
    return this._count;
  }

  /**
   * Running sample covariance.
   * @returns The current sample covariance
   * @throws {Error} If fewer than 2 observation pairs have been added
   */
  get covariance(): number {
    if (this._count < 2) throw new Error("Need at least 2 observations");
    return this._c / (this._count - 1);
  }

  /**
   * Running Pearson correlation coefficient.
   * @returns The current Pearson correlation coefficient, or 0 if variance is zero
   * @throws {Error} If fewer than 2 observation pairs have been added
   */
  get correlation(): number {
    if (this._count < 2) throw new Error("Need at least 2 observations");
    const denom = Math.sqrt(this._m2x * this._m2y);
    if (denom === 0) return 0;
    return this._c / denom;
  }

  /**
   * Running mean of the X stream.
   * @returns The current mean of X values
   * @throws {Error} If no observations have been added
   */
  get meanX(): number {
    if (this._count === 0) throw new Error("No observations");
    return this._meanX;
  }

  /**
   * Running mean of the Y stream.
   * @returns The current mean of Y values
   * @throws {Error} If no observations have been added
   */
  get meanY(): number {
    if (this._count === 0) throw new Error("No observations");
    return this._meanY;
  }

  /**
   * Add a pair of observations.
   * @param x - The X value (must not be NaN)
   * @param y - The Y value (must not be NaN)
   * @returns void
   * @throws {Error} If x or y is NaN
   */
  push(x: number, y: number): void {
    if (Number.isNaN(x) || Number.isNaN(y)) {
      throw new Error("Cannot push NaN values into OnlineCovariance");
    }
    this._count++;
    const dx = x - this._meanX;
    const dy = y - this._meanY;
    this._meanX += dx / this._count;
    this._meanY += dy / this._count;
    const dx2 = x - this._meanX;
    const dy2 = y - this._meanY;
    this._c += dx * dy2;
    this._m2x += dx * dx2;
    this._m2y += dy * dy2;
  }

  /**
   * Add multiple pairs.
   * @param xs - Array of X values
   * @param ys - Array of Y values
   * @returns void
   * @throws {Error} If arrays have different lengths
   * @throws {Error} If any value is NaN
   */
  pushAll(xs: number[], ys: number[]): void {
    if (xs.length !== ys.length) throw new Error("Arrays must have same length");
    for (let i = 0; i < xs.length; i++) this.push(xs[i], ys[i]);
  }

  /**
   * Reset all state.
   * @returns void
   */
  reset(): void {
    this._count = 0;
    this._meanX = 0;
    this._meanY = 0;
    this._c = 0;
    this._m2x = 0;
    this._m2y = 0;
  }
}

/**
 * P² algorithm for online quantile estimation.
 *
 * Estimates a quantile (e.g., median) from a stream without storing
 * all data points. Uses piecewise parabolic interpolation.
 * For fewer than 5 observations, falls back to a simple sort-based estimate.
 */
export class OnlineQuantile {
  private readonly p: number;
  private initialized = false;
  private _count = 0;
  private q = new Array<number>(5);
  private n = [0, 1, 2, 3, 4];
  private np: number[];

  /**
   * @param quantile - The quantile to estimate (0 to 1, e.g., 0.5 for median)
   * @throws {Error} If quantile is not between 0 and 1 (exclusive)
   */
  constructor(quantile = 0.5) {
    if (quantile <= 0 || quantile >= 1) {
      throw new Error("Quantile must be between 0 and 1 (exclusive)");
    }
    this.p = quantile;
    this.np = [0, 2 * quantile, 4 * quantile, 2 + 2 * quantile, 4];
  }

  /**
   * Number of observations seen so far.
   * @returns The current count
   */
  get count(): number {
    return this._count;
  }

  /**
   * Current quantile estimate.
   *
   * For count < 5 the estimate is computed via a simple sort of the
   * collected values rather than the P² algorithm, since P² requires
   * at least 5 markers to be initialized.
   *
   * @returns The estimated quantile value
   * @throws {Error} If no observations have been added
   */
  get estimate(): number {
    if (this._count === 0) throw new Error("No observations");
    if (this._count < 5) {
      // Not enough for P², use simple sort-based fallback
      const sorted = this.q.slice(0, this._count).sort((a, b) => a - b);
      const idx = Math.floor(this.p * (sorted.length - 1));
      return sorted[idx];
    }
    return this.q[2]; // The middle marker is the estimate
  }

  /**
   * Add an observation.
   * @param value - The numeric value to add (must not be NaN)
   * @returns void
   * @throws {Error} If value is NaN
   */
  push(value: number): void {
    if (Number.isNaN(value)) {
      throw new Error("Cannot push NaN value into OnlineQuantile");
    }
    this._count++;

    if (this._count <= 5) {
      this.q[this._count - 1] = value;
      if (this._count === 5) {
        // Sort initial observations
        this.q.sort((a, b) => a - b);
        this.initialized = true;
      }
      return;
    }

    // Find cell k such that q[k] <= value < q[k+1]
    let k: number;
    if (value < this.q[0]) {
      this.q[0] = value;
      k = 0;
    } else if (value >= this.q[4]) {
      this.q[4] = value;
      k = 3;
    } else {
      k = 0;
      for (let i = 1; i < 5; i++) {
        if (value < this.q[i]) {
          k = i - 1;
          break;
        }
      }
    }

    // Increment markers k+1 through 4
    for (let i = k + 1; i < 5; i++) this.n[i]++;

    // Update desired positions
    this.np[1] = this._count * this.p / 2;
    this.np[2] = this._count * this.p;
    this.np[3] = this._count * (1 + this.p) / 2;
    this.np[4] = this._count - 1;

    // Adjust markers 1, 2, 3
    for (let i = 1; i <= 3; i++) {
      const d = this.np[i] - this.n[i];
      if (
        (d >= 1 && this.n[i + 1] - this.n[i] > 1) ||
        (d <= -1 && this.n[i - 1] - this.n[i] < -1)
      ) {
        const sign = d > 0 ? 1 : -1;
        // Parabolic interpolation
        const qi = this.parabolic(i, sign);
        if (qi >= this.q[i - 1] && qi <= this.q[i + 1]) {
          this.q[i] = qi;
        } else {
          // Linear interpolation
          this.q[i] = this.linear(i, sign);
        }
        this.n[i] += sign;
      }
    }
  }

  /**
   * Add multiple observations.
   * @param values - Array of numeric values to add
   * @returns void
   * @throws {Error} If any value is NaN
   */
  pushAll(values: number[]): void {
    for (const v of values) this.push(v);
  }

  private parabolic(i: number, d: number): number {
    const qi = this.q[i];
    const qim1 = this.q[i - 1];
    const qip1 = this.q[i + 1];
    const ni = this.n[i];
    const nim1 = this.n[i - 1];
    const nip1 = this.n[i + 1];

    return (
      qi +
      (d / (nip1 - nim1)) *
        ((ni - nim1 + d) * (qip1 - qi) / (nip1 - ni) +
         (nip1 - ni - d) * (qi - qim1) / (ni - nim1))
    );
  }

  private linear(i: number, d: number): number {
    const j = i + d;
    return this.q[i] + (d * (this.q[j] - this.q[i])) / (this.n[j] - this.n[i]);
  }
}
