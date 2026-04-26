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
    if (this._count === 0) throw new Error(`Invalid state 'count': expected at least 1 observation, received ${this._count}`);
    return this._mean;
  }

  /**
   * Running sample variance.
   * @returns The current sample variance (using Bessel's correction)
   * @throws {Error} If fewer than 2 observations have been added
   */
  get variance(): number {
    if (this._count < 2) throw new Error(`Invalid state 'count': expected at least 2 observations, received ${this._count}`);
    return this._m2 / (this._count - 1);
  }

  /**
   * Running population variance.
   * @returns The current population variance
   * @throws {Error} If no observations have been added
   */
  get populationVariance(): number {
    if (this._count === 0) throw new Error(`Invalid state 'count': expected at least 1 observation, received ${this._count}`);
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
    if (this._count === 0) throw new Error(`Invalid state 'count': expected at least 1 observation, received ${this._count}`);
    return this._min;
  }

  /**
   * Maximum value seen.
   * @returns The maximum value across all observations
   * @throws {Error} If no observations have been added
   */
  get max(): number {
    if (this._count === 0) throw new Error(`Invalid state 'count': expected at least 1 observation, received ${this._count}`);
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
      throw new Error(`Invalid parameter 'value': expected a non-NaN number, received NaN`);
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
      if (Number.isNaN(v)) throw new Error(`Invalid parameter 'values': expected non-NaN numbers, received NaN`);
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
 * Uses a numerically stable one-pass algorithm based on Welford's method
 * extended to two variables. Computes running covariance, correlation,
 * and means without storing individual observations.
 *
 * @example
 * ```ts
 * const cov = new OnlineCovariance();
 * cov.push(1, 2);
 * cov.push(2, 4);
 * cov.push(3, 6);
 * cov.correlation; // 1.0 (perfect positive correlation)
 * ```
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
    if (this._count < 2) throw new Error(`Invalid state 'count': expected at least 2 observations, received ${this._count}`);
    return this._c / (this._count - 1);
  }

  /**
   * Running Pearson correlation coefficient.
   * @returns The current Pearson correlation coefficient, or 0 if variance is zero
   * @throws {Error} If fewer than 2 observation pairs have been added
   */
  get correlation(): number {
    if (this._count < 2) throw new Error(`Invalid state 'count': expected at least 2 observations, received ${this._count}`);
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
    if (this._count === 0) throw new Error(`Invalid state 'count': expected at least 1 observation, received ${this._count}`);
    return this._meanX;
  }

  /**
   * Running mean of the Y stream.
   * @returns The current mean of Y values
   * @throws {Error} If no observations have been added
   */
  get meanY(): number {
    if (this._count === 0) throw new Error(`Invalid state 'count': expected at least 1 observation, received ${this._count}`);
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
      throw new Error(`Invalid parameters 'x', 'y': expected non-NaN numbers, received x=${x}, y=${y}`);
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
    if (xs.length !== ys.length) throw new Error(`Invalid parameters 'xs', 'ys': expected same length, received xs.length=${xs.length}, ys.length=${ys.length}`);
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
 * P-squared algorithm for online quantile estimation.
 *
 * Estimates a quantile (e.g., median) from a stream without storing
 * all data points. Uses piecewise parabolic interpolation with 5 markers.
 * For fewer than 5 observations, falls back to a simple sort-based estimate.
 *
 * Reference: Jain, R. and Chlamtac, I. (1985). "The P-squared Algorithm for
 * Dynamic Calculation of Quantiles and Histograms Without Storing Observations."
 *
 * @example
 * ```ts
 * const q = new OnlineQuantile(0.5); // track median
 * for (let i = 0; i < 1000; i++) q.push(Math.random());
 * q.estimate; // approximately 0.5
 * ```
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
      throw new Error(`Invalid parameter 'quantile': expected a value in (0, 1), received ${quantile}`);
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
    if (this._count === 0) throw new Error(`Invalid state 'count': expected at least 1 observation, received ${this._count}`);
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
      throw new Error(`Invalid parameter 'value': expected a non-NaN number, received NaN`);
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

/**
 * Online skewness and kurtosis using Welford's method extended to 3rd/4th moments.
 *
 * Computes running mean, variance, skewness (adjusted Fisher-Pearson),
 * and excess kurtosis in a single pass without storing observations.
 * Numerically stable via central-moment update formulas.
 *
 * @example
 * ```ts
 * const sk = new OnlineSkewnessKurtosis();
 * for (let i = 0; i < 1000; i++) sk.push(Math.random());
 * sk.mean;     // ~0.5
 * sk.skewness; // ~0
 * sk.kurtosis; // ~-1.2 (excess kurtosis of uniform)
 * ```
 */
export class OnlineSkewnessKurtosis {
  private _count = 0;
  private _mean = 0;
  private _m2 = 0;
  private _m3 = 0;
  private _m4 = 0;

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
    if (this._count === 0) throw new Error(`Invalid state 'count': expected at least 1 observation, received ${this._count}`);
    return this._mean;
  }

  /**
   * Running sample variance.
   * @returns The current sample variance (using Bessel's correction)
   * @throws {Error} If fewer than 2 observations have been added
   */
  get variance(): number {
    if (this._count < 2) throw new Error(`Invalid state 'count': expected at least 2 observations, received ${this._count}`);
    return this._m2 / (this._count - 1);
  }

  /**
   * Running sample skewness (adjusted Fisher-Pearson standardized moment).
   *
   * Uses the formula: g1 = (n * sqrt(n-1) / (n-2)) * (M3 / M2^(3/2))
   * where M2, M3 are the 2nd and 3rd central moment sums.
   *
   * @returns The current sample skewness
   * @throws {Error} If fewer than 3 observations have been added
   */
  get skewness(): number {
    if (this._count < 3) throw new Error(`Invalid state 'count': expected at least 3 observations, received ${this._count}`);
    if (this._m2 === 0) return 0;
    const n = this._count;
    return (
      (Math.sqrt(n * (n - 1)) / (n - 2)) *
      (this._m3 / Math.pow(this._m2, 1.5)) *
      Math.sqrt(n)
    );
  }

  /**
   * Running excess kurtosis.
   *
   * Uses the formula for sample excess kurtosis:
   * G2 = ((n-1)/((n-2)(n-3))) * ((n+1) * (n * M4 / M2²) - 3*(n-1))
   *
   * @returns The current excess kurtosis
   * @throws {Error} If fewer than 4 observations have been added
   */
  get kurtosis(): number {
    if (this._count < 4) throw new Error(`Invalid state 'count': expected at least 4 observations, received ${this._count}`);
    if (this._m2 === 0) return 0;
    const n = this._count;
    const kurtPop = (n * this._m4) / (this._m2 * this._m2);
    return (
      ((n - 1) / ((n - 2) * (n - 3))) *
      ((n + 1) * kurtPop - 3 * (n - 1))
    );
  }

  /**
   * Add a single observation.
   * @param value - The numeric value to add (must not be NaN)
   * @returns void
   * @throws {Error} If value is NaN
   */
  push(value: number): void {
    if (Number.isNaN(value)) {
      throw new Error(`Invalid parameter 'value': expected a non-NaN number, received NaN`);
    }
    const n1 = this._count;
    this._count++;
    const n = this._count;
    const delta = value - this._mean;
    const deltaN = delta / n;
    const deltaN2 = deltaN * deltaN;
    const term1 = delta * deltaN * n1;

    this._mean += deltaN;
    this._m4 +=
      term1 * deltaN2 * (n * n - 3 * n + 3) +
      6 * deltaN2 * this._m2 -
      4 * deltaN * this._m3;
    this._m3 += term1 * deltaN * (n - 2) - 3 * deltaN * this._m2;
    this._m2 += term1;
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

  /**
   * Reset all state.
   * @returns void
   */
  reset(): void {
    this._count = 0;
    this._mean = 0;
    this._m2 = 0;
    this._m3 = 0;
    this._m4 = 0;
  }
}

/**
 * Online Pearson correlation for two variables.
 *
 * A convenience wrapper around the streaming covariance algorithm
 * that provides a correlation-focused API. Uses a numerically stable
 * one-pass algorithm based on Welford's method extended to two variables.
 *
 * @example
 * ```ts
 * const corr = new OnlineCorrelation();
 * for (let i = 0; i < 100; i++) corr.push(i, 2 * i + 1);
 * corr.correlation; // 1.0
 * ```
 */
export class OnlineCorrelation {
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
   * Running Pearson correlation coefficient.
   * @returns The current Pearson correlation coefficient, or 0 if variance is zero
   * @throws {Error} If fewer than 2 observation pairs have been added
   */
  get correlation(): number {
    if (this._count < 2) throw new Error(`Invalid state 'count': expected at least 2 observations, received ${this._count}`);
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
    if (this._count === 0) throw new Error(`Invalid state 'count': expected at least 1 observation, received ${this._count}`);
    return this._meanX;
  }

  /**
   * Running mean of the Y stream.
   * @returns The current mean of Y values
   * @throws {Error} If no observations have been added
   */
  get meanY(): number {
    if (this._count === 0) throw new Error(`Invalid state 'count': expected at least 1 observation, received ${this._count}`);
    return this._meanY;
  }

  /**
   * Running sample covariance.
   * @returns The current sample covariance
   * @throws {Error} If fewer than 2 observation pairs have been added
   */
  get covariance(): number {
    if (this._count < 2) throw new Error(`Invalid state 'count': expected at least 2 observations, received ${this._count}`);
    return this._c / (this._count - 1);
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
      throw new Error(`Invalid parameters 'x', 'y': expected non-NaN numbers, received x=${x}, y=${y}`);
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
    if (xs.length !== ys.length) throw new Error(`Invalid parameters 'xs', 'ys': expected same length, received xs.length=${xs.length}, ys.length=${ys.length}`);
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
 * Online covariance and correlation matrix for p variables.
 *
 * Uses a numerically stable one-pass algorithm extending Welford's method
 * to multiple dimensions. Computes the full p x p covariance matrix
 * and correlation matrix without storing individual observations.
 *
 * @example
 * ```ts
 * const mat = new OnlineCovarianceMatrix(3);
 * mat.push([1, 2, 3]);
 * mat.push([4, 5, 6]);
 * mat.push([7, 8, 9]);
 * mat.covarianceMatrix; // 3x3 sample covariance matrix
 * mat.correlationMatrix; // 3x3 correlation matrix
 * ```
 */
export class OnlineCovarianceMatrix {
  private readonly _dim: number;
  private _count = 0;
  private _mean: number[];
  private _c: number[][]; // upper triangle co-moment sums

  /**
   * @param dim - The number of variables (dimensions), must be >= 1
   * @throws {Error} If dim < 1 or is not an integer
   */
  constructor(dim: number) {
    if (!Number.isInteger(dim) || dim < 1) {
      throw new Error(`Invalid parameter 'dim': expected a positive integer, received ${dim}`);
    }
    this._dim = dim;
    this._mean = new Array<number>(dim).fill(0);
    this._c = Array.from({ length: dim }, () => new Array<number>(dim).fill(0));
  }

  /**
   * Number of observation vectors seen so far.
   * @returns The current count
   */
  get count(): number {
    return this._count;
  }

  /**
   * The dimension (number of variables).
   * @returns The dimension
   */
  get dim(): number {
    return this._dim;
  }

  /**
   * Running mean vector.
   * @returns A copy of the current mean vector
   * @throws {Error} If no observations have been added
   */
  get means(): number[] {
    if (this._count === 0) throw new Error(`Invalid state 'count': expected at least 1 observation, received ${this._count}`);
    return this._mean.slice();
  }

  /**
   * Running sample covariance matrix.
   * @returns A p x p sample covariance matrix (using Bessel's correction)
   * @throws {Error} If fewer than 2 observations have been added
   */
  get covarianceMatrix(): number[][] {
    if (this._count < 2) throw new Error(`Invalid state 'count': expected at least 2 observations, received ${this._count}`);
    const p = this._dim;
    const cov: number[][] = Array.from({ length: p }, () => new Array<number>(p));
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        cov[i][j] = this._c[i][j] / (this._count - 1);
      }
    }
    return cov;
  }

  /**
   * Running correlation matrix.
   * @returns A p x p Pearson correlation matrix
   * @throws {Error} If fewer than 2 observations have been added
   */
  get correlationMatrix(): number[][] {
    if (this._count < 2) throw new Error(`Invalid state 'count': expected at least 2 observations, received ${this._count}`);
    const p = this._dim;
    const corr: number[][] = Array.from({ length: p }, () => new Array<number>(p));
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        if (i === j) {
          corr[i][j] = 1;
        } else {
          const denom = Math.sqrt(this._c[i][i] * this._c[j][j]);
          corr[i][j] = denom === 0 ? 0 : this._c[i][j] / denom;
        }
      }
    }
    return corr;
  }

  /**
   * Add a single observation vector.
   * @param values - Array of p numeric values (must not contain NaN)
   * @returns void
   * @throws {Error} If values length does not match dimension
   * @throws {Error} If any value is NaN
   */
  push(values: number[]): void {
    if (values.length !== this._dim) {
      throw new Error(`Invalid parameter 'values': Expected ${this._dim} values, got ${values.length}`);
    }
    for (const v of values) {
      if (Number.isNaN(v)) {
        throw new Error(`Invalid parameter 'values': expected non-NaN numbers, received NaN`);
      }
    }
    this._count++;
    const n = this._count;
    const p = this._dim;
    const dx = new Array<number>(p);

    for (let i = 0; i < p; i++) {
      dx[i] = values[i] - this._mean[i];
    }

    for (let i = 0; i < p; i++) {
      this._mean[i] += dx[i] / n;
    }

    for (let i = 0; i < p; i++) {
      // dx2[j] = values[j] - updated mean[j]
      for (let j = i; j < p; j++) {
        const dx2j = values[j] - this._mean[j];
        this._c[i][j] += dx[i] * dx2j;
        if (i !== j) {
          this._c[j][i] = this._c[i][j];
        }
      }
    }
  }

  /**
   * Add multiple observation vectors.
   * @param rows - Array of observation vectors
   * @returns void
   * @throws {Error} If any vector length does not match dimension
   * @throws {Error} If any value is NaN
   */
  pushAll(rows: number[][]): void {
    for (const row of rows) this.push(row);
  }

  /**
   * Reset all state.
   * @returns void
   */
  reset(): void {
    this._count = 0;
    this._mean.fill(0);
    for (const row of this._c) row.fill(0);
  }
}
