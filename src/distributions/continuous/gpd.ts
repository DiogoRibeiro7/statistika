import { BaseContinuous } from "../base";

/**
 * Generalized Pareto Distribution (GPD).
 *
 * Used in EVT for modelling exceedances over a threshold (peaks-over-threshold).
 *
 * Parameters:
 *   mu    — location / threshold
 *   sigma — scale (> 0)
 *   xi    — shape (tail index)
 *
 * Support:
 *   xi >= 0: x >= mu
 *   xi < 0:  mu <= x <= mu - sigma/xi
 */
export class GPD extends BaseContinuous {
  readonly name: string;

  constructor(
    public readonly mu: number = 0,
    public readonly sigma: number = 1,
    public readonly xi: number = 0,
  ) {
    super();
    if (sigma <= 0) throw new Error("sigma must be positive");
    this.name = `GPD(${mu}, ${sigma}, ${xi})`;
  }

  mean(): number {
    if (this.xi >= 1) return Infinity;
    return this.mu + this.sigma / (1 - this.xi);
  }

  variance(): number {
    if (this.xi >= 0.5) return Infinity;
    return this.sigma ** 2 / ((1 - this.xi) ** 2 * (1 - 2 * this.xi));
  }

  private isInSupport(x: number): boolean {
    if (x < this.mu) return false;
    if (this.xi < 0) return x <= this.mu - this.sigma / this.xi;
    return true;
  }

  pdf(x: number): number {
    if (!this.isInSupport(x)) return 0;
    const z = (x - this.mu) / this.sigma;
    if (this.xi === 0) {
      return Math.exp(-z) / this.sigma;
    }
    const v = 1 + this.xi * z;
    if (v <= 0) return 0;
    return Math.pow(v, -(1 / this.xi + 1)) / this.sigma;
  }

  cdf(x: number): number {
    if (x < this.mu) return 0;
    if (this.xi < 0 && x >= this.mu - this.sigma / this.xi) return 1;
    const z = (x - this.mu) / this.sigma;
    if (this.xi === 0) {
      return 1 - Math.exp(-z);
    }
    return 1 - Math.pow(1 + this.xi * z, -1 / this.xi);
  }

  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return this.mu;
    if (p === 1) {
      return this.xi < 0 ? this.mu - this.sigma / this.xi : Infinity;
    }
    if (this.xi === 0) {
      return this.mu - this.sigma * Math.log(1 - p);
    }
    return this.mu + (this.sigma * (Math.pow(1 - p, -this.xi) - 1)) / this.xi;
  }

  sample(): number {
    return this.quantile(Math.random());
  }
}
