import { BaseContinuous } from "../base";
import { gamma, quantileBisect } from "../../utils/math";

/**
 * Generalized Extreme Value (GEV) distribution.
 *
 * Unifies the three extreme value types:
 *   xi = 0  → Gumbel (Type I)
 *   xi > 0  → Fréchet (Type II)
 *   xi < 0  → Reversed Weibull (Type III)
 *
 * Parameters:
 *   mu    — location
 *   sigma — scale (> 0)
 *   xi    — shape
 */
export class GEV extends BaseContinuous {
  readonly name: string;

  constructor(
    public readonly mu: number = 0,
    public readonly sigma: number = 1,
    public readonly xi: number = 0,
  ) {
    super();
    if (sigma <= 0) throw new Error("sigma must be positive");
    this.name = `GEV(${mu}, ${sigma}, ${xi})`;
  }

  mean(): number {
    if (this.xi >= 1) return Infinity;
    if (this.xi === 0) {
      // Euler-Mascheroni constant
      return this.mu + this.sigma * 0.5772156649015329;
    }
    // mu + sigma * (gamma(1 - xi) - 1) / xi
    return this.mu + (this.sigma * (gamma(1 - this.xi) - 1)) / this.xi;
  }

  variance(): number {
    if (this.xi >= 0.5) return Infinity;
    if (this.xi === 0) {
      return (this.sigma ** 2 * Math.PI ** 2) / 6;
    }
    const g1 = gamma(1 - this.xi);
    const g2 = gamma(1 - 2 * this.xi);
    return (this.sigma ** 2 * (g2 - g1 ** 2)) / this.xi ** 2;
  }

  private t(x: number): number {
    const z = (x - this.mu) / this.sigma;
    if (this.xi === 0) return Math.exp(-z);
    const v = 1 + this.xi * z;
    if (v <= 0) return this.xi > 0 ? Infinity : 0;
    return Math.pow(v, -1 / this.xi);
  }

  private isInSupport(x: number): boolean {
    if (this.xi === 0) return true;
    const z = (x - this.mu) / this.sigma;
    return 1 + this.xi * z > 0;
  }

  pdf(x: number): number {
    if (!this.isInSupport(x)) return 0;
    const tx = this.t(x);
    if (!isFinite(tx)) return 0;
    return (tx ** (this.xi + 1) * Math.exp(-tx)) / this.sigma;
  }

  cdf(x: number): number {
    if (this.xi > 0 && x <= this.mu - this.sigma / this.xi) return 0;
    if (this.xi < 0 && x >= this.mu - this.sigma / this.xi) return 1;
    return Math.exp(-this.t(x));
  }

  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) {
      return this.xi > 0 ? this.mu - this.sigma / this.xi : -Infinity;
    }
    if (p === 1) {
      return this.xi < 0 ? this.mu - this.sigma / this.xi : Infinity;
    }
    const lnp = -Math.log(p);
    if (this.xi === 0) {
      return this.mu - this.sigma * Math.log(lnp);
    }
    return this.mu + (this.sigma * (Math.pow(lnp, -this.xi) - 1)) / this.xi;
  }

  sample(): number {
    return this.quantile(Math.random());
  }
}
