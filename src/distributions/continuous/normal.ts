import { BaseContinuous } from "../base";
import { erf, quantileBisect } from "../../utils/math";
import { RandomFn } from "../../types";

export class Normal extends BaseContinuous {
  readonly name: string;

  constructor(
    public readonly mu: number = 0,
    public readonly sigma: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (sigma <= 0) throw new Error("sigma must be positive");
    this.name = `Normal(${mu}, ${sigma})`;
  }

  mean(): number {
    return this.mu;
  }

  variance(): number {
    return this.sigma ** 2;
  }

  pdf(x: number): number {
    const z = (x - this.mu) / this.sigma;
    return Math.exp(-0.5 * z * z) / (this.sigma * Math.sqrt(2 * Math.PI));
  }

  cdf(x: number): number {
    return 0.5 * (1 + erf((x - this.mu) / (this.sigma * Math.SQRT2)));
  }

  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return -Infinity;
    if (p === 1) return Infinity;
    // Rational approximation for the standard normal quantile
    return this.mu + this.sigma * standardNormalQuantile(p);
  }

  sample(): number {
    // Box-Muller transform
    const u1 = this.rng();
    const u2 = this.rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return this.mu + this.sigma * z;
  }
}

/** Rational approximation for the standard normal quantile (Beasley-Springer-Moro). */
function standardNormalQuantile(p: number): number {
  if (p < 0.5) return -standardNormalQuantile(1 - p);

  const t = Math.sqrt(-2 * Math.log(1 - p));
  const c0 = 2.515517;
  const c1 = 0.802853;
  const c2 = 0.010328;
  const d1 = 1.432788;
  const d2 = 0.189269;
  const d3 = 0.001308;

  return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
}
