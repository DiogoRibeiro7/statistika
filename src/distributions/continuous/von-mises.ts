import { BaseContinuous } from "../base";
import { quantileBisect } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Computes the modified Bessel function of the first kind, I0(x),
 * using a series approximation.
 *
 * I0(x) = sum_{m=0}^{inf} (x^2/4)^m / (m!)^2
 *
 * @param x - The argument.
 * @returns I0(x).
 */
function besselI0(x: number): number {
  const ax = Math.abs(x);
  if (ax < 3.75) {
    // Polynomial approximation for small arguments
    const t = (ax / 3.75) ** 2;
    return (
      1.0 +
      t *
        (3.5156229 +
          t *
            (3.0899424 +
              t *
                (1.2067492 +
                  t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    );
  }
  // Asymptotic approximation for large arguments
  const t = 3.75 / ax;
  return (
    (Math.exp(ax) / Math.sqrt(ax)) *
    (0.39894228 +
      t *
        (0.01328592 +
          t *
            (0.00225319 +
              t *
                (-0.00157565 +
                  t *
                    (0.00916281 +
                      t *
                        (-0.02057706 +
                          t *
                            (0.02635537 +
                              t * (-0.01647633 + t * 0.00392377))))))))
  );
}

/**
 * Von Mises distribution.
 *
 * A continuous probability distribution on the circle, often described as
 * the circular analogue of the normal distribution. Parameterized by a
 * mean direction `mu` and a concentration parameter `kappa`.
 *
 * PDF: f(x) = exp(kappa * cos(x - mu)) / (2 * pi * I0(kappa))
 *
 * Support: [-pi, pi]
 *
 * @example
 * ```ts
 * const dist = new VonMises(0, 2);
 * dist.mean();       // 0
 * dist.pdf(0);       // maximum density
 * dist.cdf(0);       // 0.5
 * dist.sample();     // random angle in [-pi, pi]
 * ```
 */
export class VonMises extends BaseContinuous {
  readonly name: string;
  private readonly i0Kappa: number; // I0(kappa)

  /**
   * Creates a new Von Mises distribution.
   *
   * @param mu - Mean direction in [-pi, pi]. Defaults to 0.
   * @param kappa - Concentration parameter (must be >= 0). Defaults to 1.
   *   - kappa = 0: uniform distribution on the circle
   *   - large kappa: approaches a wrapped normal distribution
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If kappa is negative.
   */
  constructor(
    public readonly mu: number = 0,
    public readonly kappa: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (kappa < 0) throw new Error(`Invalid parameter 'kappa': expected a non-negative number, received ${kappa}`);
    this.name = `VonMises(${mu}, ${kappa})`;
    this.i0Kappa = besselI0(kappa);
  }

  /**
   * Returns the mean direction of the Von Mises distribution.
   *
   * The circular mean is simply the location parameter `mu`.
   *
   * @returns The mean direction.
   */
  mean(): number {
    return this.mu;
  }

  /**
   * Returns the circular variance of the Von Mises distribution.
   *
   * Formula: V = 1 - I1(kappa) / I0(kappa)
   *
   * where I1 is the modified Bessel function of order 1. For simplicity,
   * this uses the approximation I1(kappa)/I0(kappa) ≈ A(kappa).
   *
   * @returns The circular variance in [0, 1].
   */
  variance(): number {
    if (this.kappa === 0) return 1;
    // I1(kappa)/I0(kappa) approximation using the ratio
    const i1Ratio = besselI1Ratio(this.kappa);
    return 1 - i1Ratio;
  }

  /**
   * Evaluates the probability density function at `x`.
   *
   * PDF: f(x) = exp(kappa * cos(x - mu)) / (2 * pi * I0(kappa))
   *
   * @param x - The angle at which to evaluate the density (any real number,
   *   interpreted modulo 2*pi).
   * @returns The probability density f(x) >= 0.
   */
  pdf(x: number): number {
    return Math.exp(this.kappa * Math.cos(x - this.mu)) / (2 * Math.PI * this.i0Kappa);
  }

  /**
   * Evaluates the cumulative distribution function at `x` via numerical integration.
   *
   * Computes F(x) = integral from -pi to x of the PDF using Simpson's rule.
   *
   * @param x - The angle at which to evaluate the CDF.
   * @returns P(X <= x) in [0, 1] for x in [-pi, pi].
   */
  cdf(x: number): number {
    if (x <= -Math.PI) return 0;
    if (x >= Math.PI) return 1;

    // Numerical integration from -pi to x using Simpson's rule
    const n = 200; // number of subintervals (must be even)
    const a = -Math.PI;
    const h = (x - a) / n;
    let sum = this.pdf(a) + this.pdf(x);

    for (let i = 1; i < n; i++) {
      const xi = a + i * h;
      sum += (i % 2 === 0 ? 2 : 4) * this.pdf(xi);
    }

    return (h / 3) * sum;
  }

  /**
   * Computes the quantile (inverse CDF) using bisection search.
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value in [-pi, pi].
   * @throws {Error} If p is not in [0, 1].
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    if (p === 0) return -Math.PI;
    if (p === 1) return Math.PI;
    return quantileBisect((x) => this.cdf(x), p, -Math.PI, Math.PI);
  }

  /**
   * Draws a single random sample using the Best-Fisher algorithm for
   * sampling from the Von Mises distribution.
   *
   * @returns A random angle in [-pi, pi].
   */
  sample(): number {
    if (this.kappa === 0) {
      // Uniform on [-pi, pi]
      return -Math.PI + 2 * Math.PI * this.rng();
    }

    // Best-Fisher algorithm for Von Mises sampling
    const tau = 1 + Math.sqrt(1 + 4 * this.kappa * this.kappa);
    const rho = (tau - Math.sqrt(2 * tau)) / (2 * this.kappa);
    const r = (1 + rho * rho) / (2 * rho);

    while (true) {
      const u1 = this.rng();
      const z = Math.cos(Math.PI * u1);
      const f = (1 + r * z) / (r + z);
      const c = this.kappa * (r - f);

      const u2 = this.rng();
      if (c * (2 - c) > u2 || Math.log(c / u2) + 1 >= c) {
        const u3 = this.rng();
        const theta = (u3 > 0.5 ? 1 : -1) * Math.acos(f) + this.mu;
        // Wrap to [-pi, pi]
        return ((theta + 3 * Math.PI) % (2 * Math.PI)) - Math.PI;
      }
    }
  }
}

/**
 * Computes the ratio I1(kappa)/I0(kappa) using a series-based approximation.
 *
 * @param kappa - The concentration parameter.
 * @returns The ratio I1(kappa)/I0(kappa).
 */
function besselI1Ratio(kappa: number): number {
  if (kappa === 0) return 0;
  if (kappa < 0.5) {
    // Series expansion for small kappa
    return kappa / 2 * (1 - kappa * kappa / 8);
  }
  // Use the recurrence: I1(x) = I0(x) - (2*0/x)*I0(x) ... or compute directly
  // For better accuracy, compute I1 directly using polynomial approximation
  const ax = Math.abs(kappa);
  let i1: number;
  if (ax < 3.75) {
    const t = (ax / 3.75) ** 2;
    i1 =
      ax *
      (0.5 +
        t *
          (0.87890594 +
            t *
              (0.51498869 +
                t *
                  (0.15084934 +
                    t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))));
  } else {
    const t = 3.75 / ax;
    i1 =
      (Math.exp(ax) / Math.sqrt(ax)) *
      (0.39894228 +
        t *
          (-0.03988024 +
            t *
              (-0.00362018 +
                t *
                  (0.00163801 +
                    t *
                      (-0.01031555 +
                        t *
                          (0.02282967 +
                            t *
                              (-0.02895312 +
                                t * (0.01787654 + t * -0.00420059))))))));
  }
  return i1 / besselI0(kappa);
}
