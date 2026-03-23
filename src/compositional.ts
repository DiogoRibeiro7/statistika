/**
 * Compositional Data Analysis (CoDA).
 *
 * Data that sums to a constant (e.g. proportions, percentages).
 * Operations in Aitchison geometry on the simplex.
 *
 * - **Log-ratio transforms** — CLR, ALR, ILR (and inverses).
 * - **Aitchison geometry** — perturbation, powering, inner product, distance.
 * - **Closure** — project to the simplex.
 * - **Centre** — geometric mean normalisation.
 * - **Variation matrix** — pairwise log-ratio variances.
 */

// ── Log-ratio Transforms ──────────────────────────────────────────────────

/**
 * Centred Log-Ratio (CLR) transform.
 *
 * clr(x)ᵢ = ln(xᵢ) − (1/D) Σ ln(xⱼ)
 *
 * @param x  Composition (all positive, length D).
 */
export function clr(x: number[]): number[] {
  validateComposition(x);
  const D = x.length;
  let sumLog = 0;
  for (let i = 0; i < D; i++) sumLog += Math.log(x[i]);
  const meanLog = sumLog / D;
  return x.map((xi) => Math.log(xi) - meanLog);
}

/**
 * Inverse CLR transform.
 *
 * clr⁻¹(y) = closure(exp(y))
 */
export function clrInverse(y: number[]): number[] {
  const exp = y.map(Math.exp);
  return closure(exp);
}

/**
 * Additive Log-Ratio (ALR) transform.
 *
 * alr(x)ᵢ = ln(xᵢ / x_D) for i = 1, ..., D−1
 *
 * @param x  Composition (all positive, length D).
 * @param ref  Reference component index (default D−1, the last).
 */
export function alr(x: number[], ref?: number): number[] {
  validateComposition(x);
  const D = x.length;
  const r = ref ?? D - 1;
  const result: number[] = [];
  for (let i = 0; i < D; i++) {
    if (i !== r) result.push(Math.log(x[i] / x[r]));
  }
  return result;
}

/**
 * Inverse ALR transform.
 *
 * @param y  ALR coordinates (length D−1).
 * @param ref  Reference component index used in forward transform.
 */
export function alrInverse(y: number[], ref?: number): number[] {
  const D = y.length + 1;
  const r = ref ?? D - 1;
  const result = new Array(D);
  let insertIdx = 0;
  for (let i = 0; i < D; i++) {
    if (i === r) {
      result[i] = 1;
    } else {
      result[i] = Math.exp(y[insertIdx++]);
    }
  }
  return closure(result);
}

/**
 * Isometric Log-Ratio (ILR) transform.
 *
 * Uses the Helmert sub-matrix as the default contrast matrix.
 * Maps D-part composition to (D−1) real coordinates.
 *
 * @param x  Composition (all positive, length D).
 */
export function ilr(x: number[]): number[] {
  validateComposition(x);
  const D = x.length;
  const logX = x.map(Math.log);

  // Helmert-based ILR coordinates
  const result: number[] = [];
  for (let i = 0; i < D - 1; i++) {
    const k = i + 1;
    let sum = 0;
    for (let j = 0; j < k; j++) sum += logX[j];
    const coef = Math.sqrt(k / (k + 1));
    result.push(coef * (sum / k - logX[k]));
  }
  return result;
}

/**
 * Inverse ILR transform.
 *
 * Reconstructs using the Helmert sub-matrix V (D × D−1)
 * where logX = V · y, then closure(exp(logX)).
 *
 * @param y  ILR coordinates (length D−1).
 */
export function ilrInverse(y: number[]): number[] {
  const D = y.length + 1;

  // Build Helmert sub-matrix V (D × D-1)
  // V[j][i] = { sqrt(1/(k(k+1)))  if j < k
  //           { -sqrt(k/(k+1))     if j = k
  //           { 0                   if j > k
  // where k = i + 1
  const logX = new Array(D).fill(0);
  for (let i = 0; i < D - 1; i++) {
    const k = i + 1;
    const a = 1 / Math.sqrt(k * (k + 1));
    const b = -Math.sqrt(k / (k + 1));
    for (let j = 0; j < k; j++) logX[j] += a * y[i];
    logX[k] += b * y[i];
  }

  const result = logX.map(Math.exp);
  return closure(result);
}

// ── Aitchison Geometry ────────────────────────────────────────────────────

/**
 * Closure operation: project a positive vector onto the simplex.
 *
 * C(x)ᵢ = xᵢ / Σ xⱼ
 */
export function closure(x: number[], total = 1): number[] {
  let sum = 0;
  for (const xi of x) sum += xi;
  if (sum === 0) throw new Error("Cannot close a zero vector");
  return x.map((xi) => (xi / sum) * total);
}

/**
 * Perturbation (Aitchison addition): x ⊕ y = C(x₁y₁, ..., xDyD).
 */
export function perturbation(x: number[], y: number[]): number[] {
  if (x.length !== y.length) throw new Error("Compositions must have same length");
  const product = x.map((xi, i) => xi * y[i]);
  return closure(product);
}

/**
 * Power transformation (Aitchison scalar multiplication): α ⊙ x = C(x₁^α, ..., xD^α).
 */
export function powering(x: number[], alpha: number): number[] {
  const powered = x.map((xi) => Math.pow(xi, alpha));
  return closure(powered);
}

/**
 * Aitchison inner product: ⟨x, y⟩_A = (1/D) Σᵢ<ⱼ ln(xᵢ/xⱼ) ln(yᵢ/yⱼ).
 */
export function aitchisonInnerProduct(x: number[], y: number[]): number {
  const D = x.length;
  if (D !== y.length) throw new Error("Compositions must have same length");
  let sum = 0;
  for (let i = 0; i < D; i++) {
    for (let j = i + 1; j < D; j++) {
      sum += Math.log(x[i] / x[j]) * Math.log(y[i] / y[j]);
    }
  }
  return sum / D;
}

/**
 * Aitchison distance: d_A(x, y) = √⟨x⊖y, x⊖y⟩_A
 * where x⊖y = perturbation(x, powering(y, -1)).
 */
export function aitchisonDistance(x: number[], y: number[]): number {
  const D = x.length;
  if (D !== y.length) throw new Error("Compositions must have same length");
  let sum = 0;
  for (let i = 0; i < D; i++) {
    for (let j = i + 1; j < D; j++) {
      const diff = Math.log(x[i] / x[j]) - Math.log(y[i] / y[j]);
      sum += diff * diff;
    }
  }
  return Math.sqrt(sum / D);
}

/**
 * Aitchison norm: ‖x‖_A = √⟨x, x⟩_A.
 */
export function aitchisonNorm(x: number[]): number {
  return Math.sqrt(aitchisonInnerProduct(x, x));
}

// ── Centre and Variation ──────────────────────────────────────────────────

/**
 * Compositional centre (geometric mean composition).
 *
 * centre(X) = C(g₁, g₂, ..., gD) where gⱼ = (∏ᵢ xᵢⱼ)^{1/n}.
 *
 * @param compositions  Array of compositions (n × D).
 */
export function compositionalCentre(compositions: number[][]): number[] {
  const n = compositions.length;
  if (n === 0) throw new Error("Need at least 1 composition");
  const D = compositions[0].length;

  const logMeans = new Array(D).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < D; j++) {
      logMeans[j] += Math.log(compositions[i][j]);
    }
  }
  for (let j = 0; j < D; j++) logMeans[j] /= n;

  return closure(logMeans.map(Math.exp));
}

/**
 * Variation matrix: T[i][j] = Var(ln(xᵢ/xⱼ)).
 *
 * @param compositions  Array of compositions (n × D).
 */
export function variationMatrix(compositions: number[][]): number[][] {
  const n = compositions.length;
  if (n < 2) throw new Error("Need at least 2 compositions");
  const D = compositions[0].length;

  const T: number[][] = Array.from({ length: D }, () => new Array(D).fill(0));

  for (let i = 0; i < D; i++) {
    for (let j = 0; j < D; j++) {
      if (i === j) continue;
      const logRatios: number[] = [];
      for (let k = 0; k < n; k++) {
        logRatios.push(Math.log(compositions[k][i] / compositions[k][j]));
      }
      const m = logRatios.reduce((a, b) => a + b, 0) / n;
      let v = 0;
      for (const lr of logRatios) v += (lr - m) ** 2;
      T[i][j] = v / (n - 1);
    }
  }

  return T;
}

// ── Helpers ───────────────────────────────────────────────────────────────

function validateComposition(x: number[]): void {
  if (x.length < 2) throw new Error("Composition must have at least 2 parts");
  for (let i = 0; i < x.length; i++) {
    if (x[i] <= 0) throw new Error(`Part x[${i}] must be positive`);
  }
}
