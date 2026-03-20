/**
 * Example: Survival analysis.
 *
 * Run: npx ts-node examples/survival-analysis.ts
 */

import { kaplanMeier, logRankTest } from "../src/survival";
import { coxRegression, CoxObservation } from "../src/cox-regression";

// --- Kaplan-Meier estimator ---
// Clinical trial: time to relapse (months)
const treatment = [
  { time: 6, event: true },
  { time: 7, event: true },
  { time: 10, event: false }, // censored (lost to follow-up)
  { time: 15, event: true },
  { time: 16, event: true },
  { time: 22, event: false },
  { time: 23, event: true },
  { time: 24, event: false },
  { time: 28, event: true },
  { time: 30, event: false },
];

const km = kaplanMeier(treatment);
console.log("=== Kaplan-Meier Estimator (treatment group) ===");
console.log(`  N: ${km.n}, Events: ${km.nEvents}, Censored: ${km.nCensored}`);
console.log(`  Median survival: ${km.medianSurvival} months`);
console.log(`  S(12 months): ${km.survivalAt(12).toFixed(4)}`);
console.log(`  S(24 months): ${km.survivalAt(24).toFixed(4)}`);
console.log("\n  Survival curve:");
for (const pt of km.curve) {
  console.log(
    `    t=${String(pt.time).padStart(2)}: S=${pt.survival.toFixed(3)}  ` +
      `95% CI [${pt.lower.toFixed(3)}, ${pt.upper.toFixed(3)}]  ` +
      `(${pt.nRisk} at risk)`,
  );
}
console.log();

// --- Log-rank test ---
// Compare treatment vs control
const control = [
  { time: 1, event: true },
  { time: 3, event: true },
  { time: 5, event: true },
  { time: 8, event: false },
  { time: 9, event: true },
  { time: 12, event: true },
  { time: 13, event: false },
  { time: 14, event: true },
  { time: 18, event: true },
  { time: 20, event: false },
];

const logrank = logRankTest(treatment, control);
console.log("=== Log-Rank Test (treatment vs control) ===");
console.log(`  Chi-squared: ${logrank.chiSquared.toFixed(4)}`);
console.log(`  p-value: ${logrank.pValue.toFixed(4)}`);
console.log(`  Reject H0 at alpha=0.05? ${logrank.rejected}`);
console.log();

// --- Cox Proportional Hazards ---
// Model: hazard depends on age and treatment (0=control, 1=treatment)
const coxData: CoxObservation[] = [
  { time: 1, event: true, covariates: [65, 0] },
  { time: 3, event: true, covariates: [58, 0] },
  { time: 5, event: true, covariates: [72, 0] },
  { time: 6, event: true, covariates: [60, 1] },
  { time: 7, event: true, covariates: [55, 1] },
  { time: 9, event: true, covariates: [68, 0] },
  { time: 10, event: false, covariates: [50, 1] },
  { time: 12, event: true, covariates: [70, 0] },
  { time: 15, event: true, covariates: [45, 1] },
  { time: 16, event: true, covariates: [62, 1] },
  { time: 22, event: false, covariates: [48, 1] },
  { time: 23, event: true, covariates: [53, 1] },
  { time: 28, event: true, covariates: [67, 0] },
  { time: 30, event: false, covariates: [42, 1] },
];

const cox = coxRegression(coxData);
const labels = ["Age", "Treatment"];
console.log("=== Cox Proportional Hazards ===");
console.log(`  Concordance: ${cox.concordance.toFixed(4)}`);
console.log(`  Log-likelihood: ${cox.logLikelihood.toFixed(4)}`);
console.log(`  Iterations: ${cox.iterations}`);
console.log("\n  Coefficients:");
for (let i = 0; i < labels.length; i++) {
  console.log(
    `    ${labels[i].padEnd(12)} beta=${cox.coefficients[i].toFixed(4)}  ` +
      `HR=${cox.hazardRatios[i].toFixed(3)}  ` +
      `95% CI [${cox.hazardRatioCIs[i][0].toFixed(3)}, ${cox.hazardRatioCIs[i][1].toFixed(3)}]  ` +
      `p=${cox.pValues[i].toFixed(4)}`,
  );
}
console.log(`\n  Hazard ratio for 60-year-old on treatment:`);
console.log(`    HR = ${cox.predictHazardRatio([60, 1]).toFixed(4)}`);
