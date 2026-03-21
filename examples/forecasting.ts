/**
 * Example: Time series forecasting.
 *
 * Run: npx ts-node examples/forecasting.ts
 */
import { adfTest, autoArima, forecastWithIntervals, seasonalDecompose } from "../src";

// Generate a simple trending series with noise
const n = 100;
let seed = 42;
const rand = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return (seed / 0x7fffffff - 0.5) * 2; };
const series = Array.from({ length: n }, (_, i) => 10 + 0.5 * i + rand() * 3);

console.log("=== Augmented Dickey-Fuller Test ===");
const adf = adfTest(series);
console.log(`  Statistic: ${adf.statistic.toFixed(4)}`);
console.log(`  p-value: ${adf.pValue.toFixed(4)}`);
console.log(`  Lags: ${adf.lags}`);
console.log(`  Stationary: ${adf.isStationary}`);
console.log();

console.log("=== Auto-ARIMA Model Selection ===");
const best = autoArima(series, { maxP: 3, maxD: 2, maxQ: 3 });
console.log(`  Selected order: (${best.selectedOrder.p}, ${best.selectedOrder.d}, ${best.selectedOrder.q})`);
console.log(`  AR coefficients: [${best.arCoefficients.map(c => c.toFixed(4)).join(", ")}]`);
console.log(`  MA coefficients: [${best.maCoefficients.map(c => c.toFixed(4)).join(", ")}]`);
console.log();

console.log("=== Forecasting with Prediction Intervals ===");
const fc = forecastWithIntervals(best, 5, 0.95);
console.log("  Step | Point    | Lower    | Upper");
console.log("  -----|----------|----------|----------");
for (let i = 0; i < fc.point.length; i++) {
  console.log(`  ${(i + 1).toString().padStart(4)} | ${fc.point[i].toFixed(4).padStart(8)} | ${fc.lower[i].toFixed(4).padStart(8)} | ${fc.upper[i].toFixed(4).padStart(8)}`);
}
console.log();

console.log("=== Seasonal Decomposition ===");
// Generate a seasonal series
const seasonal = Array.from({ length: 48 }, (_, i) => 50 + 2 * i + 10 * Math.sin((2 * Math.PI * i) / 12) + rand() * 2);
const decomp = seasonalDecompose(seasonal, 12);
const validTrend = decomp.trend.filter((v): v is number => v !== null);
console.log(`  Trend range: [${Math.min(...validTrend).toFixed(1)}, ${Math.max(...validTrend).toFixed(1)}]`);
console.log(`  Seasonal range: [${Math.min(...decomp.seasonal).toFixed(1)}, ${Math.max(...decomp.seasonal).toFixed(1)}]`);
const validResid = decomp.residual.filter((v): v is number => v !== null);
console.log(`  Residual std: ${(Math.sqrt(validResid.reduce((s, v) => s + v * v, 0) / validResid.length)).toFixed(2)}`);
