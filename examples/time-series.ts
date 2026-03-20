/**
 * Example: Time series analysis.
 *
 * Run: npx ts-node examples/time-series.ts
 */

import {
  autocorrelation,
  simpleMovingAverage,
  exponentialMovingAverage,
  difference,
  arima,
} from "../src";

// Monthly sales data (24 months with trend + seasonality)
const sales = [
  120, 135, 150, 145, 160, 175, 170, 185, 200, 195, 210, 225, 220, 235, 250,
  245, 260, 275, 270, 285, 300, 295, 310, 325,
];

// --- Autocorrelation ---
const acfResult = autocorrelation(sales, 12);
console.log("=== Autocorrelation Analysis ===");
console.log(`  95% confidence bound: +/- ${acfResult.confidenceBound.toFixed(4)}`);
console.log("\n  Lag   ACF       PACF");
for (let k = 0; k <= 6; k++) {
  const acfStr = acfResult.acf[k].toFixed(4).padStart(8);
  const pacfStr = acfResult.pacf[k].toFixed(4).padStart(8);
  const sig =
    k > 0 && Math.abs(acfResult.acf[k]) > acfResult.confidenceBound
      ? " *"
      : "";
  console.log(`  ${String(k).padStart(3)}  ${acfStr}  ${pacfStr}${sig}`);
}
console.log("  (* = significant at 95%)");
console.log();

// --- Moving Averages ---
const sma3 = simpleMovingAverage(sales, 3);
const ema = exponentialMovingAverage(sales, 0.3);

console.log("=== Moving Averages (last 6 values) ===");
console.log("  Month   Raw   SMA(3)   EMA(0.3)");
for (let i = sales.length - 6; i < sales.length; i++) {
  const smaIdx = i - 2; // SMA has 2 fewer values at start
  const smaVal = smaIdx >= 0 ? sma3.values[smaIdx].toFixed(1) : "   -";
  console.log(
    `  ${String(i + 1).padStart(5)}   ${String(sales[i]).padStart(3)}   ${String(smaVal).padStart(6)}   ${ema.values[i].toFixed(1).padStart(8)}`,
  );
}
console.log();

// --- Differencing ---
const diff1 = difference(sales, 1);
console.log("=== First Differences (first 10) ===");
console.log(`  ${diff1.slice(0, 10).join(", ")}`);
console.log();

// --- ARIMA Forecasting ---
const model = arima(sales, 2, 1, 0); // ARIMA(2,1,0)
const forecast = model.forecast(6);

console.log("=== ARIMA(2,1,0) Model ===");
console.log(`  AR coefficients: [${model.arCoefficients.map((c) => c.toFixed(4))}]`);
console.log(`  Intercept: ${model.intercept.toFixed(4)}`);
console.log(`  Residual variance: ${model.sigma2.toFixed(4)}`);
console.log(`  AIC: ${model.aic.toFixed(2)}`);
console.log(`\n  6-month forecast:`);
for (let i = 0; i < forecast.length; i++) {
  console.log(`    Month ${sales.length + i + 1}: ${forecast[i].toFixed(1)}`);
}
