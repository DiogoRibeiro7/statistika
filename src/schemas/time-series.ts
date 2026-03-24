/** JSON Schema for ARIMAResult */
export const arimaResultSchema = {
  $schema: "http://json-schema.org/draft-07/schema#",
  title: "ARIMAResult",
  description: "Result of an ARIMA(p, d, q) model fit.",
  type: "object",
  properties: {
    arCoefficients: {
      type: "array",
      items: { type: "number" },
      description: "Autoregressive (AR) coefficients of length p.",
    },
    maCoefficients: {
      type: "array",
      items: { type: "number" },
      description: "Moving average (MA) coefficients of length q.",
    },
    intercept: {
      type: "number",
      description: "Intercept / constant term of the model.",
    },
    order: {
      type: "object",
      description: "The (p, d, q) order of the ARIMA model.",
      properties: {
        p: {
          type: "number",
          description: "Autoregressive order.",
          minimum: 0,
        },
        d: {
          type: "number",
          description: "Differencing order.",
          minimum: 0,
        },
        q: {
          type: "number",
          description: "Moving average order.",
          minimum: 0,
        },
      },
      required: ["p", "d", "q"],
      additionalProperties: false,
    },
    sigma2: {
      type: "number",
      description: "Residual variance.",
      minimum: 0,
    },
    aic: {
      type: "number",
      description: "Akaike Information Criterion.",
    },
    residuals: {
      type: "array",
      items: { type: "number" },
      description: "Fitted residuals from the model.",
    },
  },
  required: [
    "arCoefficients",
    "maCoefficients",
    "intercept",
    "order",
    "sigma2",
    "aic",
    "residuals",
  ],
  additionalProperties: false,
} as const;

/** JSON Schema for ForecastResult */
export const forecastResultSchema = {
  $schema: "http://json-schema.org/draft-07/schema#",
  title: "ForecastResult",
  description: "Result of a time-series forecast, containing predicted values.",
  type: "object",
  properties: {
    values: {
      type: "array",
      items: { type: "number" },
      description: "Forecasted future values.",
    },
    steps: {
      type: "number",
      description: "Number of steps ahead that were forecast.",
      minimum: 1,
    },
  },
  required: ["values", "steps"],
  additionalProperties: false,
} as const;
