/** JSON Schema for LinearRegressionResult */
export const linearRegressionResultSchema = {
  $schema: "http://json-schema.org/draft-07/schema#",
  title: "LinearRegressionResult",
  description: "Result of a simple linear regression fit (y = slope * x + intercept).",
  type: "object",
  properties: {
    slope: {
      type: "number",
      description: "The slope of the fitted line.",
    },
    intercept: {
      type: "number",
      description: "The y-intercept of the fitted line.",
    },
    rSquared: {
      type: "number",
      description: "Coefficient of determination (R^2), measuring the proportion of variance explained.",
      minimum: 0,
      maximum: 1,
    },
  },
  required: ["slope", "intercept", "rSquared"],
  additionalProperties: false,
} as const;

/** JSON Schema for MultipleRegressionResult */
export const multipleRegressionResultSchema = {
  $schema: "http://json-schema.org/draft-07/schema#",
  title: "MultipleRegressionResult",
  description: "Result of a multiple linear regression fit.",
  type: "object",
  properties: {
    coefficients: {
      type: "array",
      items: { type: "number" },
      description: "Fitted coefficients for each predictor variable.",
    },
    intercept: {
      type: "number",
      description: "The intercept (constant) term of the fitted model.",
    },
    rSquared: {
      type: "number",
      description: "Coefficient of determination (R^2).",
      minimum: 0,
      maximum: 1,
    },
  },
  required: ["coefficients", "intercept", "rSquared"],
  additionalProperties: false,
} as const;
