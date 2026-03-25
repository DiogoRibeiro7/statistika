/** JSON Schema for TTestResult (HypothesisTestResult) */
export const tTestResultSchema = {
  $schema: "http://json-schema.org/draft-07/schema#",
  title: "TTestResult",
  description: "Result of a t-test (one-sample, two-sample, paired, or Welch's).",
  type: "object",
  properties: {
    statistic: {
      type: "number",
      description: "The t-statistic.",
    },
    pValue: {
      type: "number",
      description: "Two-tailed p-value.",
      minimum: 0,
      maximum: 1,
    },
    degreesOfFreedom: {
      type: "number",
      description: "Degrees of freedom for the test.",
      minimum: 0,
    },
    rejected: {
      type: "boolean",
      description: "Whether the null hypothesis is rejected at the given significance level.",
    },
  },
  required: ["statistic", "pValue", "degreesOfFreedom", "rejected"],
  additionalProperties: false,
} as const;

/** JSON Schema for ChiSquaredResult (HypothesisTestResult) */
export const chiSquaredResultSchema = {
  $schema: "http://json-schema.org/draft-07/schema#",
  title: "ChiSquaredResult",
  description: "Result of a chi-squared test (goodness-of-fit or independence).",
  type: "object",
  properties: {
    statistic: {
      type: "number",
      description: "The chi-squared test statistic.",
      minimum: 0,
    },
    pValue: {
      type: "number",
      description: "Upper-tail p-value.",
      minimum: 0,
      maximum: 1,
    },
    degreesOfFreedom: {
      type: "number",
      description: "Degrees of freedom for the test.",
      minimum: 0,
    },
    rejected: {
      type: "boolean",
      description: "Whether the null hypothesis is rejected at the given significance level.",
    },
  },
  required: ["statistic", "pValue", "degreesOfFreedom", "rejected"],
  additionalProperties: false,
} as const;

/** JSON Schema for AnovaResult */
export const anovaResultSchema = {
  $schema: "http://json-schema.org/draft-07/schema#",
  title: "AnovaResult",
  description: "Result of a one-way analysis of variance (ANOVA).",
  type: "object",
  properties: {
    fStatistic: {
      type: "number",
      description: "The F-statistic.",
      minimum: 0,
    },
    pValue: {
      type: "number",
      description: "P-value from the F-distribution.",
      minimum: 0,
      maximum: 1,
    },
    dfBetween: {
      type: "number",
      description: "Between-group degrees of freedom (k - 1).",
      minimum: 0,
    },
    dfWithin: {
      type: "number",
      description: "Within-group degrees of freedom (N - k).",
      minimum: 0,
    },
    ssBetween: {
      type: "number",
      description: "Between-group sum of squares.",
      minimum: 0,
    },
    ssWithin: {
      type: "number",
      description: "Within-group sum of squares.",
      minimum: 0,
    },
    msBetween: {
      type: "number",
      description: "Between-group mean square (ssBetween / dfBetween).",
      minimum: 0,
    },
    msWithin: {
      type: "number",
      description: "Within-group mean square (ssWithin / dfWithin).",
      minimum: 0,
    },
    rejected: {
      type: "boolean",
      description: "Whether the null hypothesis is rejected at the given significance level.",
    },
  },
  required: [
    "fStatistic",
    "pValue",
    "dfBetween",
    "dfWithin",
    "ssBetween",
    "ssWithin",
    "msBetween",
    "msWithin",
    "rejected",
  ],
  additionalProperties: false,
} as const;
