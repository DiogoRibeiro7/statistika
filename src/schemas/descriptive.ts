/** JSON Schema for DescribeResult (DescriptiveStats) */
export const describeResultSchema = {
  $schema: "http://json-schema.org/draft-07/schema#",
  title: "DescribeResult",
  description: "Summary of descriptive statistics for a dataset.",
  type: "object",
  properties: {
    count: {
      type: "number",
      description: "Number of observations.",
      minimum: 0,
    },
    mean: {
      type: "number",
      description: "Arithmetic mean of the dataset.",
    },
    median: {
      type: "number",
      description: "Median value of the dataset.",
    },
    variance: {
      type: "number",
      description: "Sample variance of the dataset.",
      minimum: 0,
    },
    stdDev: {
      type: "number",
      description: "Sample standard deviation of the dataset.",
      minimum: 0,
    },
    min: {
      type: "number",
      description: "Minimum value in the dataset.",
    },
    max: {
      type: "number",
      description: "Maximum value in the dataset.",
    },
  },
  required: ["count", "mean", "median", "variance", "stdDev", "min", "max"],
  additionalProperties: false,
} as const;
