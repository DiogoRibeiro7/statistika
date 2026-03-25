/** JSON Schema for MCMCResult */
export const mcmcResultSchema = {
  $schema: "http://json-schema.org/draft-07/schema#",
  title: "MCMCResult",
  description: "Result of a Metropolis-Hastings MCMC sampling run.",
  type: "object",
  properties: {
    chain: {
      type: "array",
      items: { type: "number" },
      description: "Sampled chain (after burn-in).",
    },
    acceptanceRate: {
      type: "number",
      description: "Proportion of proposed moves that were accepted.",
      minimum: 0,
      maximum: 1,
    },
    totalIterations: {
      type: "number",
      description: "Total number of iterations (including burn-in).",
      minimum: 0,
    },
    burnIn: {
      type: "number",
      description: "Number of burn-in iterations discarded.",
      minimum: 0,
    },
    posteriorMean: {
      type: "number",
      description: "Posterior mean computed from the chain.",
    },
    posteriorStd: {
      type: "number",
      description: "Posterior standard deviation computed from the chain.",
      minimum: 0,
    },
    credibleInterval: {
      type: "object",
      description: "Credible interval for the posterior distribution.",
      properties: {
        lower: {
          type: "number",
          description: "Lower bound of the credible interval.",
        },
        upper: {
          type: "number",
          description: "Upper bound of the credible interval.",
        },
        level: {
          type: "number",
          description: "Credible level (e.g. 0.95 for a 95% interval).",
          minimum: 0,
          maximum: 1,
        },
      },
      required: ["lower", "upper", "level"],
      additionalProperties: false,
    },
    effectiveSampleSize: {
      type: "number",
      description: "Estimated effective sample size accounting for autocorrelation.",
      minimum: 0,
    },
    warning: {
      type: "string",
      description: "Warning message if the acceptance rate is problematic.",
    },
  },
  required: [
    "chain",
    "acceptanceRate",
    "totalIterations",
    "burnIn",
    "posteriorMean",
    "posteriorStd",
    "credibleInterval",
    "effectiveSampleSize",
  ],
  additionalProperties: false,
} as const;
