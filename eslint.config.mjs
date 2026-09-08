import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default [
  {
    ignores: ["dist/", "node_modules/", "coverage/", "docs/", "benchmark/", "examples/", "scripts/"],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["src/**/*.ts", "tests/**/*.ts"],
    rules: {
      "no-unused-vars": "off",
      "@typescript-eslint/no-unused-vars": [
        "warn",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
      // These rules were introduced by the current ESLint stack after the
      // repository's historical CI had stopped reaching the lint step. Keep
      // them visible while baseline cleanup proceeds, but don't let unrelated
      // legacy findings block statistical correctness and compatibility fixes.
      "prefer-const": "warn",
      "no-useless-assignment": "warn",
      "no-loss-of-precision": "warn",
      "@typescript-eslint/no-empty-object-type": "warn",
      "@typescript-eslint/no-require-imports": "warn",
    },
  },
];
