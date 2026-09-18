#!/usr/bin/env node

const { execFileSync } = require("child_process");

const output = execFileSync(
  "npm",
  ["pack", "--dry-run", "--json", "--ignore-scripts"],
  { encoding: "utf8" },
);

const packs = JSON.parse(output);
if (!Array.isArray(packs) || packs.length !== 1 || !Array.isArray(packs[0].files)) {
  throw new Error("Unexpected npm pack --json output");
}

const files = new Set(
  packs[0].files.map(({ path }) => path.replaceAll("\\", "/")),
);

const required = [
  "package.json",
  "README.md",
  "LICENSE",
  "binding.gyp",
  "scripts/build-native.js",
  "dist/cjs/index.js",
  "dist/cjs/index.d.ts",
  "dist/esm/index.js",
  "native/cpp/addon.cpp",
  "native/fortran/special_functions.f90",
];

const forbiddenPrefixes = [
  ".github/",
  "benchmark/",
  "coverage/",
  "src/",
  "tests/",
];

const missing = required.filter((path) => !files.has(path));
const forbidden = [...files].filter((path) =>
  forbiddenPrefixes.some((prefix) => path.startsWith(prefix)),
);

if (missing.length > 0) {
  throw new Error(
    `Package tarball is missing required files:\n${missing.map((path) => `  - ${path}`).join("\n")}`,
  );
}

if (forbidden.length > 0) {
  throw new Error(
    `Package tarball contains development-only files:\n${forbidden.map((path) => `  - ${path}`).join("\n")}`,
  );
}

console.log(
  `Package contents verified: ${files.size} files, ${packs[0].size} bytes packed.`,
);
