#!/usr/bin/env node

/**
 * Post-build step for ESM output.
 *
 * Node.js ESM requires explicit file extensions in import specifiers.
 * TypeScript emits bare specifiers like `from "./utils"` which resolve
 * via index.js in CJS but fail in ESM.
 *
 * This script rewrites all relative import/export paths in dist/esm/ to
 * include explicit .js extensions, resolving directory imports to /index.js.
 */

const fs = require("fs");
const path = require("path");

const ESM_DIR = path.resolve(__dirname, "..", "dist", "esm");

function walk(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...walk(full));
    } else if (entry.name.endsWith(".js")) {
      files.push(full);
    }
  }
  return files;
}

// Match: from "./foo" or from "../foo" (with single or double quotes)
// Also match: export * from "./foo" and dynamic import("./foo")
const IMPORT_RE = /(from\s+["'])(\.\.?\/[^"']+)(["'])/g;
const EXPORT_RE = /(export\s+\*\s+from\s+["'])(\.\.?\/[^"']+)(["'])/g;

function fixImports(filePath) {
  let content = fs.readFileSync(filePath, "utf8");
  let changed = false;

  function replacer(match, prefix, specifier, suffix) {
    // Skip if already has .js extension
    if (specifier.endsWith(".js")) return match;

    // Check if specifier points to a directory (has an index.js)
    const resolved = path.resolve(path.dirname(filePath), specifier);
    if (fs.existsSync(resolved) && fs.statSync(resolved).isDirectory()) {
      changed = true;
      return `${prefix}${specifier}/index.js${suffix}`;
    }

    // Check if specifier.js exists
    if (fs.existsSync(resolved + ".js")) {
      changed = true;
      return `${prefix}${specifier}.js${suffix}`;
    }

    return match;
  }

  content = content.replace(IMPORT_RE, replacer);
  content = content.replace(EXPORT_RE, replacer);

  if (changed) {
    fs.writeFileSync(filePath, content);
  }
  return changed;
}

function main() {
  if (!fs.existsSync(ESM_DIR)) {
    console.warn("⚠  dist/esm/ not found — skipping ESM import fix.");
    return;
  }

  const files = walk(ESM_DIR);
  let fixedCount = 0;
  for (const file of files) {
    if (fixImports(file)) fixedCount++;
  }
  console.log(`✓  Fixed ESM imports in ${fixedCount}/${files.length} files.`);
}

main();
