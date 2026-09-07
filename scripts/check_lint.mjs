// Preserve existing debt by exact diagnostic/source fingerprints, never by total count.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ui = path.join(root, 'theseus-ui');
const result = spawnSync(process.execPath, ['node_modules/eslint/bin/eslint.js', '.', '-f', 'json'], { cwd: ui, encoding: 'utf8', maxBuffer: 20 * 1024 * 1024 });
if (result.error || result.status > 1 || !result.stdout) throw new Error(result.stderr || String(result.error));
const baseline = JSON.parse(fs.readFileSync(path.join(ui, 'eslint-baseline.json'), 'utf8'));
const current = {};
for (const file of JSON.parse(result.stdout)) {
  const lines = fs.readFileSync(file.filePath, 'utf8').split('\n');
  for (const diagnostic of file.messages) {
    const key = JSON.stringify([path.relative(ui, file.filePath), diagnostic.ruleId, diagnostic.message, (lines[diagnostic.line - 1] || '').trim()]);
    current[key] = (current[key] || 0) + 1;
  }
}
const added = Object.entries(current).filter(([key, count]) => count > (baseline[key] || 0));
if (added.length) {
  for (const [key, count] of added) console.error(`New lint diagnostic (${count}): ${key}`);
  process.exitCode = 1;
} else {
  console.log(`No new lint diagnostics (${Object.values(current).reduce((a, b) => a + b, 0)} existing).`);
}
