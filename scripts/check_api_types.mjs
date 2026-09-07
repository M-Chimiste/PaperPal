import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ui = path.join(root, 'theseus-ui');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'theseus-types-'));
try {
  const output = path.join(temporary, 'schema.d.ts');
  const result = spawnSync(process.execPath, ['node_modules/openapi-typescript/bin/cli.js', 'openapi.json', '-o', output], { cwd: ui, encoding: 'utf8' });
  if (result.status !== 0) throw new Error(result.stderr || 'API type generation failed');
  if (fs.readFileSync(output, 'utf8') !== fs.readFileSync(path.join(ui, 'src/services/generated/schema.d.ts'), 'utf8')) {
    throw new Error('Generated API types have drifted. Run make generate-api and review the diff.');
  }
  console.log('Generated API types match the schema.');
} finally {
  fs.rmSync(temporary, { recursive: true, force: true });
}
