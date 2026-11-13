// Pydantic JSON Schemas are the only source of domain types.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const sources = ['config', 'execute_python', 'rpc_initialize_result', 'rpc_prepare_tool_result'];
function type(s) {
  if (s.$ref) return s.$ref.split('/').at(-1);
  if ('const' in s) return JSON.stringify(s.const);
  if (s.enum) return s.enum.map(v=>JSON.stringify(v)).join(' | ');
  if (s.anyOf || s.oneOf) return (s.anyOf ?? s.oneOf).map(type).join(' | ');
  if (s.type === 'array') return `Array<${type(s.items ?? {})}>`;
  if (s.type === 'object' || s.properties) {
    if (!s.properties) return `Record<string, ${s.additionalProperties === true ? 'unknown' : type(s.additionalProperties ?? {})}>`;
    return `{${Object.entries(s.properties).map(([k,v])=>`${JSON.stringify(k)}${s.required?.includes(k)?'':'?'}: ${type(v)}`).join(';')}}`;
  }
  return ({string:'string',integer:'number',number:'number',boolean:'boolean',null:'null'})[s.type] ?? 'unknown';
}
let result = '// Generated from Pydantic schemas. Do not edit.\n';
const definitions = new Map();
for (const name of sources) {
  const data = await readFile(`schemas/${name}.json`,'utf8');
  const schema = JSON.parse(data);
  result += `// ${name} sha256: ${createHash('sha256').update(data).digest('hex')}\n`;
  for (const [key,value] of Object.entries(schema.$defs ?? {})) definitions.set(key,value);
  definitions.set(schema.title ?? name,schema);
}
for (const [key,value] of definitions) result += `export type ${key} = ${type(value)};\n`;
await mkdir('packages/agent/src/generated',{recursive:true});
const path='packages/agent/src/generated/contracts.ts';
if (process.argv.includes('--check')) {
  if (await readFile(path,'utf8') !== result) throw new Error('Generated TypeScript schema drift');
} else await writeFile(path,result);
