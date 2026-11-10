import { test } from 'node:test';
import assert from 'node:assert/strict';
import { providerBaseUrl } from '../src/provider-url.ts';
test('provider prefixes append v1 exactly once',()=>{
  assert.equal(providerBaseUrl('http://localhost:9999/v1','anthropic-messages'),'http://localhost:9999');
  assert.equal(providerBaseUrl('http://localhost:9999/v1/','openai-responses'),'http://localhost:9999/v1');
  assert.equal(providerBaseUrl('https://example.org/api/v1','openai-chat'),'https://example.org/api/v1');
});
test('resource loader has no ambient resources',async context=>{
  const {strippedLoader}=await import('../dist/controller.js');
  const loader=strippedLoader('Only execute_python');
  assert.deepEqual(loader.getSkills().skills,[]);
  assert.deepEqual(loader.getExtensions().extensions,[]);
  assert.deepEqual(loader.getAgentsFiles().agentsFiles,[]);
  assert.deepEqual(loader.getPrompts().prompts,[]);
  assert.equal(loader.getSystemPrompt(),'Only execute_python');
});
