import { test } from 'node:test';
import assert from 'node:assert/strict';
import { toolResult, SYSTEM_PROMPT_RULES } from '../dist/controller.js';

test('displayed images become native image content and leave the text JSON small', () => {
  const observation = {status:'ok', stdout:'plot', images:[{media_type:'image/png', data_base64:'AAAA'}]};
  const result = toolResult(observation);
  assert.equal(result.isError, false);
  assert.equal(result.content.length, 2);
  assert.deepEqual(result.content[1], {type:'image', data:'AAAA', mimeType:'image/png'});
  const text = JSON.parse(result.content[0].text);
  assert.equal(text.images_attached, 1);
  assert.equal('images' in text, false);
  assert.equal(result.details, observation);
});

test('observations without images keep the original single text block', () => {
  const observation = {status:'timeout', stdout:'', operators:[{atom_call_id:'a', done:3, total:9}]};
  const result = toolResult(observation);
  assert.equal(result.isError, true);
  assert.deepEqual(result.content, [{type:'text', text:JSON.stringify(observation)}]);
});

test('system prompt states the working discipline', () => {
  for (const phrase of ['dry_run=True', 'Write the deliverable as soon as', 'tag_async', '"locate"', 'row_context=',
                        'Review selectively', 'verbatim', 'canonical value', 'check_deliverable']) {
    assert.ok(SYSTEM_PROMPT_RULES.includes(phrase), phrase);
  }
});
