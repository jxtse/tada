export function providerBaseUrl(prefix:string, protocol:string):string {
  // Anthropic's SDK appends /v1/messages; OpenAI SDKs append /responses or /chat/completions.
  const clean = prefix.replace(/\/$/,'');
  return protocol === 'anthropic-messages' ? clean.replace(/\/v1$/,'') : clean;
}
