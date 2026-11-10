# Third-party notices

TADA's new harness uses the published Pi SDK and standard Python packages. Their license files remain in the installed distributions. The lockfiles identify the versions used.

## Pi

- Source: https://github.com/earendil-works/pi
- Reference revision: `f9a26365568127461c47a07a37dcb42c140a3ac5` (version 1.5.0).
- License: MIT.
- Use: the installed `@earendil-works/pi-coding-agent` SDK supplies the agent session, model adaptation, and compaction. The integration was informed by `packages/coding-agent/src/core/sdk.ts`, `agent-session.ts`, `examples/sdk/12-full-control.ts`, and `packages/agent/src/agent.ts`.

## DeepSeek Harness

- Source: https://github.com/deepseek-ai/deepseek-harness
- Reference revision: `fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159`.
- License: MIT.
- Design reference: typed append-only events, distinct failed attempts and visible messages, single-writer durability, explicit flush barriers, and incomplete physical-tail recovery. The implementation review included the session and LLM types, `session-persistence/src/storage-contract.ts`, and `session-persistence-jsonl/src/storage.ts`, in addition to the subsystem documentation.
- TADA's Python collector and contracts are an independent implementation; no DeepSeek runtime is installed or vendored.

## Prime Agent

- Source: https://github.com/PrimeIntellect-ai/prime-agent
- Reference revision: `42a07c13ce492a45bdb3eeb8e536a717b53fc753`.
- License: MIT.
- Design reference: persistent namespace ownership and explicit lifecycle/restoration reporting, inspected in `prime-agent-runtime/src/rlm/repl.py` and `crates/pa-core/src/kernel/state_snapshot.rs`.
- No Prime code is vendored. In particular, TADA does not adopt Prime's pickle/dill snapshot representation or fork its Rust daemon.

## TADA skill assets

The existing TADA skill prompts and batching approach were reviewed at revision `8df9cc2d4ed3e134f353b569cd322ec2ce511498`. The new harness preserves the two semantic operations while imposing stricter row identities, output types, validation, and explicit failure handling. Existing experiment and paper artifacts retain their original licensing.
