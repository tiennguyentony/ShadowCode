---
inclusion: always
---

# Kiro Log Rule

`docs/kiro-log.md` is the project log of Kiro work. Keep it updated.

## When to write an entry

Append one entry for each of these that changed code or project config:

- a spec task
- a hook run
- a chat session

Write the entry at the end of the session, before the final reply. Do this even if the session was short. Do not write entries for sessions that only read or explain files.

## Entry format

Add a new `## YYYY-MM-DD` heading at the bottom of the file. Use today's date from the system clock. If today's heading already exists, add a new `**Feature:**` block under it.

Each entry has these parts:

- **Feature:** which Kiro feature was used: spec, steering, hook, chat, Kiro Crew, or MCP.
- **Output:** what was produced or changed, with file paths in backticks.
- **Caught, corrected, or changed my mind about:** issues found, fixes made, and decisions reversed during the work.
- **Went wrong or had to redo:** failures, rework, and failing checks. Write "Nothing recorded." if there were none.

## Style

- Facts only. Name files, counts, and results. Give no praise and no marketing language.
- Do not edit or delete earlier entries. If an earlier entry is wrong, add a correction in a new entry.
- Do not log secrets, tokens, or the contents of `.env` files.
