---
inclusion: always
---

# Boundaries

## Ownership

- Kiro (IDE, CLI and Crew) owns `/web`, `/.kiro`, `/print`, the repo root config, and `/docs/message-format.md`.

## Training folder

- Kiro never creates, edits, or deletes anything under `/training`.
- `/training` is written by a separate tool (Claude Code). It holds the PyTorch training code.
- The only thing that enters the app from `/training` is a model bundle: the folder `/training/export/<bundle_id>/`.
- A spec task copies a bundle unchanged into `/web/public/models/<bundle_id>/`.
- Never edit the files of a bundle.

## Message size

- The Message is exactly 64 bits in total.
- That number is frozen, because neural models are trained against it.

## Secrets, artifacts, and git

- Never commit secrets, `.env` files, model checkpoints, or datasets.
- Never add `.kiro` to `.gitignore`.
- Commit messages for app work start with `web:`.
- Never commit or push directly to master. Work on a branch; only Tony merges.
