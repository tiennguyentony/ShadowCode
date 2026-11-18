# Project Structure

## Directories

| Path | Contents |
|---|---|
| `/web/src/codec` | Encode and decode logic |
| `/web/src/codec/__tests__` | Codec tests |
| `/web/src/ui` | Screens and UI components |
| `/training` | PyTorch training code |
| `/print` | Printable tile sheets |
| `/docs` | Notes and documentation |

## Codec isolation rule

Codec code (`/web/src/codec`) must have zero imports from UI modules or camera APIs. It operates on plain image data only. This keeps it fully testable without a browser environment.
