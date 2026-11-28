# Print pages for the race demo

Open these files directly from the file system in a desktop browser and print them on A4, portrait, at 100% (actual size, no "fit to page"). Spec: `.kiro/specs/race/`.

| File | What it prints |
|---|---|
| `qr-codes.html` | One page: the three QR codes (decimal Product_ID, 42 mm code, 4-module quiet zone) with name, ID and dashed cut lines. Stick one code on one side of each box. |
| `labels.html` | The Tilemark labels at 114 x 149 mm, two per sheet, turned 90 degrees; copies per product on the page (default 6, one per side of a box). Each sheet has a 100 mm ruler line: measure it after printing. The page shows a size check on screen. |

`qr-codes.html` and `products.js` are generated from `web/src/config.ts` (product IDs and names). After changing a product name there, run `npm run print:gen` in `/web`. `npm test` fails while they are out of date.

The label images go in `labels/` (see `labels/README.md`).
