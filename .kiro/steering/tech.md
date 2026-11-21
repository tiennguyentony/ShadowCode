   # Tech Stack and Constraints

## Web

- Language: TypeScript
- Bundler: Vite
- Runs entirely in the browser — no server-side processing
- Camera input via `getUserMedia`
- Image processing: OpenCV.js is allowed
- Neural decoder: ONNX Runtime Web

## Training

- Language: Python, framework: PyTorch
- Lives in `/training`
- The decoder must be exported to ONNX for use in the browser

## Tests

- Runner: Vitest
- Property-based tests: fast-check
- Every codec change must keep all property tests passing

## Licensing

- No paid or restrictively licensed dependencies
- All dependencies must be MIT-compatible

## Performance targets

- One decode attempt: under 100 ms on a laptop
- Successful read (possibly multiple attempts): under 500 ms
