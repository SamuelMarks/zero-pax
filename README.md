# zero-pax

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/zero-pax/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-pax/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-81.5%25-yellowgreen.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)

**Version:** 0.1.0

## Why this project exists

`zero-pax` is a foundational component of the **Abstract ML Machine Ecosystem**. The ecosystem solves the N-to-M translation problem in Machine Learning: instead of writing bespoke translators for every frontend framework (JAX, PyTorch, Keras) to every hardware target (WASM, WebGPU, TensorRT), computations are traced from N frontends into a strictly defined Intermediate Representation (IR, via `ml-switcheroo-ir`), which is then consumed by M backends.

`zero-pax` provides a **zero-dependency, pure Python implementation of the Pax/Praxis API surface**. It accurately mirrors the mathematical semantics of the 108 Praxis layers—ranging from standard Activations and Convolutions to complex Attention mechanisms and Transformer blocks. 

Standard ML frameworks often bring heavy dependency chains, making them brittle and challenging to deploy in constrained or browser-based environments. `zero-pax` bypasses this by:
1. Relying solely on the Python Standard Library and `numpy` for eager evaluations.
2. Replacing complex configurations with strict `numpy` array validation and pure mathematical forward passes.
3. Hooking seamlessly into the `ml-switcheroo-compiler` to trace operations via proxy tensors, enabling Reverse-mode automatic differentiation (AD), static shape inference, and deterministic Graph generation.

This allows complex Pax/Praxis models to be statically compiled, optimized, and deployed as source-to-browser payloads (like WASM or WGSL shaders) without any third-party overhead.

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
