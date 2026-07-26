# sglang-mm

Rust-accelerated multimodal preprocessing for SGLang. Fused image decode,
fetch, resize, patchify, normalize, and content hash — all parallel and
GIL-released.

Built two ways:

- **PyO3 extension** `sglang.srt.multimodal._core` (feature `python`, default)
  via setuptools-rust when installing sglang — used by Python processors and
  parity tests.
- **Pure-Rust `rlib`** (`default-features = false`) linked by `sglang-server`'s
  native MM worker path — no pyo3 in that dependency graph.

## Architecture

```
src/
├── lib.rs                    # module root; PyO3 module (_core) feature-gated
├── family.rs                 # the model-family seam: MmFamilyProcessor trait
│                             #   + the carriers (NamedTensors, TokenLayout, ...)
├── driver.rs                 # model-independent request driver (fetch →
│                             #   decode → process_item → layout → positions)
├── registry.rs               # ImageProcessorSpec registry (Python-facing)
│                             #   + pipeline_from_spec (family factory)
├── common/
│   ├── mod.rs                # thread pool, image decode, content hash, base64
│   ├── fetch.rs              # media source → bytes (data:/base64/file/http)
│   ├── resize.rs             # PIL-exact Lanczos + Bicubic resize
│   ├── tokens.rs             # TokenLayout mechanics (apply_layout + helpers)
│   └── transforms.rs         # reusable primitives: normalize, pad, extract_patches
└── <model>/
    └── mod.rs                # model-specific processor (inkling, qwen_vl, ...)
```

## Native server pipeline architecture

The pipeline that `sglang-server`'s MM workers drive is built to eventually
carry **every** model family the Python multimodal processors serve today.
Its one design rule: **families produce data, the driver owns control flow.**

`driver::process` is the fixed request skeleton — parallel fetch/decode
fan-out, layout application, position computation, failure semantics — and
contains zero model knowledge. A model family implements the
`MmFamilyProcessor` trait (`family.rs`) and only describes *what*, never
*how*:

- **`process_item`** — decoded media in, `ProcessedItem` out, mirroring
  Python's `MultimodalDataItem`: the primary feature tensor (hashed by the
  driver for item identity, standing in for `hash_feature`), named auxiliary
  tensors for the model runner (`image_grid_thw`, and for other families
  `image_sizes`, `tgt_sizes`, ... — the `model_specific_data` analogue), and
  a `Geometry` value for the family's own later hooks.
- **`layout`** — prompt geometry *as data*: a [`TokenLayout`] of
  `Text`/`Media` segments, where a media span is either `Repeat` (qwen's
  `<|image_pad|>` × N) or `Explicit` ids (tile markers, row separators,
  wrapper tokens — the minicpm/internvl-style structured schemes). The
  driver applies it mechanically (`common::tokens::apply_layout`), so final
  input ids and per-item offsets derive from one declarative structure and
  the family cannot get expansion, offsets, and positions out of sync.
  `layout` sees the whole prompt and all items, so whole-request schemes are
  expressible without giving families the control flow.
- **`positions`** — a position *scheme*, not a computation slot: `Rope1D`
  (default, scheduler needs nothing) or `MRope` (qwen).
- **`capabilities`** — modalities the family accepts; the server's message
  layer rejects everything else per family.

Why this instead of Python's "each family overrides
`process_mm_data_async`": in a server core there is no exception handler
upstream — every request must resolve to exactly one accept/reject with its
buffers parked in order, and that invariant is structural only if the driver
owns the flow. The wide seam also decays (Python's base class has been
steadily pulling duplicated expansion/offset logic back out of subclasses);
the narrow seam starts where that converged.

Growth path (deliberately not built speculatively): each carrier is an enum
that grows a variant when a real family needs it — `DecodedMedia` per
modality (video/audio), `Geometry` per family style (tile sets),
`TensorData` per dtype. What stays in Python permanently: HF config
resolution (families are configured by a spec JSON of resolved params,
selected via `registry::pipeline_from_spec`) and the thin drain adapter
mapping feature + aux tensors to model kwargs.

Anything outside a family's declared scope (video/audio, precomputed
features, unknown source shapes, placeholder mismatches) is rejected back to
the client as a 400 — there is no Python fallback path.

Supported families: `qwen_vl` (Qwen2-VL / 2.5-VL / 3-VL / 3.5; images only).
Adding one = a `MmFamilyProcessor` impl in `src/<model>/mod.rs` plus a
`family` arm in `pipeline_from_spec`.

`common::fetch` matches the Python `get_image_bytes` semantics
(`REQUEST_TIMEOUT` env, proxy env vars) with two deliberate differences:
HTTP downloads are capped at 64 MiB, and `NO_PROXY` is not honored (ureq
limitation).

## Python API

```python
from sglang.srt.multimodal._core import common, inkling

# Common (model-agnostic)
common.resize_rgb(arr, out_w, out_h)
common.scaled_dims(w, h, rescale_frac, rescale_cap)
common.image_decode_rgb(bytes)          # -> (h, w, ndarray)
common.data_hash(bytes)                 # -> u64 SHA256
common.base64_decode(str)               # -> bytes

# Model-specific
inkling.preprocess_images(list[bytes], ps, frac, cap)  # -> [(h, w, bits, hash), ...]
inkling.decode_patchify(bytes, ps, frac, cap)
inkling.decode_patchify_batch(list[bytes], ps, frac, cap)
inkling.patchify_rgb(arr, patch_size)
```

## Adding a new model

1. Create `src/<model_name>/mod.rs`:

```rust
use crate::common;
use crate::registry::ImageProcessorSpec;
use rayon::prelude::*;

pub struct MyModelProcessor;

impl ImageProcessorSpec for MyModelProcessor {
    fn name(&self) -> &'static str {
        "my_model"
    }

    fn preprocess_batch(
        &self,
        datas: &[Vec<u8>],
        patch_size: usize,
        rescale_frac: Option<f64>,
        rescale_cap: Option<i64>,
    ) -> Result<Vec<(usize, usize, Vec<u16>, u64)>, String> {
        common::pool().install(|| {
            datas.par_iter().map(|data| {
                let hash = common::sha256_u64(data);
                let (rgb, h, w) = common::decode_rescale(data, rescale_frac, rescale_cap)?;
                // Use common::transforms::* or model-specific logic
                let patches = my_patchify(&rgb, h, w, patch_size);
                Ok((h, w, patches, hash))
            }).collect()
        })
    }
}
```

2. Register in `src/registry.rs` `default_registry()`.

3. Add PyO3 bindings in `src/<model_name>/mod.rs` with a `register()` function.

4. Wire up in `src/lib.rs`: `mod my_model;` and `my_model::register(m)?;`.

5. Add Python processor class that calls `from sglang.srt.multimodal._core import my_model`.

## Available transform primitives (`common::transforms`)

| Function | Description |
|----------|-------------|
| `normalize_rgb_f32` | Single-pass `(pixel/255 - mean) / std` |
| `pad_to_grid` | Pad HWC image to grid-aligned dimensions |
| `extract_patches_hwc` | Reshape padded image into `[N, ph, pw, C]` patches |
| `patch_grid` | Compute `(nph, npw)` for given image and patch size |

## Design notes

- Thread pool capped at `min(8, cores)`. Override: `SGL_MM_RS_THREADS`.
- PNG decode is bit-exact vs PIL; JPEG may differ by ±1 LSB.
- Lanczos resize is a bit-exact clone of PIL's fixed-point implementation.

## Build

Automatically built when installing sglang:
```bash
pip install -e "python"
```

Or standalone for development:
```bash
cd rust/sglang-mm
pip install maturin
maturin develop --release
```

## Test

```bash
cd rust/sglang-mm
cargo test --no-default-features  # pure-Rust unit tests (CI: pr-test-rust-exts)
python tests/generate_golden.py   # regenerate fixtures
pytest tests/test_golden.py       # regression tests
python bench/bench_parity.py      # parity + benchmark
```

Scheduler-boundary parity tests against the real HF processors live in
`test/registered/unit/multimodal/rust/`.
