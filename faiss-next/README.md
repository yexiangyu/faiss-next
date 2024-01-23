## `faiss-next` 

A light weighted [facebookresearch/faiss](https://github.com/facebookresearch/faiss) `c_api` rust wrapper.

`faiss-next` is created for purpose of convinience of integrating [facebookresearch/faiss](https://github.com/facebookresearch/faiss) more easily in some other rust projects. 

[Enet4/faiss-rs](https://github.com/Enet4/faiss-rs) inspired this crate.

Use this crates on YOUR OWN RISK please.

### Platform supported: 

| Arch    | x86_64 | Apple Silicon | ARM | CUDA |
|---------|--------|---------------|-----|------|
| Linux   |   ✓    |      N/A      |  ⨯  |  ✓   |
| Macos   |   ?    |       ✓       |   ✓ |  N/A |
| Windows |   TODO |       N/A     |  N/A|  TODO |

### Benchmark results

base: 10485760 * 128 features
query: 1 * 128 feature

- Macos
  - Apple Silicon M2/16GB/Macbook Air: duration=`474.09647ms`, times=10
- Linux
  - CPU: Intel(R) Xeon(R) Platinum 8176 CPU @ 2.10GHz/692GB: duration=`413.781354ms`, times=10
  - GPU: Intel(R) Xeon(R) Platinum 8176 CPU @ 2.10GHz/692GB/3090: duration=`23.235486ms`, times=10