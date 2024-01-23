## `faiss-next` 

A light weighted [facebookresearch/faiss](https://github.com/facebookresearch/faiss) `c_api` rust wrapper.

### Platform supported: 

| Arch    | x86_64 | Apple Silicon | ARM | CUDA |
|---------|--------|---------------|-----|------|
| Linux   |   ✓    |      N/A      |  ⨯  |  ✓   |
| Macos   |   ?    |       ✓       |   ✓ |  N/A |
| Windows |   TODO |       N/A     |  N/A|  TODO |

### Benchmark result

base: 1048576 * 128 features
query: 1 * 128 feature

- Macos + Apple Silicon: duration=11.415016ms, times=10
