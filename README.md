# faiss-next

`faiss-next` is light weighted [facebookresearch/faiss](https://github.com/facebookresearch/faiss) c_api wrapper in rust. it's inspired by [`faiss-rs`](https://github.com/Enet4/faiss-rs).

[`faiss-rs`](https://github.com/Enet4/faiss-rs) is great already, this crate intend to create a convinient way to integrate [facebookresearch/faiss](https://github.com/facebookresearch/faiss) for some other `rust` projects of my own, to avoid to create some ugly self-referencing struct with `Index` and `StandardGpuResources`