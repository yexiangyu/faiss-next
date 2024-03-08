## Index Types

|Name|Created|Refer Index | Refer Others| Need train |
|-|-|-|-|-|
|`FaissIndexImpl`|`faiss_index_factory`|depends on `description`|depends on `description`|depends on `depends on description`|
|`FaissIndexFlatImpl`|`FaissIndexFlatImpl::new`|N|N|N|
|`FaissIndexFlatIP`|`FaissIndexFlatIP::new`|N|N|N|
|`FaissIndexFlatL2`|`FaissIndexFlatL2::new`|N|N|N|
|`FaissIndexFlat1D`|`FaissIndexFlat1D::new`|N|N|N|
|`FaissIndexRefineFlatImpl`|`FaissIndexRefineFlatImpl::new`|Y|N|N|
|`FaissIndexIVFImpl`|`FaissIndexIVFImpl::downcast`|Y, `quantizer`|N|Y|
|`FaissIndexIVFFlat`|`FaissIndexIVFFlat::new` or `FaissIndexIVFFlat::new_with` or `FaissIndexIVFFlat::new_with_metric` |Y, `quantizer`|N|Y|
|`FaissIndexLSH`|`FaissIndexLSH::new` or `FaissIndexLSH::new_with_options`|N|N|Y|
|`FaissIndexPreTransformImpl`|`FaissIndexPreTransformImpl::new` or `FaissIndexPreTransform::new_with` or `FaissIndexPreTransform::new_with_transform`|Y `index`| Y `transformer`|Y|