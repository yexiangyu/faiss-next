use std::env;
use std::path::PathBuf;

fn main() {
    if env::var("DOC_RS").is_ok() {
        return;
    }

    build_extension();
    link_faiss();

    #[cfg(feature = "bindgen")]
    generate_bindings();
}

fn faiss_dir() -> Option<PathBuf> {
    if let Ok(dir) = env::var("FAISS_DIR") {
        let path = PathBuf::from(dir);
        if path.join("include").is_dir() && path.join("lib").is_dir() {
            return Some(path);
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let path = PathBuf::from("/opt/homebrew/opt/faiss");
        if path.join("include").is_dir() && path.join("lib").is_dir() {
            return Some(path);
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        let path = PathBuf::from("/usr/local/opt/faiss");
        if path.join("include").is_dir() && path.join("lib").is_dir() {
            return Some(path);
        }
    }

    #[cfg(target_os = "linux")]
    {
        let path = PathBuf::from("/usr/local");
        if path.join("include/faiss").is_dir() {
            return Some(path);
        }
    }

    None
}

#[cfg(feature = "cuda")]
fn cuda_dir() -> Option<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        let path = PathBuf::from("/usr/local/cuda");
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn build_extension() {
    let mut build = cxx_build::bridge("src/extension/mod.rs");

    let src_dir = PathBuf::from("src/extension");
    build
        .file(src_dir.join("index_binary_factory.cpp"))
        .std("c++14");

    if let Some(faiss_path) = faiss_dir() {
        build.include(faiss_path.join("include"));
    }

    build.compile("faiss_next_extension");

    println!("cargo:rerun-if-changed=src/extension/mod.rs");
    println!("cargo:rerun-if-changed=src/extension/index_binary_factory.cpp");
    println!("cargo:rerun-if-changed=src/extension/index_binary_factory.hpp");

    #[cfg(feature = "cuda")]
    build_cuda_extension();
}

#[cfg(feature = "cuda")]
fn build_cuda_extension() {
    let mut build = cxx_build::bridge("src/extension/cuda.rs");

    let src_dir = PathBuf::from("src/extension");
    build
        .file(src_dir.join("gpu_distance.cpp"))
        .std("c++14")
        .define("USE_CUDA", None);

    if let Some(faiss_path) = faiss_dir() {
        build.include(faiss_path.join("include"));
    }

    if let Some(cuda_path) = cuda_dir() {
        build.include(cuda_path.join("include"));
    }

    build.compile("faiss_next_cuda_extension");

    println!("cargo:rerun-if-changed=src/extension/cuda.rs");
    println!("cargo:rerun-if-changed=src/extension/gpu_distance.cpp");
    println!("cargo:rerun-if-changed=src/extension/gpu_distance.hpp");
}

fn link_faiss() {
    let mut found_faiss = false;
    let mut found_faiss_c = false;

    for env_var in ["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"] {
        if found_faiss && found_faiss_c {
            break;
        }

        if let Ok(paths) = env::var(env_var) {
            for path in paths.split(':') {
                let path = PathBuf::from(path);

                if !found_faiss {
                    found_faiss = path.join("libfaiss.so").exists()
                        || path.join("libfaiss.dylib").exists()
                        || path.join("faiss.dll").exists();
                }

                if !found_faiss_c {
                    found_faiss_c = path.join("libfaiss_c.so").exists()
                        || path.join("libfaiss_c.dylib").exists()
                        || path.join("faiss_c.dll").exists();
                }

                if found_faiss || found_faiss_c {
                    println!("cargo:rustc-link-search=native={}", path.display());
                }
            }
        }
    }

    if let Some(faiss_path) = faiss_dir() {
        println!(
            "cargo:rustc-link-search=native={}",
            faiss_path.join("lib").display()
        );
    }

    println!("cargo:rustc-link-lib=dylib=faiss");
    println!("cargo:rustc-link-lib=dylib=faiss_c");
}

#[cfg(feature = "bindgen")]
fn generate_bindings() {
    println!("cargo:rerun-if-changed=wrapper.h");

    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    let out_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("src")
        .join("bindings")
        .join(&os);

    std::fs::create_dir_all(&out_dir).unwrap();

    let output_file = if cfg!(feature = "cuda") {
        #[cfg(target_os = "macos")]
        panic!("CUDA is not supported on macOS");
        out_dir.join(format!("{}_cuda.rs", arch))
    } else {
        out_dir.join(format!("{}.rs", arch))
    };

    let mut builder = bindgen::builder()
        .header("wrapper.h")
        .derive_default(true)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        .layout_tests(false)
        .allowlist_function("faiss_.*")
        .allowlist_type("idx_t|Faiss.*")
        .opaque_type("FILE");

    if let Some(faiss_path) = faiss_dir() {
        builder = builder.clang_arg(format!("-I{}", faiss_path.join("include").display()));
    }

    #[cfg(feature = "cuda")]
    {
        if let Some(cuda_path) = cuda_dir() {
            builder = builder
                .clang_arg("-DUSE_CUDA")
                .clang_arg(format!("-I{}", cuda_path.join("include").display()));
        }
    }

    builder
        .generate()
        .expect("Failed to generate bindings")
        .write_to_file(output_file)
        .expect("Failed to write bindings");
}
