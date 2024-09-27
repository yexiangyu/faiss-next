#![allow(unused)]

fn main() {
    if std::env::var("DOC_RS").is_ok() {
        #[allow(clippy::needless_return)]
        return;
    }

    #[cfg(feature = "bindgen")]
    {
        create_bindgen();
    }

    do_build_extension();
    do_link();
}

fn do_build_extension() {
    cxx_build::bridge("src/extension/mod.rs")
        .file("src/extension/index_binary_factory.cpp")
        .file("src/extension/gpu_distance.cpp")
        .include(faiss_dir().unwrap().join("include"))
        .std("c++11")
        .compile("faiss_next_extension");
    println!("cargo:rerun-if-changed=src/extension/mod.rs");
    println!("cargo:rerun-if-changed=src/extension/index_bianry_factory.cpp");
    println!("cargo:rerun-if-changed=src/extension/index_bianry_factory.hpp");
    println!("cargo:rerun-if-changed=src/extension/gpu_distance.cpp");
    println!("cargo:rerun-if-changed=src/extension/gpu_distance.hpp");
}

fn do_link() {
    let mut found_faiss = false;
    let mut found_faiss_c = false;

    for env_var in ["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH", "Path"] {
        if found_faiss && found_faiss_c {
            break;
        }

        let mut sp = ':';
        #[cfg(target_os = "windows")]
        {
            sp = ":"
        }
        if let Ok(paths) = std::env::var(env_var) {
            for path in paths.split(':') {
                let path = std::path::PathBuf::from(path);
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
                    println!("cargo:rustc-link-search=native={}", path.to_str().unwrap());
                }
            }
        }
    }

    if let Some(faiss_dir) = faiss_dir() {
        println!(
            "cargo:rustc-link-search=native={}",
            faiss_dir.join("lib").to_str().unwrap()
        );
    }

    println!("cargo:rustc-link-lib=dylib=faiss");
    println!("cargo:rustc-link-lib=dylib=faiss_c");
}

fn create_bindgen() {
    println!("cargo:rerun-if-changed=faissw.h");

    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    let output_dir = std::path::PathBuf::from("src").join(os);

    std::fs::create_dir_all(&output_dir).unwrap();

    let mut output_bindings = output_dir.join(format!("{arch}.rs"));

    #[cfg(feature = "cuda")]
    {
        #[cfg(target_os = "macos")]
        {
            panic!("CUDA is not supported on macOS");
        }
        output_bindings = output_dir.join(format!("{arch}_cuda.rs"));
    }

    let mut builder = bindgen::builder()
        .header("faissw.h")
        .derive_default(true)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        .layout_tests(false)
        .allowlist_function("faiss_.*")
        .allowlist_type("idx_t|Faiss.*")
        .opaque_type("FILE");

    if let Some(faiss_dir) = faiss_dir() {
        let include_dir = faiss_dir.join("include");
        builder = builder.clang_arg(format!("-I{}", include_dir.to_str().unwrap()));
    }

    #[cfg(feature = "cuda")]
    {
        #[cfg(not(target_os = "macos"))]
        {
            let cuda_dir = cuda_dir().unwrap();
            builder = builder
                .clang_arg("-DUSE_CUDA")
                .clang_arg(format!("-I{}", cuda_dir.join("include").to_str().unwrap()));
        }
    }

    builder
        .generate()
        .expect("unable to generate bindings")
        .write_to_file(output_bindings)
        .expect("unable to write bindings");
}

fn cuda_dir() -> Option<std::path::PathBuf> {
    #[cfg(target_os = "linux")]
    {
        let cuda = std::path::PathBuf::from("/usr/local/cuda");
        if cuda.exists() {
            return Some(cuda);
        }
    }

    None
}

fn faiss_dir() -> Option<std::path::PathBuf> {
    if let Ok(faiss_dir) = std::env::var("FAISS_DIR") {
        let faiss_dir = std::path::PathBuf::from(faiss_dir);

        if faiss_dir.is_dir()
            && faiss_dir.join("include").is_dir()
            && faiss_dir.join("lib").is_dir()
        {
            return Some(faiss_dir);
        }
    }

    #[cfg(target_os = "macos")]
    {
        #[cfg(target_arch = "aarch64")]
        {
            let faiss_dir = std::path::PathBuf::from("/opt/homebrew/opt/faiss");
            if faiss_dir.join("include").is_dir() && faiss_dir.join("lib").is_dir() {
                return Some(faiss_dir);
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            let faiss_dir = std::path::PathBuf::from("/usr/local/homebrew/opt/faiss");
            if faiss_dir.join("include").is_dir() && faiss_dir.join("lib").is_dir() {
                return Some(faiss_dir);
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        return Some(std::path::PathBuf::from("/usr/local/include"));
    }

    None
}
