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

    do_link();
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

    let output_bindings = output_dir.join(format!("{arch}.rs"));

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

    builder
        .generate()
        .expect("unable to generate bindings")
        .write_to_file(output_bindings)
        .expect("unable to write bindings");
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

    None
}
