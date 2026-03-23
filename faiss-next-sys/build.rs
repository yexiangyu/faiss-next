use std::env;
use std::path::PathBuf;

const FAISS_MIN_VERSION: (u32, u32, u32) = (1, 14, 0);
const FAISS_MAX_TESTED_VERSION: (u32, u32, u32) = (1, 14, 99);
const BINDING_VERSION: &str = "v1_14";

fn main() {
    println!("cargo:rustc-check-cfg=cfg(faiss_binding, values(\"v1_14\", \"v1_15\"))");
    println!(
        "cargo:rustc-check-cfg=cfg(faiss_version, values(\"1_14_0\", \"1_14_1\", \"1_15_0\"))"
    );

    if env::var("DOC_RS").is_ok() {
        return;
    }

    let faiss_path = find_faiss();

    if let Some(ref path) = faiss_path {
        println!(
            "cargo:rustc-link-search=native={}",
            path.join("lib").display()
        );

        if let Some(version) = detect_version(path) {
            check_version(&version);
            emit_version_cfg(&version);
            select_binding_version(&version);
        } else {
            println!(
                "cargo:warning=Could not detect Faiss version, using {} bindings",
                BINDING_VERSION
            );
            println!("cargo:rustc-cfg=faiss_binding=\"{}\"", BINDING_VERSION);
        }
    } else {
        println!(
            "cargo:warning=Faiss not found via standard paths, using {} bindings",
            BINDING_VERSION
        );
        println!("cargo:rustc-cfg=faiss_binding=\"{}\"", BINDING_VERSION);
    }

    search_library_paths();

    println!("cargo:rustc-link-lib=dylib=faiss");
    println!("cargo:rustc-link-lib=dylib=faiss_c");

    #[cfg(feature = "bindgen")]
    generate_bindings(&faiss_path);

    println!("cargo:rerun-if-changed=wrapper.h");
}

fn find_faiss() -> Option<PathBuf> {
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

        if pkg_config::probe_library("faiss").is_ok() {
            return None;
        }
    }

    None
}

fn detect_version(faiss_path: &std::path::Path) -> Option<(u32, u32, u32)> {
    let index_h = faiss_path.join("include/faiss/Index.h");
    if let Ok(content) = std::fs::read_to_string(&index_h) {
        let mut major: Option<u32> = None;
        let mut minor: Option<u32> = None;
        let mut patch: Option<u32> = None;

        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                if parts.get(1) == Some(&"FAISS_VERSION_MAJOR") {
                    major = parts.get(2).and_then(|s| s.parse().ok());
                }
                if parts.get(1) == Some(&"FAISS_VERSION_MINOR") {
                    minor = parts.get(2).and_then(|s| s.parse().ok());
                }
                if parts.get(1) == Some(&"FAISS_VERSION_PATCH") {
                    patch = parts.get(2).and_then(|s| s.parse().ok());
                }
            }
        }

        if let (Some(m), Some(mi), Some(p)) = (major, minor, patch) {
            return Some((m, mi, p));
        }
    }

    let version_file = faiss_path.join("include/faiss/impl/platform_macros.h");
    if let Ok(content) = std::fs::read_to_string(&version_file) {
        if let Some(line) = content.lines().find(|l| l.contains("FAISS_VERSION")) {
            let parts: Vec<&str> = line.split('"').collect();
            if parts.len() >= 2 {
                let version_str = parts[1];
                let nums: Vec<u32> = version_str
                    .split('.')
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if nums.len() >= 3 {
                    return Some((nums[0], nums[1], nums[2]));
                }
            }
        }
    }
    None
}

fn check_version(version: &(u32, u32, u32)) {
    let (maj, min, patch) = *version;

    if version < &FAISS_MIN_VERSION {
        panic!(
            "Faiss version {}.{}.{} is not supported. Minimum required: {}.{}.{}",
            maj, min, patch, FAISS_MIN_VERSION.0, FAISS_MIN_VERSION.1, FAISS_MIN_VERSION.2
        );
    }

    if version > &FAISS_MAX_TESTED_VERSION {
        println!(
            "cargo:warning=Faiss version {}.{}.{} is newer than tested versions ({}.{}.x).",
            maj, min, patch, FAISS_MIN_VERSION.0, FAISS_MIN_VERSION.1
        );
        println!(
            "cargo:warning=Using {} bindings. Compatibility not guaranteed.",
            BINDING_VERSION
        );
    }
}

fn emit_version_cfg(version: &(u32, u32, u32)) {
    let (maj, min, patch) = *version;
    println!(
        "cargo:rustc-cfg=faiss_version=\"{}_{}_{}\"",
        maj, min, patch
    );
    println!("cargo:rustc-env=FAISS_VERSION={}.{}.{}", maj, min, patch);
}

fn select_binding_version(version: &(u32, u32, u32)) {
    let (_, minor, _) = *version;

    let binding_version = match minor {
        14 => "v1_14",
        15 => "v1_15",
        _ if minor > 15 => {
            println!(
                "cargo:warning=Faiss 1.{}.x detected, using latest available bindings (v1_14)",
                minor
            );
            "v1_14"
        }
        _ => {
            panic!("Unsupported Faiss version 1.{}.x", minor);
        }
    };

    println!("cargo:rustc-cfg=faiss_binding=\"{}\"", binding_version);
}

fn search_library_paths() {
    for env_var in ["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"] {
        if let Ok(paths) = env::var(env_var) {
            for path in paths.split(':') {
                let path = PathBuf::from(path);
                let has_faiss = path.join("libfaiss.so").exists()
                    || path.join("libfaiss.dylib").exists()
                    || path.join("faiss.dll").exists();
                let has_faiss_c = path.join("libfaiss_c.so").exists()
                    || path.join("libfaiss_c.dylib").exists()
                    || path.join("faiss_c.dll").exists();

                if has_faiss || has_faiss_c {
                    println!("cargo:rustc-link-search=native={}", path.display());
                }
            }
        }
    }
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

#[cfg(feature = "bindgen")]
fn generate_bindings(faiss_path: &Option<PathBuf>) {
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    let out_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("src")
        .join("bindings")
        .join(BINDING_VERSION);

    std::fs::create_dir_all(&out_dir).unwrap();

    let output_file = if cfg!(feature = "cuda") {
        #[cfg(target_os = "macos")]
        panic!("CUDA is not supported on macOS");
        out_dir.join(format!("{}_cuda.rs", os))
    } else {
        match os.as_str() {
            "macos" => out_dir.join(format!("macos_{}.rs", arch)),
            "linux" => out_dir.join(format!("linux_{}.rs", arch)),
            _ => out_dir.join(format!("{}_{}.rs", os, arch)),
        }
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

    if let Some(path) = faiss_path {
        builder = builder.clang_arg(format!("-I{}", path.join("include").display()));
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
