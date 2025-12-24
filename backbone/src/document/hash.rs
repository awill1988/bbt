use crate::error::Result;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Compute SHA256 hash of a file
///
/// # Arguments
/// * `path` - Path to the file
///
/// # Returns
/// Hex-encoded SHA256 hash string
pub fn hash_file<P: AsRef<Path>>(path: P) -> Result<String> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();

    // read file in chunks and update hasher
    let mut buffer = [0; 8192]; // 8 KiB buffer
    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    // finalize and convert to hex string
    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}

/// Compute SHA256 hash of a string
///
/// # Arguments
/// * `text` - The text to hash
///
/// # Returns
/// Hex-encoded SHA256 hash string
pub fn hash_string(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_hash_string() {
        let hash1 = hash_string("hello world");
        let hash2 = hash_string("hello world");
        let hash3 = hash_string("hello world!");

        // same input -> same hash
        assert_eq!(hash1, hash2);

        // different input -> different hash
        assert_ne!(hash1, hash3);

        // verify length (sha256 is 64 hex chars)
        assert_eq!(hash1.len(), 64);
    }

    #[test]
    fn test_hash_string_known_value() {
        // known sha256 of "hello world" (without newline)
        let hash = hash_string("hello world");
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_hash_file() -> Result<()> {
        // create temporary file
        let mut temp_file = NamedTempFile::new()?;
        temp_file.write_all(b"hello world")?;
        temp_file.flush()?;

        let hash = hash_file(temp_file.path())?;

        // should match the known hash of "hello world"
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );

        Ok(())
    }

    #[test]
    fn test_hash_file_empty() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let hash = hash_file(temp_file.path())?;

        // hash of empty string
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        Ok(())
    }

    #[test]
    fn test_hash_file_large() -> Result<()> {
        // create a larger file to test chunked reading
        let mut temp_file = NamedTempFile::new()?;
        let data = vec![b'a'; 100000]; // 100 KB of 'a'
        temp_file.write_all(&data)?;
        temp_file.flush()?;

        let hash = hash_file(temp_file.path())?;

        // verify it's a valid sha256 (64 hex chars)
        assert_eq!(hash.len(), 64);

        // hash should be deterministic
        let hash2 = hash_file(temp_file.path())?;
        assert_eq!(hash, hash2);

        Ok(())
    }

    #[test]
    fn test_hash_file_nonexistent() {
        let result = hash_file("/nonexistent/file/path");
        assert!(result.is_err());
    }
}
