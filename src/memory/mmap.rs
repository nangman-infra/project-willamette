use anyhow::Result;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Zero-copy model weights loader backed by `memmap2::Mmap`.
///
/// Enables lazy-loading pages of the weights from the storage without memory copying overhead.
pub struct ModelMmap {
    #[allow(dead_code)]
    file: File,
    mmap: Mmap,
}

impl ModelMmap {
    /// Opens the file at the specified path and maps it into the virtual address space.
    ///
    /// # Safety
    ///
    /// Mapping a file is inherently unsafe because:
    /// 1. If another process or thread truncates or modifies the file while it is mapped,
    ///    accessing the mapped slice will lead to Undefined Behavior (UB), such as bus errors (SIGBUS).
    /// 2. The caller must guarantee that the file contents are not concurrently modified.
    /// We assume the model weights file is static and immutable during the runtime inference.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;

        // SAFETY: The Mmap mapping operation is unsafe because concurrent external writes or
        // truncations to the underlying file will trigger undefined behavior or a crash.
        // We assume the file is immutable and dedicated exclusively to this runtime process.
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self { file, mmap })
    }

    /// Accesses the underlying memory-mapped file as a byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Returns the size of the memory-mapped file in bytes.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Returns `true` if the memory-mapped file is empty.
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
}

/*
32-bit CPU 및 OS 아키텍처 제약 사항:
----------------------------------
32비트 아키텍처(예: i686-unknown-linux-gnu 등)에서는 프로세스가 활용 가능한 가상 주소 공간이 최대 4GB로 제한됩니다.
(실제 OS 커널 공간 등을 제외하면 사용자 공간은 약 2~3GB만 사용 가능합니다.)
따라서 파일 크기가 수십 기가바이트(GB) 이상인 대형 LLM 가중치 파일을 통째로 메모리 맵핑(`mmap`)하려 시도할 경우,
가상 주소 영역 부족으로 인해 `Out of Memory (OOM)`나 `Address space exhausted` 에러를 발생시키며 매핑 자체가 실패하게 됩니다.

향후 Windowed Mmap(슬라이딩 윈도우) 방식으로의 확장 방안:
------------------------------------------------------
이 제약을 해결하기 위해서는 모델 가중치 파일 전체를 한 번에 메모리에 올리지 않고,
현재 추론 연산을 진행 중인 레이어(Layer)의 가중치 데이터만 필요한 시점에 동적으로 메모리에 매핑하고 해제(unmap)하는
슬라이딩 윈도우(Windowed Mmap) 형태의 아키텍처 확장이 필요합니다.

구현 예시:
```rust
pub struct WindowedModelLoader {
    file: File,
}

impl WindowedModelLoader {
    /// 파일의 특정 offset부터 size만큼의 영역만 부분 매핑합니다.
    pub fn map_window(&self, offset: u64, size: usize) -> Result<Mmap> {
        // SAFETY: 파일이 수정되지 않는 환경에서만 안전합니다.
        unsafe {
            memmap2::MmapOptions::new()
                .offset(offset)
                .len(size)
                .map(&self.file)
                .map_err(Into::into)
        }
    }
}
```
위와 같이 각 레이어 또는 텐서 단위로 `map_window`를 호출하여 계산한 뒤 해당 `Mmap` 객체가 드롭(Drop)되도록 설계하면,
32비트 환경에서도 가상 주소 공간을 재활용하면서 대용량 모델의 순차적 추론을 매끄럽게 처리할 수 있습니다.
*/
