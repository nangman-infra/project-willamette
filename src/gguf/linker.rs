//! GGUF artifact planning, relocation, and streaming output.

use crate::error::WillametteError;
use crate::gguf::reader::GgufFile;
use crate::gguf::tensor::TensorView;
use crate::gguf::types::GgmlType;
use crate::model::graph::ModelGraph;
use crate::model::primitives::f16_to_f32;
use crate::model::q6_k;
use std::fmt;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A closed set of artifact policies supported by the linker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactProfile {
    /// Convert the tied F16 token embedding to standard Q6_K.
    EmbeddingQ6K,
}

impl ArtifactProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EmbeddingQ6K => "embedding-q6-k",
        }
    }
}

impl fmt::Display for ArtifactProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ArtifactProfile {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "embedding-q6-k" => Ok(Self::EmbeddingQ6K),
            other => Err(format!(
                "unsupported artifact profile {other:?}; expected embedding-q6-k"
            )),
        }
    }
}

/// One tensor whose representation changes in an artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorChange {
    pub name: String,
    pub source_type: GgmlType,
    pub output_type: GgmlType,
    pub source_bytes: u64,
    pub output_bytes: u64,
    pub source_offset: u64,
    pub output_offset: u64,
}

/// Operation selected for one physical tensor slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorAction {
    Copy,
    QuantizeF16ToQ6K,
}

/// Source and output layout for one tensor, in physical output order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorPlan {
    pub name: String,
    pub action: TensorAction,
    pub source_type: GgmlType,
    pub output_type: GgmlType,
    pub source_offset: u64,
    pub output_offset: u64,
    pub source_primary_bytes: u64,
    pub output_primary_bytes: u64,
    /// Full source span through the next physical tensor offset (or EOF).
    pub source_slot_bytes: u64,
    /// Full output span through the next tensor, including linker padding.
    pub output_slot_bytes: u64,
}

/// Validated artifact layout computed before an output file is opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkPlan {
    pub profile: ArtifactProfile,
    pub architecture: String,
    pub source_bytes: u64,
    pub output_bytes: u64,
    pub alignment: u64,
    pub tensor_count: u64,
    pub tensors: Vec<TensorPlan>,
    pub changes: Vec<TensorChange>,
}

/// Result of publishing one linked artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkReport {
    pub plan: LinkPlan,
}

#[derive(Debug, Clone, Copy)]
struct PlannedTensor {
    source_index: usize,
    source_slot_end: u64,
    output_relative_offset: u64,
    output_type: GgmlType,
    output_primary_bytes: u64,
    action: TensorAction,
}

fn align_offset(offset: u64, alignment: u64) -> Result<u64, WillametteError> {
    let remainder = offset % alignment;
    if remainder == 0 {
        Ok(offset)
    } else {
        offset.checked_add(alignment - remainder).ok_or_else(|| {
            WillametteError::GgufParse("artifact tensor offset overflow".to_string())
        })
    }
}

fn validate_finite_f16(tensor: &TensorView<'_>) -> Result<(), WillametteError> {
    for (index, bytes) in tensor.data.chunks_exact(2).enumerate() {
        let value = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
        if !value.is_finite() {
            return Err(WillametteError::GgufParse(format!(
                "tensor {:?} contains non-finite value {value} at element {index}",
                tensor.name
            )));
        }
    }
    Ok(())
}

fn validate_q6k_embedding(embedding: &TensorView<'_>) -> Result<u64, WillametteError> {
    if embedding.ggml_type != GgmlType::F16 || embedding.shape.len() != 2 {
        return Err(WillametteError::GgufParse(
            "token_embd.weight must be a 2-D F16 tensor".to_string(),
        ));
    }
    if embedding.shape[0] == 0 {
        return Err(WillametteError::GgufParse(
            "token_embd.weight row width must be greater than zero".to_string(),
        ));
    }
    let output_bytes = TensorView::q6k_expected_byte_len(&embedding.shape)?;
    validate_finite_f16(embedding)?;
    Ok(output_bytes)
}

fn profile_transform(
    gguf: &GgufFile<'_>,
    profile: ArtifactProfile,
) -> Result<(String, usize, GgmlType, u64), WillametteError> {
    match profile {
        ArtifactProfile::EmbeddingQ6K => {
            let graph = ModelGraph::from_gguf(gguf)?;
            if !graph.lm_head_is_tied() || graph.has_output_weight_tensor {
                return Err(WillametteError::GgufParse(
                    "embedding-q6-k requires no separate output.weight tensor".to_string(),
                ));
            }
            let architecture = graph.config.architecture.clone();
            let source_index = gguf
                .tensors
                .iter()
                .position(|tensor| tensor.name == "token_embd.weight")
                .ok_or_else(|| {
                    WillametteError::GgufParse("missing token_embd.weight".to_string())
                })?;
            let embedding = &gguf.tensors[source_index];
            let output_bytes = validate_q6k_embedding(embedding)?;
            Ok((architecture, source_index, GgmlType::Q6K, output_bytes))
        }
    }
}

fn build_plan<'a>(
    source: &'a [u8],
    profile: ArtifactProfile,
) -> Result<(GgufFile<'a>, LinkPlan, Vec<PlannedTensor>), WillametteError> {
    let gguf = GgufFile::parse(source)?;
    let (architecture, transform_index, output_type, transformed_bytes) =
        profile_transform(&gguf, profile)?;
    let mut physical_indices: Vec<usize> = (0..gguf.tensors.len()).collect();
    physical_indices.sort_unstable_by_key(|&index| gguf.tensors[index].offset);

    let mut running_relative_offset = 0u64;
    let mut planned = Vec::new();
    planned
        .try_reserve_exact(physical_indices.len())
        .map_err(|error| {
            WillametteError::GgufParse(format!("allocating artifact plan: {error}"))
        })?;
    let mut changes = Vec::with_capacity(1);
    let mut tensor_plans: Vec<TensorPlan> = Vec::new();
    tensor_plans
        .try_reserve_exact(physical_indices.len())
        .map_err(|error| {
            WillametteError::GgufParse(format!("allocating public tensor plan: {error}"))
        })?;

    for (physical_index, &source_index) in physical_indices.iter().enumerate() {
        let tensor = &gguf.tensors[source_index];
        let source_slot_end = physical_indices
            .get(physical_index + 1)
            .map(|&next_index| gguf.tensors[next_index].offset)
            .unwrap_or(source.len() as u64);
        let source_primary_end = tensor.offset.checked_add(tensor.byte_len).ok_or_else(|| {
            WillametteError::GgufParse(format!("tensor {:?} end overflow", tensor.name))
        })?;
        let suffix_bytes = source_slot_end
            .checked_sub(source_primary_end)
            .ok_or_else(|| {
                WillametteError::GgufParse(format!("tensor {:?} slot underflow", tensor.name))
            })?;
        let output_relative_offset = align_offset(running_relative_offset, gguf.alignment)?;
        if let Some(previous) = tensor_plans.last_mut() {
            let previous_relative_offset = previous
                .output_offset
                .checked_sub(gguf.data_section_start)
                .ok_or_else(|| {
                    WillametteError::GgufParse(
                        "public artifact tensor offset underflow".to_string(),
                    )
                })?;
            previous.output_slot_bytes = output_relative_offset
                .checked_sub(previous_relative_offset)
                .ok_or_else(|| {
                    WillametteError::GgufParse("public artifact slot size underflow".to_string())
                })?;
        }
        let (action, output_tensor_type, output_primary_bytes) = if source_index == transform_index
        {
            (
                TensorAction::QuantizeF16ToQ6K,
                output_type,
                transformed_bytes,
            )
        } else {
            (TensorAction::Copy, tensor.ggml_type, tensor.byte_len)
        };
        let output_slot_bytes =
            output_primary_bytes
                .checked_add(suffix_bytes)
                .ok_or_else(|| {
                    WillametteError::GgufParse(format!(
                        "artifact slot size overflow at {:?}",
                        tensor.name
                    ))
                })?;
        running_relative_offset = output_relative_offset
            .checked_add(output_slot_bytes)
            .ok_or_else(|| {
                WillametteError::GgufParse(format!(
                    "artifact offset overflow after {:?}",
                    tensor.name
                ))
            })?;

        let output_offset = gguf
            .data_section_start
            .checked_add(output_relative_offset)
            .ok_or_else(|| {
                WillametteError::GgufParse("artifact absolute tensor offset overflow".to_string())
            })?;
        if source_index == transform_index {
            changes.push(TensorChange {
                name: tensor.name.clone(),
                source_type: tensor.ggml_type,
                output_type: output_tensor_type,
                source_bytes: tensor.byte_len,
                output_bytes: output_primary_bytes,
                source_offset: tensor.offset,
                output_offset,
            });
        }
        tensor_plans.push(TensorPlan {
            name: tensor.name.clone(),
            action,
            source_type: tensor.ggml_type,
            output_type: output_tensor_type,
            source_offset: tensor.offset,
            output_offset,
            source_primary_bytes: tensor.byte_len,
            output_primary_bytes,
            source_slot_bytes: source_slot_end - tensor.offset,
            output_slot_bytes,
        });
        planned.push(PlannedTensor {
            source_index,
            source_slot_end,
            output_relative_offset,
            output_type: output_tensor_type,
            output_primary_bytes,
            action,
        });
    }

    let output_bytes = gguf
        .data_section_start
        .checked_add(running_relative_offset)
        .ok_or_else(|| WillametteError::GgufParse("artifact size overflow".to_string()))?;
    let plan = LinkPlan {
        profile,
        architecture,
        source_bytes: source.len() as u64,
        output_bytes,
        alignment: gguf.alignment,
        tensor_count: gguf.tensor_count,
        tensors: tensor_plans,
        changes,
    };
    Ok((gguf, plan, planned))
}

/// Validate an artifact profile and compute every output tensor offset.
pub fn plan_artifact(source: &[u8], profile: ArtifactProfile) -> Result<LinkPlan, WillametteError> {
    let (_, plan, _) = build_plan(source, profile)?;
    Ok(plan)
}

fn patch_u32(data: &mut [u8], offset: u64, value: u32) -> Result<(), WillametteError> {
    let offset = usize::try_from(offset)
        .map_err(|_| WillametteError::GgufParse("header offset overflow".to_string()))?;
    let destination = data.get_mut(offset..offset + 4).ok_or_else(|| {
        WillametteError::GgufParse("descriptor dtype patch is out of bounds".to_string())
    })?;
    destination.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn patch_u64(data: &mut [u8], offset: u64, value: u64) -> Result<(), WillametteError> {
    let offset = usize::try_from(offset)
        .map_err(|_| WillametteError::GgufParse("header offset overflow".to_string()))?;
    let destination = data.get_mut(offset..offset + 8).ok_or_else(|| {
        WillametteError::GgufParse("descriptor tensor-offset patch is out of bounds".to_string())
    })?;
    destination.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn patched_header(
    source: &[u8],
    gguf: &GgufFile<'_>,
    planned: &[PlannedTensor],
) -> Result<Vec<u8>, WillametteError> {
    let data_start = usize::try_from(gguf.data_section_start)
        .map_err(|_| WillametteError::GgufParse("data start does not fit usize".to_string()))?;
    let mut header = source
        .get(..data_start)
        .ok_or_else(|| WillametteError::GgufParse("data start is out of bounds".to_string()))?
        .to_vec();
    for tensor in planned {
        let descriptor = gguf
            .tensor_descriptors
            .get(tensor.source_index)
            .ok_or_else(|| {
                WillametteError::GgufParse("missing tensor descriptor location".to_string())
            })?;
        patch_u32(
            &mut header,
            descriptor.dtype_offset,
            tensor.output_type.to_raw(),
        )?;
        patch_u64(
            &mut header,
            descriptor.relative_offset_offset,
            tensor.output_relative_offset,
        )?;
    }
    Ok(header)
}

fn source_slice(source: &[u8], start: u64, end: u64) -> Result<&[u8], WillametteError> {
    let start = usize::try_from(start)
        .map_err(|_| WillametteError::GgufParse("source offset overflow".to_string()))?;
    let end = usize::try_from(end)
        .map_err(|_| WillametteError::GgufParse("source offset overflow".to_string()))?;
    source.get(start..end).ok_or_else(|| {
        WillametteError::GgufParse("source tensor slot is out of bounds".to_string())
    })
}

fn write_zeros(output: &mut impl Write, count: u64) -> Result<(), WillametteError> {
    const ZEROS: [u8; 4096] = [0; 4096];
    let mut remaining = count;
    while remaining > 0 {
        let chunk = remaining.min(ZEROS.len() as u64) as usize;
        output.write_all(&ZEROS[..chunk])?;
        remaining -= chunk as u64;
    }
    Ok(())
}

fn write_q6k_tensor(
    output: &mut impl Write,
    tensor: &TensorView<'_>,
) -> Result<(), WillametteError> {
    let row_elements = usize::try_from(tensor.shape[0])
        .map_err(|_| WillametteError::GgufParse("embedding row size overflow".to_string()))?;
    let f16_row_bytes = row_elements
        .checked_mul(2)
        .ok_or_else(|| WillametteError::GgufParse("F16 row size overflow".to_string()))?;
    let q6_row_bytes = row_elements / TensorView::Q6K_ELEMENTS_PER_BLOCK as usize
        * TensorView::Q6K_BYTES_PER_BLOCK as usize;
    let mut row = vec![0.0_f32; row_elements];
    let mut quantized = vec![0u8; q6_row_bytes];
    for f16_bytes in tensor.data.chunks_exact(f16_row_bytes) {
        for (value, bytes) in row.iter_mut().zip(f16_bytes.chunks_exact(2)) {
            *value = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
        }
        q6_k::quantize_row(&row, &mut quantized)?;
        output.write_all(&quantized)?;
    }
    Ok(())
}

fn write_artifact(
    source: &[u8],
    gguf: &GgufFile<'_>,
    plan: &LinkPlan,
    planned: &[PlannedTensor],
    temp_path: &Path,
    output_path: &Path,
    temp_created: &mut bool,
) -> Result<(), WillametteError> {
    let header = patched_header(source, gguf, planned)?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temp_path)?;
    *temp_created = true;
    let mut output = BufWriter::with_capacity(1024 * 1024, file);
    output.write_all(&header)?;
    let mut position = header.len() as u64;

    for tensor_plan in planned {
        let output_offset = gguf
            .data_section_start
            .checked_add(tensor_plan.output_relative_offset)
            .ok_or_else(|| {
                WillametteError::GgufParse("artifact output offset overflow".to_string())
            })?;
        let padding = output_offset.checked_sub(position).ok_or_else(|| {
            WillametteError::GgufParse("artifact output offsets are not monotonic".to_string())
        })?;
        write_zeros(&mut output, padding)?;
        position = output_offset;

        let source_tensor = &gguf.tensors[tensor_plan.source_index];
        match tensor_plan.action {
            TensorAction::Copy => {
                let slot = source_slice(source, source_tensor.offset, tensor_plan.source_slot_end)?;
                output.write_all(slot)?;
                position = position.checked_add(slot.len() as u64).ok_or_else(|| {
                    WillametteError::GgufParse("artifact position overflow".to_string())
                })?;
            }
            TensorAction::QuantizeF16ToQ6K => {
                write_q6k_tensor(&mut output, source_tensor)?;
                position = position
                    .checked_add(tensor_plan.output_primary_bytes)
                    .ok_or_else(|| {
                        WillametteError::GgufParse("artifact position overflow".to_string())
                    })?;
                let source_primary_end = source_tensor
                    .offset
                    .checked_add(source_tensor.byte_len)
                    .ok_or_else(|| {
                    WillametteError::GgufParse("source tensor end overflow".to_string())
                })?;
                let suffix = source_slice(source, source_primary_end, tensor_plan.source_slot_end)?;
                output.write_all(suffix)?;
                position = position.checked_add(suffix.len() as u64).ok_or_else(|| {
                    WillametteError::GgufParse("artifact position overflow".to_string())
                })?;
            }
        }
    }
    if position != plan.output_bytes {
        return Err(WillametteError::GgufParse(format!(
            "artifact wrote {position} bytes, planned {}",
            plan.output_bytes
        )));
    }
    output.flush()?;
    output.get_ref().sync_all()?;
    drop(output);

    // A hard link publishes atomically without replacing an existing path.
    std::fs::hard_link(temp_path, output_path)?;
    Ok(())
}

/// Build and atomically publish an artifact for the selected profile.
pub fn link_artifact(
    source: &[u8],
    output_path: &Path,
    profile: ArtifactProfile,
) -> Result<LinkReport, WillametteError> {
    let (gguf, plan, planned) = build_plan(source, profile)?;
    if output_path.try_exists()? {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("output already exists: {}", output_path.display()),
        )
        .into());
    }
    let file_name = output_path
        .file_name()
        .ok_or_else(|| WillametteError::GgufParse("output path has no file name".to_string()))?;
    let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp_path = output_path.with_file_name(format!(
        ".{}.{}.{}.tmp",
        file_name.to_string_lossy(),
        std::process::id(),
        counter
    ));

    let mut temp_created = false;
    let write_result = write_artifact(
        source,
        &gguf,
        &plan,
        &planned,
        &temp_path,
        output_path,
        &mut temp_created,
    );
    let cleanup_result = temp_created.then(|| std::fs::remove_file(&temp_path));
    match (write_result, cleanup_result) {
        (Ok(()), _) => Ok(LinkReport { plan }),
        (Err(write_error), Some(Err(cleanup_error))) => Err(WillametteError::GgufParse(format!(
            "{write_error}; also failed to remove temporary file {}: {cleanup_error}",
            temp_path.display()
        ))),
        (Err(write_error), _) => Err(write_error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{build_gguf_for_config, build_gguf_with_output_weight, Preset};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn compact_source() -> Vec<u8> {
        let mut config = Preset::Small.config();
        config.n_layers = 1;
        config.vocab_size = 5;
        build_gguf_for_config(config, false)
    }

    fn move_embedding_to_second_physical_slot(source: &[u8]) -> Vec<u8> {
        let gguf = GgufFile::parse(source).unwrap();
        let mut physical: Vec<usize> = (0..gguf.tensors.len()).collect();
        physical.sort_unstable_by_key(|&index| gguf.tensors[index].offset);
        assert_eq!(gguf.tensors[physical[0]].name, "token_embd.weight");
        assert_eq!(gguf.tensors[physical[1]].name, "output_norm.weight");
        physical.swap(0, 1);

        let data_start = gguf.data_section_start as usize;
        let mut reordered = source[..data_start].to_vec();
        let original_physical = {
            let mut indices: Vec<usize> = (0..gguf.tensors.len()).collect();
            indices.sort_unstable_by_key(|&index| gguf.tensors[index].offset);
            indices
        };
        let mut relative_offset = 0u64;
        for source_index in physical {
            relative_offset = align_offset(relative_offset, gguf.alignment).unwrap();
            let descriptor = gguf.tensor_descriptors[source_index];
            patch_u64(
                &mut reordered,
                descriptor.relative_offset_offset,
                relative_offset,
            )
            .unwrap();
            let physical_position = original_physical
                .iter()
                .position(|&index| index == source_index)
                .unwrap();
            let slot_end = original_physical
                .get(physical_position + 1)
                .map(|&next_index| gguf.tensors[next_index].offset)
                .unwrap_or(source.len() as u64);
            let absolute_offset = gguf.data_section_start + relative_offset;
            reordered.resize(absolute_offset as usize, 0);
            let slot = source_slice(source, gguf.tensors[source_index].offset, slot_end).unwrap();
            reordered.extend_from_slice(slot);
            relative_offset += slot.len() as u64;
        }
        reordered
    }

    fn temp_path(label: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "willamette-linker-{label}-{}-{nonce}.gguf",
            std::process::id()
        ))
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    #[test]
    fn profile_name_round_trips() {
        let profile: ArtifactProfile = "embedding-q6-k".parse().unwrap();
        assert_eq!(profile, ArtifactProfile::EmbeddingQ6K);
        assert_eq!(profile.to_string(), "embedding-q6-k");
        assert!("unknown".parse::<ArtifactProfile>().is_err());
    }

    #[test]
    fn plans_non_alignment_sized_embedding_reduction() {
        let source = compact_source();
        let plan = plan_artifact(&source, ArtifactProfile::EmbeddingQ6K).unwrap();
        assert_eq!(plan.architecture, "bitnet-b1.58");
        assert_eq!(
            plan.tensor_count as usize,
            GgufFile::parse(&source).unwrap().tensors.len()
        );
        assert_eq!(plan.changes.len(), 1);
        assert_eq!(plan.tensors.len(), plan.tensor_count as usize);
        assert_eq!(
            plan.tensors
                .iter()
                .filter(|tensor| tensor.action == TensorAction::QuantizeF16ToQ6K)
                .count(),
            1
        );
        assert_eq!(plan.changes[0].source_bytes, 2_560);
        assert_eq!(plan.changes[0].output_bytes, 1_050);
        let embedding_plan = plan
            .tensors
            .iter()
            .find(|tensor| tensor.name == "token_embd.weight")
            .unwrap();
        assert_eq!(embedding_plan.output_primary_bytes, 1_050);
        assert_eq!(embedding_plan.output_slot_bytes, 1_056);
        assert!(
            !(plan.changes[0].source_bytes - plan.changes[0].output_bytes)
                .is_multiple_of(plan.alignment)
        );
    }

    #[test]
    fn tied_profile_rejects_separate_output_weight() {
        let mut config = Preset::Small.config();
        config.n_layers = 1;
        config.vocab_size = 5;
        let source = build_gguf_with_output_weight(config);
        let error = plan_artifact(&source, ArtifactProfile::EmbeddingQ6K).unwrap_err();
        assert!(error.to_string().contains("separate output.weight"));
    }

    #[test]
    fn q6k_profile_rejects_zero_width_before_writing() {
        let embedding = TensorView {
            name: "token_embd.weight".to_string(),
            shape: vec![0, 1],
            ggml_type: GgmlType::F16,
            offset: 0,
            byte_len: 0,
            data: &[],
            scale_data: None,
        };
        let error = validate_q6k_embedding(&embedding).unwrap_err();
        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn compact_artifact_checksum_is_stable() {
        let source = compact_source();
        let output_path = temp_path("checksum");
        link_artifact(&source, &output_path, ArtifactProfile::EmbeddingQ6K).unwrap();
        let output = std::fs::read(&output_path).unwrap();
        std::fs::remove_file(output_path).unwrap();
        assert_eq!(fnv1a64(&output), 18_102_645_404_363_191_156);
    }

    #[test]
    fn links_embedding_from_middle_and_preserves_other_tensors() {
        let mut source = move_embedding_to_second_physical_slot(&compact_source());
        let padding_offset = {
            let gguf = GgufFile::parse(&source).unwrap();
            let i2s = gguf
                .tensors
                .iter()
                .find(|tensor| tensor.ggml_type == GgmlType::BitNetI2S)
                .unwrap();
            (i2s.offset + i2s.byte_len + 7) as usize
        };
        source[padding_offset] = 0xa5;
        let source_gguf = GgufFile::parse(&source).unwrap();
        let embedding = source_gguf
            .tensors
            .iter()
            .find(|tensor| tensor.name == "token_embd.weight")
            .unwrap();
        assert!(source_gguf
            .tensors
            .iter()
            .any(|tensor| tensor.offset < embedding.offset));

        let output_path = temp_path("middle");
        let report = link_artifact(&source, &output_path, ArtifactProfile::EmbeddingQ6K).unwrap();
        let output = std::fs::read(&output_path).unwrap();
        std::fs::remove_file(output_path).unwrap();
        assert_eq!(output.len() as u64, report.plan.output_bytes);

        let output_gguf = GgufFile::parse(&output).unwrap();
        let output_embedding = output_gguf
            .tensors
            .iter()
            .find(|tensor| tensor.name == "token_embd.weight")
            .unwrap();
        assert_eq!(output_embedding.ggml_type, GgmlType::Q6K);
        for source_tensor in source_gguf
            .tensors
            .iter()
            .filter(|tensor| tensor.name != "token_embd.weight")
        {
            let output_tensor = output_gguf
                .tensors
                .iter()
                .find(|tensor| tensor.name == source_tensor.name)
                .unwrap();
            assert_eq!(
                output_tensor.data, source_tensor.data,
                "{}",
                source_tensor.name
            );
            assert_eq!(
                output_tensor.scale_data, source_tensor.scale_data,
                "{} scale block",
                source_tensor.name
            );
            assert_eq!(
                (output_tensor.offset - output_gguf.data_section_start) % output_gguf.alignment,
                0,
                "{} alignment",
                source_tensor.name
            );
        }
        ModelGraph::from_gguf(&output_gguf).unwrap();
    }
}
