use std::fmt;

/// GGML tensor data type identifiers.
///
/// Standard types from the upstream GGML spec, plus BitNet-specific types from
/// the bitnet.cpp / llama.cpp fork.
///
/// **Authoritative source for BitNet types:**
///   - GGML_TYPE_I2_S  = 36
///   - GGML_TYPE_I8_S  = 37
///   - GGML_TYPE_TL1   = 38
///   - GGML_TYPE_TL2   = 39
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    // Q4_2 removed
    // Q4_3 removed
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ1S,
    IQ4NL,
    IQ3S,
    IQ2S,
    IQ4XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1M,
    BF16,

    // ── BitNet-specific types (bitnet.cpp fork) ──
    /// Ternary weight packing, 2 bits per weight, block-interleaved for SIMD.
    /// 128 ternary elements per 32-byte block.
    BitNetI2S,
    /// 8-bit signed integer activations for BitNet.
    BitNetI8S,
    /// BitNet TL1 quantisation layout.
    BitNetTL1,
    /// BitNet TL2 quantisation layout.
    BitNetTL2,

    /// Any type we haven't enumerated.
    Unknown(u32),
}

impl GgmlType {
    /// Convert from the raw u32 type tag stored in the GGUF tensor info.
    pub fn from_raw(raw: u32) -> Self {
        match raw {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            15 => GgmlType::Q8K,
            16 => GgmlType::IQ2XXS,
            17 => GgmlType::IQ2XS,
            18 => GgmlType::IQ3XXS,
            19 => GgmlType::IQ1S,
            20 => GgmlType::IQ4NL,
            21 => GgmlType::IQ3S,
            22 => GgmlType::IQ2S,
            23 => GgmlType::IQ4XS,
            24 => GgmlType::I8,
            25 => GgmlType::I16,
            26 => GgmlType::I32,
            27 => GgmlType::I64,
            28 => GgmlType::F64,
            29 => GgmlType::IQ1M,
            30 => GgmlType::BF16,
            // ── BitNet fork ──
            36 => GgmlType::BitNetI2S,
            37 => GgmlType::BitNetI8S,
            38 => GgmlType::BitNetTL1,
            39 => GgmlType::BitNetTL2,
            other => GgmlType::Unknown(other),
        }
    }

    /// Convert back to the raw u32 value.
    pub fn to_raw(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q4_1 => 3,
            GgmlType::Q5_0 => 6,
            GgmlType::Q5_1 => 7,
            GgmlType::Q8_0 => 8,
            GgmlType::Q8_1 => 9,
            GgmlType::Q2K => 10,
            GgmlType::Q3K => 11,
            GgmlType::Q4K => 12,
            GgmlType::Q5K => 13,
            GgmlType::Q6K => 14,
            GgmlType::Q8K => 15,
            GgmlType::IQ2XXS => 16,
            GgmlType::IQ2XS => 17,
            GgmlType::IQ3XXS => 18,
            GgmlType::IQ1S => 19,
            GgmlType::IQ4NL => 20,
            GgmlType::IQ3S => 21,
            GgmlType::IQ2S => 22,
            GgmlType::IQ4XS => 23,
            GgmlType::I8 => 24,
            GgmlType::I16 => 25,
            GgmlType::I32 => 26,
            GgmlType::I64 => 27,
            GgmlType::F64 => 28,
            GgmlType::IQ1M => 29,
            GgmlType::BF16 => 30,
            GgmlType::BitNetI2S => 36,
            GgmlType::BitNetI8S => 37,
            GgmlType::BitNetTL1 => 38,
            GgmlType::BitNetTL2 => 39,
            GgmlType::Unknown(v) => v,
        }
    }

    /// Human-readable name for display/logging.
    pub fn name(&self) -> String {
        match self {
            GgmlType::F32 => "F32".into(),
            GgmlType::F16 => "F16".into(),
            GgmlType::Q4_0 => "Q4_0".into(),
            GgmlType::Q4_1 => "Q4_1".into(),
            GgmlType::Q5_0 => "Q5_0".into(),
            GgmlType::Q5_1 => "Q5_1".into(),
            GgmlType::Q8_0 => "Q8_0".into(),
            GgmlType::Q8_1 => "Q8_1".into(),
            GgmlType::Q2K => "Q2_K".into(),
            GgmlType::Q3K => "Q3_K".into(),
            GgmlType::Q4K => "Q4_K".into(),
            GgmlType::Q5K => "Q5_K".into(),
            GgmlType::Q6K => "Q6_K".into(),
            GgmlType::Q8K => "Q8_K".into(),
            GgmlType::IQ2XXS => "IQ2_XXS".into(),
            GgmlType::IQ2XS => "IQ2_XS".into(),
            GgmlType::IQ3XXS => "IQ3_XXS".into(),
            GgmlType::IQ1S => "IQ1_S".into(),
            GgmlType::IQ4NL => "IQ4_NL".into(),
            GgmlType::IQ3S => "IQ3_S".into(),
            GgmlType::IQ2S => "IQ2_S".into(),
            GgmlType::IQ4XS => "IQ4_XS".into(),
            GgmlType::I8 => "I8".into(),
            GgmlType::I16 => "I16".into(),
            GgmlType::I32 => "I32".into(),
            GgmlType::I64 => "I64".into(),
            GgmlType::F64 => "F64".into(),
            GgmlType::IQ1M => "IQ1_M".into(),
            GgmlType::BF16 => "BF16".into(),
            GgmlType::BitNetI2S => "I2_S (BitNet)".into(),
            GgmlType::BitNetI8S => "I8_S (BitNet)".into(),
            GgmlType::BitNetTL1 => "TL1 (BitNet)".into(),
            GgmlType::BitNetTL2 => "TL2 (BitNet)".into(),
            GgmlType::Unknown(v) => format!("Unknown({})", v),
        }
    }

    /// Returns true if this is a BitNet-specific quantisation type.
    pub fn is_bitnet(&self) -> bool {
        matches!(
            self,
            GgmlType::BitNetI2S | GgmlType::BitNetI8S | GgmlType::BitNetTL1 | GgmlType::BitNetTL2
        )
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ── GGUF metadata value type tags (spec v3) ──

/// The type-tag byte that precedes each metadata value in a GGUF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufMetadataValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
    Unknown(u32),
}

impl GgufMetadataValueType {
    pub fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            other => Self::Unknown(other),
        }
    }
}
