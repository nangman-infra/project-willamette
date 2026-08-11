use project_willamette::model::architecture::{
    resolve, ForwardVariant, LayerTensorRole, ModelArchitecture,
};
use project_willamette::model::BitNetConfig;

fn assert_bitnet_contract(architecture: &dyn ModelArchitecture, alias: &str) {
    assert!(architecture.architecture_strings().contains(&alias));
    assert_eq!(architecture.metadata_prefix(alias), alias);
    assert_eq!(
        architecture.forward_variant(),
        ForwardVariant::BitNetSubNorm
    );

    let roles = architecture.layer_tensor_roles();
    assert_eq!(roles.len(), 11);
    assert!(roles.contains(&LayerTensorRole::AttnSubNorm));
    assert!(roles.contains(&LayerTensorRole::FfnSubNorm));
}

#[test]
fn bitnet_family_aliases_share_the_graph_contract() {
    for alias in [BitNetConfig::ARCHITECTURE, "bitnet-25", "bitnet"] {
        let architecture = resolve(alias).expect("BitNet alias must resolve");
        assert_bitnet_contract(architecture, alias);
    }
}

#[test]
fn layer_roles_map_to_expected_gguf_suffixes() {
    let architecture = resolve(BitNetConfig::ARCHITECTURE).unwrap();
    let suffixes = architecture
        .layer_tensor_roles()
        .iter()
        .map(|role| role.suffix())
        .collect::<Vec<_>>();

    assert_eq!(
        suffixes,
        [
            "attn_norm",
            "attn_sub_norm",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_norm",
            "ffn_sub_norm",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ]
    );
}

#[test]
fn unregistered_architectures_remain_rejected() {
    assert!(resolve("phi3").is_none());
    assert!(resolve("gemma").is_none());
}

#[test]
fn llama_declares_the_vanilla_nine_role_contract() {
    let architecture = resolve("llama").expect("Llama must resolve");
    assert_eq!(architecture.metadata_prefix("llama"), "llama");
    assert_eq!(architecture.forward_variant(), ForwardVariant::VanillaLlama);
    assert_eq!(architecture.layer_tensor_roles().len(), 9);
    assert!(!architecture
        .layer_tensor_roles()
        .contains(&LayerTensorRole::AttnSubNorm));
    assert!(!architecture
        .layer_tensor_roles()
        .contains(&LayerTensorRole::FfnSubNorm));
}
