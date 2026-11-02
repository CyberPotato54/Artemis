use qwen36::{
    dims::Dims,
    taps::{expected_taps, llama_filter, Key, TapName},
};

#[test]
fn names_round_trip() {
    for t in expected_taps(&Dims::QWEN36) {
        let s = t.to_string();
        assert_eq!(s.parse::<TapName>(), Ok(t), "{s}");
    }
    assert_eq!(
        TapName::Layer(12, Key::AttnNorm).to_string(),
        "l12.attn_norm"
    );
    assert_eq!(TapName::Layer(0, Key::GdnQkv).to_string(), "l0.gdn.qkv");
    assert!("l3.nonsense".parse::<TapName>().is_err());
    assert!("x".parse::<TapName>().is_err());
}

#[test]
fn which_taps_each_layer_has() {
    // DeltaNet layers have 24 taps and attention layers 19, plus embd, final_norm and logits.
    assert_eq!(expected_taps(&Dims::tiny()).len(), 3 * 24 + 19 + 3);
    assert_eq!(expected_taps(&Dims::QWEN36).len(), 30 * 24 + 10 * 19 + 3);
    let taps = expected_taps(&Dims::QWEN36);
    assert!(
        taps.windows(2).all(|w| w[0] < w[1]),
        "expected_taps is in graph order"
    );
    assert_eq!(taps[0], TapName::Embd);
    assert_eq!(taps[taps.len() - 1], TapName::Logits);
}

#[test]
fn llama_names() {
    let d = Dims::QWEN36;
    let n = |t: TapName| t.llama(&d);
    assert_eq!(n(TapName::Embd).as_deref(), Some("model.input_embed"));
    assert_eq!(n(TapName::Logits).as_deref(), Some("result_output"));
    assert_eq!(
        n(TapName::Layer(3, Key::MixerOut)).as_deref(),
        Some("attn_output-3")
    );
    assert_eq!(
        n(TapName::Layer(0, Key::MixerOut)).as_deref(),
        Some("linear_attn_out-0")
    );
    assert_eq!(
        n(TapName::Layer(0, Key::GdnO)).as_deref(),
        Some("attn_output-0")
    );
    assert_eq!(n(TapName::Layer(7, Key::AttnQ)).as_deref(), Some("Qcur-7"));
    assert_eq!(n(TapName::Layer(39, Key::Out)).as_deref(), Some("l_out-39"));
    assert_eq!(
        n(TapName::Layer(5, Key::MoeSwiglu)).as_deref(),
        Some("ffn_moe_swiglu-5")
    );
    assert_eq!(
        n(TapName::Layer(5, Key::ShexpSwiglu)).as_deref(),
        Some("ffn_swiglu-5")
    );
    assert_eq!(n(TapName::Layer(3, Key::GdnQkv)), None);
    assert_eq!(n(TapName::Layer(0, Key::AttnQ)), None);
}

#[test]
fn values_per_position() {
    let d = Dims::QWEN36;
    assert_eq!(TapName::Layer(0, Key::GdnQkv).per_pos(&d), 8192);
    assert_eq!(TapName::Layer(0, Key::GdnQ).per_pos(&d), 2048);
    assert_eq!(TapName::Layer(0, Key::GdnO).per_pos(&d), 4096);
    assert_eq!(TapName::Layer(0, Key::GdnBeta).per_pos(&d), 32);
    assert_eq!(TapName::Layer(3, Key::AttnQ).per_pos(&d), 4096);
    assert_eq!(TapName::Layer(3, Key::AttnK).per_pos(&d), 512);
    assert_eq!(TapName::Layer(3, Key::RouterProbs).per_pos(&d), 256);
    assert_eq!(TapName::Layer(3, Key::TopK).per_pos(&d), 8);
    assert_eq!(TapName::Layer(3, Key::MoeSwiglu).per_pos(&d), 8 * 512);
    assert_eq!(TapName::Layer(3, Key::ShexpSwiglu).per_pos(&d), 512);
    assert_eq!(TapName::Logits.per_pos(&d), 248_320);
}

#[test]
fn the_oracle_filter_names_every_llama_tensor() {
    let f = llama_filter();
    assert!(
        f.starts_with("^(model\\.input_embed|result_norm|result_output|attn_norm|"),
        "{f}"
    );
    assert!(f.ends_with(")(-[0-9]+)?$"), "{f}");
    for t in expected_taps(&Dims::QWEN36) {
        let name = t.llama(&Dims::QWEN36).unwrap();
        let base = name
            .rsplit_once('-')
            .map_or(name.as_str(), |(b, _)| b)
            .replace('.', "\\.");
        assert!(
            f.contains(&format!("|{base}|"))
                || f.contains(&format!("({base}|"))
                || f.contains(&format!("|{base})")),
            "{base}"
        );
    }
}

#[test]
fn the_mtp_taps_have_names_sizes_and_oracle_names() {
    use qwen36::taps::{expected_mtp_taps, mtp_llama_filter, MtpKey};
    let d = Dims::QWEN36;
    let all = expected_mtp_taps();
    assert_eq!(all.len(), 25);
    assert!(all.windows(2).all(|w| w[0] < w[1]), "graph order");
    assert!(
        all.iter().all(|t| *t > TapName::Logits),
        "after the main taps"
    );
    for t in &all {
        assert_eq!(t.to_string().parse::<TapName>().as_ref(), Ok(t));
        assert!(
            !expected_taps(&d).contains(t),
            "the main list stays main-only"
        );
    }
    assert_eq!(TapName::Mtp(MtpKey::EhProj).to_string(), "mtp.eh_proj");
    assert!("mtp.nonsense".parse::<TapName>().is_err());
    let n = |k: MtpKey| TapName::Mtp(k).llama(&d).unwrap();
    assert_eq!(n(MtpKey::Enorm), "mtp_enorm-40");
    assert_eq!(n(MtpKey::Q), "mtp_Qcur_normed-40");
    assert_eq!(n(MtpKey::K), "mtp_Kcur_normed-40");
    assert_eq!(n(MtpKey::V), "mtp_Vcur-40");
    assert_eq!(n(MtpKey::AttnGated), "mtp_attn_gated-40");
    assert_eq!(n(MtpKey::ShexpGated), "mtp_ffn_shexp_gated-40");
    assert_eq!(n(MtpKey::PostFfn), "mtp_post_ffn-40");
    assert_eq!(n(MtpKey::HeadNorm), "h_nextn");
    assert_eq!(n(MtpKey::Logits), "result_output");
    assert_eq!(TapName::Mtp(MtpKey::Concat).per_pos(&d), 4096);
    assert_eq!(TapName::Mtp(MtpKey::Q).per_pos(&d), 4096);
    assert_eq!(TapName::Mtp(MtpKey::K).per_pos(&d), 512);
    assert_eq!(TapName::Mtp(MtpKey::Logits).per_pos(&d), 248_320);
    let f = mtp_llama_filter();
    for k in MtpKey::ALL {
        assert!(f.contains(k.llama()), "{}", k.llama());
    }
    assert!(
        f.ends_with(")(-[0-9]+)?$") && f.starts_with("^(mtp_enorm|"),
        "{f}"
    );
    // The main filter stays main-only.
    assert!(!llama_filter().contains("mtp_"));
}
