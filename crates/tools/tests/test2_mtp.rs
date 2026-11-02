//! Test 2's MTP half, in CI: synthetic tap sets and the tiny model, no real model and no oracle.
use artemis_tools::{
    tapdir::{self, TapSet},
    test2::{dump_mtp, forced_bound, test2_mtp, Status, FORCED_BOUND, FORCED_BOUND_ATTN_PREGATE},
};
use gguf::Gguf;
use qwen36::{
    dims::Dims,
    model::Model,
    reference::{MtpReference, Reference, Rounding},
    taps::{expected_mtp_taps, mtp_llama_filter, MtpKey, NullTap, Tap, TapName},
    testing::{tiny_model, Rng},
};
use std::{fs, path::PathBuf};

const ROWS: usize = 3;

fn temp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("artemis-test2-mtp-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

/// A full MTP tap set for the tiny model: distinct routed experts and router logits 0, 1, …, 15 (every gap is 1).
fn base(seed: u64) -> TapSet {
    let d = Dims::tiny();
    let mut rng = Rng::new(seed);
    let mut set = TapSet {
        n_tokens: ROWS,
        ..Default::default()
    };
    for t in expected_mtp_taps() {
        let n = t.per_pos(&d) * ROWS;
        let data: Vec<f32> = match t {
            TapName::Mtp(MtpKey::TopK) => (0..ROWS)
                .flat_map(|p| (0..4).map(move |j| ((p + 3 * j) % 16) as f32))
                .collect(),
            TapName::Mtp(MtpKey::TopKW) => vec![0.25; n],
            TapName::Mtp(MtpKey::RouterLogits) => (0..n).map(|i| (i % 16) as f32).collect(),
            _ => (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect(),
        };
        set.taps.insert(t, data);
    }
    set
}

fn scaled(s: &TapSet, key: MtpKey, eps: f32) -> TapSet {
    let mut out = s.clone();
    for x in out.taps.get_mut(&TapName::Mtp(key)).unwrap() {
        *x *= 1.0 + eps;
    }
    out
}

#[test]
fn identical_mtp_taps_pass() {
    let d = Dims::tiny();
    let s = base(1);
    let v = test2_mtp(&s, &s, &d).unwrap();
    assert_eq!(v.status, Status::Pass, "{v:#?}");
    assert_eq!(v.rows.len(), 24, "every tap but the routed ids is a row");
    assert!(v.missing.is_empty() && v.topk_mismatches.is_empty());
}

#[test]
fn a_tap_past_its_bound_fails_by_name() {
    let d = Dims::tiny();
    let s = base(2);
    // 1e-4 is ten times the 1e-5 bound of a projection, and the shared expert's SwiGLU is one of the new taps.
    for key in [MtpKey::EhProj, MtpKey::ShexpSwiglu, MtpKey::Logits] {
        let v = test2_mtp(&s, &scaled(&s, key, 1e-4), &d).unwrap();
        assert_eq!(v.status, Status::Fail, "{key:?}");
        let failing: Vec<&str> = v
            .rows
            .iter()
            .filter(|r| !r.pass)
            .map(|r| r.tap.as_str())
            .collect();
        assert_eq!(failing, [format!("mtp.{}", key.name())], "{key:?}");
    }
}

#[test]
fn the_attention_pregate_has_the_attention_bound() {
    let d = Dims::tiny();
    let s = base(3);
    assert_eq!(
        forced_bound(TapName::Mtp(MtpKey::AttnPregate)),
        FORCED_BOUND_ATTN_PREGATE
    );
    assert_eq!(forced_bound(TapName::Mtp(MtpKey::EhProj)), FORCED_BOUND);
    // 5e-4: past a projection's bound, inside the flash attention's.
    let v = test2_mtp(&s, &scaled(&s, MtpKey::AttnPregate, 5e-4), &d).unwrap();
    assert_eq!(v.status, Status::Pass, "{v:#?}");
    let v = test2_mtp(&s, &scaled(&s, MtpKey::AttnPregate, 5e-3), &d).unwrap();
    assert_eq!(v.status, Status::Fail);
}

#[test]
fn a_missing_mtp_tap_fails() {
    let d = Dims::tiny();
    let s = base(4);
    let mut cand = s.clone();
    cand.taps.remove(&TapName::Mtp(MtpKey::MoeSwiglu));
    let v = test2_mtp(&s, &cand, &d).unwrap();
    assert_eq!(v.status, Status::Fail);
    assert!(
        v.missing.iter().any(|m| m.contains("mtp.moe_swiglu")),
        "{:?}",
        v.missing
    );
    // And one the oracle lacks.
    let mut oracle = s.clone();
    oracle.taps.remove(&TapName::Mtp(MtpKey::RouterProbs));
    let v = test2_mtp(&oracle, &s, &d).unwrap();
    assert_eq!(v.status, Status::Fail);
    assert!(v.missing.iter().any(|m| m.contains("mtp.router_probs")));
}

#[test]
fn a_routing_swap_outside_a_tie_fails_and_one_within_a_tie_is_reported() {
    let d = Dims::tiny();
    let s = base(5);
    let ids = TapName::Mtp(MtpKey::TopK);
    // Row 0 routes {0, 3, 6, 9}; logits 0..15 put the 4th largest (the cutoff) at expert 12 in a gap-1 field, so
    // naming expert 15 instead of 9 isn't a tie.
    let mut cand = s.clone();
    cand.taps.get_mut(&ids).unwrap()[3] = 15.0;
    let v = test2_mtp(&s, &cand, &d).unwrap();
    assert_eq!(v.status, Status::Fail);
    assert_eq!(v.topk_mismatches, vec![(d.n_layer as u16, 0)]);
    assert!(v.near_ties.is_empty());

    // Make experts 9 and 15 tie with the cutoff in the oracle's logits: the same swap is a reported near-tie.
    let mut oracle = s.clone();
    let n = d.n_expert;
    let kth = {
        let mut v = s.taps[&TapName::Mtp(MtpKey::RouterLogits)][..n].to_vec();
        v.sort_by(|a, b| b.total_cmp(a));
        v[3]
    };
    for set in [&mut oracle, &mut cand] {
        let logits = set
            .taps
            .get_mut(&TapName::Mtp(MtpKey::RouterLogits))
            .unwrap();
        logits[9] = kth;
        logits[15] = kth;
    }
    let v = test2_mtp(&oracle, &cand, &d).unwrap();
    assert_eq!(v.status, Status::Pass, "{v:#?}");
    assert_eq!(v.near_ties.len(), 1);
    assert_eq!((v.near_ties[0].0, v.near_ties[0].1), (d.n_layer as u16, 0));
}

#[test]
fn the_routing_weights_are_paired_with_the_oracles_experts() {
    // The reference was forced with the oracle's TopK, so its weights are the oracle's experts' and are compared expert
    // by expert, whatever order the oracle listed them in.
    let d = Dims::tiny();
    let mut oracle = base(6);
    oracle.taps.insert(
        TapName::Mtp(MtpKey::TopKW),
        [0.1, 0.2, 0.3, 0.4].repeat(ROWS),
    );
    let mut cand = oracle.clone();
    let v = test2_mtp(&oracle, &cand, &d).unwrap();
    assert_eq!(v.status, Status::Pass, "{v:#?}");
    // A wrong weight for one expert fails.
    cand.taps.get_mut(&TapName::Mtp(MtpKey::TopKW)).unwrap()[0] *= 1.5;
    assert_eq!(test2_mtp(&oracle, &cand, &d).unwrap().status, Status::Fail);
}

#[test]
fn the_mtp_keys_map_to_the_oracles_names() {
    let d = Dims::QWEN36;
    let name = |k: MtpKey| TapName::Mtp(k).llama(&d).unwrap();
    // The names the oracle's callbacks use (b11509: qwen35moe.cpp's `mtp_*`, and llama-graph.cpp's own for the MoE).
    assert_eq!(name(MtpKey::RouterLogits), "ffn_moe_logits-40");
    assert_eq!(name(MtpKey::RouterProbs), "ffn_moe_probs-40");
    assert_eq!(name(MtpKey::TopK), "ffn_moe_topk-40");
    assert_eq!(name(MtpKey::TopKW), "ffn_moe_weights_norm-40");
    assert_eq!(name(MtpKey::MoeSwiglu), "ffn_moe_swiglu-40");
    assert_eq!(name(MtpKey::ShexpSwiglu), "ffn_swiglu-40");
    assert_eq!(name(MtpKey::Q), "mtp_Qcur_normed-40");
    assert_eq!(name(MtpKey::K), "mtp_Kcur_normed-40");
    assert_eq!(name(MtpKey::AttnPregate), "mtp_attn_pregate-40");
    assert_eq!(name(MtpKey::HeadNorm), "h_nextn");
    assert_eq!(name(MtpKey::Logits), "result_output");

    // The filter takes every one of those and none of the main graph's per-layer names.
    let re = mtp_llama_filter();
    for k in MtpKey::ALL {
        assert!(re.contains(k.llama()), "{k:?} is in the filter");
    }
    assert!(re.starts_with("^(") && re.ends_with(")(-[0-9]+)?$"), "{re}");
    assert!(!re.contains("attn_norm-"), "no layer suffix is baked in");

    // An oracle directory in llama.cpp's format reads back as the MTP set, last occurrence of a name winning.
    let tiny = Dims::tiny();
    let want = base(7);
    let dir = temp("names");
    fs::create_dir_all(&dir).unwrap();
    let mut entries = Vec::new();
    for (i, (t, data)) in want.taps.iter().enumerate() {
        let file = format!("{i:05}.bin");
        let (dtype, bytes): (&str, Vec<u8>) = if matches!(t, TapName::Mtp(MtpKey::TopK)) {
            (
                "i32",
                data.iter()
                    .flat_map(|&v| (v as i32).to_le_bytes())
                    .collect(),
            )
        } else {
            ("f32", data.iter().flat_map(|v| v.to_le_bytes()).collect())
        };
        fs::write(dir.join(&file), bytes).unwrap();
        let n = t.llama(&tiny).unwrap();
        // A stale first occurrence of the same name, to be overridden.
        if i == 0 {
            let stale = "stale.bin".to_string();
            fs::write(dir.join(&stale), vec![0u8; data.len() * 4]).unwrap();
            entries.push(format!(
                r#"{{"name":"{n}","occurrence":0,"ne":[1,1,1,1],"dtype":"f32","file":"{stale}"}}"#
            ));
        }
        entries.push(format!(
            r#"{{"name":"{n}","occurrence":{},"ne":[1,1,1,1],"dtype":"{dtype}","file":"{file}"}}"#,
            (i == 0) as u32
        ));
    }
    // A main-graph tensor the MTP table doesn't name is ignored.
    entries.push(
        r#"{"name":"attn_norm-3","occurrence":0,"ne":[1,1,1,1],"dtype":"f32","file":"stale.bin"}"#
            .into(),
    );
    fs::write(
        dir.join("index.json"),
        format!(
            r#"{{"source":"llama.cpp b11509","backend":"cpu","n_tokens":{ROWS},"tensors":[{}]}}"#,
            entries.join(",")
        ),
    )
    .unwrap();
    let got = tapdir::read_mtp(&dir, &tiny).unwrap();
    assert_eq!(got, want);
    fs::remove_dir_all(&dir).unwrap();
}

// ---- the dump (`artemis dump --cpu --mtp`) on the tiny model ----

const TOKENS: [u32; 6] = [10, 300, 7, 506, 42, 1];

/// The main model's final hidden row at each position: what `mtp-taps` writes as `hidden.bin`.
fn hidden_rows(m: &Model) -> Vec<f32> {
    let mut r = Reference::with_options(m, 2, Rounding::Exact);
    let mut out = Vec::new();
    for &t in &TOKENS {
        r.step(t, &mut NullTap).unwrap();
        out.extend_from_slice(r.hidden().unwrap());
    }
    out
}

#[test]
fn the_dump_pairs_each_token_with_the_previous_hidden_row() {
    let d = Dims::tiny();
    let g = Gguf::from_bytes(tiny_model(1)).unwrap();
    let m = Model::open(&g, &d).unwrap();
    let h = hidden_rows(&m);
    let set = dump_mtp(&m, &TOKENS, &h, None, 2, Rounding::Exact).unwrap();
    assert_eq!(set.n_tokens, TOKENS.len() - 1, "rows q = 1 .. n - 1");

    // Row i is position i + 1: token i + 1 with the hidden row at i, over the earlier rows only.
    struct Last(Vec<f32>);
    impl Tap for Last {
        fn tap(&mut self, name: TapName, _: u32, data: &[f32]) {
            if name == TapName::Mtp(MtpKey::Logits) {
                self.0 = data.to_vec();
            }
        }
    }
    let mut mtp = MtpReference::new(&m, 2, Rounding::Exact);
    let e = d.n_embd;
    let logits = &set.taps[&TapName::Mtp(MtpKey::Logits)];
    for q in 1..TOKENS.len() {
        let mut last = Last(Vec::new());
        mtp.row(q as u32, TOKENS[q], &h[(q - 1) * e..q * e], &mut last);
        assert_eq!(
            &logits[(q - 1) * d.n_vocab..q * d.n_vocab],
            last.0.as_slice(),
            "row {q}"
        );
    }
    // It is a complete MTP set, readable as one.
    let dir = temp("dump");
    tapdir::write(&set, &dir, "test").unwrap();
    assert_eq!(tapdir::read_mtp(&dir, &d).unwrap(), set);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_forced_dump_passes_against_the_run_it_was_forced_from() {
    let d = Dims::tiny();
    let g = Gguf::from_bytes(tiny_model(1)).unwrap();
    let m = Model::open(&g, &d).unwrap();
    let h = hidden_rows(&m);
    // Stand in for llama.cpp: the exact run, then the same run forced from it.
    let oracle = dump_mtp(&m, &TOKENS, &h, None, 2, Rounding::Exact).unwrap();
    let forced = dump_mtp(&m, &TOKENS, &h, Some(&oracle), 2, Rounding::Exact).unwrap();
    let v = test2_mtp(&oracle, &forced, &d).unwrap();
    assert_eq!(v.status, Status::Pass, "{v:#?}");
    assert!(
        v.rows.iter().all(|r| r.err == 0.0),
        "the same arithmetic, forced"
    );
}

#[test]
fn the_dump_refuses_inputs_that_disagree() {
    let d = Dims::tiny();
    let g = Gguf::from_bytes(tiny_model(1)).unwrap();
    let m = Model::open(&g, &d).unwrap();
    let h = hidden_rows(&m);
    let err = |r: anyhow::Result<TapSet>| format!("{:#}", r.unwrap_err());
    let e = err(dump_mtp(
        &m,
        &TOKENS,
        &h[d.n_embd..],
        None,
        2,
        Rounding::Exact,
    ));
    assert!(e.contains("hidden"), "{e}");
    let e = err(dump_mtp(
        &m,
        &TOKENS[..1],
        &h[..d.n_embd],
        None,
        2,
        Rounding::Exact,
    ));
    assert!(e.contains("at least 2"), "{e}");
    let wrong = base(8); // 3 rows, but 6 tokens make 5
    let e = err(dump_mtp(&m, &TOKENS, &h, Some(&wrong), 2, Rounding::Exact));
    assert!(e.contains("3") && e.contains("5"), "{e}");
    let e = err(dump_mtp(
        &m,
        &[1, d.n_vocab as u32],
        &h[..2 * d.n_embd],
        None,
        2,
        Rounding::Exact,
    ));
    assert!(e.contains("vocabulary"), "{e}");
}

#[test]
fn test2_mtp_refuses_a_reference_dumped_the_wrong_way() {
    use artemis_tools::test2::check_mtp_reference_source as check;
    assert!(check("artemis reference MTP (LlamaCpu rounding), teacher-forced").is_ok());
    let e = check("artemis reference MTP (LlamaCpu rounding)")
        .unwrap_err()
        .to_string();
    assert!(e.contains("--force-from"), "{e}");
    let e = check("artemis reference MTP (Exact rounding), teacher-forced")
        .unwrap_err()
        .to_string();
    assert!(e.contains("--llama-rounding"), "{e}");
    let e = check("artemis reference (LlamaCpu rounding), teacher-forced")
        .unwrap_err()
        .to_string();
    assert!(e.contains("--mtp"), "{e}");
    // The source string `dump --mtp` writes is read back by `tapdir::source`.
    let dir = temp("source");
    tapdir::write(
        &base(9),
        &dir,
        "artemis reference MTP (LlamaCpu rounding), teacher-forced",
    )
    .unwrap();
    check(&tapdir::source(&dir).unwrap()).unwrap();
    fs::remove_dir_all(&dir).unwrap();
}

/// An earlier miss: `wo`'s Q8_0 activation was computed by the reference from an unforced gate, so a 1e-7 difference
/// could flip a quantum. A forced `attn_out` must depend only on forced taps: here the oracle's `attn_norm` is
/// perturbed (so the reference's own gate, from `attn_q · attn_norm`, differs), and `attn_out` must still equal the
/// oracle's bit for bit.
#[test]
fn a_forced_attn_out_reads_only_forced_taps() {
    let d = Dims::tiny();
    let g = Gguf::from_bytes(tiny_model(1)).unwrap();
    let m = Model::open(&g, &d).unwrap();
    let h = hidden_rows(&m);
    let oracle = dump_mtp(&m, &TOKENS, &h, None, 2, Rounding::LlamaCpu).unwrap();
    let mut perturbed = oracle.clone();
    for (i, x) in perturbed
        .taps
        .get_mut(&TapName::Mtp(MtpKey::AttnNorm))
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        *x *= 1.0 + if i % 2 == 0 { 1e-3 } else { -1e-3 };
    }
    let forced = dump_mtp(&m, &TOKENS, &h, Some(&perturbed), 2, Rounding::LlamaCpu).unwrap();
    let key = TapName::Mtp(MtpKey::AttnOut);
    let bits = |s: &TapSet| s.taps[&key].iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&forced),
        bits(&oracle),
        "attn_out read an unforced input"
    );
}
