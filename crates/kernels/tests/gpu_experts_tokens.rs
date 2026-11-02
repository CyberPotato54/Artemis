//! Test 1 on the GPU for the token expert kernels: `moe_unique`, `experts_gate_up_tokens` and
//! `experts_down_tokens` against their twins, and against decode's `experts_gate_up` / `experts_down` run token by
//! token. Each expert is read once for all the tokens that chose it, from HOST and VRAM slots alike.
use kernels::{
    dispatch::{
        read_status, slot_entry, ExpertStatus, SlotLayout, STATUS_BYTES, STATUS_CLEAR, WHERE_CPU,
        WHERE_HOST, WHERE_VRAM,
    },
    testing::{check, identity, rand_f32, random_weights, Gpu},
    tokens::moe_unique_bytes,
    twin::moe,
    Kernel,
};
use qwen36::{quant::WType, testing::Rng};

const LAYER: usize = 1;

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The real model's types at its shapes (gate/up Q4_K, down Q5_K, and Q6_K in the last layers), and a mix of the rest
/// at the tiny model's.
fn layouts() -> Vec<SlotLayout> {
    let real = |down| SlotLayout {
        gate: WType::Q4K,
        up: WType::Q4K,
        down,
        n_embd: 2048,
        n_ff: 512,
    };
    vec![
        real(WType::Q5K),
        real(WType::Q6K),
        SlotLayout {
            gate: WType::Q8_0,
            up: WType::Bf16,
            down: WType::F32,
            n_embd: 256,
            n_ff: 256,
        },
    ]
}

fn case_name(lay: &SlotLayout, tokens: usize) -> String {
    format!(
        "{}/{}/{} {} t={tokens}",
        lay.gate.name(),
        lay.up.name(),
        lay.down.name(),
        lay.n_embd
    )
}

/// `n` random experts of `lay` packed in one buffer (VRAM or pinned host memory), and a slot table for layer `LAYER`
/// whose entries point at them with `loc`; layer 0's entries are garbage that must never be read. Returns (table, the
/// slots' bytes).
fn setup(
    g: &Gpu,
    rng: &mut Rng,
    lay: &SlotLayout,
    n: usize,
    loc: u32,
) -> (vk::Slice, Vec<Vec<u8>>) {
    let slots: Vec<Vec<u8>> = (0..n)
        .map(|_| {
            let mut s = random_weights(rng, lay.gate, lay.n_ff, lay.n_embd);
            s.extend(random_weights(rng, lay.up, lay.n_ff, lay.n_embd));
            s.extend(random_weights(rng, lay.down, lay.n_embd, lay.n_ff));
            s
        })
        .collect();
    let packed: Vec<u8> = slots.concat();
    let at = if loc == WHERE_VRAM {
        g.bytes(&packed)
    } else {
        g.host_bytes(&packed)
    };
    let mut table = vec![0xEEu8; 2 * n * 16];
    for e in 0..n {
        let entry = slot_entry(at.addr + (e * lay.bytes()) as u64, loc);
        table[(n + e) * 16..(n + e + 1) * 16].copy_from_slice(&entry);
    }
    (g.bytes(&table), slots)
}

/// `[tokens][k]` routed ids as f32: each token's `k` distinct experts of `n_expert`, so tokens share some experts and
/// not others.
fn routes(rng: &mut Rng, tokens: usize, k: usize, n_expert: usize) -> Vec<f32> {
    let mut ids = Vec::new();
    for _ in 0..tokens {
        let mut pool: Vec<usize> = (0..n_expert).collect();
        for _ in 0..k {
            let at = (rng.next_u64() % pool.len() as u64) as usize;
            ids.push(pool.swap_remove(at) as f32);
        }
    }
    ids
}

fn status_of(g: &Gpu, status: vk::Slice) -> ExpertStatus {
    let b: [u8; STATUS_BYTES] = g.dev.download(status).unwrap().try_into().unwrap();
    read_status(&b)
}

/// What the three token kernels wrote.
struct Out {
    list: Vec<u32>,
    act: Vec<f32>,
    ye: Vec<f32>,
}

/// Runs `moe_unique`, then the gate/up and down kernels (on the gate/up kernel's own activations) over `[tokens][k]`
/// `ids` and `[tokens][n_embd]` `x`. Every output buffer starts as garbage no kernel is allowed to leave.
#[allow(clippy::too_many_arguments)]
fn run_tokens(
    g: &Gpu,
    table: vk::Slice,
    status: vk::Slice,
    lay: &SlotLayout,
    n: usize,
    ids: &[f32],
    x: &[f32],
    k: usize,
    tokens: usize,
) -> Out {
    let list_bytes = moe_unique_bytes(tokens, k) as usize;
    let (gids, gx) = (g.f32s(ids), g.f32s(x));
    let list = g.bytes(&vec![0xEEu8; list_bytes]);
    let act = g.f32s(&vec![7.0; tokens * k * lay.n_ff]);
    let ye = g.f32s(&vec![7.0; tokens * k * lay.n_embd]);
    g.run(|kk, r| {
        kk.moe_unique(r, gids.addr, list.addr, tokens, k);
        r.barrier();
        kk.experts_gate_up_tokens(
            r,
            table.addr,
            LAYER,
            n,
            gids.addr,
            list.addr,
            gx.addr,
            act.addr,
            status.addr,
            lay,
            k,
            tokens,
        );
        r.barrier();
        kk.experts_down_tokens(
            r,
            table.addr,
            LAYER,
            n,
            gids.addr,
            list.addr,
            act.addr,
            ye.addr,
            status.addr,
            lay,
            k,
            tokens,
        );
    });
    let list = g
        .dev
        .download(list)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    Out {
        list,
        act: g.read(act),
        ye: g.read(ye),
    }
}

/// The kernel's list, as the twin's (expert, mask) pairs: the count word, then that many entries.
fn listed(list: &[u32]) -> Vec<(u32, u32)> {
    list[1..1 + 2 * list[0] as usize]
        .chunks(2)
        .map(|p| (p[0], p[1]))
        .collect()
}

const N_EXPERT: usize = 20;
const K: usize = 8;
const TOKENS: [usize; 4] = [1, 2, 5, 16];

#[test]
#[ignore = "needs the GPU"]
fn experts_tokens_against_their_twins() {
    let g = Gpu::new();
    let mut rng = Rng::new(71);
    for lay in layouts() {
        for loc in [WHERE_HOST, WHERE_VRAM] {
            // The same seed for both, so the HOST and VRAM slots hold the same experts.
            let (table, slots) = setup(&g, &mut Rng::new(99), &lay, N_EXPERT, loc);
            let slot = |e: usize| slots[e].as_slice();
            for tokens in TOKENS {
                let case = format!(
                    "{} {}",
                    case_name(&lay, tokens),
                    if loc == WHERE_HOST { "host" } else { "vram" }
                );
                let ids = routes(&mut rng, tokens, K, N_EXPERT);
                let x = rand_f32(&mut rng, tokens * lay.n_embd, -1.0, 1.0);
                let status = g.bytes(&STATUS_CLEAR);
                let out = run_tokens(&g, table, status, &lay, N_EXPERT, &ids, &x, K, tokens);

                let want = moe::moe_unique(&ids, tokens, K);
                assert_eq!(listed(&out.list), want, "{case}: moe_unique's list");
                let t_act = moe::experts_gate_up_tokens(slot, &lay, &x, &ids, K, tokens);
                check(Kernel::ExpertsGateUpTokens, &case, &out.act, &t_act);
                let t_ye = moe::experts_down_tokens(slot, &lay, &out.act, &ids, K, tokens);
                check(Kernel::ExpertsDownTokens, &case, &out.ye, &t_ye);
                // On the kernel's own activations, down is its twin bit for bit (unit_dot's sums are `precise`).
                assert!(
                    same_bits(&out.ye, &t_ye),
                    "{case}: experts_down_tokens differs from its twin"
                );
                assert_eq!(
                    status_of(&g, status),
                    ExpertStatus::default(),
                    "{case}: nothing was refused"
                );
            }
        }
    }
}

/// `moe_unique` alone over `ids`: the kernel's list, as the twin's (expert, mask) pairs. It needs no slots.
fn unique_on_gpu(g: &Gpu, ids: &[f32], tokens: usize, k: usize) -> Vec<(u32, u32)> {
    let gids = g.f32s(ids);
    let list = g.bytes(&vec![0xEEu8; moe_unique_bytes(tokens, k) as usize]);
    g.run(|kk, r| kk.moe_unique(r, gids.addr, list.addr, tokens, k));
    let list: Vec<u32> = g
        .dev
        .download(list)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    listed(&list)
}

#[test]
#[ignore = "needs the GPU"]
fn moe_unique_lists_up_to_128_experts_at_16_tokens() {
    // The routed tests above never list more than N_EXPERT = 20; the real model lists up to 16 tokens x 8 = 128.
    let g = Gpu::new();
    let k = 8;
    for t in [2usize, 5, 16] {
        // Every token on the same 8 experts, in a different order each: 8 listed, every mask all t tokens.
        let mut ids = Vec::new();
        for i in 0..t {
            ids.extend((0..k).map(|j| ((i + j) % k) as f32));
        }
        let got = unique_on_gpu(&g, &ids, t, k);
        assert_eq!(got.len(), 8, "t={t}: shared routes list 8");
        assert_eq!(got, moe::moe_unique(&ids, t, k), "t={t}: shared routes");

        // Disjoint routes: token i on experts 8i .. 8i + 8, so 8·t listed (128 at t = 16), one bit each.
        let ids: Vec<f32> = (0..t * k).map(|v| v as f32).collect();
        let got = unique_on_gpu(&g, &ids, t, k);
        assert_eq!(got.len(), 8 * t, "t={t}: disjoint routes list 8·t");
        assert_eq!(got, moe::moe_unique(&ids, t, k), "t={t}: disjoint routes");
    }
}

#[test]
#[ignore = "needs the GPU"]
fn experts_tokens_host_equals_vram_bit_for_bit() {
    // Test 5's rule: where an expert lives never changes the arithmetic.
    let g = Gpu::new();
    let mut rng = Rng::new(72);
    for lay in layouts() {
        for tokens in TOKENS {
            let case = case_name(&lay, tokens);
            let ids = routes(&mut rng, tokens, K, N_EXPERT);
            let x = rand_f32(&mut rng, tokens * lay.n_embd, -1.0, 1.0);
            let outs: Vec<Out> = [WHERE_HOST, WHERE_VRAM]
                .into_iter()
                .map(|loc| {
                    let (table, _) = setup(&g, &mut Rng::new(99), &lay, N_EXPERT, loc);
                    let status = g.bytes(&STATUS_CLEAR);
                    run_tokens(&g, table, status, &lay, N_EXPERT, &ids, &x, K, tokens)
                })
                .collect();
            assert!(
                same_bits(&outs[0].act, &outs[1].act),
                "{case}: HOST and VRAM gate/up must give bit-identical results"
            );
            assert!(
                same_bits(&outs[0].ye, &outs[1].ye),
                "{case}: HOST and VRAM down must give bit-identical results"
            );
            assert_eq!(outs[0].list, outs[1].list, "{case}: the same list");
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn experts_tokens_write_decodes_bits_per_token() {
    // Reported, not asserted: token i's rows against decode's own kernels run on that token alone.
    let g = Gpu::new();
    let mut rng = Rng::new(73);
    for lay in layouts() {
        let (table, _) = setup(&g, &mut Rng::new(99), &lay, N_EXPERT, WHERE_HOST);
        for tokens in TOKENS {
            let case = case_name(&lay, tokens);
            let ids = routes(&mut rng, tokens, K, N_EXPERT);
            let x = rand_f32(&mut rng, tokens * lay.n_embd, -1.0, 1.0);
            let status = g.bytes(&STATUS_CLEAR);
            let out = run_tokens(&g, table, status, &lay, N_EXPERT, &ids, &x, K, tokens);
            let (gids, gx) = (g.f32s(&ids), g.f32s(&x));
            let act = g.f32s(&out.act);
            let (mut same_act, mut same_ye) = (true, true);
            for i in 0..tokens {
                let (want_act, want_ye) = (g.zeros(K * lay.n_ff), g.zeros(K * lay.n_embd));
                let ids_i = gids.addr + (4 * i * K) as u64;
                let x_i = gx.addr + (4 * i * lay.n_embd) as u64;
                let act_i = act.addr + (4 * i * K * lay.n_ff) as u64;
                g.run(|kk, r| {
                    kk.experts_gate_up(
                        r,
                        table.addr,
                        LAYER,
                        N_EXPERT,
                        ids_i,
                        x_i,
                        want_act.addr,
                        status.addr,
                        &lay,
                        K,
                        0,
                        0,
                        0,
                    );
                    kk.experts_down(
                        r,
                        table.addr,
                        LAYER,
                        N_EXPERT,
                        ids_i,
                        act_i,
                        want_ye.addr,
                        status.addr,
                        &lay,
                        K,
                        0,
                        0,
                        0,
                    );
                });
                let at = i * K * lay.n_ff..(i + 1) * K * lay.n_ff;
                same_act &= same_bits(&out.act[at], &g.read(want_act));
                let at = i * K * lay.n_embd..(i + 1) * K * lay.n_embd;
                same_ye &= same_bits(&out.ye[at], &g.read(want_ye));
            }
            identity(Kernel::ExpertsGateUpTokens.name(), &case, same_act);
            identity(Kernel::ExpertsDownTokens.name(), &case, same_ye);
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn experts_tokens_refuse_an_unusable_slot() {
    let g = Gpu::new();
    let mut rng = Rng::new(74);
    let lay = SlotLayout {
        gate: WType::Q4K,
        up: WType::Q4K,
        down: WType::Q5K,
        n_embd: 2048,
        n_ff: 512,
    };
    let (n, k, tokens) = (6, 4, 3);
    let (table, _) = setup(&g, &mut rng, &lay, n, WHERE_HOST);
    // Layer 1's experts 5 and 2 are CPU entries (null address, never read), 3 has a `where` nothing defines, and
    // 1 is HOST with no address. Experts 4 and 0 are fine.
    let mut t = g.dev.download(table).unwrap();
    let at3 = u64::from_le_bytes(t[(n + 3) * 16..(n + 3) * 16 + 8].try_into().unwrap());
    for e in [5, 2] {
        t[(n + e) * 16..(n + e + 1) * 16].copy_from_slice(&slot_entry(0, WHERE_CPU));
    }
    t[(n + 3) * 16..(n + 4) * 16].copy_from_slice(&slot_entry(at3, 7));
    t[(n + 1) * 16..(n + 2) * 16].copy_from_slice(&slot_entry(0, WHERE_HOST));
    let table = g.bytes(&t);
    #[rustfmt::skip]
    let ids = [
        4.0, 0.0, 5.0, 2.0,
        5.0, 1.0, 2.0, 3.0,
        4.0, 3.0, 0.0, 1.0,
    ];
    let x = rand_f32(&mut rng, tokens * lay.n_embd, -1.0, 1.0);
    // Several rounds: the refused experts' workgroups run in parallel and the status names the lowest, every time.
    for round in 0..8 {
        let status = g.bytes(&STATUS_CLEAR);
        let out = run_tokens(&g, table, status, &lay, n, &ids, &x, k, tokens);
        assert_eq!(
            status_of(&g, status),
            ExpertStatus {
                cpu: Some((LAYER as u32, 2)),
                bad: Some((LAYER as u32, 1)),
            },
            "round {round}: the lowest CPU and the lowest bad expert"
        );
        for (p, &e) in ids.iter().enumerate() {
            let usable = e == 4.0 || e == 0.0;
            let act = &out.act[p * lay.n_ff..(p + 1) * lay.n_ff];
            let ye = &out.ye[p * lay.n_embd..(p + 1) * lay.n_embd];
            if usable {
                assert!(
                    act.iter().chain(ye).all(|v| v.is_finite()),
                    "round {round}: expert {e} (choice {p}) still runs"
                );
            } else {
                assert!(
                    act.iter().chain(ye).all(|v| v.is_nan()),
                    "round {round}: expert {e} (choice {p}) is NaN"
                );
            }
        }
    }
}

/// Calls each token kernel's dispatch with `tokens` tokens of 8 choices; the shape check refuses before anything is
/// recorded, so the buffers are placeholders.
fn dispatch_with(which: Kernel, tokens: usize) {
    let g = Gpu::new();
    let lay = SlotLayout {
        gate: WType::Q4K,
        up: WType::Q4K,
        down: WType::Q5K,
        n_embd: 256,
        n_ff: 256,
    };
    let buf = g.zeros(4096);
    let a = buf.addr;
    g.run(|kk, r| match which {
        Kernel::MoeUnique => kk.moe_unique(r, a, a, tokens, 8),
        Kernel::ExpertsGateUpTokens => {
            kk.experts_gate_up_tokens(r, a, 1, 6, a, a, a, a, a, &lay, 8, tokens)
        }
        _ => kk.experts_down_tokens(r, a, 1, 6, a, a, a, a, a, &lay, 8, tokens),
    });
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "moe_unique: 0 tokens")]
fn moe_unique_refuses_no_tokens() {
    dispatch_with(Kernel::MoeUnique, 0);
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "moe_unique: 17 tokens")]
fn moe_unique_refuses_seventeen_tokens() {
    dispatch_with(Kernel::MoeUnique, 17);
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "experts_gate_up_tokens: 0 tokens")]
fn experts_gate_up_tokens_refuses_no_tokens() {
    dispatch_with(Kernel::ExpertsGateUpTokens, 0);
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "experts_gate_up_tokens: 17 tokens")]
fn experts_gate_up_tokens_refuses_seventeen_tokens() {
    dispatch_with(Kernel::ExpertsGateUpTokens, 17);
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "experts_down_tokens: 0 tokens")]
fn experts_down_tokens_refuses_no_tokens() {
    dispatch_with(Kernel::ExpertsDownTokens, 0);
}

#[test]
#[ignore = "needs the GPU"]
#[should_panic(expected = "experts_down_tokens: 17 tokens")]
fn experts_down_tokens_refuses_seventeen_tokens() {
    dispatch_with(Kernel::ExpertsDownTokens, 17);
}
