//! CI's half of test 1 for the token expert kernels: their twins against decode's twins applied token
//! by token (bit for bit), and `moe_unique`'s list.
use kernels::{
    dispatch::SlotLayout,
    testing::{rand_f32, random_weights},
    twin::moe,
};
use qwen36::{quant::WType, testing::Rng};

/// `n` random experts of `lay`, packed as a slot holds them.
fn experts(rng: &mut Rng, lay: &SlotLayout, n: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|_| {
            let mut s = random_weights(rng, lay.gate, lay.n_ff, lay.n_embd);
            s.extend(random_weights(rng, lay.up, lay.n_ff, lay.n_embd));
            s.extend(random_weights(rng, lay.down, lay.n_embd, lay.n_ff));
            s
        })
        .collect()
}

/// `[tokens][k]` routed ids as f32: each token's `k` distinct experts out of `n_expert`, drawn at random, so tokens
/// share some experts and not others.
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

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn layouts() -> [SlotLayout; 3] {
    [
        // The real model's types (experts' gate and up Q4_K, down Q5_K, Q6_K in a few layers) at the tiny shapes.
        SlotLayout {
            gate: WType::Q4K,
            up: WType::Q4K,
            down: WType::Q5K,
            n_embd: 256,
            n_ff: 256,
        },
        SlotLayout {
            gate: WType::Q4K,
            up: WType::Q4K,
            down: WType::Q6K,
            n_embd: 256,
            n_ff: 256,
        },
        // A mix of the rest.
        SlotLayout {
            gate: WType::Q8_0,
            up: WType::Bf16,
            down: WType::F32,
            n_embd: 256,
            n_ff: 256,
        },
    ]
}

#[test]
fn the_tokens_twins_are_decodes_per_token() {
    // Token i's rows are what decode's twin gives that token alone, in its routed order: the twins differ from decode's
    // in the grouping (each distinct expert once, the tokens of its mask) and in where each token's rows land.
    let mut rng = Rng::new(70);
    let (k, n_expert) = (8, 12);
    for lay in layouts() {
        let slots = experts(&mut rng, &lay, n_expert);
        for tokens in [1, 2, 5, 16] {
            let case = format!(
                "{}/{}/{} t={tokens}",
                lay.gate.name(),
                lay.up.name(),
                lay.down.name()
            );
            let x = rand_f32(&mut rng, tokens * lay.n_embd, -1.0, 1.0);
            let act = rand_f32(&mut rng, tokens * k * lay.n_ff, -1.0, 1.0);
            let ids = routes(&mut rng, tokens, k, n_expert);
            let slot = |e: usize| slots[e].as_slice();

            let got = moe::experts_gate_up_tokens(slot, &lay, &x, &ids, k, tokens);
            let want: Vec<f32> = (0..tokens)
                .flat_map(|i| {
                    let chosen: Vec<&[u8]> = ids[i * k..(i + 1) * k]
                        .iter()
                        .map(|&e| slots[e as usize].as_slice())
                        .collect();
                    moe::experts_gate_up(&chosen, &lay, &x[i * lay.n_embd..(i + 1) * lay.n_embd])
                })
                .collect();
            assert_eq!(got.len(), tokens * k * lay.n_ff, "{case}");
            assert!(
                same_bits(&got, &want),
                "{case}: gate/up differs from decode's twin"
            );

            let got = moe::experts_down_tokens(slot, &lay, &act, &ids, k, tokens);
            let want: Vec<f32> = (0..tokens)
                .flat_map(|i| {
                    let chosen: Vec<&[u8]> = ids[i * k..(i + 1) * k]
                        .iter()
                        .map(|&e| slots[e as usize].as_slice())
                        .collect();
                    moe::experts_down(
                        &chosen,
                        &lay,
                        &act[i * k * lay.n_ff..(i + 1) * k * lay.n_ff],
                    )
                })
                .collect();
            assert_eq!(got.len(), tokens * k * lay.n_embd, "{case}");
            assert!(
                same_bits(&got, &want),
                "{case}: down differs from decode's twin"
            );
        }
    }
}

#[test]
fn moe_unique_counts_shared_and_disjoint_routes() {
    let k = 8;
    for t in [1usize, 2, 5, 16] {
        // Every token on the same 8 experts, in a different order each: 8 unique, every mask all t tokens.
        let mut ids = Vec::new();
        for i in 0..t {
            ids.extend((0..k).map(|j| ((i + j) % k) as f32));
        }
        let list = moe::moe_unique(&ids, t, k);
        assert_eq!(list.len(), 8, "t={t}: shared routes");
        assert!(
            list.iter().all(|&(_, mask)| mask == (1u32 << t) - 1),
            "t={t}: every token chose each of the 8"
        );

        // Disjoint routes: token i on experts 8i .. 8i + 8, so 8·t unique, one bit each.
        let ids: Vec<f32> = (0..t * k).map(|v| v as f32).collect();
        let list = moe::moe_unique(&ids, t, k);
        assert_eq!(list.len(), 8 * t, "t={t}: disjoint routes");
        for (u, &(expert, mask)) in list.iter().enumerate() {
            assert_eq!(expert as usize, u, "t={t}: first-appearance order");
            assert_eq!(
                mask,
                1u32 << (u / k),
                "t={t}: expert {u} was chosen by one token"
            );
        }
    }
}

#[test]
fn moe_unique_lists_experts_in_first_appearance_order_with_one_bit_per_token() {
    // Token 0 routes to 5 3 9 1, token 1 to 3 7 5 0, token 2 to 2 5 3 8.
    let ids = [5., 3., 9., 1., 3., 7., 5., 0., 2., 5., 3., 8.];
    let list = moe::moe_unique(&ids, 3, 4);
    assert_eq!(
        list,
        vec![
            (5, 0b111),
            (3, 0b111),
            (9, 0b001),
            (1, 0b001),
            (7, 0b010),
            (0, 0b010),
            (2, 0b100),
            (8, 0b100),
        ]
    );
}
