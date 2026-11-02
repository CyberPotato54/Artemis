//! Twins of the MoE kernels.
use super::{
    matvec::{dot_in_order, dot_split_pairs, row_dot, split_all},
    sigmoid, silu,
    silu_det::silu_det,
    tree, tree_max,
};
use crate::dispatch::{slot_entry, SlotLayout, ENTRY_BYTES, MOE_TILE, UPDATE_BYTES};
use crate::tokens::MAX_TOKENS;

/// `router_topk`: (probabilities, ids as f32, weights).
pub fn router_topk(logits: &[f32], k: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = logits.len();
    assert!(
        n <= 256 && k <= 16,
        "router_topk: at most 256 experts and 16 chosen"
    );
    let mut xs = vec![f32::NEG_INFINITY; 256];
    xs[..n].copy_from_slice(logits);
    let m = tree_max(&mut xs.clone());
    let es: Vec<f32> = (0..256)
        .map(|t| if t < n { (xs[t] - m).exp() } else { 0.0 })
        .collect();
    let s = tree(&mut es.clone());
    let probs: Vec<f32> = es[..n].iter().map(|e| e / s).collect();
    let mut pv: Vec<f32> = (0..256)
        .map(|t| if t < n { probs[t] } else { -1.0 })
        .collect();
    let (mut ids, mut sel) = (Vec::with_capacity(k), Vec::with_capacity(k));
    for _ in 0..k {
        // The maximum under (value descending, id ascending), which the shader's tree finds whatever its shape.
        let mut best = 0;
        for t in 1..256 {
            if pv[t] > pv[best] {
                best = t;
            }
        }
        ids.push(best as f32);
        sel.push(pv[best]);
        pv[best] = -1.0;
    }
    let sum = sel.iter().fold(0f32, |a, v| a + v);
    // 2⁻¹⁴ (6.103515625e-5), exact in f32: llama.cpp's floor on the sum.
    let den = sum.max(1.0 / 16384.0);
    let w = sel.iter().map(|v| v / den).collect();
    (probs, ids, w)
}

/// `swiglu`.
pub fn swiglu(g: &[f32], u: &[f32]) -> Vec<f32> {
    g.iter().zip(u).map(|(g, u)| silu(*g) * u).collect()
}

/// `experts_gate_up`: for each slot (one routed expert's packed bytes, in routed order), silu_det(gate·x) ⊙ (up·x).
pub fn experts_gate_up(slots: &[&[u8]], lay: &SlotLayout, x: &[f32]) -> Vec<f32> {
    let (rg, ru) = (lay.gate.row_bytes(lay.n_embd), lay.up.row_bytes(lay.n_embd));
    let mut y = Vec::with_capacity(slots.len() * lay.n_ff);
    for s in slots {
        for row in 0..lay.n_ff {
            let g = row_dot(lay.gate, &s[row * rg..(row + 1) * rg], x);
            let up = &s[lay.up_offset() + row * ru..lay.up_offset() + (row + 1) * ru];
            y.push(silu_det(g) * row_dot(lay.up, up, x));
        }
    }
    y
}

/// `experts_down`: for each slot j, down_j · act[j].
pub fn experts_down(slots: &[&[u8]], lay: &SlotLayout, act: &[f32]) -> Vec<f32> {
    let rd = lay.down.row_bytes(lay.n_ff);
    let mut y = Vec::with_capacity(slots.len() * lay.n_embd);
    for (j, s) in slots.iter().enumerate() {
        let x = &act[j * lay.n_ff..(j + 1) * lay.n_ff];
        for row in 0..lay.n_embd {
            let at = lay.down_offset() + row * rd;
            y.push(row_dot(lay.down, &s[at..at + rd], x));
        }
    }
    y
}

/// `moe_unique`: a layer's distinct routed experts over `tokens` tokens of `k` choices (`ids` is `[tokens][k]` f32),
/// as (expert, mask) in order of first appearance, token-major then by choice. Bit `i` of the mask is set when token
/// `i` chose the expert.
pub fn moe_unique(ids: &[f32], tokens: usize, k: usize) -> Vec<(u32, u32)> {
    assert!(
        tokens <= MAX_TOKENS && ids.len() == tokens * k,
        "moe_unique: {} ids for {tokens} tokens of {k} choices",
        ids.len()
    );
    let mut list: Vec<(u32, u32)> = Vec::new();
    for (idx, &v) in ids.iter().enumerate() {
        let (e, bit) = (v as u32, 1u32 << (idx / k));
        match list.iter_mut().find(|(x, _)| *x == e) {
            Some((_, mask)) => *mask |= bit,
            None => list.push((e, bit)),
        }
    }
    list
}

/// Token `i`'s choice that names expert `e`: the first of its `k` ids that does.
fn routed_index(ids: &[f32], i: usize, k: usize, e: u32) -> usize {
    ids[i * k..(i + 1) * k]
        .iter()
        .position(|&v| v as u32 == e)
        .expect("a token in an expert's mask chose it")
}

/// `experts_gate_up_tokens`: `[tokens][k][n_ff]`. Each distinct expert of `ids` (`[tokens][k]`) is taken once
/// (`slot(e)` gives its packed bytes), and for every token that chose it, `experts_gate_up`'s values for that token
/// land at the token's routed index: token `i`'s rows are what `experts_gate_up` gives it alone, bit for bit.
pub fn experts_gate_up_tokens<'a>(
    slot: impl Fn(usize) -> &'a [u8],
    lay: &SlotLayout,
    x: &[f32],
    ids: &[f32],
    k: usize,
    tokens: usize,
) -> Vec<f32> {
    let mut y = vec![0f32; tokens * k * lay.n_ff];
    for (e, mask) in moe_unique(ids, tokens, k) {
        let s = slot(e as usize);
        for i in (0..tokens).filter(|i| mask >> i & 1 == 1) {
            let row = experts_gate_up(&[s], lay, &x[i * lay.n_embd..(i + 1) * lay.n_embd]);
            let at = (i * k + routed_index(ids, i, k, e)) * lay.n_ff;
            y[at..at + lay.n_ff].copy_from_slice(&row);
        }
    }
    y
}

/// `experts_down_tokens`: `[tokens][k][n_embd]`. As `experts_gate_up_tokens`, with `act` `[tokens][k][n_ff]`: token
/// `i`'s input to expert `e` is `act[i][j]`, `j` its routed index, and the result goes to `y[i][j]`.
pub fn experts_down_tokens<'a>(
    slot: impl Fn(usize) -> &'a [u8],
    lay: &SlotLayout,
    act: &[f32],
    ids: &[f32],
    k: usize,
    tokens: usize,
) -> Vec<f32> {
    let mut y = vec![0f32; tokens * k * lay.n_embd];
    for (e, mask) in moe_unique(ids, tokens, k) {
        let s = slot(e as usize);
        for i in (0..tokens).filter(|i| mask >> i & 1 == 1) {
            let j = routed_index(ids, i, k, e);
            let a = (i * k + j) * lay.n_ff;
            let row = experts_down(&[s], lay, &act[a..a + lay.n_ff]);
            let at = (i * k + j) * lay.n_embd;
            y[at..at + lay.n_embd].copy_from_slice(&row);
        }
    }
    y
}

/// `moe_combine`: (Σ_j w_j · ye[j] in order, sh · sigmoid(sg)).
pub fn moe_combine(ye: &[f32], w: &[f32], sh: &[f32], sg: f32, k: usize) -> (Vec<f32>, Vec<f32>) {
    let n = sh.len();
    let moe = (0..n)
        .map(|i| (0..k).fold(0f32, |m, j| m + w[j] * ye[j * n + i]))
        .collect();
    let gate = sigmoid(sg);
    (moe, sh.iter().map(|v| v * gate).collect())
}

/// `apply_slot_updates`: each of the first `count` updates ({ u32 index; u32 where; u64 addr }) rewrites table entry
/// `index` as { addr, where, 0 }; an index past the table is skipped.
pub fn apply_slot_updates(table: &mut [u8], updates: &[u8], count: usize) {
    let n = table.len() / ENTRY_BYTES;
    for u in updates.as_chunks::<UPDATE_BYTES>().0.iter().take(count) {
        let index = u32::from_le_bytes(u[..4].try_into().expect("4 bytes")) as usize;
        if index >= n {
            continue;
        }
        let loc = u32::from_le_bytes(u[4..8].try_into().expect("4 bytes"));
        let addr = u64::from_le_bytes(u[8..].try_into().expect("8 bytes"));
        table[index * ENTRY_BYTES..(index + 1) * ENTRY_BYTES]
            .copy_from_slice(&slot_entry(addr, loc));
    }
}

/// `moe_group`: (list, tiles). The list holds each expert's entries (indices into `ids`) in order, experts in order;
/// the tiles cut each expert's run into pieces of up to `MOE_TILE`, as [expert, first, count, 0], then empty tiles up
/// to `max_tiles`. An id that names no expert is dropped.
pub fn moe_group(ids: &[f32], n_expert: usize, max_tiles: usize) -> (Vec<u32>, Vec<[u32; 4]>) {
    assert!(
        max_tiles >= crate::dispatch::moe_tiles(ids.len(), n_expert),
        "moe_group: {max_tiles} tiles are too few for {} entries over {n_expert} experts",
        ids.len()
    );
    let (mut list, mut tiles) = (Vec::with_capacity(ids.len()), Vec::new());
    for e in 0..n_expert {
        let first = list.len();
        list.extend(
            ids.iter()
                .enumerate()
                .filter(|&(_, &v)| v >= 0.0 && v < n_expert as f32 && v as usize == e)
                .map(|(i, _)| i as u32),
        );
        let count = list.len() - first;
        for q in (0..count).step_by(MOE_TILE) {
            tiles.push([
                e as u32,
                (first + q) as u32,
                MOE_TILE.min(count - q) as u32,
                0,
            ]);
        }
    }
    tiles.resize(max_tiles, [0; 4]);
    (list, tiles)
}

/// `experts_matmul_gate_up`: for each entry i of `list` (token list[i] / k, expert ids[list[i]], whose packed bytes
/// `slot` gives), silu_det(gate·x) ⊙ (up·x) with matmul's sums; `[list.len()][n_ff]`.
pub fn experts_matmul_gate_up<'a>(
    slot: impl Fn(usize) -> &'a [u8],
    lay: &SlotLayout,
    x: &[f32],
    ids: &[f32],
    list: &[u32],
    k: usize,
) -> Vec<f32> {
    let (rg, ru) = (lay.gate.row_bytes(lay.n_embd), lay.up.row_bytes(lay.n_embd));
    let mut y = Vec::with_capacity(list.len() * lay.n_ff);
    for &entry in list {
        let s = slot(ids[entry as usize] as usize);
        let t = entry as usize / k;
        let xt = &x[t * lay.n_embd..(t + 1) * lay.n_embd];
        for row in 0..lay.n_ff {
            let g = dot_in_order(lay.gate, &s[row * rg..(row + 1) * rg], xt);
            let at = lay.up_offset() + row * ru;
            y.push(silu_det(g) * dot_in_order(lay.up, &s[at..at + ru], xt));
        }
    }
    y
}

/// `experts_matmul_down`: row list[i] of the result (`[ids.len()][n_embd]`) ← down · act[i] of entry i's expert, with
/// matmul's sums. Rows no entry names stay zero.
pub fn experts_matmul_down<'a>(
    slot: impl Fn(usize) -> &'a [u8],
    lay: &SlotLayout,
    act: &[f32],
    ids: &[f32],
    list: &[u32],
) -> Vec<f32> {
    let rd = lay.down.row_bytes(lay.n_ff);
    let mut y = vec![0f32; ids.len() * lay.n_embd];
    for (i, &entry) in list.iter().enumerate() {
        let s = slot(ids[entry as usize] as usize);
        let a = &act[i * lay.n_ff..(i + 1) * lay.n_ff];
        for row in 0..lay.n_embd {
            let at = lay.down_offset() + row * rd;
            y[entry as usize * lay.n_embd + row] = dot_in_order(lay.down, &s[at..at + rd], a);
        }
    }
    y
}

/// `experts_matmul_cm_gate_up`: `experts_matmul_gate_up`'s rows, `[list.len()][n_ff]`, from split-f16 weights (as
/// `dequant_row` gives them) and inputs (`matvec::split_all`, then `matvec::dot_split_pairs`).
pub fn experts_matmul_cm_gate_up<'a>(
    slot: impl Fn(usize) -> &'a [u8],
    lay: &SlotLayout,
    x: &[f32],
    ids: &[f32],
    list: &[u32],
    k: usize,
) -> Vec<f32> {
    let (rg, ru) = (lay.gate.row_bytes(lay.n_embd), lay.up.row_bytes(lay.n_embd));
    let mut row = vec![0f32; lay.n_embd];
    let mut y = Vec::with_capacity(list.len() * lay.n_ff);
    for &entry in list {
        let s = slot(ids[entry as usize] as usize);
        let t = entry as usize / k;
        let xt = split_all(&x[t * lay.n_embd..(t + 1) * lay.n_embd]);
        for r in 0..lay.n_ff {
            qwen36::quant::dequant_row(lay.gate, &s[r * rg..(r + 1) * rg], &mut row);
            let g = dot_split_pairs(&split_all(&row), &xt);
            let at = lay.up_offset() + r * ru;
            qwen36::quant::dequant_row(lay.up, &s[at..at + ru], &mut row);
            y.push(silu_det(g) * dot_split_pairs(&split_all(&row), &xt));
        }
    }
    y
}

/// `experts_matmul_cm_down`: row list[i] of the result (`[ids.len()][n_embd]`) ← down · act[i] of entry i's expert,
/// from split-f16 weights and inputs. Rows no entry names stay zero.
pub fn experts_matmul_cm_down<'a>(
    slot: impl Fn(usize) -> &'a [u8],
    lay: &SlotLayout,
    act: &[f32],
    ids: &[f32],
    list: &[u32],
) -> Vec<f32> {
    let rd = lay.down.row_bytes(lay.n_ff);
    let mut row = vec![0f32; lay.n_ff];
    let mut y = vec![0f32; ids.len() * lay.n_embd];
    for (i, &entry) in list.iter().enumerate() {
        let s = slot(ids[entry as usize] as usize);
        let a = split_all(&act[i * lay.n_ff..(i + 1) * lay.n_ff]);
        for r in 0..lay.n_embd {
            let at = lay.down_offset() + r * rd;
            qwen36::quant::dequant_row(lay.down, &s[at..at + rd], &mut row);
            y[entry as usize * lay.n_embd + r] = dot_split_pairs(&split_all(&row), &a);
        }
    }
    y
}
