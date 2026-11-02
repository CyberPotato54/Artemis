//! Test 1 on real tensors: one real tensor per weight type, and layer 0's `ssm_out` (k = 4,096, the widest
//! k prefill's matmuls take), through its matvec, its matmul and its matmul_cm (five tokens), a real embedding row,
//! and decode's expert kernels and the split-f16 grouped ones on real slots of layers 0 and 34 (Q5_K and Q6_K down) in
//! pinned host memory. Reads ARTEMIS_MODEL.
use gguf::Gguf;
use kernels::{
    dispatch::{
        moe_tiles, read_status, slot_entry, ExpertStatus, MatW, SlotLayout, STATUS_CLEAR,
        WHERE_HOST,
    },
    testing::{check, rand_f32, Gpu},
    twin, Kernel,
};
use qwen36::{
    dims::Dims,
    model::{Mixer, Model, Weight},
    quant::WType,
    testing::Rng,
};

fn open() -> Gguf {
    let path = std::env::var("ARTEMIS_MODEL").expect("set ARTEMIS_MODEL to the model's GGUF");
    Gguf::open(std::path::Path::new(&path)).unwrap()
}

/// One real tensor, uploaded once, through its matvec and then its matmul and matmul_cm over `TOKENS` random input
/// vectors, each against its twin.
fn real_matvec_and_matmul(g: &Gpu, rng: &mut Rng, name: &str, w: Weight) {
    const TOKENS: usize = 5;
    let gw = g.bytes(w.bytes);
    let m = MatW {
        addr: gw.addr,
        ty: w.ty,
        rows: w.rows,
        cols: w.cols,
    };
    let x = rand_f32(rng, w.cols, -1.0, 1.0);
    let (gx, y) = (g.f32s(&x), g.zeros(w.rows));
    g.run(|k, r| k.matvec(r, m, gx.addr, y.addr));
    let want = twin::matvec::matvec(w.ty, w.bytes, w.rows, &x);
    check(
        Kernel::matvec(w.ty),
        &format!("real {name}"),
        &g.read(y),
        &want,
    );
    let x = rand_f32(rng, TOKENS * w.cols, -1.0, 1.0);
    let (gx, y) = (g.f32s(&x), g.zeros(TOKENS * w.rows));
    g.run(|k, r| k.matmul(r, m, gx.addr, y.addr, TOKENS));
    let want = twin::matvec::matmul(w.ty, w.bytes, w.rows, &x, TOKENS);
    check(
        Kernel::matmul(w.ty),
        &format!("real {name} matmul"),
        &g.read(y),
        &want,
    );
    assert!(
        g.k.coop(),
        "the RX 9070 XT runs the cooperative-matrix kernels"
    );
    let y = g.zeros(TOKENS * w.rows);
    g.run(|k, r| k.matmul_cm(r, m, gx.addr, y.addr, TOKENS));
    let want = twin::matvec::matmul_cm(w.ty, w.bytes, w.rows, &x, TOKENS);
    check(
        Kernel::matmul_cm(w.ty),
        &format!("real {name} matmul_cm"),
        &g.read(y),
        &want,
    );
}

#[test]
#[ignore = "needs the GPU and ARTEMIS_MODEL"]
fn one_real_tensor_per_weight_type() {
    let gg = open();
    let m = Model::open(&gg, &Dims::QWEN36).unwrap();
    let g = Gpu::new();
    let mut rng = Rng::new(30);
    let l0 = &m.layers[0];
    let Mixer::DeltaNet(dn) = &l0.mixer else {
        panic!("layer 0 is a DeltaNet layer")
    };
    let cases = [
        ("blk.0.ffn_gate_inp", l0.moe.router, WType::F32),
        ("blk.40.ffn_gate_inp", m.mtp.layer.moe.router, WType::Bf16),
        ("blk.0.attn_qkv", dn.qkv, WType::Q8_0),
        ("blk.0.ssm_out", dn.out, WType::Q8_0),
        (
            "blk.0.ffn_gate_exps[0]",
            l0.moe.gate_exps.expert(0, 512),
            WType::Q4K,
        ),
        (
            "blk.0.ffn_down_exps[0]",
            l0.moe.down_exps.expert(0, 2048),
            WType::Q5K,
        ),
        ("output", m.output, WType::Q6K),
    ];
    for (name, w, ty) in cases {
        assert_eq!(w.ty, ty, "{name} is {}, not {}", w.ty.name(), ty.name());
        real_matvec_and_matmul(&g, &mut rng, name, w);
    }
    // The embedding: three real rows in pinned host memory, the middle one dequantized.
    let rb = m.embd.ty.row_bytes(m.embd.cols);
    let t = 151_643;
    let host = g.host_bytes(&m.embd.bytes[(t - 1) * rb..(t + 2) * rb]);
    let y = g.zeros(m.embd.cols);
    g.run(|k, r| k.embed_row(r, host.addr + rb as u64, m.embd.ty, m.embd.cols, y.addr));
    check(
        Kernel::EmbedRow,
        &format!("real token_embd row {t}"),
        &g.read(y),
        &twin::embed_row(m.embd.ty, m.embd.row(t), m.embd.cols),
    );
}

#[test]
#[ignore = "needs the GPU and ARTEMIS_MODEL"]
fn experts_on_real_slots_from_host_memory() {
    let gg = open();
    let d = Dims::QWEN36;
    let m = Model::open(&gg, &d).unwrap();
    let g = Gpu::new();
    let mut rng = Rng::new(31);
    let ids = [3.0f32, 17.0, 100.0, 200.0, 255.0, 0.0, 64.0, 128.0];
    for l in [0usize, 34] {
        let moe = &m.layers[l].moe;
        let lay = SlotLayout {
            gate: moe.gate_exps.ty,
            up: moe.up_exps.ty,
            down: moe.down_exps.ty,
            n_embd: d.n_embd,
            n_ff: d.n_ff_exp,
        };
        // Pack the routed experts' slots, and point their entries (in a one-layer table of 256) at them.
        let slot = |e: usize| -> Vec<u8> {
            let mut s = moe.gate_exps.expert(e, d.n_ff_exp).bytes.to_vec();
            s.extend_from_slice(moe.up_exps.expert(e, d.n_ff_exp).bytes);
            s.extend_from_slice(moe.down_exps.expert(e, d.n_embd).bytes);
            s
        };
        let slots: Vec<Vec<u8>> = ids.iter().map(|&e| slot(e as usize)).collect();
        assert_eq!(slots[0].len(), lay.bytes());
        let packed = g.host_bytes(&slots.concat());
        let mut table = vec![0u8; d.n_expert * 16];
        for (j, &e) in ids.iter().enumerate() {
            let entry = slot_entry(packed.addr + (j * lay.bytes()) as u64, WHERE_HOST);
            table[e as usize * 16..(e as usize + 1) * 16].copy_from_slice(&entry);
        }
        let table = g.bytes(&table);
        let status = g.bytes(&STATUS_CLEAR);
        let x = rand_f32(&mut rng, d.n_embd, -1.0, 1.0);
        let (gids, gx) = (g.f32s(&ids), g.f32s(&x));
        let (act, ye) = (g.zeros(8 * d.n_ff_exp), g.zeros(8 * d.n_embd));
        g.run(|k, r| {
            k.experts_gate_up(
                r,
                table.addr,
                0,
                d.n_expert,
                gids.addr,
                gx.addr,
                act.addr,
                status.addr,
                &lay,
                8,
                0,
                0,
                0,
            );
            r.barrier();
            k.experts_down(
                r,
                table.addr,
                0,
                d.n_expert,
                gids.addr,
                act.addr,
                ye.addr,
                status.addr,
                &lay,
                8,
                0,
                0,
                0,
            );
        });
        let refs: Vec<&[u8]> = slots.iter().map(|s| s.as_slice()).collect();
        let case = format!("real blk.{l} ({} down)", lay.down.name());
        let act_v = g.read(act);
        check(
            Kernel::ExpertsGateUp,
            &case,
            &act_v,
            &twin::moe::experts_gate_up(&refs, &lay, &x),
        );
        check(
            Kernel::ExpertsDown,
            &case,
            &g.read(ye),
            &twin::moe::experts_down(&refs, &lay, &act_v),
        );
        let st = g.dev.download(status).unwrap().try_into().unwrap();
        assert_eq!(
            read_status(&st),
            ExpertStatus::default(),
            "{case}: nothing refused"
        );
    }
}

#[test]
#[ignore = "needs the GPU and ARTEMIS_MODEL"]
fn grouped_cm_experts_on_real_slots() {
    // The split-f16 grouped kernels on real experts of layers 0 and 34 (Q5_K and Q6_K down), read from pinned host
    // memory through a one-layer table: three tokens, eight experts each, five of them shared, so some tiles hold
    // several entries.
    let gg = open();
    let d = Dims::QWEN36;
    let m = Model::open(&gg, &d).unwrap();
    let g = Gpu::new();
    assert!(
        g.k.coop(),
        "the RX 9070 XT runs the cooperative-matrix kernels"
    );
    let mut rng = Rng::new(32);
    let ids: Vec<f32> = [
        3, 17, 100, 200, 255, 0, 64, 128, 3, 17, 100, 200, 255, 1, 2, 5, 0, 64, 128, 3, 17, 7, 9,
        11,
    ]
    .iter()
    .map(|&e| e as f32)
    .collect();
    let k = 8;
    let u32s = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    for l in [0usize, 34] {
        let moe = &m.layers[l].moe;
        let lay = SlotLayout {
            gate: moe.gate_exps.ty,
            up: moe.up_exps.ty,
            down: moe.down_exps.ty,
            n_embd: d.n_embd,
            n_ff: d.n_ff_exp,
        };
        let slot = |e: usize| -> Vec<u8> {
            let mut s = moe.gate_exps.expert(e, d.n_ff_exp).bytes.to_vec();
            s.extend_from_slice(moe.up_exps.expert(e, d.n_ff_exp).bytes);
            s.extend_from_slice(moe.down_exps.expert(e, d.n_embd).bytes);
            s
        };
        let mut used: Vec<usize> = ids.iter().map(|&e| e as usize).collect();
        used.sort();
        used.dedup();
        let slots: Vec<Vec<u8>> = (0..d.n_expert)
            .map(|e| {
                if used.contains(&e) {
                    slot(e)
                } else {
                    Vec::new()
                }
            })
            .collect();
        let packed: Vec<u8> = used.iter().flat_map(|&e| slots[e].clone()).collect();
        let host = g.host_bytes(&packed);
        let mut table = vec![0u8; d.n_expert * 16];
        for (j, &e) in used.iter().enumerate() {
            let entry = slot_entry(host.addr + (j * lay.bytes()) as u64, WHERE_HOST);
            table[e * 16..(e + 1) * 16].copy_from_slice(&entry);
        }
        let table = g.bytes(&table);
        let max = moe_tiles(ids.len(), d.n_expert);
        let (list, tiles) = twin::moe::moe_group(&ids, d.n_expert, max);
        let (glist, gtiles) = (g.bytes(&u32s(&list)), g.bytes(&u32s(tiles.as_flattened())));
        let x = rand_f32(&mut rng, 3 * d.n_embd, -1.0, 1.0);
        let gx = g.f32s(&x);
        let (act, ye) = (
            g.zeros(ids.len() * d.n_ff_exp),
            g.zeros(ids.len() * d.n_embd),
        );
        let status = g.bytes(&STATUS_CLEAR);
        g.run(|kk, r| {
            kk.experts_matmul_cm_gate_up(
                r,
                table.addr,
                0,
                d.n_expert,
                glist.addr,
                gtiles.addr,
                max,
                gx.addr,
                act.addr,
                status.addr,
                &lay,
                k,
            );
            r.barrier();
            kk.experts_matmul_cm_down(
                r,
                table.addr,
                0,
                d.n_expert,
                glist.addr,
                gtiles.addr,
                max,
                act.addr,
                ye.addr,
                status.addr,
                &lay,
            );
        });
        let case = format!("real blk.{l} ({} down), split f16", lay.down.name());
        let act_v = g.read(act);
        check(
            Kernel::ExpertsMatmulCmGateUp,
            &case,
            &act_v,
            &twin::moe::experts_matmul_cm_gate_up(
                |e| slots[e].as_slice(),
                &lay,
                &x,
                &ids,
                &list,
                k,
            ),
        );
        check(
            Kernel::ExpertsMatmulCmDown,
            &case,
            &g.read(ye),
            &twin::moe::experts_matmul_cm_down(|e| slots[e].as_slice(), &lay, &act_v, &ids, &list),
        );
        let st = g.dev.download(status).unwrap().try_into().unwrap();
        assert_eq!(
            read_status(&st),
            ExpertStatus::default(),
            "{case}: nothing refused"
        );
    }
}
