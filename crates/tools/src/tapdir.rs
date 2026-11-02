//! Tap sets on disk. Artemis writes `artemis-taps-1` directories: one f32 file per tap, positions in
//! order. The oracle writes one file per llama.cpp callback tensor. Both load into a `TapSet` under our names.
use anyhow::{anyhow, bail, Context, Result};
use qwen36::{
    dims::Dims,
    taps::{expected_mtp_taps, expected_taps, Key, MtpKey, Tap, TapName},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

/// Each tap's values for `n_tokens` positions, position-major.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TapSet {
    pub n_tokens: usize,
    pub taps: BTreeMap<TapName, Vec<f32>>,
}

/// Collects taps in step order, from the reference now and from the GPU engine.
#[derive(Default)]
pub struct Recorder {
    pub set: TapSet,
}

impl Tap for Recorder {
    fn tap(&mut self, name: TapName, pos: u32, data: &[f32]) {
        self.set
            .taps
            .entry(name)
            .or_default()
            .extend_from_slice(data);
        self.set.n_tokens = self.set.n_tokens.max(pos as usize + 1);
    }
}

/// Records a run while forcing it with an oracle's values at every tap the oracle has: teacher forcing.
pub struct Forcer<'a> {
    pub rec: Recorder,
    oracle: &'a TapSet,
    d: Dims,
}

impl<'a> Forcer<'a> {
    pub fn new(oracle: &'a TapSet, d: &Dims) -> Forcer<'a> {
        Forcer {
            rec: Recorder::default(),
            oracle,
            d: d.clone(),
        }
    }
}

impl Tap for Forcer<'_> {
    fn tap(&mut self, name: TapName, pos: u32, data: &[f32]) {
        self.rec.tap(name, pos, data);
    }

    fn forced(&self, name: TapName, pos: u32) -> Option<&[f32]> {
        let n = name.per_pos(&self.d);
        self.oracle
            .taps
            .get(&name)?
            .get(pos as usize * n..(pos as usize + 1) * n)
    }
}

/// Presents a run whose positions start at `by` as a tap set whose rows start at 0: the MTP's rows are at positions
/// `1..n` (row `q` pairs token `q` with the hidden row at `q − 1`), and its tap sets, the oracle's and ours, count rows.
pub(crate) struct Shifted<'a> {
    pub(crate) inner: &'a mut dyn Tap,
    pub(crate) by: u32,
}

impl Shifted<'_> {
    fn row(&self, pos: u32) -> u32 {
        pos.checked_sub(self.by)
            .expect("a position before the first MTP row")
    }
}

impl Tap for Shifted<'_> {
    fn tap(&mut self, name: TapName, pos: u32, data: &[f32]) {
        self.inner.tap(name, self.row(pos), data);
    }

    fn forced(&self, name: TapName, pos: u32) -> Option<&[f32]> {
        self.inner.forced(name, self.row(pos))
    }
}

/// `artemis-oracle mtp-taps`'s `hidden.bin`: the main model's final hidden row at each position, raw little-endian
/// f32, `[n][n_embd]`.
pub fn read_hidden(path: &Path, n_embd: usize) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let v = f32s(&bytes).with_context(|| format!("reading {}", path.display()))?;
    if v.is_empty() || !v.len().is_multiple_of(n_embd) {
        bail!(
            "{} holds {} values, not a whole number of {n_embd}-wide rows",
            path.display(),
            v.len()
        );
    }
    Ok(v)
}

#[derive(Serialize, Deserialize)]
struct OurIndex {
    format: String,
    source: String,
    n_tokens: usize,
    taps: Vec<OurEntry>,
}

#[derive(Serialize, Deserialize)]
struct OurEntry {
    name: String,
    per_pos: usize,
    file: String,
}

#[derive(Deserialize)]
struct LlamaIndex {
    n_tokens: usize,
    tensors: Vec<LlamaEntry>,
}

#[derive(Deserialize)]
struct LlamaEntry {
    name: String,
    occurrence: u32,
    ne: [i64; 4],
    dtype: String,
    file: String,
}

pub const FORMAT: &str = "artemis-taps-1";

/// The `source` string an Artemis tap directory records (how it was made), if it has one.
pub fn source(dir: &Path) -> Result<String> {
    let path = dir.join("index.json");
    let v: serde_json::Value = serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
    )
    .with_context(|| format!("parsing {}", path.display()))?;
    Ok(v["source"].as_str().unwrap_or("").to_string())
}

pub fn write(set: &TapSet, dir: &Path, source: &str) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let mut taps = Vec::new();
    for (i, (name, data)) in set.taps.iter().enumerate() {
        let file = format!("t{i:05}.f32");
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(dir.join(&file), bytes)?;
        taps.push(OurEntry {
            name: name.to_string(),
            per_pos: data.len() / set.n_tokens.max(1),
            file,
        });
    }
    let index = OurIndex {
        format: FORMAT.into(),
        source: source.into(),
        n_tokens: set.n_tokens,
        taps,
    };
    fs::write(dir.join("index.json"), serde_json::to_vec_pretty(&index)?)?;
    Ok(())
}

/// Loads either format. llama.cpp names map to ours through `TapName::llama`; when llama.cpp wrote a name more than
/// once, the last occurrence wins. Taps the dump lacks are left out, and `compare` reports them as missing.
pub fn read(dir: &Path, d: &Dims) -> Result<TapSet> {
    read_expecting(dir, d, &expected_taps(d))
}

/// `read` for a directory of the MTP block's taps (`artemis-oracle mtp-taps`, or `dump --cpu --mtp`): llama.cpp's
/// names are matched against the MTP table, not the main model's. Its `n_tokens` counts rows, not positions: row `i`
/// is the MTP's row at position `i + 1`.
pub fn read_mtp(dir: &Path, d: &Dims) -> Result<TapSet> {
    read_expecting(dir, d, &expected_mtp_taps())
}

fn read_expecting(dir: &Path, d: &Dims, expected: &[TapName]) -> Result<TapSet> {
    let path = dir.join("index.json");
    let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let v: serde_json::Value =
        serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    let set = if v.get("format").and_then(|f| f.as_str()) == Some(FORMAT) {
        read_ours(dir, serde_json::from_value(v)?)?
    } else {
        read_llama(dir, serde_json::from_value(v)?, d, expected)?
    };
    validate(&set, d).with_context(|| format!("checking {}", dir.display()))?;
    Ok(set)
}

/// Every tap the size these dims give it, and routed expert ids that are experts, so nothing downstream can index out
/// of range on a dump of another model or a short tap.
fn validate(set: &TapSet, d: &Dims) -> Result<()> {
    for (tap, data) in &set.taps {
        let want = tap.per_pos(d) * set.n_tokens;
        if data.len() != want {
            bail!(
                "tap {tap} has {} values, but these dims need {} per position × {} positions",
                data.len(),
                tap.per_pos(d),
                set.n_tokens
            );
        }
        if matches!(
            tap,
            TapName::Layer(_, Key::TopK) | TapName::Mtp(MtpKey::TopK)
        ) {
            if let Some(bad) = data
                .iter()
                .find(|&&id| id < 0.0 || id.fract() != 0.0 || id >= d.n_expert as f32)
            {
                bail!(
                    "tap {tap} holds {bad}, which is not an expert id (0..{})",
                    d.n_expert
                );
            }
        }
    }
    Ok(())
}

fn read_ours(dir: &Path, index: OurIndex) -> Result<TapSet> {
    let mut set = TapSet {
        n_tokens: index.n_tokens,
        ..Default::default()
    };
    for e in index.taps {
        let name: TapName = e.name.parse().map_err(|m: String| anyhow!(m))?;
        let data = f32s(&fs::read(dir.join(&e.file))?)?;
        if data.len() != e.per_pos * index.n_tokens {
            bail!(
                "tap {} has {} values, expected {} per position × {}",
                e.name,
                data.len(),
                e.per_pos,
                index.n_tokens
            );
        }
        set.taps.insert(name, data);
    }
    Ok(set)
}

fn read_llama(dir: &Path, index: LlamaIndex, d: &Dims, expected: &[TapName]) -> Result<TapSet> {
    let mut last: HashMap<&str, &LlamaEntry> = HashMap::new();
    for e in &index.tensors {
        if last
            .get(e.name.as_str())
            .is_none_or(|prev| e.occurrence > prev.occurrence)
        {
            last.insert(&e.name, e);
        }
    }
    let mut set = TapSet {
        n_tokens: index.n_tokens,
        ..Default::default()
    };
    for &tap in expected {
        let Some(e) = tap.llama(d).and_then(|n| last.get(n.as_str()).copied()) else {
            continue;
        };
        let bytes = fs::read(dir.join(&e.file)).with_context(|| format!("reading {}", e.file))?;
        let data: Vec<f32> = match e.dtype.as_str() {
            "f32" => f32s(&bytes)?,
            "i32" => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c) as f32)
                .collect(),
            other => bail!("{}: dtype {other}", e.name),
        };
        let want = tap.per_pos(d) * index.n_tokens;
        if data.len() != want {
            bail!(
                "llama.cpp tensor {} (ne {:?}) has {} values; tap {tap} needs {} per position × {} positions",
                e.name,
                e.ne,
                data.len(),
                tap.per_pos(d),
                index.n_tokens
            );
        }
        set.taps.insert(tap, data);
    }
    Ok(set)
}

fn f32s(b: &[u8]) -> Result<Vec<f32>> {
    if !b.len().is_multiple_of(4) {
        bail!("{} bytes is not a whole number of f32s", b.len());
    }
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen36::taps::Key;

    fn temp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("artemis-tapdir-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn the_forcer_supplies_the_oracles_value_at_each_position() {
        let d = Dims::tiny();
        let mut oracle = TapSet {
            n_tokens: 2,
            ..Default::default()
        };
        oracle.taps.insert(
            TapName::Layer(0, Key::GdnBeta),
            (0..16).map(|i| i as f32).collect(),
        );
        let mut f = Forcer::new(&oracle, &d);
        assert_eq!(
            f.forced(TapName::Layer(0, Key::GdnBeta), 1),
            Some(&oracle.taps[&TapName::Layer(0, Key::GdnBeta)][8..16])
        );
        assert_eq!(
            f.forced(TapName::Layer(0, Key::GdnBeta), 2),
            None,
            "past the oracle's positions"
        );
        assert_eq!(
            f.forced(TapName::Layer(0, Key::GdnG), 0),
            None,
            "a tap the oracle lacks"
        );
        f.tap(TapName::Layer(0, Key::GdnBeta), 0, &[1.0; 8]);
        assert_eq!(
            f.rec.set.taps[&TapName::Layer(0, Key::GdnBeta)],
            vec![1.0; 8],
            "records the run's own values"
        );
    }

    #[test]
    fn a_tap_sized_for_other_dims_is_an_error() {
        // Two values per position of embd: right for no model, so a dump of the tiny model read as the real one, or a
        // short GPU tap, is refused instead of panicking later.
        let mut rec = Recorder::default();
        rec.tap(TapName::Embd, 0, &[1.0, 2.0]);
        let dir = temp("dims");
        write(&rec.set, &dir, "test").unwrap();
        let e = format!("{:#}", read(&dir, &Dims::tiny()).unwrap_err());
        assert!(e.contains("embd") && e.contains("256"), "{e}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn routed_expert_ids_must_be_experts() {
        let d = Dims::tiny();
        for bad in [16.0f32, -1.0, 2.5] {
            let mut rec = Recorder::default();
            rec.tap(TapName::Layer(0, Key::TopK), 0, &[0.0, 1.0, 2.0, bad]);
            let dir = temp("ids");
            write(&rec.set, &dir, "test").unwrap();
            let e = format!("{:#}", read(&dir, &d).unwrap_err());
            assert!(e.contains("l0.topk") && e.contains("expert"), "{bad}: {e}");
            fs::remove_dir_all(&dir).unwrap();
        }
    }

    #[test]
    fn our_format_round_trips() {
        let mut rec = Recorder::default();
        for p in 0..3 {
            rec.tap(TapName::Embd, p, &[p as f32; 256]);
            rec.tap(TapName::Layer(2, Key::Out), p, &[-1.0; 256]);
        }
        let dir = temp("ours");
        write(&rec.set, &dir, "test").unwrap();
        let back = read(&dir, &Dims::tiny()).unwrap();
        assert_eq!(back, rec.set);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn llama_dumps_map_to_our_names_and_take_the_last_occurrence() {
        let d = Dims::tiny();
        let dir = temp("llama");
        fs::create_dir_all(&dir).unwrap();
        let k1: Vec<u8> = vec![0.0f32; 64 * 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let k2: Vec<u8> = vec![1.0f32; 64 * 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let ids: Vec<u8> = [3i32, 1, 4, 0, 2, 5, 6, 7]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        fs::write(dir.join("00000.bin"), k1).unwrap();
        fs::write(dir.join("00001.bin"), k2).unwrap();
        fs::write(dir.join("00002.bin"), ids).unwrap();
        let index = r#"{"source":"llama.cpp b11509","backend":"cpu","n_tokens":2,"tensors":[
{"name":"Kcur-3","occurrence":0,"ne":[64,2,1,1],"dtype":"f32","file":"00000.bin"},
{"name":"Kcur-3","occurrence":1,"ne":[64,1,2,1],"dtype":"f32","file":"00001.bin"},
{"name":"ffn_moe_topk-0","occurrence":0,"ne":[4,2,1,1],"dtype":"i32","file":"00002.bin"}
]}"#;
        fs::write(dir.join("index.json"), index).unwrap();
        let set = read(&dir, &d).unwrap();
        assert_eq!(set.n_tokens, 2);
        assert!(set.taps[&TapName::Layer(3, Key::AttnK)]
            .iter()
            .all(|&v| v == 1.0));
        assert_eq!(
            set.taps[&TapName::Layer(0, Key::TopK)],
            [3.0, 1.0, 4.0, 0.0, 2.0, 5.0, 6.0, 7.0]
        );
        assert_eq!(set.taps.len(), 2);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_wrong_size_is_an_error() {
        let d = Dims::tiny();
        let dir = temp("size");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("00000.bin"), [0u8; 12]).unwrap();
        let index = r#"{"n_tokens":2,"tensors":[{"name":"l_out-0","occurrence":0,"ne":[3,1,1,1],"dtype":"f32","file":"00000.bin"}]}"#;
        fs::write(dir.join("index.json"), index).unwrap();
        let e = read(&dir, &d).unwrap_err().to_string();
        assert!(
            e.contains("l_out-0") && e.contains("256 per position"),
            "{e}"
        );
        fs::remove_dir_all(&dir).unwrap();
    }
}
