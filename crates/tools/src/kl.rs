//! Test 4: Artemis's distribution against llama.cpp's CPU backend, as mean KL divergence over
//! the KL corpus. `llama-perplexity --kl-divergence-base` writes llama.cpp's tokens and, for the second half of every
//! chunk, each position's log-probabilities quantized to 16 bits. `artemis kl` reads that file and scores its own logits
//! the way `llama-perplexity --kl-divergence` does (b11509, `tools/perplexity/perplexity.cpp`), so the two means
//! compare like for like.
use crate::test3::Status;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

pub const MAGIC: &[u8; 8] = b"_logits_";

/// A base file's header: the context each chunk was evaluated with, the vocabulary, and the number of chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub n_ctx: usize,
    pub n_vocab: usize,
    pub n_chunk: usize,
}

impl Header {
    /// u16s per record: two f32s (the scale and the smallest log-probability), then one per token, the vocabulary
    /// padded to an even count.
    pub fn record_u16s(&self) -> usize {
        2 * self.n_vocab.div_ceil(2) + 4
    }

    /// A chunk's first scored position: llama.cpp scores the second half of each chunk.
    pub fn first(&self) -> usize {
        self.n_ctx / 2
    }

    /// Scored positions per chunk: `first` to n_ctx − 2, each predicting the token after it.
    pub fn scored(&self) -> usize {
        self.n_ctx - 1 - self.first()
    }

    /// The file's length: the 20-byte header, the tokens, then every chunk's records.
    pub fn file_bytes(&self) -> u64 {
        20 + 4 * (self.n_ctx * self.n_chunk) as u64
            + 2 * (self.n_chunk * self.scored() * self.record_u16s()) as u64
    }
}

/// `-c` must be the base's context: llama.cpp's scores come from chunks of exactly that many tokens.
pub fn check_ctx(h: &Header, c: usize) -> Result<()> {
    if c != h.n_ctx {
        bail!(
            "-c {c} differs from the base's context of {}: the chunks must be the ones llama.cpp scored",
            h.n_ctx
        );
    }
    Ok(())
}

/// The base's tokens must start our tokenization of the corpus, as llama-perplexity reads it: special tokens as text
/// (no special parsing) and no escape processing (`--no-escape`).
pub fn check_corpus(base: &[u32], ours: &[u32]) -> Result<()> {
    if let Some(i) = base.iter().zip(ours).position(|(a, b)| a != b) {
        bail!(
            "the corpus's token {i} is {}, the base's {}: the base wasn't made from this corpus",
            ours[i],
            base[i]
        );
    }
    if ours.len() < base.len() {
        bail!(
            "the corpus gives {} tokens, fewer than the base's {}",
            ours.len(),
            base.len()
        );
    }
    Ok(())
}

/// A llama-perplexity KL base, read in file order.
pub struct Base {
    pub header: Header,
    pub tokens: Vec<u32>,
    reader: BufReader<File>,
    buf: Vec<u8>,
}

impl Base {
    /// Opens a KL base, refusing a file without the magic, an empty header, or a length its header doesn't explain.
    pub fn open(path: &Path) -> Result<Base> {
        let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let len = f.metadata()?.len();
        let mut reader = BufReader::with_capacity(1 << 20, f);
        let mut head = [0u8; 20];
        reader
            .read_exact(&mut head)
            .with_context(|| format!("{} is shorter than a KL base's header", path.display()))?;
        if &head[..8] != MAGIC {
            bail!(
                "{} is not a llama-perplexity KL base: it doesn't start with \"_logits_\"",
                path.display()
            );
        }
        let word = |i: usize| u32::from_le_bytes(head[i..i + 4].try_into().expect("4 bytes"));
        let (n_ctx, n_vocab, n_chunk) = (word(8), word(12) as i32, word(16) as i32);
        if n_ctx < 2 || n_vocab < 1 || n_chunk < 1 {
            bail!(
                "{}: a base of {n_ctx} positions, {n_vocab} tokens and {n_chunk} chunks scores nothing",
                path.display()
            );
        }
        let header = Header {
            n_ctx: n_ctx as usize,
            n_vocab: n_vocab as usize,
            n_chunk: n_chunk as usize,
        };
        if len != header.file_bytes() {
            bail!(
                "{}: {len} bytes, but its header ({n_ctx} positions, {n_vocab} tokens, {n_chunk} chunks) says {}",
                path.display(),
                header.file_bytes()
            );
        }
        let mut tb = vec![0u8; 4 * header.n_ctx * header.n_chunk];
        reader.read_exact(&mut tb)?;
        let tokens = tb
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        Ok(Base {
            header,
            tokens,
            reader,
            buf: Vec::new(),
        })
    }

    /// Chunk `c`'s tokens.
    pub fn chunk_tokens(&self, c: usize) -> &[u32] {
        let n = self.header.n_ctx;
        &self.tokens[c * n..(c + 1) * n]
    }

    /// The next record, in file order: chunk by chunk, each chunk's scored positions in order.
    pub fn next_record(&mut self, out: &mut Vec<u16>) -> Result<()> {
        let n = self.header.record_u16s();
        self.buf.resize(2 * n, 0);
        self.reader
            .read_exact(&mut self.buf)
            .context("reading the next record of the KL base")?;
        out.clear();
        out.extend(
            self.buf
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c)),
        );
        Ok(())
    }
}

/// ggml's nearest_int: round half to even through the float mantissa, for values below 2²².
fn nearest_int(f: f32) -> i32 {
    let v = f + 12_582_912.0;
    (v.to_bits() & 0x007f_ffff) as i32 - 0x0040_0000
}

/// The record llama.cpp writes for `logits` (perplexity.cpp's `log_softmax` with a uint16 output): the scale and the
/// smallest kept log-probability as two f32s, then each token's log-probability above it in 65,535 steps. Logits more
/// than 16 below the top are stored as 0. For tests and synthetic bases.
pub fn base_record(logits: &[f32]) -> Vec<u16> {
    let (mut max, mut min) = (logits[0], logits[0]);
    for &l in &logits[1..] {
        max = max.max(l);
        min = min.min(l);
    }
    let min = min.max(max - 16.0);
    let sum_exp: f64 = logits.iter().map(|&l| (l - max).exp() as f64).sum();
    let lse = sum_exp.ln() as f32;
    let min_log_prob = min - max - lse;
    let scale = (max - min) / 65535.0;
    let n = 2 * logits.len().div_ceil(2) + 4;
    let mut r = vec![0u16; n];
    let (s, m) = (scale.to_bits(), min_log_prob.to_bits());
    r[..4].copy_from_slice(&[s as u16, (s >> 16) as u16, m as u16, (m >> 16) as u16]);
    if scale != 0.0 {
        let inv = 1.0 / scale;
        for (q, &l) in r[4..].iter_mut().zip(logits) {
            *q = if l > min {
                nearest_int(inv * (l - min)) as u16
            } else {
                0
            };
        }
    }
    r
}

/// Writes a KL base in llama-perplexity's format: for tests and synthetic bases.
pub fn write_base(path: &Path, h: &Header, tokens: &[u32], records: &[Vec<u16>]) -> Result<()> {
    if tokens.len() != h.n_ctx * h.n_chunk || records.len() != h.n_chunk * h.scored() {
        bail!(
            "a base of {} chunks of {} needs {} tokens and {} records, not {} and {}",
            h.n_chunk,
            h.n_ctx,
            h.n_ctx * h.n_chunk,
            h.n_chunk * h.scored(),
            tokens.len(),
            records.len()
        );
    }
    let mut w =
        BufWriter::new(File::create(path).with_context(|| format!("creating {}", path.display()))?);
    w.write_all(MAGIC)?;
    for v in [h.n_ctx, h.n_vocab, h.n_chunk] {
        w.write_all(&(v as u32).to_le_bytes())?;
    }
    for t in tokens {
        w.write_all(&t.to_le_bytes())?;
    }
    for r in records {
        if r.len() != h.record_u16s() {
            bail!(
                "a record of {} u16s; this base's hold {}",
                r.len(),
                h.record_u16s()
            );
        }
        for q in r {
            w.write_all(&q.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

/// Running sums over scored positions, as `llama-perplexity --kl-divergence` keeps them.
#[derive(Clone, Debug, Default, Serialize)]
pub struct KlSum {
    pub count: usize,
    pub sum_kld: f64,
    pub sum_kld2: f64,
    pub sum_nll: f64,
    pub sum_nll2: f64,
    pub sum_nll_base: f64,
    pub sum_nll_base2: f64,
    pub same_top: usize,
}

/// llama.cpp's mean and its uncertainty: (sum/count, sqrt(variance/(count − 1))), the uncertainty 0 below 11 values.
fn mean_and_uncertainty(sum: f64, sum2: f64, count: usize) -> (f64, f64) {
    if count < 1 {
        return (0.0, 0.0);
    }
    let f = sum / count as f64;
    let df = sum2 / count as f64 - f * f;
    let df = if df > 0.0 && count > 10 {
        (df / (count - 1) as f64).sqrt()
    } else {
        0.0
    };
    (f, df)
}

impl KlSum {
    /// Scores one position: `logits` are ours, `record` is the base's for the same position, and `next_token` is the
    /// token that follows it. The arithmetic is perplexity.cpp's (f32 values, f64 sums): KL(base ‖ ours) over the tokens
    /// whose base log-probability is above −16.
    pub fn add(&mut self, logits: &[f32], record: &[u16], next_token: u32) {
        assert!(
            record.len() >= 4 + logits.len(),
            "a record of {} u16s for {} logits",
            record.len(),
            logits.len()
        );
        let tok = next_token as usize;
        let (mut max, mut imax) = (logits[0], 0);
        for (i, &l) in logits.iter().enumerate().skip(1) {
            if l > max {
                max = l;
                imax = i;
            }
        }
        let sum_exp: f64 = logits.iter().map(|&l| (l - max).exp() as f64).sum();
        let lse = sum_exp.ln() as f32;
        let scale = f32::from_bits(record[0] as u32 | (record[1] as u32) << 16);
        let min_log_prob = f32::from_bits(record[2] as u32 | (record[3] as u32) << 16);
        let q = &record[4..4 + logits.len()];
        let nll = max + lse - logits[tok];
        self.sum_nll += nll as f64;
        self.sum_nll2 += (nll * nll) as f64;
        let nll_base = -(scale * q[tok] as f32 + min_log_prob);
        self.sum_nll_base += nll_base as f64;
        self.sum_nll_base2 += (nll_base * nll_base) as f64;
        let top = max + lse;
        let (mut sum, mut imax_base, mut best) = (0f64, 0usize, 0f32);
        for (i, (&qi, &l)) in q.iter().zip(logits).enumerate() {
            let p_log_base = scale * qi as f32 + min_log_prob;
            if i == 0 || p_log_base > best {
                best = p_log_base;
                imax_base = i;
            }
            if p_log_base > -16.0 {
                sum += (p_log_base.exp() * (p_log_base - l + top)) as f64;
            }
        }
        self.sum_kld += sum;
        self.sum_kld2 += sum * sum;
        self.count += 1;
        if imax == imax_base {
            self.same_top += 1;
        }
    }

    /// Mean KL divergence and its uncertainty.
    pub fn kld(&self) -> (f64, f64) {
        mean_and_uncertainty(self.sum_kld, self.sum_kld2, self.count)
    }

    /// Our perplexity on the scored tokens and its uncertainty.
    pub fn ppl(&self) -> (f64, f64) {
        let (f, df) = mean_and_uncertainty(self.sum_nll, self.sum_nll2, self.count);
        (f.exp(), f.exp() * df)
    }

    /// The share of positions where our top token is the base's.
    pub fn same_top(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.same_top as f64 / self.count as f64
        }
    }
}

/// What `artemis kl` writes.
#[derive(Debug, Serialize, Deserialize)]
pub struct KlReport {
    pub n_ctx: usize,
    pub chunks: usize,
    /// The base's `n_chunk`. Test 4 needs `chunks == base_chunks`. Reports written before this field existed lack
    /// it, and test 4 refuses them.
    #[serde(default)]
    pub base_chunks: Option<usize>,
    pub count: usize,
    /// A NaN is written as null (JSON has none) and read back as NaN, so test 4 can fail it cleanly.
    #[serde(deserialize_with = "nan_if_null")]
    pub kld: f64,
    #[serde(deserialize_with = "nan_if_null")]
    pub kld_unc: f64,
    #[serde(deserialize_with = "nan_if_null")]
    pub ppl: f64,
    pub same_top: f64,
    pub seconds: f64,
    pub cache_slots: usize,
}

/// A number, or null for the NaN that `serde_json` writes in its place.
fn nan_if_null<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
}

/// The mean KL divergence and its uncertainty from llama-perplexity's summary line, "Mean    KLD:   0.001234 ±
/// 0.000012". The ± may arrive mangled by a console code page, so the two numbers are read as the line's first two.
pub fn parse_llama_kld(text: &str) -> Result<(f64, f64)> {
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with("Mean") && l.contains("KLD:"))
        .context("no \"Mean KLD\" line in llama-perplexity's output; did the --kl-divergence run finish?")?;
    let after = line.split_once("KLD:").expect("the line holds KLD:").1;
    // Anything but a number's characters separates: the ± may be glued to the digits, or mangled into several bytes.
    let spaced: String = after
        .chars()
        .map(|c| {
            if c.is_ascii_digit() || "+-.eE".contains(c) {
                c
            } else {
                ' '
            }
        })
        .collect();
    let nums: Vec<f64> = spaced
        .split_whitespace()
        .filter_map(|w| w.parse().ok())
        .collect();
    match nums[..] {
        [mean, unc, ..] => Ok((mean, unc)),
        _ => bail!("can't read the mean KL divergence from {line:?}"),
    }
}

/// Test 4's factor: Artemis's mean KL from llama.cpp's CPU may be at most 1.5 × llama.cpp's own
/// Vulkan-vs-CPU mean KL.
pub const KL_FACTOR: f64 = 1.5;

#[derive(Debug, Serialize)]
pub struct Verdict4 {
    pub status: Status,
    pub artemis: f64,
    pub floor: f64,
    pub bound: f64,
    /// Why the measurement can't be judged (a partial or old report); the status is then a fail.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub problem: Option<String>,
}

/// What is wrong with a report as a test 4 measurement: it must cover every chunk of the base (decision of
/// 2026-10-09).
pub fn report_problem(r: &KlReport) -> Option<String> {
    if !r.kld.is_finite() {
        return Some(
            "the KL report's mean KL divergence is not a number; the run scored a non-finite value"
                .into(),
        );
    }
    match r.base_chunks {
        None => Some(
            "the KL report has no base_chunks field (written by an older `artemis kl`); rerun it"
                .into(),
        ),
        Some(n) if r.chunks != n => Some(format!(
            "scored {} of the KL base's {n} chunks; test 4 needs all of them",
            r.chunks
        )),
        Some(_) => None,
    }
}

/// Test 4's rule: pass iff the report covers the whole base and artemis ≤ KL_FACTOR × floor. A NaN fails.
pub fn test4(report: &KlReport, floor: f64) -> Verdict4 {
    let artemis = report.kld;
    let bound = KL_FACTOR * floor;
    let problem = report_problem(report);
    Verdict4 {
        status: if problem.is_none() && artemis <= bound {
            Status::Pass
        } else {
            Status::Fail
        },
        artemis,
        floor,
        bound,
        problem,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen36::testing::Rng;

    /// A fresh path in the temp directory for one test's base.
    fn temp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("artemis-kl-{}-{name}.bin", std::process::id()))
    }

    fn logits(rng: &mut Rng, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.uniform(-8.0, 8.0)).collect()
    }

    /// The exact KL(p ‖ q) of two logit vectors, in f64.
    fn exact_kl(p: &[f32], q: &[f32]) -> f64 {
        let lsm = |x: &[f32]| {
            let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = x.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
            x.iter().map(|&v| v as f64 - lse).collect::<Vec<f64>>()
        };
        let (lp, lq) = (lsm(p), lsm(q));
        lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
    }

    const H: Header = Header {
        n_ctx: 8,
        n_vocab: 1001,
        n_chunk: 2,
    };

    /// A base of H's shape over random logits, and the logits behind each record.
    fn synthetic(path: &Path) -> Vec<Vec<f32>> {
        let mut rng = Rng::new(4);
        let tokens: Vec<u32> = (0..H.n_ctx * H.n_chunk)
            .map(|_| (rng.next_u64() % H.n_vocab as u64) as u32)
            .collect();
        let ls: Vec<Vec<f32>> = (0..H.n_chunk * H.scored())
            .map(|_| logits(&mut rng, H.n_vocab))
            .collect();
        let records: Vec<Vec<u16>> = ls.iter().map(|l| base_record(l)).collect();
        write_base(path, &H, &tokens, &records).unwrap();
        ls
    }

    #[test]
    fn the_header_sizes_follow_llama_cpp() {
        // Test 4's base: 25 chunks of 4,096, the real vocabulary.
        let h = Header {
            n_ctx: 4096,
            n_vocab: 248_320,
            n_chunk: 25,
        };
        assert_eq!(
            (h.first(), h.scored(), h.record_u16s()),
            (2048, 2047, 248_324)
        );
        assert_eq!(h.file_bytes(), 25_416_371_020);
        assert_eq!(H.record_u16s(), 1006, "an odd vocabulary is padded to even");
    }

    #[test]
    fn nearest_int_rounds_half_to_even() {
        assert_eq!(
            [0.4, 0.5, 1.5, 2.5, 65534.6].map(nearest_int),
            [0, 0, 2, 2, 65535]
        );
    }

    #[test]
    fn a_synthetic_base_round_trips_and_scores_itself_as_zero() {
        let path = temp("round-trip");
        let ls = synthetic(&path);
        let mut b = Base::open(&path).unwrap();
        assert_eq!(b.header, H);
        let mut sum = KlSum::default();
        let mut rec = Vec::new();
        for c in 0..H.n_chunk {
            let toks = b.chunk_tokens(c).to_vec();
            for j in 0..H.scored() {
                b.next_record(&mut rec).unwrap();
                assert_eq!(rec, base_record(&ls[c * H.scored() + j]));
                sum.add(&ls[c * H.scored() + j], &rec, toks[H.first() + j + 1]);
            }
        }
        assert!(b.next_record(&mut rec).is_err(), "every record was read");
        std::fs::remove_file(&path).unwrap();
        let (kld, _) = sum.kld();
        // llama.cpp's formula sums f32 terms, so even identical distributions score about 1e-6 (−1.2e-6 here), three
        // orders below the KL test 4 measures.
        eprintln!("KL of the base's own logits: {kld:.3e}");
        assert!(kld.abs() < 1e-5, "{kld:e}");
        assert_eq!(sum.same_top(), 1.0);
        assert_eq!(sum.count, 6);
    }

    #[test]
    fn shifted_logits_score_the_exact_kl() {
        let mut rng = Rng::new(9);
        let mut sum = KlSum::default();
        let mut exact = 0f64;
        for _ in 0..20 {
            let p = logits(&mut rng, 1001);
            let q: Vec<f32> = p.iter().map(|&v| v + rng.uniform(-0.3, 0.3)).collect();
            sum.add(&q, &base_record(&p), 7);
            exact += exact_kl(&p, &q);
        }
        let exact = exact / 20.0;
        let (kld, _) = sum.kld();
        eprintln!("mean KL {kld:.6e}, exact {exact:.6e}");
        assert!((kld - exact).abs() < 1e-3 * exact, "{kld:e} vs {exact:e}");
    }

    #[test]
    fn a_base_of_the_wrong_length_is_refused() {
        let path = temp("short");
        synthetic(&path);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 2]).unwrap();
        let e = Base::open(&path).err().unwrap().to_string();
        std::fs::remove_file(&path).unwrap();
        assert!(e.contains("12154 bytes, but its header"), "{e}");
        assert!(
            e.contains("8 positions, 1001 tokens, 2 chunks) says 12156"),
            "{e}"
        );
    }

    #[test]
    fn a_file_without_the_magic_is_refused() {
        let path = temp("magic");
        synthetic(&path);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] = b'X';
        std::fs::write(&path, &bytes).unwrap();
        let e = Base::open(&path).err().unwrap().to_string();
        std::fs::remove_file(&path).unwrap();
        assert!(e.contains("is not a llama-perplexity KL base"), "{e}");
    }

    #[test]
    fn kl_refuses_a_context_other_than_the_bases() {
        let h = Header {
            n_ctx: 4096,
            n_vocab: 248_320,
            n_chunk: 25,
        };
        assert!(check_ctx(&h, 4096).is_ok());
        let e = check_ctx(&h, 2048).unwrap_err().to_string();
        assert!(e.contains("-c 2048") && e.contains("4096"), "{e}");
        assert!(!e.contains('\n'), "{e}");
    }

    #[test]
    fn the_corpus_check_names_the_first_difference() {
        assert!(check_corpus(&[1, 2, 3], &[1, 2, 3, 4]).is_ok());
        let e = check_corpus(&[1, 2, 3], &[1, 9, 3])
            .unwrap_err()
            .to_string();
        assert!(e.contains("token 1 is 9, the base's 2"), "{e}");
        let e = check_corpus(&[1, 2, 3], &[1, 2]).unwrap_err().to_string();
        assert!(e.contains("2 tokens, fewer than the base's 3"), "{e}");
    }

    #[test]
    fn parse_reads_llama_perplexitys_summary() {
        let tail = "====== KL divergence statistics ======\n\
                    Mean    KLD:   0.012345 ±   0.000321\n\
                    Maximum KLD:   3.141593\n\
                    Median  KLD:   0.004321\n";
        assert_eq!(parse_llama_kld(tail).unwrap(), (0.012345, 0.000321));
        let mangled = "Mean    KLD:   0.002000 Â±   0.000010\n";
        assert_eq!(parse_llama_kld(mangled).unwrap(), (0.002, 0.00001));
        let e = parse_llama_kld("Final estimate: PPL = 7.1\n")
            .unwrap_err()
            .to_string();
        assert!(e.contains("no \"Mean KLD\" line"), "{e}");
    }

    fn report(kld: f64, chunks: usize, base_chunks: Option<usize>) -> KlReport {
        KlReport {
            n_ctx: 4096,
            chunks,
            base_chunks,
            count: chunks * 2047,
            kld,
            kld_unc: 0.0,
            ppl: 7.0,
            same_top: 0.9,
            seconds: 1.0,
            cache_slots: 100,
        }
    }

    #[test]
    fn test4_passes_up_to_one_and_a_half_times_the_floor() {
        let full = |kld| report(kld, 25, Some(25));
        assert_eq!(test4(&full(0.375), 0.25).status, Status::Pass);
        assert_eq!(test4(&full(0.375_000_1), 0.25).status, Status::Fail);
        assert_eq!(test4(&full(f64::NAN), 0.25).status, Status::Fail);
        let v = test4(&full(0.1), 0.25);
        assert_eq!((v.artemis, v.floor, v.bound), (0.1, 0.25, 0.375));
        assert!(v.problem.is_none());
    }

    #[test]
    fn test4_fails_a_partial_measurement_however_good() {
        let v = test4(&report(0.0001, 2, Some(25)), 0.25);
        assert_eq!(v.status, Status::Fail);
        assert_eq!(
            v.problem.as_deref(),
            Some("scored 2 of the KL base's 25 chunks; test 4 needs all of them")
        );
    }

    #[test]
    fn parse_reads_a_plus_minus_glued_to_the_digits() {
        let glued = "Mean    KLD:   0.002000\u{b1}0.000010\n";
        assert_eq!(parse_llama_kld(glued).unwrap(), (0.002, 0.00001));
    }

    #[test]
    fn test4_fails_a_nan_measurement_cleanly() {
        // serde_json writes a NaN as null.
        let json = serde_json::to_string(&report(f64::NAN, 25, Some(25))).unwrap();
        assert!(json.contains("\"kld\":null"), "{json}");
        let r: KlReport = serde_json::from_str(&json).unwrap();
        assert!(r.kld.is_nan());
        let v = test4(&r, 0.25);
        assert_eq!(v.status, Status::Fail);
        assert!(v.problem.unwrap().contains("not a number"));
    }

    #[test]
    fn test4_fails_a_report_without_base_chunks() {
        let old = r#"{"n_ctx":4096,"chunks":25,"count":51175,"kld":0.001,"kld_unc":0.0,"ppl":7.0,
            "same_top":0.9,"seconds":1.0,"cache_slots":100}"#;
        let r: KlReport = serde_json::from_str(old).unwrap();
        let v = test4(&r, 0.25);
        assert_eq!(v.status, Status::Fail);
        assert!(v.problem.unwrap().contains("base_chunks"));
    }
}
