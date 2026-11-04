//! The per-user tuned file (milestone-6b spec 7): what `artemis setup` measured on this machine, layered under
//! `artemis.toml`. It lives in the local (not roaming) state folder because its values belong to one PC. The toml crate
//! here has no serializer, so `render` writes the file by hand.
use crate::config::{one_line, var, Env, ServeArgs};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The environment variable that moves the tuned file (a path) or turns the layer off (`off`).
pub const TUNED_ENV: &str = "ARTEMIS_TUNED";

/// The schema of the file's `[setup]` table; a file with another number is refused.
pub const TUNED_SCHEMA: u32 = 1;

/// The version a tuned file is stamped with and compared against: this build's.
pub const ARTEMIS_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The words every refusal of a tuned file ends with.
pub(crate) const REMEDY: &str = "run artemis setup --force";

/// Where the tuned file is, or that there is none to have.
#[derive(Clone, Debug, PartialEq)]
pub enum TunedLocation {
    /// `ARTEMIS_TUNED=off`: nothing is read or written.
    Off,
    /// The base variable of the default path is unset: nothing is read or written, and `serve` says so once.
    Unplaced,
    /// The file's path (it may not exist).
    At(PathBuf),
}

/// The `config_warning` text for an unplaced location: it names the variable this OS's default path needs.
pub fn unplaced_warning(windows: bool) -> String {
    let base = if windows { "LOCALAPPDATA" } else { "HOME" };
    format!(
        "{base} is not set, so the tuned settings are not read or written; \
set ARTEMIS_TUNED to a file path to use them"
    )
}

/// `ARTEMIS_TUNED` (`off` in any case, or a path), else `%LOCALAPPDATA%\artemis\tuned.toml` on Windows, else
/// `$XDG_STATE_HOME/artemis/tuned.toml`, else `$HOME/.local/state/artemis/tuned.toml`; `Unplaced` when the base
/// variable is unset.
pub fn tuned_location(env: Env<'_>, windows: bool) -> TunedLocation {
    if let Some(value) = env(TUNED_ENV)
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
    {
        return if value.eq_ignore_ascii_case("off") {
            TunedLocation::Off
        } else {
            TunedLocation::At(PathBuf::from(value))
        };
    }
    let base = if windows {
        var(env, "LOCALAPPDATA")
    } else {
        var(env, "XDG_STATE_HOME")
            .or_else(|| var(env, "HOME").map(|h| h.join(".local").join("state")))
    };
    match base {
        Some(base) => TunedLocation::At(base.join("artemis").join("tuned.toml")),
        None => TunedLocation::Unplaced,
    }
}

/// The settings a setup run chose. Every one is optional: a file may choose only some.
#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TunedSpeculation {
    pub enabled: Option<bool>,
    pub sources: Option<Vec<String>>,
    pub cost_slope_ms: Option<f64>,
    pub mtp_cost_ms: Option<f64>,
}

/// What setup measured and saw: the file's `[setup]` table. Nothing here is applied to the config; it says which
/// machine and build the choices belong to.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetupFacts {
    pub schema: u32,
    pub artemis_version: String,
    pub finished: String,
    pub seconds: f64,
    pub limit_hit: bool,
    pub target: bool,
    pub gpu_name: String,
    pub gpu_uuid: String,
    pub driver: String,
    pub vram_bytes: u64,
    pub budget_bytes: u64,
    pub cache_slots: u32,
    pub link_h2d_gbps: f64,
    pub host_free_bytes: u64,
    pub pool_gib: u32,
    pub model_bytes: u64,
    pub expected_tok_s: f64,
    pub measured_slope_ms: Option<f64>,
    pub measured_mtp_d1_ms: Option<f64>,
    pub scores: BTreeMap<String, f64>,
}

/// A tuned file.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tuned {
    pub device_uuid: Option<String>,
    #[serde(default)]
    pub speculation: TunedSpeculation,
    pub setup: SetupFacts,
}

/// The file's text: a header, each chosen value under a `#` line from `why` (keyed `device_uuid`,
/// `speculation.enabled`, ...), then the `[setup]` facts. Keys and tables carry the names `artemis.toml` uses, so a
/// user can copy a line across.
pub fn render(t: &Tuned, why: &BTreeMap<String, String>) -> String {
    use std::fmt::Write;
    let mut o = String::new();
    let facts = &t.setup;
    let _ = writeln!(
        o,
        "# Written by `artemis setup` on {}. These are the settings it measured on this PC.",
        facts.finished.replace(['\n', '\r'], " ")
    );
    o.push_str("# A key in artemis.toml, or a flag, wins over the same key here.\n");
    o.push_str("# Run `artemis setup --force` to measure again. The [setup] table records what was measured.\n");
    // A reason goes above its value, one `#` line for each line of text.
    let reason = |o: &mut String, key: &str| {
        if let Some(text) = why.get(key) {
            for line in text.lines() {
                let _ = writeln!(o, "# {line}");
            }
        }
    };
    if let Some(uuid) = &t.device_uuid {
        o.push('\n');
        reason(&mut o, "device_uuid");
        let _ = writeln!(o, "device_uuid = {}", quote(uuid));
    }
    let sp = &t.speculation;
    if sp != &TunedSpeculation::default() {
        o.push_str("\n[speculation]\n");
        if let Some(v) = sp.enabled {
            reason(&mut o, "speculation.enabled");
            let _ = writeln!(o, "enabled = {v}");
        }
        if let Some(v) = &sp.sources {
            reason(&mut o, "speculation.sources");
            let list: Vec<String> = v.iter().map(|s| quote(s)).collect();
            let _ = writeln!(o, "sources = [{}]", list.join(", "));
        }
        if let Some(v) = sp.cost_slope_ms {
            reason(&mut o, "speculation.cost_slope_ms");
            let _ = writeln!(o, "cost_slope_ms = {}", float(v));
        }
        if let Some(v) = sp.mtp_cost_ms {
            reason(&mut o, "speculation.mtp_cost_ms");
            let _ = writeln!(o, "mtp_cost_ms = {}", float(v));
        }
    }
    o.push_str("\n[setup]\n");
    let _ = writeln!(o, "schema = {}", facts.schema);
    let _ = writeln!(o, "artemis_version = {}", quote(&facts.artemis_version));
    let _ = writeln!(o, "finished = {}", quote(&facts.finished));
    let _ = writeln!(o, "seconds = {}", float(facts.seconds));
    let _ = writeln!(o, "limit_hit = {}", facts.limit_hit);
    let _ = writeln!(o, "target = {}", facts.target);
    let _ = writeln!(o, "gpu_name = {}", quote(&facts.gpu_name));
    let _ = writeln!(o, "gpu_uuid = {}", quote(&facts.gpu_uuid));
    let _ = writeln!(o, "driver = {}", quote(&facts.driver));
    let _ = writeln!(o, "vram_bytes = {}", facts.vram_bytes);
    let _ = writeln!(o, "budget_bytes = {}", facts.budget_bytes);
    let _ = writeln!(o, "cache_slots = {}", facts.cache_slots);
    let _ = writeln!(o, "link_h2d_gbps = {}", float(facts.link_h2d_gbps));
    let _ = writeln!(o, "host_free_bytes = {}", facts.host_free_bytes);
    let _ = writeln!(o, "pool_gib = {}", facts.pool_gib);
    let _ = writeln!(o, "model_bytes = {}", facts.model_bytes);
    let _ = writeln!(o, "expected_tok_s = {}", float(facts.expected_tok_s));
    if let Some(v) = facts.measured_slope_ms {
        let _ = writeln!(o, "measured_slope_ms = {}", float(v));
    }
    if let Some(v) = facts.measured_mtp_d1_ms {
        let _ = writeln!(o, "measured_mtp_d1_ms = {}", float(v));
    }
    o.push_str("\n[setup.scores]\n");
    for (name, score) in &facts.scores {
        let _ = writeln!(o, "{} = {}", key(name), float(*score));
    }
    o
}

/// The 1-based line of a byte offset.
fn line_of(text: &str, offset: usize) -> usize {
    text.as_bytes()[..offset.min(text.len())]
        .iter()
        .filter(|&&b| b == b'\n')
        .count()
        + 1
}

/// A TOML basic string.
fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04X}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A table key: bare when it can be, else quoted.
fn key(name: &str) -> String {
    if !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        name.to_string()
    } else {
        quote(name)
    }
}

/// A float as TOML spells it: the shortest text that reads back as the same number, and always a float.
fn float(v: f64) -> String {
    if v.is_nan() {
        "nan".into()
    } else if v.is_infinite() {
        if v > 0.0 { "inf" } else { "-inf" }.into()
    } else {
        format!("{v:?}")
    }
}

/// Reads a tuned file's text. An error names `path` and the key, and ends with `run artemis setup --force`.
pub fn parse(text: &str, path: &Path) -> Result<Tuned, String> {
    let fail = |what: String| one_line(&format!("{}: {what}", path.display())) + "; " + REMEDY;
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let table: toml::Table = text.parse().map_err(|e: toml::de::Error| {
        let line = e.span().map_or(String::new(), |s| {
            format!("line {}: ", line_of(text, s.start))
        });
        fail(format!("{line}{}", e.message()))
    })?;
    // The schema first: a newer file may hold keys this build does not know, and the schema is the better thing to say.
    if let Some(schema) = table.get("setup").and_then(|v| v.get("schema")) {
        if schema.as_integer() != Some(i64::from(TUNED_SCHEMA)) {
            let found = schema
                .as_integer()
                .map_or("not a whole number".to_string(), |n| n.to_string());
            return Err(fail(format!(
                "setup.schema is {found} and this Artemis reads schema {TUNED_SCHEMA}"
            )));
        }
    }
    serde_path_to_error::deserialize::<_, Tuned>(toml::Value::Table(table)).map_err(|e| {
        let key = e.path().to_string();
        let message = e.inner().to_string();
        if key == "." {
            fail(message)
        } else {
            fail(format!("{key}: {message}"))
        }
    })
}

/// Writes `text` to `path` so that a reader sees the old file, the new file or none: the folder is created, a
/// `<file name>.tmp-*` left by a killed run is removed, the text goes to `<path>.tmp-<pid>`, which is synced and then
/// renamed over `path`. On any error the temp file is removed and the error names `path`.
pub fn write_atomic(path: &Path, text: &str) -> Result<(), String> {
    let fail = |e: &dyn std::fmt::Display| format!("cannot write {}: {e}", path.display());
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty());
    if let Some(dir) = dir {
        std::fs::create_dir_all(dir).map_err(|e| fail(&e))?;
    }
    let name = path
        .file_name()
        .ok_or_else(|| fail(&"the path has no file name"))?
        .to_string_lossy()
        .into_owned();
    // A temp file a killed run left behind goes before this run makes its own.
    let prefix = format!("{name}.tmp-");
    if let Ok(entries) = std::fs::read_dir(dir.unwrap_or(Path::new("."))) {
        for entry in entries.flatten() {
            if entry.file_name().to_string_lossy().starts_with(&prefix) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
    let tmp = path.with_file_name(format!("{prefix}{}", std::process::id()));
    let written = (|| -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp, path)
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(fail(&e));
    }
    // Where a folder can be synced, the rename reaches the disk too; Windows has no such call.
    #[cfg(unix)]
    if let Some(dir) = dir {
        if let Ok(d) = std::fs::File::open(dir) {
            let _ = d.sync_all();
        }
    }
    Ok(())
}

/// The dotted keys the user has set, by their file (`file`, parsed) or a flag: `speculation.enabled` for
/// `--speculation-enabled`, `speculation.sources`, `device_uuid`, and the engine keys `ctx`, `cache_slots` and
/// `f32_prefill`.
pub fn user_keys(file: Option<&toml::Table>, args: &ServeArgs) -> BTreeSet<String> {
    fn walk(prefix: &str, table: &toml::Table, out: &mut BTreeSet<String>) {
        for (k, v) in table {
            let name = if prefix.is_empty() {
                k.clone()
            } else {
                format!("{prefix}.{k}")
            };
            match v {
                toml::Value::Table(t) => walk(&name, t, out),
                _ => {
                    out.insert(name);
                }
            }
        }
    }
    let mut keys = BTreeSet::new();
    if let Some(file) = file {
        walk("", file, &mut keys);
    }
    let flags = [
        (args.speculation_enabled.is_some(), "speculation.enabled"),
        (args.speculation_sources.is_some(), "speculation.sources"),
        (args.device_uuid.is_some(), "device_uuid"),
        (args.ctx.is_some(), "ctx"),
        (args.cache_slots.is_some(), "cache_slots"),
        (args.mtp_expert_slots.is_some(), "mtp_expert_slots"),
        (args.f32_prefill.is_some(), "f32_prefill"),
    ];
    for (set, name) in flags {
        if set {
            keys.insert(name.to_string());
        }
    }
    keys
}

/// One sentence for each way the file is out of date for this build: another Artemis version, or a model of another
/// size (`model_bytes` is `None` when the model's size is unknown). The GPU and the driver are compared when the engine
/// loads.
pub fn stale(facts: &SetupFacts, version: &str, model_bytes: Option<u64>) -> Vec<String> {
    let mut out = Vec::new();
    if facts.artemis_version != version {
        out.push(format!(
            "The tuned settings were measured by Artemis {}; this is Artemis {version}. \
             Run artemis setup --force to measure again.",
            facts.artemis_version
        ));
    }
    if let Some(bytes) = model_bytes.filter(|&b| b != facts.model_bytes) {
        out.push(format!(
            "The tuned settings were measured with a model of {} bytes; this model has {bytes}. \
             Run artemis setup --force to measure again.",
            facts.model_bytes
        ));
    }
    out
}

#[cfg(test)]
pub(crate) fn sample() -> Tuned {
    Tuned {
        device_uuid: Some("0123456789abcdef0123456789abcdef".into()),
        speculation: TunedSpeculation {
            enabled: Some(true),
            sources: Some(vec!["lookup".into(), "mtp".into()]),
            cost_slope_ms: Some(4.25),
            mtp_cost_ms: Some(1.5),
        },
        setup: SetupFacts {
            schema: TUNED_SCHEMA,
            artemis_version: ARTEMIS_VERSION.into(),
            finished: "2026-11-02T10:11:12Z".into(),
            seconds: 187.5,
            limit_hit: false,
            target: true,
            gpu_name: "AMD Radeon RX 9070 XT".into(),
            gpu_uuid: "0123456789abcdef0123456789abcdef".into(),
            driver: "2.0.355".into(),
            vram_bytes: 17_095_000_000,
            budget_bytes: 16_106_127_360,
            cache_slots: 112,
            link_h2d_gbps: 25.5,
            host_free_bytes: 20_000_000_000,
            pool_gib: 8,
            model_bytes: 22_000_000_000,
            expected_tok_s: 137.0,
            measured_slope_ms: Some(4.5),
            measured_mtp_d1_ms: None,
            scores: BTreeMap::from([
                ("lookup".to_string(), 120.5),
                ("lookup+mtp".to_string(), 137.0),
            ]),
        },
    }
}

#[cfg(test)]
pub(crate) fn sample_why() -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            "device_uuid".to_string(),
            "two discrete GPUs were found; this one measured fastest".to_string(),
        ),
        (
            "speculation.enabled".to_string(),
            "speculation measured 1.24 times faster than plain decode".to_string(),
        ),
        (
            "speculation.sources".to_string(),
            "lookup with the MTP block beat lookup alone".to_string(),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Scratch {
            let dir = std::env::temp_dir()
                .join(format!("artemis-tuned-test-{}-{name}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let owned: Vec<(String, String)> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |name| {
            owned
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone())
        }
    }

    #[test]
    fn tuned_location_rules() {
        let at = |p: &str| TunedLocation::At(PathBuf::from(p));
        // off, in any case, and a path
        for off in ["off", "OFF", "Off"] {
            let env = env_of(&[("ARTEMIS_TUNED", off), ("LOCALAPPDATA", "C:/L")]);
            assert_eq!(tuned_location(&env, true), TunedLocation::Off);
        }
        let env = env_of(&[("ARTEMIS_TUNED", "x/y.toml"), ("LOCALAPPDATA", "C:/L")]);
        assert_eq!(tuned_location(&env, true), at("x/y.toml"));
        // an empty value is unset
        let env = env_of(&[("ARTEMIS_TUNED", ""), ("LOCALAPPDATA", "C:/L")]);
        assert_eq!(
            tuned_location(&env, true),
            TunedLocation::At(Path::new("C:/L").join("artemis").join("tuned.toml"))
        );
        // the Windows default
        let env = env_of(&[("LOCALAPPDATA", "C:/L"), ("HOME", "/h")]);
        assert_eq!(
            tuned_location(&env, true),
            TunedLocation::At(Path::new("C:/L").join("artemis").join("tuned.toml"))
        );
        // the two others
        let env = env_of(&[("XDG_STATE_HOME", "/xs"), ("HOME", "/h")]);
        assert_eq!(
            tuned_location(&env, false),
            TunedLocation::At(Path::new("/xs").join("artemis").join("tuned.toml"))
        );
        let env = env_of(&[("HOME", "/h")]);
        assert_eq!(
            tuned_location(&env, false),
            TunedLocation::At(
                Path::new("/h")
                    .join(".local")
                    .join("state")
                    .join("artemis")
                    .join("tuned.toml")
            )
        );
        // no base variable
        assert_eq!(tuned_location(&env_of(&[]), false), TunedLocation::Unplaced);
        assert_eq!(
            tuned_location(&env_of(&[("HOME", "/h")]), true),
            TunedLocation::Unplaced
        );
    }

    #[test]
    fn render_then_parse_round_trips() {
        let t = sample();
        let why = sample_why();
        let text = render(&t, &why);
        assert_eq!(parse(&text, Path::new("tuned.toml")).unwrap(), t);
        for line in why.values() {
            assert!(text.contains(&format!("# {line}")), "{line}\n{text}");
        }
        assert!(!text.contains('\u{2014}') && !text.contains('\u{2013}'));
        assert!(text.contains("artemis setup --force"), "{text}");
        assert!(text.contains("2026-11-02T10:11:12Z"), "{text}");
        assert!(text.lines().all(|l| l.len() <= 120), "{text}");
        // keys are named as artemis.toml names them
        assert!(text.contains("\n[speculation]\n"), "{text}");
        assert!(
            text.contains("\nsources = [\"lookup\", \"mtp\"]\n"),
            "{text}"
        );
        assert!(
            text.contains("\ndevice_uuid = \"0123456789abcdef"),
            "{text}"
        );

        // the options at their other values, and an awkward string
        let mut u = sample();
        u.device_uuid = None;
        u.speculation = TunedSpeculation::default();
        u.setup.measured_slope_ms = None;
        u.setup.measured_mtp_d1_ms = Some(1.25);
        u.setup.limit_hit = true;
        u.setup.gpu_name = "Card \"X\" \\ one\ttwo".into();
        u.setup.seconds = 3.0;
        u.setup.scores.insert("a b\"c".into(), 0.5);
        let text = render(&u, &BTreeMap::new());
        assert_eq!(parse(&text, Path::new("tuned.toml")).unwrap(), u);
        assert!(!text.contains("[speculation]"), "{text}");
        assert!(!text.contains("device_uuid"), "{text}");
    }

    #[test]
    fn an_unknown_key_or_schema_is_refused() {
        let good = render(&sample(), &sample_why());
        let path = Path::new("T/tuned.toml");
        let refused = |text: &str, needles: &[&str]| {
            let e = parse(text, path).unwrap_err();
            assert!(e.contains("T/tuned.toml"), "{e}");
            assert!(e.ends_with("run artemis setup --force"), "{e}");
            for n in needles {
                assert!(e.contains(n), "{n} in {e}");
            }
            assert!(!e.contains('\n'), "{e}");
        };
        // an unknown key at the top, in a table, and in [setup]
        refused(&format!("bogus = 1\n{good}"), &["bogus"]);
        refused(
            &good.replace("[speculation]\n", "[speculation]\nmax_draft = 4\n"),
            &["max_draft"],
        );
        refused(
            &good.replace("[setup]\n", "[setup]\nnew_fact = 1\n"),
            &["new_fact"],
        );
        // another schema, which names the schema even if the newer file has keys this one lacks
        refused(&good.replace("schema = 1", "schema = 2"), &["schema"]);
        refused(
            &good
                .replace("schema = 1", "schema = 2")
                .replace("[setup]\n", "[setup]\nnew_fact = 1\n"),
            &["schema"],
        );
        // a wrong type names the key
        refused(
            &good.replace("sources = [\"lookup\", \"mtp\"]", "sources = 3"),
            &["speculation.sources"],
        );
        // a missing fact, and text that is not TOML
        refused(
            &good.replace("cache_slots = 112\n", ""),
            &["setup", "cache_slots"],
        );
        refused("this is [not toml", &[]);
    }

    #[test]
    fn a_stale_file_warns() {
        let facts = sample().setup;
        assert!(stale(&facts, ARTEMIS_VERSION, Some(facts.model_bytes)).is_empty());
        assert!(stale(&facts, ARTEMIS_VERSION, None).is_empty());
        let other = stale(&facts, "9.9.9", Some(facts.model_bytes));
        assert_eq!(other.len(), 1);
        assert!(other[0].contains("9.9.9") && other[0].contains(ARTEMIS_VERSION));
        assert!(other[0].contains("artemis setup --force"), "{}", other[0]);
        let size = stale(&facts, ARTEMIS_VERSION, Some(facts.model_bytes + 1));
        assert_eq!(size.len(), 1);
        assert!(size[0].contains("model"), "{}", size[0]);
        assert!(size[0].contains("artemis setup --force"), "{}", size[0]);
        assert_eq!(stale(&facts, "9.9.9", Some(1)).len(), 2);
    }

    #[test]
    fn write_atomic_leaves_one_file_or_none() {
        let s = Scratch::new("atomic");
        let path = s.0.join("sub").join("tuned.toml");
        // the folder is created; one file is left
        write_atomic(&path, "one\n").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "one\n");
        let names = |dir: &Path| -> Vec<String> {
            let mut v: Vec<String> = std::fs::read_dir(dir)
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            v.sort();
            v
        };
        assert_eq!(names(path.parent().unwrap()), ["tuned.toml"]);

        // a temp file a killed run left is removed by the next write; another file stays
        let stale = path.parent().unwrap().join("tuned.toml.tmp-1234");
        std::fs::write(&stale, "half").unwrap();
        let other = path.parent().unwrap().join("other.toml.tmp-1234");
        std::fs::write(&other, "x").unwrap();
        write_atomic(&path, "two\n").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "two\n");
        assert_eq!(
            names(path.parent().unwrap()),
            ["other.toml.tmp-1234", "tuned.toml"]
        );

        // a folder that is a file: Err naming the path, no temp file anywhere
        let blocker = s.0.join("blocker");
        std::fs::write(&blocker, "file").unwrap();
        let bad = blocker.join("tuned.toml");
        let e = write_atomic(&bad, "x").unwrap_err();
        assert!(e.contains(&bad.display().to_string()), "{e}");
        assert_eq!(names(&s.0), ["blocker", "sub"]);

        // a rename that fails (the target is a directory) removes the temp file and names the path
        let dir_target = s.0.join("dir.toml");
        std::fs::create_dir(&dir_target).unwrap();
        let e = write_atomic(&dir_target, "x").unwrap_err();
        assert!(e.contains(&dir_target.display().to_string()), "{e}");
        assert_eq!(names(&s.0), ["blocker", "dir.toml", "sub"]);
    }
}
