//! Response and call ids, seeds and UTC time formatting. No random-number crate: an id is the process's
//! start time and a counter, so it is unique per process and across restarts.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

const BASE62: &[u8; 62] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
/// The characters after the prefix.
const ID_CHARS: usize = 24;
/// The counter's share of an id's bits; the start time sits above it.
const COUNTER_BITS: u32 = 40;

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// When this process first asked, in nanoseconds since the Unix epoch.
fn start_ns() -> u64 {
    static START: OnceLock<u64> = OnceLock::new();
    *START.get_or_init(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    })
}

/// `prefix` and 24 base-62 characters from the process start (ns) and a counter.
pub fn new_id(prefix: &str) -> String {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed) & ((1 << COUNTER_BITS) - 1);
    let mut v = (u128::from(start_ns()) << COUNTER_BITS) | u128::from(n);
    let mut digits = [b'0'; ID_CHARS];
    for d in digits.iter_mut().rev() {
        *d = BASE62[(v % 62) as usize];
        v /= 62;
    }
    let mut id = String::with_capacity(prefix.len() + ID_CHARS);
    id.push_str(prefix);
    id.extend(digits.iter().map(|&b| b as char));
    id
}

/// A sampling seed: the start time and the counter through splitmix64.
pub fn new_seed() -> u64 {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut z = start_ns().wrapping_add(n.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seconds since the Unix epoch.
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The proleptic Gregorian date of a day count since 1970-01-01 (Howard Hinnant's `civil_from_days`).
fn civil(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z % 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    (year, month, day)
}

/// The day count since 1970-01-01 of a proleptic Gregorian date (Howard Hinnant's `days_from_civil`); `civil` is its
/// inverse. The month is 1 to 12 and the day 1 to 31, unchecked here.
pub fn days_from_civil(year: u64, month: u64, day: u64) -> i64 {
    let y = year as i64 - i64::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = (month as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The day count since 1970-01-01 of a `YYYY-MM-DD` text, `None` for any other text or a date that does not exist
/// (month 13, 31 June, 29 February in a common year).
pub fn parse_date(text: &str) -> Option<i64> {
    let b = text.as_bytes();
    let digits = |r: std::ops::Range<usize>| -> Option<u64> {
        let part = b.get(r)?;
        part.iter()
            .all(u8::is_ascii_digit)
            .then(|| part.iter().fold(0, |n, d| n * 10 + u64::from(d - b'0')))
    };
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let (y, m, d) = (digits(0..4)?, digits(5..7)?, digits(8..10)?);
    let days = days_from_civil(y, m, d);
    // A date that does not exist comes back as another one.
    (days >= 0 && civil(days as u64) == (y, m, d)).then_some(days)
}

/// `YYYY-MM-DD`, UTC.
pub fn utc_date(unix: u64) -> String {
    let (y, m, d) = civil(unix / 86_400);
    format!("{y:04}-{m:02}-{d:02}")
}

/// `YYYY-MM-DDTHH:MM:SSZ`, UTC.
pub fn iso_utc(unix: u64) -> String {
    let s = unix % 86_400;
    format!(
        "{}T{:02}:{:02}:{:02}Z",
        utc_date(unix),
        s / 3_600,
        s % 3_600 / 60,
        s % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn ids_are_24_base62_and_unique() {
        let mut seen = HashSet::new();
        for _ in 0..20_000 {
            let id = new_id("chatcmpl-");
            let rest = id.strip_prefix("chatcmpl-").unwrap();
            assert_eq!(rest.len(), 24, "{id}");
            assert!(rest.bytes().all(|b| b.is_ascii_alphanumeric()), "{id}");
            assert!(seen.insert(id));
        }
        // No prefix: the 24 characters alone.
        assert_eq!(new_id("").len(), 24);
    }

    #[test]
    fn seeds_differ() {
        let seeds: HashSet<u64> = (0..1_000).map(|_| new_seed()).collect();
        assert_eq!(seeds.len(), 1_000);
    }

    #[test]
    fn dates_convert_both_ways() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2026, 10, 18), 1_792_281_600 / 86_400);
        assert_eq!(days_from_civil(2000, 2, 29), 951_782_400 / 86_400);
        for days in (0..60_000).step_by(7) {
            let (y, m, d) = civil(days);
            assert_eq!(days_from_civil(y, m, d), days as i64);
        }
        assert_eq!(
            parse_date("2026-11-14"),
            Some(days_from_civil(2026, 11, 14))
        );
        assert_eq!(parse_date("2000-02-29"), Some(951_782_400 / 86_400));
        for bad in [
            "2026-13-01",
            "2026-00-10",
            "2026-02-29",
            "2026-06-31",
            "2026-11-00",
            "2026-1-01",
            "2026/11/01",
            "2026-11-011",
            "+026-11-01",
            "2026-11-0a",
            "",
            "1969-12-31",
        ] {
            assert_eq!(parse_date(bad), None, "{bad}");
        }
    }

    #[test]
    fn iso_and_dates_from_unix() {
        assert_eq!(iso_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso_utc(1_792_281_600), "2026-10-18T00:00:00Z");
        assert_eq!(utc_date(1_792_281_600), "2026-10-18");
        // The last second of a day, a leap day and the end of a year.
        assert_eq!(iso_utc(1_792_367_999), "2026-10-18T23:59:59Z");
        assert_eq!(iso_utc(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(iso_utc(1_798_761_599), "2026-12-31T23:59:59Z");
        assert_eq!(utc_date(1_798_761_600), "2027-01-01");
        assert_eq!(iso_utc(1_700_000_000), "2023-11-14T22:13:20Z");
    }
}
