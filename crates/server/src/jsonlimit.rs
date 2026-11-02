//! A cap on how many JSON values a request body holds.
//!
//! A parsed `serde_json::Value` costs 72 bytes a node plus its allocations, so a 16 MiB body of `[[],[],...]` or
//! `[0,0,...]` becomes about 600 MiB. The byte limit alone does not bound that; the number of values does. `check`
//! counts them in one pass that builds nothing, before the real parse.
use crate::error::{ApiError, ErrorType};
use serde::de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};
use std::cell::Cell;
use std::fmt;

/// The most values (scalars, arrays, objects and object keys, each counting one) a request body may hold.
pub const MAX_JSON_VALUES: usize = 1_000_000;

/// Counts the values of `body`; a body over `MAX_JSON_VALUES` is a 400. A body that is not valid JSON passes: the real
/// parse reports it, as it did before this check.
pub fn check(body: &[u8]) -> Result<(), ApiError> {
    let seen = Cell::new(0usize);
    let over = Cell::new(false);
    let mut de = serde_json::Deserializer::from_slice(body);
    let _ = Count {
        seen: &seen,
        over: &over,
    }
    .deserialize(&mut de);
    if over.get() {
        return Err(ApiError::new(
            400,
            ErrorType::InvalidRequest,
            "the request holds more than 1,000,000 JSON values",
        ));
    }
    Ok(())
}

/// The seed (and the visitor) that counts the value it is given and everything inside it.
#[derive(Clone, Copy)]
struct Count<'a> {
    seen: &'a Cell<usize>,
    over: &'a Cell<bool>,
}

impl Count<'_> {
    /// Counts one value; false once the cap is passed.
    fn one(&self) -> bool {
        let n = self.seen.get() + 1;
        self.seen.set(n);
        if n > MAX_JSON_VALUES {
            self.over.set(true);
            return false;
        }
        true
    }

    fn scalar<E: de::Error>(&self) -> Result<(), E> {
        if self.one() {
            Ok(())
        } else {
            Err(E::custom("too many values"))
        }
    }
}

impl<'de> DeserializeSeed<'de> for Count<'_> {
    type Value = ();

    fn deserialize<D: de::Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Count<'_> {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("any JSON value")
    }

    fn visit_bool<E: de::Error>(self, _: bool) -> Result<(), E> {
        self.scalar()
    }

    fn visit_i64<E: de::Error>(self, _: i64) -> Result<(), E> {
        self.scalar()
    }

    fn visit_u64<E: de::Error>(self, _: u64) -> Result<(), E> {
        self.scalar()
    }

    fn visit_f64<E: de::Error>(self, _: f64) -> Result<(), E> {
        self.scalar()
    }

    fn visit_str<E: de::Error>(self, _: &str) -> Result<(), E> {
        self.scalar()
    }

    fn visit_unit<E: de::Error>(self) -> Result<(), E> {
        self.scalar()
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        self.scalar()?;
        while seq.next_element_seed(self)?.is_some() {}
        Ok(())
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        self.scalar()?;
        while map.next_key::<IgnoredAny>()?.is_some() {
            self.scalar()?;
            map.next_value_seed(self)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zeros(n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(2 * n + 2);
        v.push(b'[');
        for i in 0..n {
            if i > 0 {
                v.push(b',');
            }
            v.push(b'0');
        }
        v.push(b']');
        v
    }

    #[test]
    fn the_cap_counts_the_array_and_each_element() {
        // One array and 999,999 zeros: 1,000,000 values, the most that passes.
        assert!(check(&zeros(MAX_JSON_VALUES - 1)).is_ok());
        let e = check(&zeros(MAX_JSON_VALUES)).unwrap_err();
        assert_eq!(e.status, 400);
        assert!(e.message.contains("1,000,000"), "{}", e.message);
    }

    #[test]
    fn object_keys_count_as_values() {
        let mut body = String::from("{");
        for i in 0..500_000 {
            if i > 0 {
                body.push(',');
            }
            body.push_str(&format!("\"k{i}\":0"));
        }
        body.push('}');
        // The object, 500,000 keys and 500,000 values: 1,000,001.
        assert!(check(body.as_bytes()).is_err());
        let fewer = body.replacen("\"k0\":0,", "", 1);
        assert!(check(fewer.as_bytes()).is_ok());
    }

    #[test]
    fn a_syntax_error_is_left_to_the_real_parse() {
        assert!(check(b"{\"a\":").is_ok());
        assert!(check(b"").is_ok());
        assert!(check(b"not json").is_ok());
    }

    #[test]
    fn strings_and_nesting_are_values_too() {
        assert!(check(br#"{"a":[1,"x",null,true,{"b":2.5}]}"#).is_ok());
        let mut deep = "[".repeat(100);
        deep.push_str(&"]".repeat(100));
        assert!(check(deep.as_bytes()).is_ok());
    }
}
