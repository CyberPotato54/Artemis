//! Python-style text for the template's `tojson` and `string` filters: what `json.dumps(v, ensure_ascii=False)` and
//! `str(v)` give, because that is what the oracle renders and what the model was trained on. The one exception is a
//! float, which the oracle writes as C++ does (`float_text`).
use serde_json::Value as Json;

/// Python's `json.dumps(v, ensure_ascii=False)`: separators `", "` and `": "`, insertion order, non-ASCII raw. A float
/// is the exception: it is written as `float_text` writes it (C++'s `%g`), not as Python's `repr`.
pub fn dumps(v: &Json) -> String {
    let mut out = String::new();
    dump(v, &mut out);
    out
}

fn dump(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => out.push_str(&i.to_string()),
            (None, Some(u)) => out.push_str(&u.to_string()),
            // serde_json holds no NaN or infinity, so an f64 here is finite. An integer beyond u64 parses as an f64
            // here and in nlohmann alike (`100000000000000000000` writes as `1e+20` in both). Known divergence: the
            // literal `-0` is an integer 0 in nlohmann (`0`) but a float -0.0 in serde_json (`-0`); a `Number`
            // can't tell the two spellings apart.
            _ => out.push_str(&float_text(n.as_f64().unwrap_or(f64::NAN))),
        },
        Json::String(s) => string(s, out),
        Json::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump(x, out);
            }
            out.push(']');
        }
        Json::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                string(k, out);
                out.push_str(": ");
                dump(x, out);
            }
            out.push('}');
        }
    }
}

/// A JSON string as Python writes it with `ensure_ascii=False`: only `"`, `\` and the control characters are escaped.
fn string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `str()` of a template value: `True`, `False`, `None`, integers as digits, floats as `float_text`, strings
/// unchanged, undefined as nothing, and anything else as the JSON `dumps` writes. That last one departs from Python
/// on purpose (`str({'a': [1, None]})` is `{'a': [1, None]}` there): the template sends maps and sequences to
/// `tojson`, so `string` only meets them where JSON is the better text.
pub fn py_str(v: &minijinja::Value) -> String {
    use minijinja::value::ValueKind;
    match v.kind() {
        ValueKind::Undefined => String::new(),
        ValueKind::None => "None".into(),
        ValueKind::Bool => (if v.is_true() { "True" } else { "False" }).into(),
        ValueKind::Number if v.is_integer() => v.to_string(),
        ValueKind::Number => float_text(f64::try_from(v.clone()).unwrap_or(f64::NAN)),
        ValueKind::String => v.as_str().unwrap_or_default().to_string(),
        _ => serde_json::to_value(v).map_or_else(|_| v.to_string(), |j| dumps(&j)),
    }
}

/// A float as llama.cpp's template engine writes it, both in `string` and in `tojson`: C++'s `operator<<(double)` with
/// the default precision, `%g` with 6 significant digits. `1.0` is `1`, `1e-07` stays, `1.23456789` is `1.23457` and
/// `1234567.0` is `1.23457e+06`. That is the oracle's text, not Python's `repr`.
pub fn float_text(x: f64) -> String {
    if !x.is_finite() {
        return if x.is_nan() {
            "nan"
        } else if x < 0.0 {
            "-inf"
        } else {
            "inf"
        }
        .into();
    }
    let mut out = String::new();
    if x.is_sign_negative() {
        out.push('-');
    }
    // Six significant digits, rounded on the exact value (ties to even), as printf does.
    let sci = format!("{:.5e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    if x == 0.0 {
        out.push('0');
    } else if (-4..6).contains(&exp) {
        if exp < 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-exp - 1) as usize));
            out.push_str(digits);
        } else {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                out.push_str(digits);
                out.extend(std::iter::repeat_n('0', int_len - digits.len()));
            } else {
                out.push_str(&digits[..int_len]);
                out.push('.');
                out.push_str(&digits[int_len..]);
            }
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exp.abs()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dumps_writes_json_as_the_oracle_does() {
        let v = serde_json::json!({"b": 1, "a": [true, null, 1.0, 1e-7, 0.5, -0.0, 1e16, 1.23456789],
                                   "s": "é 日本 <tag> & 'q' \"d\" \\ \n\t\u{1}"});
        assert_eq!(
            dumps(&v),
            [
                r#"{"b": 1, "a": [true, null, 1, 1e-07, 0.5, -0, 1e+16, 1.23457], "#,
                r#""s": "é 日本 <tag> & 'q' \"d\" \\ \n\t\u0001"}"#,
            ]
            .concat()
        );
    }

    #[test]
    fn string_of_scalars_is_the_oracles() {
        use minijinja::Value;
        assert_eq!(py_str(&Value::from(true)), "True");
        assert_eq!(py_str(&Value::from(false)), "False");
        assert_eq!(py_str(&Value::from(())), "None");
        assert_eq!(py_str(&Value::from(1.0)), "1");
        assert_eq!(py_str(&Value::from(1e-7)), "1e-07");
        assert_eq!(py_str(&Value::from(42)), "42");
        assert_eq!(py_str(&Value::from("text")), "text");
        assert_eq!(py_str(&Value::UNDEFINED), "");
        // Anything else is its JSON.
        assert_eq!(
            py_str(&Value::from(minijinja::value::Serde(
                serde_json::json!({"a": [1, null]})
            ))),
            r#"{"a": [1, null]}"#
        );
    }

    #[test]
    fn float_text_follows_cpp_default_ostream() {
        for (x, want) in [
            (1.0, "1"),
            (0.5, "0.5"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1e-7, "1e-07"),
            (12345.0, "12345"),
            (123456.0, "123456"),
            (1234567.0, "1.23457e+06"),
            (999999.5, "1e+06"),
            // An exact decimal tie rounds to even, as printf does.
            (1234565.0, "1.23456e+06"),
            (1.23456789, "1.23457"),
            (-2.5, "-2.5"),
            (0.1 + 0.2, "0.3"),
            (1.5e300, "1.5e+300"),
            (5e-324, "4.94066e-324"),
            (-0.0, "-0"),
            (0.0, "0"),
            (100.0, "100"),
            (1e16, "1e+16"),
        ] {
            assert_eq!(float_text(x), want, "{x:e}");
        }
    }
}
