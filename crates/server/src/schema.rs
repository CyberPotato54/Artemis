//! What a tool's JSON schema says about each parameter, and the typing of a parameter's text by it.
use crate::convo::ToolDef;
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};

/// A JSON type a parameter may take.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Kind {
    String,
    Number,
    Integer,
    Boolean,
    Object,
    Array,
    Null,
}

/// The types a parameter's schema allows: any, or exactly these.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParamTypes {
    Any,
    Only(BTreeSet<Kind>),
}

/// The declared tools of a request, by name.
#[derive(Clone, Debug, Default)]
pub struct ToolSet {
    tools: HashMap<String, Value>,
}

impl ToolSet {
    pub fn new(tools: &[ToolDef]) -> ToolSet {
        ToolSet {
            tools: tools
                .iter()
                .map(|t| (t.name.clone(), t.parameters.clone()))
                .collect(),
        }
    }

    pub fn has(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// The types of `tool`'s parameter `key`; `Any` for an undeclared tool or parameter, or a schema that can't be
    /// resolved.
    pub fn param(&self, tool: &str, key: &str) -> ParamTypes {
        let Some(root) = self.tools.get(tool) else {
            return ParamTypes::Any;
        };
        let Some(schema) = root.get("properties").and_then(|p| p.get(key)) else {
            return ParamTypes::Any;
        };
        match kinds(schema, root, true) {
            Some(set) => ParamTypes::Only(set),
            None => ParamTypes::Any,
        }
    }
}

fn kind_by_name(name: &str) -> Option<Kind> {
    Some(match name {
        "string" => Kind::String,
        "number" => Kind::Number,
        "integer" => Kind::Integer,
        "boolean" => Kind::Boolean,
        "object" => Kind::Object,
        "array" => Kind::Array,
        "null" => Kind::Null,
        _ => return None,
    })
}

fn kind_of(v: &Value) -> Kind {
    match v {
        Value::String(_) => Kind::String,
        Value::Bool(_) => Kind::Boolean,
        Value::Null => Kind::Null,
        Value::Object(_) => Kind::Object,
        Value::Array(_) => Kind::Array,
        Value::Number(n) if n.is_i64() || n.is_u64() => Kind::Integer,
        Value::Number(_) => Kind::Number,
    }
}

/// The kinds a schema allows, `None` for "any" (no type information, or none that can be resolved). A `$ref` is
/// followed while `refs` is set, which its target clears: one level.
fn kinds(schema: &Value, root: &Value, refs: bool) -> Option<BTreeSet<Kind>> {
    let obj = schema.as_object()?;
    if let Some(r) = obj.get("$ref").and_then(Value::as_str) {
        if !refs || !(r.starts_with("#/$defs/") || r.starts_with("#/definitions/")) {
            return None;
        }
        return kinds(root.pointer(&r[1..])?, root, false);
    }
    let set: BTreeSet<Kind> = if let Some(t) = obj.get("type") {
        match t {
            Value::String(name) => BTreeSet::from([kind_by_name(name)?]),
            Value::Array(names) => names
                .iter()
                .map(|n| kind_by_name(n.as_str()?))
                .collect::<Option<_>>()?,
            _ => return None,
        }
    } else if let Some(branches) = ["anyOf", "oneOf"]
        .iter()
        .find_map(|k| obj.get(*k).and_then(Value::as_array))
    {
        let mut set = BTreeSet::new();
        for b in branches {
            set.extend(kinds(b, root, refs)?);
        }
        set
    } else {
        let values = obj.get("enum").and_then(Value::as_array)?;
        values.iter().map(kind_of).collect()
    };
    (!set.is_empty()).then_some(set)
}

/// Whether a parsed non-string value is of an allowed kind (an integer is also a number; a number with a fraction or
/// an exponent is not an integer).
fn allowed(v: &Value, set: Option<&BTreeSet<Kind>>) -> bool {
    let Some(set) = set else { return true };
    match kind_of(v) {
        Kind::Integer => set.contains(&Kind::Integer) || set.contains(&Kind::Number),
        k => set.contains(&k),
    }
}

/// `-0` is the integer `0`, where serde_json holds `-0.0`: a text that is a minus sign and digits only (no `.`, `e` or
/// `E`) and parses to zero becomes the integer `0`; every other value is unchanged.
/// An integer past u64 stays a number (serde_json holds it as a float), so an `integer`-only type falls back to the
/// text.
fn negative_zero(raw: &str, v: Value) -> Value {
    let digits = raw.trim().strip_prefix('-').unwrap_or("");
    let is_integer_text = !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit());
    match &v {
        Value::Number(n) if is_integer_text && n.as_f64() == Some(0.0) => Value::from(0),
        _ => v,
    }
}

/// A parameter's text as JSON of its schema's types, and the repair if it had to fall back to the text.
pub fn convert(raw: &str, t: &ParamTypes) -> (Value, Option<String>) {
    let set = match t {
        ParamTypes::Any => None,
        ParamTypes::Only(set) => Some(set),
    };
    let has = |k: Kind| set.is_none_or(|s| s.contains(&k));
    if set.is_some_and(|s| s.len() == 1 && s.contains(&Kind::String)) {
        return (Value::String(raw.to_string()), None);
    }
    // Python's literals, whole-value only, where the kind is allowed
    match raw {
        "True" if has(Kind::Boolean) => return (Value::Bool(true), None),
        "False" if has(Kind::Boolean) => return (Value::Bool(false), None),
        "None" if has(Kind::Null) => return (Value::Null, None),
        _ => {}
    }
    match serde_json::from_str::<Value>(raw).map(|v| negative_zero(raw, v)) {
        Ok(v) if !v.is_string() && allowed(&v, set) => (v, None),
        _ => {
            let repair = (!has(Kind::String))
                .then(|| "value does not fit the declared type, kept as a string".to_string());
            (Value::String(raw.to_string()), repair)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(parameters: Value) -> ToolSet {
        ToolSet::new(&[ToolDef {
            name: "f".into(),
            description: String::new(),
            parameters,
        }])
    }

    fn only(kinds: &[Kind]) -> ParamTypes {
        ParamTypes::Only(kinds.iter().copied().collect())
    }

    #[test]
    fn param_types_follow_the_schema() {
        let ts = tool(json!({
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "m": {"type": ["string", "integer"]},
                "u": {"description": "no type"},
                "r": {"$ref": "#/$defs/Mode"},
                "d": {"$ref": "#/definitions/Num"},
                "e": {"enum": [1, 2]},
                "ef": {"enum": ["a", 1.5, null]},
                "a": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "o": {"oneOf": [{"type": "boolean"}, {"$ref": "#/$defs/Mode"}]},
                "any_branch": {"anyOf": [{"type": "integer"}, {"description": "anything"}]},
                "deep": {"$ref": "#/$defs/Deep"},
                "gone": {"$ref": "#/$defs/Missing"},
                "other": {"$ref": "http://example.com/x"},
                "weird": {"type": "bogus"},
                "typed_enum": {"type": "string", "enum": [1]},
                "flag": true
            },
            "$defs": {"Mode": {"type": "string"}, "Deep": {"$ref": "#/$defs/Mode"}},
            "definitions": {"Num": {"type": "number"}}
        }));
        assert!(ts.has("f") && !ts.has("g"));
        assert_eq!(ts.param("f", "s"), only(&[Kind::String]));
        assert_eq!(ts.param("f", "m"), only(&[Kind::String, Kind::Integer]));
        assert_eq!(ts.param("f", "u"), ParamTypes::Any);
        assert_eq!(ts.param("f", "r"), only(&[Kind::String]));
        assert_eq!(ts.param("f", "d"), only(&[Kind::Number]));
        assert_eq!(ts.param("f", "e"), only(&[Kind::Integer]));
        assert_eq!(
            ts.param("f", "ef"),
            only(&[Kind::String, Kind::Number, Kind::Null])
        );
        assert_eq!(ts.param("f", "a"), only(&[Kind::Integer, Kind::Null]));
        assert_eq!(ts.param("f", "o"), only(&[Kind::Boolean, Kind::String]));
        assert_eq!(ts.param("f", "any_branch"), ParamTypes::Any);
        // a reference resolves one level: a reference inside the target counts as unresolvable
        assert_eq!(ts.param("f", "deep"), ParamTypes::Any);
        assert_eq!(ts.param("f", "gone"), ParamTypes::Any);
        assert_eq!(ts.param("f", "other"), ParamTypes::Any);
        assert_eq!(ts.param("f", "weird"), ParamTypes::Any);
        assert_eq!(ts.param("f", "typed_enum"), only(&[Kind::String]));
        assert_eq!(ts.param("f", "flag"), ParamTypes::Any);
        // undeclared parameter, undeclared tool
        assert_eq!(ts.param("f", "nope"), ParamTypes::Any);
        assert_eq!(ts.param("g", "s"), ParamTypes::Any);
    }

    #[test]
    fn a_tool_without_properties_types_everything_as_any() {
        let ts = tool(json!({}));
        assert_eq!(ts.param("f", "x"), ParamTypes::Any);
        let ts = tool(Value::Null);
        assert!(ts.has("f"));
        assert_eq!(ts.param("f", "x"), ParamTypes::Any);
    }

    #[test]
    fn convert_a_string_is_never_parsed() {
        let t = only(&[Kind::String]);
        for raw in ["123", "true", "None", "{\"k\":1}", "\"q\"", ""] {
            assert_eq!(convert(raw, &t), (json!(raw), None), "{raw}");
        }
    }

    #[test]
    fn convert_without_string_parses_the_kinds_allowed() {
        let int = only(&[Kind::Integer]);
        assert_eq!(convert("3", &int), (json!(3), None));
        assert_eq!(convert(" -7\n", &int), (json!(-7), None));
        for raw in ["3.0", "1e3", "0.5", "three", "", "\"3\"", "true"] {
            let (v, repair) = convert(raw, &int);
            assert_eq!(v, json!(raw), "{raw}");
            assert!(repair.is_some(), "{raw}");
        }
        let num = only(&[Kind::Number]);
        assert_eq!(convert("3", &num), (json!(3), None));
        assert_eq!(convert("0.5", &num), (json!(0.5), None));
        assert_eq!(convert("1e3", &num), (json!(1000.0), None));
        assert!(convert("abc", &num).1.is_some());
        let obj = only(&[Kind::Object]);
        assert_eq!(convert("{\"k\": [1]}", &obj), (json!({"k": [1]}), None));
        assert!(convert("[1]", &obj).1.is_some());
        let arr = only(&[Kind::Array]);
        assert_eq!(convert("[1, \"a\"]", &arr), (json!([1, "a"]), None));
        assert!(convert("{", &arr).1.is_some());
        let null_int = only(&[Kind::Integer, Kind::Null]);
        assert_eq!(convert("null", &null_int), (Value::Null, None));
        assert_eq!(convert("5", &null_int), (json!(5), None));
    }

    #[test]
    fn convert_decides_integers_by_their_text() {
        let int = only(&[Kind::Integer]);
        assert_eq!(convert("-0", &int), (json!(0), None));
        assert_eq!(convert("-0", &only(&[Kind::Number])), (json!(0), None));
        assert_eq!(convert("-0", &ParamTypes::Any), (json!(0), None));
        assert_eq!(convert("0", &int), (json!(0), None));
        // a fraction or an exponent is never an integer, even a zero one
        for raw in ["-0.0", "0.0", "-0e0", "1E2"] {
            assert!(convert(raw, &int).1.is_some(), "{raw}");
        }
        assert_eq!(convert("-0.0", &only(&[Kind::Number])).1, None);
        // up to u64 and down to i64 stay integers
        assert_eq!(
            convert("18446744073709551615", &int),
            (json!(18446744073709551615u64), None)
        );
        assert_eq!(
            convert("-9223372036854775808", &int),
            (json!(i64::MIN), None)
        );
        // past that serde_json holds a float: not an integer, so the text, with the repair
        let huge = "100000000000000000000";
        let (v, repair) = convert(huge, &int);
        assert_eq!(v, json!(huge));
        assert!(repair.is_some());
        assert!(convert(huge, &only(&[Kind::Number])).0.is_number());
    }

    #[test]
    fn convert_mixed_or_any_tries_the_non_string_forms_first() {
        let m = only(&[Kind::String, Kind::Integer]);
        assert_eq!(convert("42", &m), (json!(42), None));
        assert_eq!(convert("4.5", &m), (json!("4.5"), None));
        assert_eq!(convert("hello", &m), (json!("hello"), None));
        // a JSON string literal is not a non-string form: the raw text, quotes and all
        assert_eq!(convert("\"x\"", &m), (json!("\"x\""), None));
        assert_eq!(convert("123", &ParamTypes::Any), (json!(123), None));
        assert_eq!(
            convert("{\"a\":[1]}", &ParamTypes::Any),
            (json!({"a": [1]}), None)
        );
        assert_eq!(convert("hello", &ParamTypes::Any), (json!("hello"), None));
        assert_eq!(convert("[1", &ParamTypes::Any), (json!("[1"), None));
        // an object is not an allowed kind here
        assert_eq!(convert("{\"a\":1}", &m), (json!("{\"a\":1}"), None));
    }

    #[test]
    fn convert_python_literals_follow_the_allowed_kinds() {
        let any = ParamTypes::Any;
        assert_eq!(convert("True", &any), (json!(true), None));
        assert_eq!(convert("False", &any), (json!(false), None));
        assert_eq!(convert("None", &any), (Value::Null, None));
        assert_eq!(
            convert("True", &only(&[Kind::Boolean])),
            (json!(true), None)
        );
        assert_eq!(
            convert("None", &only(&[Kind::Integer, Kind::Null])),
            (Value::Null, None)
        );
        assert_eq!(
            convert("True", &only(&[Kind::String, Kind::Boolean])),
            (json!(true), None)
        );
        // not allowed: not normalized
        assert_eq!(
            convert("True", &only(&[Kind::String])),
            (json!("True"), None)
        );
        assert_eq!(
            convert("True", &only(&[Kind::String, Kind::Integer])),
            (json!("True"), None)
        );
        let (v, repair) = convert("True", &only(&[Kind::Integer]));
        assert_eq!(v, json!("True"));
        assert!(repair.is_some());
        let (v, repair) = convert("None", &only(&[Kind::Boolean]));
        assert_eq!(v, json!("None"));
        assert!(repair.is_some());
        // only the whole value is a literal
        assert_eq!(convert("True ", &any), (json!("True "), None));
    }

    #[test]
    fn a_repair_never_carries_the_value() {
        let (_, repair) = convert("secret-value", &only(&[Kind::Integer]));
        assert!(!repair.unwrap().contains("secret"));
    }
}
