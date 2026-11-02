//! Test 7's conformance cases on the fake engine's side: the chat template against the oracle's prompts.
mod common;

use common::{template, TESTDATA};
use serde_json::{json, Value};

/// A message's content as `build_golden_set.py`'s `content()` joins it: a string, or the parts' text and files.
fn content(c: &Value) -> Value {
    match c {
        Value::String(_) => c.clone(),
        Value::Array(parts) => Value::String(
            parts
                .iter()
                .map(|p| match p.get("text").and_then(Value::as_str) {
                    Some(t) => t.to_string(),
                    None => {
                        let f = p["file"].as_str().unwrap();
                        String::from_utf8(std::fs::read(format!("{TESTDATA}{f}")).unwrap()).unwrap()
                    }
                })
                .collect(),
        ),
        other => panic!("content {other}"),
    }
}

/// The bytes around the first difference of two byte strings, for a failure message.
fn first_difference(got: &[u8], want: &[u8]) -> String {
    let at = got
        .iter()
        .zip(want)
        .position(|(a, b)| a != b)
        .unwrap_or(got.len().min(want.len()));
    let around = |b: &[u8]| {
        String::from_utf8_lossy(&b[at.saturating_sub(40)..(at + 40).min(b.len())]).into_owned()
    };
    format!(
        "first difference at byte {at}\n got: {:?}\nwant: {:?}",
        around(got),
        around(want)
    )
}

/// The ids in a `.tokens` file: little-endian u32s.
fn read_ids(path: &str) -> Vec<u32> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// Fails with the first differing token when a prompt's ids aren't the oracle's.
fn assert_same_ids(id: &str, got: &[u32], want: &[u32]) {
    let at = got
        .iter()
        .zip(want)
        .position(|(a, b)| a != b)
        .unwrap_or(got.len().min(want.len()));
    assert!(
        got == want,
        "{id}: first token difference at {at} ({} got, {} want)",
        got.len(),
        want.len()
    );
}

#[test]
fn t1_golden_prompts_without_tools_render_byte_for_byte() {
    let set: Value =
        serde_json::from_slice(&std::fs::read(format!("{TESTDATA}golden/prompts.json")).unwrap())
            .unwrap();
    let mut checked = 0;
    for p in set["prompts"].as_array().unwrap() {
        let id = p["id"].as_str().unwrap();
        let n: u32 = id[..2].parse().unwrap();
        if !(5..=15).contains(&n) && n != 19 && n != 20 {
            continue;
        }
        assert!(p.get("tools").is_none(), "{id}");
        let messages: Vec<Value> = p["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                let mut m = m.clone();
                m["content"] = content(&m["content"]);
                m
            })
            .collect();
        let ctx = json!({
            "messages": messages,
            "add_generation_prompt": true,
            "enable_thinking": p["enable_thinking"],
        });
        let got = template()
            .render_json(&ctx)
            .unwrap_or_else(|e| panic!("{id}: {}", e.message));
        let want = std::fs::read(format!("{TESTDATA}golden/prompts/{id}.txt")).unwrap();
        assert!(
            got.as_bytes() == want.as_slice(),
            "{id}: {}",
            first_difference(got.as_bytes(), &want)
        );
        checked += 1;
    }
    assert_eq!(checked, 13);
}

#[test]
fn an_empty_message_list_raises() {
    let e = template()
        .render_json(&json!({"messages": []}))
        .unwrap_err();
    assert!(e.raised);
    assert_eq!(e.message, "No messages provided.");
}

/// The body `build_golden_set.py` posted for a chat prompt of the golden set.
fn body(p: &Value) -> Vec<u8> {
    let messages: Vec<Value> = p["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            let mut m = m.clone();
            m["content"] = content(&m["content"]);
            m
        })
        .collect();
    let mut b = json!({"messages": messages, "chat_template_kwargs": {"enable_thinking": p["enable_thinking"]}});
    if let Some(t) = p.get("tools") {
        b["tools"] = t.clone();
    }
    serde_json::to_vec(&b).unwrap()
}

/// The chat prompts (05 to 20) with their prompt and the oracle's text and ids.
fn chat_prompts() -> Vec<(String, Value, String, Vec<u32>)> {
    let set: Value =
        serde_json::from_slice(&std::fs::read(format!("{TESTDATA}golden/prompts.json")).unwrap())
            .unwrap();
    let mut out = Vec::new();
    for p in set["prompts"].as_array().unwrap() {
        let id = p["id"].as_str().unwrap().to_string();
        if id[..2].parse::<u32>().unwrap() < 5 {
            continue;
        }
        let txt =
            String::from_utf8(std::fs::read(format!("{TESTDATA}golden/prompts/{id}.txt")).unwrap())
                .unwrap();
        let ids = read_ids(&format!("{TESTDATA}golden/prompts/{id}.tokens"));
        out.push((id, p.clone(), txt, ids));
    }
    assert_eq!(out.len(), 16);
    out
}

fn defaults() -> server::params::Defaults {
    server::params::Defaults::new("artemis", 65536)
}

fn prompt_of(p: &Value) -> (server::convo::Conversation, server::prompt::Prompt) {
    prompt_of_body(&body(p))
}

/// The prompt of an OpenAI request body, through `parse`, `conversation` and `build_prompt`.
fn prompt_of_body(body: &[u8]) -> (server::convo::Conversation, server::prompt::Prompt) {
    let convo = server::openai::request::parse(body)
        .and_then(|mut r| {
            let f = r.fields(&defaults())?;
            r.conversation(&f)
        })
        .unwrap();
    let prompt = server::prompt::build_prompt(
        &convo,
        template(),
        common::tokenizer(),
        common::specials(),
        65536,
    )
    .unwrap();
    (convo, prompt)
}

#[test]
fn specials_are_the_specs_ids() {
    let sp = common::specials();
    assert_eq!(
        (
            sp.im_start,
            sp.im_end,
            sp.think,
            sp.end_think,
            sp.tool_call,
            sp.end_tool_call
        ),
        (248045, 248046, 248068, 248069, 248058, 248059)
    );
    assert_eq!(sp.eog, vec![248044, 248046, 248063, 248064, 248065]);
    assert_eq!(
        common::tokenizer().decode(&sp.think_close),
        "\n</think>\n\n"
    );
}

#[test]
fn t1_golden_prompts_render_and_tokenize_as_the_oracle() {
    for (id, p, want_text, want_ids) in chat_prompts() {
        let (_, prompt) = prompt_of(&p);
        assert!(
            prompt.text.as_bytes() == want_text.as_bytes(),
            "{id}: {}",
            first_difference(prompt.text.as_bytes(), want_text.as_bytes())
        );
        assert_same_ids(&id, &prompt.tokens, &want_ids);
    }
}

/// The oracle fixtures: each request body under `template/requests/` with the id, the bytes of its prompt and the
/// oracle's ids.
fn oracle_fixtures() -> Vec<(String, Vec<u8>, String, Vec<u32>)> {
    let dir = format!("{TESTDATA}template/requests");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    paths.sort();
    assert_eq!(paths.len(), 18, "{paths:?}");
    paths
        .into_iter()
        .map(|path| {
            let id = path.file_stem().unwrap().to_string_lossy().into_owned();
            let prompts = format!("{TESTDATA}template/prompts/{id}");
            let text = String::from_utf8(std::fs::read(format!("{prompts}.txt")).unwrap()).unwrap();
            let ids = read_ids(&format!("{prompts}.tokens"));
            (id, std::fs::read(&path).unwrap(), text, ids)
        })
        .collect()
}

#[test]
fn t1_oracle_fixtures_render_and_tokenize_as_the_oracle() {
    for (id, body, want_text, want_ids) in oracle_fixtures() {
        let (_, prompt) = prompt_of_body(&body);
        assert!(
            prompt.text.as_bytes() == want_text.as_bytes(),
            "{id}: {}",
            first_difference(prompt.text.as_bytes(), want_text.as_bytes())
        );
        assert_same_ids(&id, &prompt.tokens, &want_ids);
    }
}

#[test]
fn r1_oracle_fixture_cuts() {
    let mut seen = 0;
    for (id, body, _, _) in oracle_fixtures() {
        let (convo, prompt) = prompt_of_body(&body);
        let cut = prompt.cut.unwrap_or_else(|| panic!("{id}: no cut"));
        // The flag follows its definition, evaluated on the template's own last_query_index.
        let (_, want) = common::template_user_query(&convo.template_context(true)["messages"]);
        assert_eq!(cut.user_query, want, "{id}");
        let expect = match &id[..3] {
            "t01" | "t05" | "t17" | "t18" => Some(true),
            "t03" | "t04" | "t06" | "t07" => Some(false),
            _ => None,
        };
        if let Some(expect) = expect {
            assert_eq!(cut.user_query, expect, "{id}");
            seen += 1;
        }
    }
    assert_eq!(seen, 8);
}

#[test]
fn r1_golden_cut_and_user_query() {
    for (id, p, _, _) in chat_prompts() {
        let (convo, prompt) = prompt_of(&p);
        let cut = prompt.cut.unwrap_or_else(|| panic!("{id}: no cut"));
        let last = prompt.tokens.iter().rposition(|&t| t == 248045).unwrap();
        assert_eq!(cut.at as usize, last, "{id}");
        // The cut is exactly the end of the render without the generation prompt.
        let plain = template().render(&convo, false).unwrap();
        assert_eq!(
            cut.at as usize,
            common::tokenizer().encode(&plain, true).len(),
            "{id}"
        );
        let ctx = convo.template_context(true);
        let (_, want) = common::template_user_query(&ctx["messages"]);
        assert_eq!(cut.user_query, want, "{id}");
        // The cut is the generation prompt: <|im_start|>assistant\n then <think>.
        assert!(
            prompt.text.ends_with("<|im_start|>assistant\n<think>\n")
                || prompt.text.ends_with("<think>\n\n</think>\n\n")
        );
        let expect = !id.starts_with("17-");
        assert_eq!(cut.user_query, expect, "{id}");
    }
}

#[test]
fn r1_is_user_query_cases() {
    use server::convo::{is_user_query, Content, Message, ToolCallIn};
    let m = |role: &str, text: &str, call: bool| Message {
        role: role.into(),
        content: Content::Text(text.into()),
        reasoning_content: None,
        tool_calls: if call {
            vec![ToolCallIn {
                name: "f".into(),
                arguments: json!({}),
            }]
        } else {
            vec![]
        },
    };
    let wrapped = "<tool_response>\nok\n</tool_response>";
    assert!(is_user_query(&[m("user", "q", false)]));
    assert!(is_user_query(&[
        m("user", "q", false),
        m("assistant", "a", false),
        m("user", "q2", false)
    ]));
    assert!(!is_user_query(&[
        m("user", "q", false),
        m("assistant", "", true),
        m("tool", "r", false)
    ]));
    assert!(!is_user_query(&[
        m("user", "q", false),
        m("assistant", "", true),
        m("tool", "r", false),
        m("assistant", "done", false)
    ]));
    assert!(is_user_query(&[
        m("user", "q", false),
        m("assistant", "", true),
        m("user", "q2", false),
        m("tool", "A", false),
        m("tool", "B", false)
    ]));
    assert!(!is_user_query(&[
        m("user", "q", false),
        m("assistant", "a", false),
        m("user", wrapped, false)
    ]));
    // Only user message tool-response-wrapped: the default (last message) applies.
    assert!(is_user_query(&[m("user", wrapped, false)]));
    assert!(!is_user_query(&[
        m("user", wrapped, false),
        m("assistant", "a", false)
    ]));
    assert!(!is_user_query(&[m("system", "s", false)]));
    assert!(!is_user_query(&[]));
    // Claude Code's `system` messages after a turn: the template drops them, so they don't end the query.
    assert!(is_user_query(&[
        m("user", "q", false),
        m("system", "s", false)
    ]));
    assert!(!is_user_query(&[
        m("user", "q", false),
        m("assistant", "", true),
        m("tool", "r", false),
        m("system", "s", false)
    ]));
    assert!(is_user_query(&[
        m("user", "q", false),
        m("assistant", "", true),
        m("user", "q2", false),
        m("tool", "r", false),
        m("system", "s", false),
        m("tool", "r2", false)
    ]));
    // Leading systems before the query are merged into the system prompt; they never follow it.
    assert!(is_user_query(&[
        m("system", "s", false),
        m("system", "s2", false),
        m("user", "q", false)
    ]));
    assert!(!is_user_query(&[
        m("system", "s", false),
        m("user", "q", false),
        m("assistant", "a", false),
        m("system", "s2", false)
    ]));
    assert!(is_user_query(&[
        m("system", "s", false),
        m("user", "q", false),
        m("developer", "d", false)
    ]));
}

#[test]
fn r1_claude_code_fixtures_match_the_template() {
    let dir = format!("{TESTDATA}clients/claude-code");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    assert_eq!(files.len(), 4);
    for path in files {
        let id = path.file_name().unwrap().to_string_lossy().into_owned();
        let capture: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let body = serde_json::to_vec(&capture["body"]).unwrap();
        let (convo, prompt) = anthropic_prompt(&body);
        let cut = prompt.cut.unwrap_or_else(|| panic!("{id}: no cut"));
        let (_, want) = common::template_user_query(&convo.template_context(true)["messages"]);
        assert_eq!(cut.user_query, want, "{id}");
        // Every request ends in a `system` message. 001 is `[system, user, system]` (the query, then a dropped
        // system: true); 002 and 004 end in a tool loop (false); 003 ends in a new user turn (true).
        assert_eq!(convo.messages.last().unwrap().role, "system", "{id}");
        let expected = [("001", true), ("002", false), ("003", true), ("004", false)]
            .iter()
            .find(|(p, _)| id.starts_with(p))
            .unwrap_or_else(|| panic!("{id}: unexpected fixture"))
            .1;
        assert_eq!(cut.user_query, expected, "{id}");
    }
}

/// The prompt of an Anthropic body, through `parse`, `conversation` and `build_prompt`.
fn anthropic_prompt(body: &[u8]) -> (server::convo::Conversation, server::prompt::Prompt) {
    let convo = server::anthropic::request::parse(body)
        .and_then(|mut r| {
            let f = r.fields(&defaults(), true)?;
            r.conversation(&f)
        })
        .unwrap();
    let prompt = server::prompt::build_prompt(
        &convo,
        template(),
        common::tokenizer(),
        common::specials(),
        65536,
    )
    .unwrap();
    (convo, prompt)
}

#[test]
fn r1_text_beside_tool_results_is_the_last_query() {
    let body = json!({"messages": [
        {"role": "user", "content": "Check a and b."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_a", "name": "read_file", "input": {"path": "a.txt"}},
            {"type": "tool_use", "id": "toolu_b", "name": "read_file", "input": {"path": "b.txt"}}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Also check b."},
            {"type": "tool_result", "tool_use_id": "toolu_a", "content": "contents of a"},
            {"type": "tool_result", "tool_use_id": "toolu_b", "content": "contents of b"}]}]});
    let (mut convo, prompt) = anthropic_prompt(&serde_json::to_vec(&body).unwrap());
    let roles: Vec<_> = convo.messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant", "user", "tool", "tool"]);
    // The last message is a tool message, yet the text before it is the last query.
    let (lqi, want) = common::template_user_query(&convo.template_context(true)["messages"]);
    assert_eq!(lqi, 2);
    assert!(want);
    assert_eq!(prompt.cut.expect("a cut").user_query, want);
    // An assistant turn between the text and its tool messages ends the query.
    let assistant = server::convo::Message {
        role: "assistant".into(),
        content: server::convo::Content::Text("hm".into()),
        reasoning_content: None,
        tool_calls: vec![],
    };
    convo.messages.insert(3, assistant);
    let prompt = server::prompt::build_prompt(
        &convo,
        template(),
        common::tokenizer(),
        common::specials(),
        65536,
    )
    .unwrap();
    let (_, want) = common::template_user_query(&convo.template_context(true)["messages"]);
    assert!(!want);
    assert_eq!(prompt.cut.expect("a cut").user_query, want);
}

#[test]
fn m2_anthropic_bodies_count_the_oracle_tokens() {
    let dir = format!("{TESTDATA}template/anthropic");
    let mut bodies: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    bodies.sort();
    assert!(bodies.len() >= 4, "{bodies:?}");
    let (mut checked, mut skipped) = (0, Vec::new());
    for path in bodies {
        let id = path.file_stem().unwrap().to_string_lossy().into_owned();
        // Every body must convert and render; the oracle's count is compared where its run has written one.
        let (_, prompt) = anthropic_prompt(&std::fs::read(&path).unwrap());
        let Ok(count) = std::fs::read_to_string(path.with_extension("count")) else {
            skipped.push(id);
            continue;
        };
        let want: usize = count
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("{id}.count"));
        assert_eq!(prompt.tokens.len(), want, "{id}");
        checked += 1;
    }
    assert!(skipped.is_empty(), "bodies without a .count: {skipped:?}");
    assert_eq!(checked, 4);
}

#[test]
fn m6_the_billing_header_is_normalized_in_the_prompt() {
    let path = format!("{TESTDATA}template/anthropic/a01-system-blocks-and-billing-header.json");
    let (_, prompt) = anthropic_prompt(&std::fs::read(path).unwrap());
    assert!(
        prompt.text.contains(
            "x-anthropic-billing-header: cc_version=1.2.3; cc_entrypoint=cli; cch=fffff;You are a careful assistant."
        ),
        "{}",
        prompt.text
    );
    assert!(!prompt.text.contains("ab12c"));
}

#[test]
fn m2_a_null_input_schema_renders_as_null() {
    let body = json!({"messages": [{"role": "user", "content": "q"}], "tools": [
        {"name": "absent"}, {"name": "nulled", "input_schema": null}]});
    let (convo, prompt) = anthropic_prompt(&serde_json::to_vec(&body).unwrap());
    assert_eq!(convo.tools[0].parameters, json!({}));
    assert_eq!(convo.tools[1].parameters, Value::Null);
    assert!(prompt
        .text
        .contains(r#""name": "absent", "description": "", "parameters": {}"#));
    assert!(prompt
        .text
        .contains(r#""name": "nulled", "description": "", "parameters": null"#));
}

/// `special_tokens_in_content = false` changes no prompt whose client strings hold no CONTROL text. All
/// 18 OpenAI fixtures and the 4 Anthropic bodies give the same tokens, text and cut in both modes.
#[test]
fn plain_mode_changes_no_fixture_prompt() {
    use server::prompt::{build_prompt_with, PromptOptions};
    let build = |convo: &server::convo::Conversation, flavour, plain: bool| {
        let o = PromptOptions {
            special_tokens_in_content: !plain,
            flavour,
        };
        build_prompt_with(
            convo,
            template(),
            common::tokenizer(),
            common::specials(),
            65536,
            &o,
        )
        .unwrap()
    };
    let mut checked = 0;
    for (id, body, _, _) in oracle_fixtures() {
        let (convo, _) = prompt_of_body(&body);
        let parity = build(&convo, server::error::Flavour::OpenAi, false);
        let plain = build(&convo, server::error::Flavour::OpenAi, true);
        assert_eq!(parity.tokens, plain.tokens, "{id}");
        assert_eq!(parity, plain, "{id}");
        checked += 1;
    }
    let dir = format!("{TESTDATA}template/anthropic");
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let (convo, _) = anthropic_prompt(&std::fs::read(&path).unwrap());
        let parity = build(&convo, server::error::Flavour::Anthropic, false);
        let plain = build(&convo, server::error::Flavour::Anthropic, true);
        assert_eq!(parity.tokens, plain.tokens, "{path:?}");
        assert_eq!(parity, plain, "{path:?}");
        checked += 1;
    }
    assert_eq!(checked, 22);
}
