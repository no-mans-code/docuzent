//! Pulling JSON out of what a model wrote. Models wrap it in prose, code
//! fences and apologies; these find the first balanced object or array
//! (aware of strings and escapes) so a single stray sentence does not lose
//! an otherwise good answer.

fn first_balanced(s: &str, open: char, close: char) -> Option<&str> {
    let start = s.find(open)?;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    for (i, c) in s[start..].char_indices() {
        if in_str {
            match (escaped, c) {
                (true, _) => escaped = false,
                (false, '\\') => escaped = true,
                (false, '"') => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            c if c == open => depth += 1,
            c if c == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..start + i + c.len_utf8()]);
                }
            }
            _ => {}
        }
    }
    None
}

/// The first `{...}` in `s`.
pub fn first_object(s: &str) -> Option<&str> {
    first_balanced(s, '{', '}')
}

/// The first `[...]` in `s`.
pub fn first_array(s: &str) -> Option<&str> {
    first_balanced(s, '[', ']')
}

/// Every complete `{...}` in `s`, in order - what can be saved from an array the model was cut off writing (its
/// token limit reached mid-way): the objects it finished, without the one it did not.
pub fn complete_objects(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(o) = first_object(rest) {
        out.push(o);
        let end = o.as_ptr() as usize - rest.as_ptr() as usize + o.len();
        rest = &rest[end..];
    }
    out
}

/// A list field the model may have written as an array, one string, or with
/// junk in it: always a clean `Vec<String>`.
pub fn string_list(v: &serde_json::Value) -> Vec<String> {
    match v {
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|i| match i {
                serde_json::Value::String(s) => Some(s.trim().to_string()),
                serde_json::Value::Null => None,
                other => Some(other.to_string()),
            })
            .filter(|s| !s.is_empty())
            .collect(),
        serde_json::Value::String(s) if !s.trim().is_empty() => vec![s.trim().to_string()],
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_finished_objects_of_a_cut_off_array_are_kept() {
        let cut = r#"[{"n": 1, "facts": ["a {brace} in a string"]}, {"n": 2, "x": {"y": 1}}, {"n": 3, "context": "the model ran out of to"#;
        assert_eq!(first_array(cut), None);
        assert_eq!(complete_objects(cut), vec![r#"{"n": 1, "facts": ["a {brace} in a string"]}"#, r#"{"n": 2, "x": {"y": 1}}"#]);
        assert!(complete_objects("no json here").is_empty());
    }

    #[test]
    fn finds_json_inside_prose_and_fences() {
        let raw = "Sure!\n```json\n[{\"a\": \"x ] y\"}, {\"b\": 2}]\n```\nDone.";
        assert_eq!(first_array(raw), Some("[{\"a\": \"x ] y\"}, {\"b\": 2}]"), "a bracket inside a string does not end the array");
        assert_eq!(first_object("noise {\"k\": \"v\\\"}\"} tail"), Some("{\"k\": \"v\\\"}\"}"));
    }

    #[test]
    fn cut_off_or_missing_json_is_none() {
        assert_eq!(first_array("no json"), None);
        assert_eq!(first_object("{\"a\": \"unterminated"), None);
    }

    #[test]
    fn lists_are_tolerated_in_any_shape() {
        assert_eq!(string_list(&serde_json::json!([1, null, "a", "  "])), vec!["1", "a"]);
        assert_eq!(string_list(&serde_json::json!("solo")), vec!["solo"]);
        assert!(string_list(&serde_json::json!({"x": 1})).is_empty());
    }
}
