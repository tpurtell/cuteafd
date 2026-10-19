//! Hugging Face-compatible chat-template rendering over minijinja.
//!
//! Transformers renders `chat_template` in a sandboxed Jinja2 environment with
//! `trim_blocks`, `lstrip_blocks`, loop controls, a `tojson` filter backed by
//! `json.dumps` (non-ASCII kept, `", "`/`": "` separators), and the
//! `raise_exception`/`strftime_now` globals. This module reproduces that
//! environment so a checkpoint's own template renders byte-identical prompts.
use minijinja::value::{Kwargs, Value};
use minijinja::{AutoEscape, Environment, Error, ErrorKind};
use serde_json::Value as Json;

const NAME: &str = "chat_template";

/// A compiled chat template.
pub struct ChatTemplate {
    env: Environment<'static>,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ChatTemplate")
    }
}

impl ChatTemplate {
    /// Compile `source` with the Transformers environment settings.
    pub fn new(source: impl Into<String>) -> Result<Self, Error> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_keep_trailing_newline(false);
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_function("raise_exception", |message: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        });
        env.add_function("strftime_now", strftime_now);
        env.add_function("__hf_generation", |state: &minijinja::State, kwargs: Kwargs| -> Result<Value, Error> {
            let caller: Value = kwargs.get("caller")?;
            kwargs.assert_all_used()?;
            caller.call(state, &[])
        });
        env.add_template_owned(NAME, generation_annotations(source.into())?)?;
        Ok(Self { env })
    }

    /// Render the template with `context` as its top-level variables.
    pub fn render(&self, context: &Json) -> Result<String, Error> {
        self.env.get_template(NAME)?.render(context)
    }
}

/// Lower Transformers' assistant-mask annotations to equivalent caller blocks.
/// Use the pinned minijinja lexer, not textual replacement: quoted delimiters,
/// comments and raw blocks must stay literal, and whitespace controls stay intact.
fn generation_annotations(source: String) -> Result<String, Error> {
    use minijinja::machinery::{tokenize, Span, Token, WhitespaceConfig};

    let mut block_start = false;
    let mut annotation: Option<(&str, Span)> = None;
    let mut edits = Vec::new();
    let mut depth = 0usize;
    for item in tokenize(&source, false, Default::default(), WhitespaceConfig::default()) {
        let (token, span) = item?;
        if let Some((name, name_span)) = annotation.take() {
            if !matches!(token, Token::BlockEnd) {
                return Err(Error::new(ErrorKind::SyntaxError, "generation annotations take no arguments"));
            }
            let replacement = if name == "generation" {
                depth += 1;
                "call __hf_generation()"
            } else {
                depth = depth.checked_sub(1).ok_or_else(|| {
                    Error::new(ErrorKind::SyntaxError, "unexpected endgeneration annotation")
                })?;
                "endcall"
            };
            edits.push((name_span.start_offset as usize, name_span.end_offset as usize, replacement));
        }
        if block_start {
            if let Token::Ident(name @ ("generation" | "endgeneration")) = token {
                annotation = Some((name, span));
            }
        }
        block_start = matches!(token, Token::BlockStart);
    }
    if depth != 0 || annotation.is_some() {
        return Err(Error::new(ErrorKind::SyntaxError, "unclosed generation annotation"));
    }
    if edits.is_empty() { return Ok(source); }
    let mut lowered = String::with_capacity(source.len());
    let mut offset = 0;
    for (start, end, replacement) in edits {
        lowered.push_str(&source[offset..start]);
        lowered.push_str(replacement);
        offset = end;
    }
    lowered.push_str(&source[offset..]);
    Ok(lowered)
}

/// `strftime_now(format)`: only the date directives templates use in practice.
fn strftime_now(format: String) -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or(0);
    let (year, month, day) = civil_from_days(seconds.div_euclid(86_400));
    const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December"];
    let mut out = String::new();
    let mut chars = format.chars();
    while let Some(c) = chars.next() {
        if c != '%' { out.push(c); continue; }
        match chars.next() {
            Some('Y') => out.push_str(&year.to_string()),
            Some('m') => out.push_str(&format!("{month:02}")),
            Some('d') => out.push_str(&format!("{day:02}")),
            Some('B') => out.push_str(MONTHS[month as usize - 1]),
            Some('b') => out.push_str(&MONTHS[month as usize - 1][..3]),
            Some('%') => out.push('%'),
            Some(other) => { out.push('%'); out.push(other); }
            None => out.push('%'),
        }
    }
    out
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(month <= 2), month, day)
}

/// `tojson(ensure_ascii=False, indent=None, separators=None, sort_keys=False)`
/// with Python `json.dumps` output.
fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    let indent: Option<Value> = kwargs.get("indent")?;
    let separators: Option<Vec<String>> = kwargs.get("separators")?;
    kwargs.assert_all_used()?;
    let indent = match indent {
        None => None,
        Some(value) if value.is_none() || value.is_undefined() => None,
        Some(value) => Some(match value.as_str() {
            Some(text) => text.to_owned(),
            None => " ".repeat(usize::try_from(value).map_err(|_| {
                Error::new(ErrorKind::InvalidOperation, "tojson indent must be a string or a non-negative integer")
            })?),
        }),
    };
    let (item, key) = match separators {
        Some(pair) if pair.len() == 2 => (pair[0].clone(), pair[1].clone()),
        Some(_) => return Err(Error::new(ErrorKind::InvalidOperation, "tojson separators must be a pair")),
        None if indent.is_some() => (",".to_owned(), ": ".to_owned()),
        None => (", ".to_owned(), ": ".to_owned()),
    };
    let json = serde_json::to_value(&value)
        .map_err(|error| Error::new(ErrorKind::InvalidOperation, format!("tojson: {error}")))?;
    let options = PyJson { ensure_ascii: ensure_ascii.unwrap_or(false), sort_keys: sort_keys.unwrap_or(false),
        indent, item, key };
    let mut out = String::new();
    options.write(&mut out, &json, 0);
    Ok(Value::from(out))
}

/// Native checkpoint encoders use the same Python JSON spelling as templates.
pub(crate) fn python_json(value: &Json, compact: bool) -> String {
    let options = PyJson { ensure_ascii: false, sort_keys: false, indent: None,
        item: if compact { "," } else { ", " }.into(),
        key: if compact { ":" } else { ": " }.into() };
    let mut out = String::new();
    options.write(&mut out, value, 0);
    out
}

/// `json.dumps` formatting.
struct PyJson {
    ensure_ascii: bool,
    sort_keys: bool,
    indent: Option<String>,
    item: String,
    key: String,
}

impl PyJson {
    fn newline(&self, out: &mut String, depth: usize) {
        if let Some(indent) = &self.indent {
            out.push('\n');
            for _ in 0..depth { out.push_str(indent); }
        }
    }

    fn write(&self, out: &mut String, value: &Json, depth: usize) {
        match value {
            Json::Null => out.push_str("null"),
            Json::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Json::Number(number) => {
                if let Some(value) = number.as_i64() { out.push_str(&value.to_string()); }
                else if let Some(value) = number.as_u64() { out.push_str(&value.to_string()); }
                else { out.push_str(&python_float(number.as_f64().unwrap_or(f64::NAN))); }
            }
            Json::String(text) => python_string(out, text, self.ensure_ascii),
            Json::Array(items) => {
                if items.is_empty() { out.push_str("[]"); return; }
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 { out.push_str(&self.item); }
                    self.newline(out, depth + 1);
                    self.write(out, item, depth + 1);
                }
                self.newline(out, depth);
                out.push(']');
            }
            Json::Object(map) => {
                if map.is_empty() { out.push_str("{}"); return; }
                let mut entries: Vec<_> = map.iter().collect();
                if self.sort_keys { entries.sort_by(|a, b| a.0.cmp(b.0)); }
                out.push('{');
                for (index, (key, item)) in entries.into_iter().enumerate() {
                    if index > 0 { out.push_str(&self.item); }
                    self.newline(out, depth + 1);
                    python_string(out, key, self.ensure_ascii);
                    out.push_str(&self.key);
                    self.write(out, item, depth + 1);
                }
                self.newline(out, depth);
                out.push('}');
            }
        }
    }
}

/// Python `float.__repr__`: shortest round-trip digits, scientific notation
/// outside `1e-4 <= |x| < 1e16`, and a two-digit signed exponent.
fn python_float(value: f64) -> String {
    if value.is_nan() { return "NaN".into(); }
    if value.is_infinite() { return if value > 0.0 { "Infinity" } else { "-Infinity" }.into(); }
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').expect("LowerExp has an exponent");
    let exponent: i32 = exponent.parse().expect("LowerExp exponent is an integer");
    let (sign, mantissa) = match mantissa.strip_prefix('-') { Some(rest) => ("-", rest), None => ("", mantissa) };
    if (-4..16).contains(&exponent) {
        let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
        let point = exponent + 1;
        let (whole, fraction) = if point <= 0 {
            ("0".to_owned(), format!("{}{digits}", "0".repeat((-point) as usize)))
        } else if point as usize >= digits.len() {
            (format!("{digits}{}", "0".repeat(point as usize - digits.len())), String::new())
        } else {
            (digits[..point as usize].to_owned(), digits[point as usize..].to_owned())
        };
        let fraction = if fraction.is_empty() { "0".to_owned() } else { fraction };
        format!("{sign}{whole}.{fraction}")
    } else {
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{exponent_sign}{:02}", exponent.abs())
    }
}

fn python_string(out: &mut String, text: &str, ensure_ascii: bool) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (ensure_ascii && !(' '..='~').contains(&c)) => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats_match_python_repr() {
        for (value, expected) in [(1.0, "1.0"), (0.5, "0.5"), (1e16, "1e+16"), (1.5e-5, "1.5e-05"),
            (0.0001, "0.0001"), (-0.0, "-0.0"), (123456789.125, "123456789.125"), (1e-7, "1e-07"),
            (9999999999999998.0, "9999999999999998.0"), (1.2345678901234568e+17, "1.2345678901234568e+17"),
            (0.1, "0.1"), (-2.5e-300, "-2.5e-300"), (100.0, "100.0")] {
            assert_eq!(python_float(value), expected);
        }
    }

    #[test]
    fn tojson_options_match_json_dumps() {
        let template = ChatTemplate::new(concat!(
            "{{ v | tojson }}|{{ v | tojson(ensure_ascii=True) }}|{{ v | tojson(indent=2) }}|",
            "{{ v | tojson(separators=[',', ':'], sort_keys=True) }}")).unwrap();
        let rendered = template.render(&json!({"v": {"b": "é\u{1F4A1}", "a": [1, 2.0, {}], "c": []}})).unwrap();
        assert_eq!(rendered, concat!(
            r#"{"b": "é💡", "a": [1, 2.0, {}], "c": []}|"#,
            r#"{"b": "\u00e9\ud83d\udca1", "a": [1, 2.0, {}], "c": []}|"#,
            "{\n  \"b\": \"é💡\",\n  \"a\": [\n    1,\n    2.0,\n    {}\n  ],\n  \"c\": []\n}|",
            r#"{"a":[1,2.0,{}],"b":"é💡","c":[]}"#));
    }

    // Expected strings from Transformers 4.57.1's AssistantTracker extension
    // with Jinja2 3.1.6, without assistant-mask tracking.
    #[test]
    fn generation_annotations_match_transformers() {
        for (source, context, expected) in [
            ("prefix{% generation %}hello {{ who }}{% endgeneration %}suffix", json!({"who": "world"}), "prefixhello worldsuffix"),
            ("A \n  {%- generation +%}\n {{ who }} \n{%- endgeneration -%}\n B\n", json!({"who": "world"}), "A\n worldB"),
            ("{% set x = 'outside' %}{% generation %}{% set x = 'inside' %}{{x}}{% endgeneration %}|{{x}}", json!({}), "inside|outside"),
            ("{% for x in [1,2] %}{% generation %}{{loop.index}}={{x}}{% endgeneration %};{% endfor %}", json!({}), "1=1;2=2;"),
            ("{% generation %}a{% generation %}b{% endgeneration %}c{% endgeneration %}", json!({}), "abc"),
            ("{{ '{% generation %}' }}{# {% generation %} #}{% raw %}{% generation %}literal{% endgeneration %}{% endraw %}", json!({}), "{% generation %}{% generation %}literal{% endgeneration %}"),
            ("é{% generation %}\u{1f4a1}{{x}}{% endgeneration %}", json!({"x": "tail"}), "é\u{1f4a1}tail"),
        ] {
            assert_eq!(ChatTemplate::new(source).unwrap().render(&context).unwrap(), expected, "{source}");
        }
    }

    #[test]
    fn generation_annotations_reject_invalid_blocks() {
        for source in ["{% generation %}", "{% endgeneration %}",
            "{% generation extra %}{% endgeneration %}",
            "{% generation %}{% endgeneration extra %}",
            "{% generation %}{% endif %}{% endgeneration %}"] {
            assert!(ChatTemplate::new(source).is_err(), "{source}");
        }
    }

    #[test]
    fn raise_exception_fails_rendering() {
        let template = ChatTemplate::new("{{ raise_exception('bad role') }}").unwrap();
        assert!(template.render(&json!({})).unwrap_err().to_string().contains("bad role"));
    }
}
