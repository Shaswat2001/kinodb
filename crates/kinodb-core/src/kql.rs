//! KQL — Kino Query Language.
//!
//! A minimal query language for filtering episodes by their metadata.
//!
//! ## Syntax
//!
//! ```text
//! <field> <op> <value> [AND <field> <op> <value> ...]
//! ```
//!
//! ## Supported fields
//!
//! | Field        | Type     | Example                          |
//! |--------------|----------|----------------------------------|
//! | embodiment   | string   | `embodiment = 'franka'`          |
//! | task         | string   | `task CONTAINS 'pick'`           |
//! | success      | bool     | `success = true`                 |
//! | num_frames   | int      | `num_frames > 50`                |
//! | action_dim   | int      | `action_dim = 7`                 |
//! | fps          | float    | `fps >= 10.0`                    |
//! | total_reward | float    | `total_reward > 0.5`             |
//!
//! ## Operators
//!
//! | Op           | Applies to      |
//! |--------------|-----------------|
//! | `=`          | all types       |
//! | `!=`         | all types       |
//! | `>` `<` `>=` `<=` | numbers  |
//! | `CONTAINS`   | strings         |
//!
//! ## Example
//!
//! ```ignore
//! use kinodb_core::kql::{parse, evaluate};
//!
//! let query = parse("embodiment = 'franka' AND success = true AND num_frames > 50")?;
//! let matches = evaluate(&query, &episode_meta);
//! ```

use crate::EpisodeMeta;

// ── Values ──────────────────────────────────────────────────

/// A literal value in a KQL expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}

// ── Operators ───────────────────────────────────────────────

/// Comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Eq,       // =
    Neq,      // !=
    Gt,       // >
    Lt,       // <
    Gte,      // >=
    Lte,      // <=
    Contains, // CONTAINS (string only)
}

// ── AST ─────────────────────────────────────────────────────

/// A single condition: field op value.
#[derive(Debug, Clone, PartialEq)]
pub struct Condition {
    pub field: String,
    pub op: Op,
    pub value: Value,
}

/// A query is a list of conditions joined by AND.
/// (We can add OR later if needed, but AND-only covers 99% of use cases.)
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    pub conditions: Vec<Condition>,
}

// ── Parse errors ────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    EmptyQuery,
    ExpectedField { position: usize },
    ExpectedOperator { position: usize, got: String },
    ExpectedValue { position: usize },
    UnterminatedString { position: usize },
    UnknownField { field: String },
    InvalidNumber { text: String },
    ExpectedAnd { position: usize, got: String },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::EmptyQuery => write!(f, "empty query"),
            ParseError::ExpectedField { position } => {
                write!(f, "expected field name at position {}", position)
            }
            ParseError::ExpectedOperator { position, got } => {
                write!(
                    f,
                    "expected operator at position {}, got '{}'",
                    position, got
                )
            }
            ParseError::ExpectedValue { position } => {
                write!(f, "expected value at position {}", position)
            }
            ParseError::UnterminatedString { position } => {
                write!(f, "unterminated string starting at position {}", position)
            }
            ParseError::UnknownField { field } => {
                write!(
                    f,
                    "unknown field '{}'. Valid: embodiment, task, success, num_frames, action_dim, fps, total_reward",
                    field
                )
            }
            ParseError::InvalidNumber { text } => {
                write!(f, "invalid number: '{}'", text)
            }
            ParseError::ExpectedAnd { position, got } => {
                write!(f, "expected AND at position {}, got '{}'", position, got)
            }
        }
    }
}

impl std::error::Error for ParseError {}

// ── Known fields ────────────────────────────────────────────

const KNOWN_FIELDS: &[&str] = &[
    "embodiment",
    "task",
    "success",
    "num_frames",
    "action_dim",
    "fps",
    "total_reward",
];

fn is_known_field(s: &str) -> bool {
    KNOWN_FIELDS.contains(&s)
}

// ── Tokenizer ───────────────────────────────────────────────

/// A token from the KQL input.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Word(String),   // field names, AND, CONTAINS, true, false, null
    Str(String),    // 'quoted string'
    Number(String), // 123 or 1.5
    Eq,             // =
    Neq,            // !=
    Gt,             // >
    Lt,             // <
    Gte,            // >=
    Lte,            // <=
}

fn tokenize(input: &str) -> Result<Vec<(Token, usize)>, ParseError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Skip whitespace
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }

        let pos = i;

        // String literal: 'xxx' or "xxx"
        if chars[i] == '\'' || chars[i] == '"' {
            let quote = chars[i];
            i += 1;
            let start = i;
            while i < chars.len() && chars[i] != quote {
                i += 1;
            }
            if i >= chars.len() {
                return Err(ParseError::UnterminatedString { position: pos });
            }
            let s: String = chars[start..i].iter().collect();
            tokens.push((Token::Str(s), pos));
            i += 1; // skip closing quote
            continue;
        }

        // Operators
        if chars[i] == '!' && i + 1 < chars.len() && chars[i + 1] == '=' {
            tokens.push((Token::Neq, pos));
            i += 2;
            continue;
        }
        if chars[i] == '>' && i + 1 < chars.len() && chars[i + 1] == '=' {
            tokens.push((Token::Gte, pos));
            i += 2;
            continue;
        }
        if chars[i] == '<' && i + 1 < chars.len() && chars[i + 1] == '=' {
            tokens.push((Token::Lte, pos));
            i += 2;
            continue;
        }
        if chars[i] == '=' {
            tokens.push((Token::Eq, pos));
            i += 1;
            continue;
        }
        if chars[i] == '>' {
            tokens.push((Token::Gt, pos));
            i += 1;
            continue;
        }
        if chars[i] == '<' {
            tokens.push((Token::Lt, pos));
            i += 1;
            continue;
        }

        // Number (possibly negative, possibly float)
        if chars[i].is_ascii_digit()
            || (chars[i] == '-' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit())
        {
            let start = i;
            if chars[i] == '-' {
                i += 1;
            }
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            let s: String = chars[start..i].iter().collect();
            tokens.push((Token::Number(s), pos));
            continue;
        }

        // Word (field name, AND, CONTAINS, true, false, null)
        if chars[i].is_alphabetic() || chars[i] == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let s: String = chars[start..i].iter().collect();
            tokens.push((Token::Word(s), pos));
            continue;
        }

        // Unknown character — skip
        i += 1;
    }

    Ok(tokens)
}

// ── Parser ──────────────────────────────────────────────────

/// Parse a KQL query string into a Query AST.
pub fn parse(input: &str) -> Result<Query, ParseError> {
    let input = input.trim();
    if input.is_empty() {
        return Err(ParseError::EmptyQuery);
    }

    let tokens = tokenize(input)?;
    if tokens.is_empty() {
        return Err(ParseError::EmptyQuery);
    }

    let mut conditions = Vec::new();
    let mut i = 0;

    loop {
        // Expect: field
        if i >= tokens.len() {
            return Err(ParseError::ExpectedField {
                position: input.len(),
            });
        }

        let field = match &tokens[i].0 {
            Token::Word(w) => {
                let lower = w.to_lowercase();
                if !is_known_field(&lower) {
                    return Err(ParseError::UnknownField { field: lower });
                }
                lower
            }
            _ => {
                return Err(ParseError::ExpectedField {
                    position: tokens[i].1,
                });
            }
        };
        i += 1;

        // Expect: operator
        if i >= tokens.len() {
            return Err(ParseError::ExpectedOperator {
                position: input.len(),
                got: "end of input".to_string(),
            });
        }

        let op = match &tokens[i].0 {
            Token::Eq => Op::Eq,
            Token::Neq => Op::Neq,
            Token::Gt => Op::Gt,
            Token::Lt => Op::Lt,
            Token::Gte => Op::Gte,
            Token::Lte => Op::Lte,
            Token::Word(w) if w.to_uppercase() == "CONTAINS" => Op::Contains,
            other => {
                return Err(ParseError::ExpectedOperator {
                    position: tokens[i].1,
                    got: format!("{:?}", other),
                });
            }
        };
        i += 1;

        // Expect: value
        if i >= tokens.len() {
            return Err(ParseError::ExpectedValue {
                position: input.len(),
            });
        }

        let value = match &tokens[i].0 {
            Token::Str(s) => Value::Str(s.clone()),
            Token::Number(s) => {
                if s.contains('.') {
                    Value::Float(
                        s.parse::<f64>()
                            .map_err(|_| ParseError::InvalidNumber { text: s.clone() })?,
                    )
                } else {
                    Value::Int(
                        s.parse::<i64>()
                            .map_err(|_| ParseError::InvalidNumber { text: s.clone() })?,
                    )
                }
            }
            Token::Word(w) => match w.to_lowercase().as_str() {
                "true" => Value::Bool(true),
                "false" => Value::Bool(false),
                "null" | "none" => Value::Null,
                _ => Value::Str(w.clone()), // treat bare words as strings
            },
            _ => {
                return Err(ParseError::ExpectedValue {
                    position: tokens[i].1,
                });
            }
        };
        i += 1;

        conditions.push(Condition { field, op, value });

        // Done or expect AND
        if i >= tokens.len() {
            break;
        }

        match &tokens[i].0 {
            Token::Word(w) if w.to_uppercase() == "AND" => {
                i += 1; // consume AND, loop
            }
            _ => {
                return Err(ParseError::ExpectedAnd {
                    position: tokens[i].1,
                    got: format!("{:?}", tokens[i].0),
                });
            }
        }
    }

    Ok(Query { conditions })
}

// ── Evaluator ───────────────────────────────────────────────

/// Evaluate a parsed query against one episode's metadata.
/// Returns true if all conditions match.
pub fn evaluate(query: &Query, meta: &EpisodeMeta) -> bool {
    query.conditions.iter().all(|c| eval_condition(c, meta))
}

fn eval_condition(cond: &Condition, meta: &EpisodeMeta) -> bool {
    match cond.field.as_str() {
        "embodiment" => eval_string(&meta.embodiment, &cond.op, &cond.value),
        "task" => eval_string(&meta.language_instruction, &cond.op, &cond.value),
        "success" => eval_option_bool(meta.success, &cond.op, &cond.value),
        "num_frames" => eval_number(meta.num_frames as f64, &cond.op, &cond.value),
        "action_dim" => eval_number(meta.action_dim as f64, &cond.op, &cond.value),
        "fps" => eval_number(meta.fps as f64, &cond.op, &cond.value),
        "total_reward" => match meta.total_reward {
            Some(r) => eval_number(r as f64, &cond.op, &cond.value),
            None => matches!(cond.value, Value::Null) && cond.op == Op::Eq,
        },
        _ => false, // unknown field never matches
    }
}

fn eval_string(actual: &str, op: &Op, value: &Value) -> bool {
    match (op, value) {
        (Op::Eq, Value::Str(s)) => actual == s,
        (Op::Neq, Value::Str(s)) => actual != s,
        (Op::Contains, Value::Str(s)) => actual.to_lowercase().contains(&s.to_lowercase()),
        _ => false,
    }
}

fn eval_number(actual: f64, op: &Op, value: &Value) -> bool {
    let target = match value {
        Value::Int(n) => *n as f64,
        Value::Float(n) => *n,
        _ => return false,
    };

    match op {
        Op::Eq => (actual - target).abs() < 1e-6,
        Op::Neq => (actual - target).abs() >= 1e-6,
        Op::Gt => actual > target,
        Op::Lt => actual < target,
        Op::Gte => actual >= target,
        Op::Lte => actual <= target,
        Op::Contains => false, // not valid for numbers
    }
}

fn eval_option_bool(actual: Option<bool>, op: &Op, value: &Value) -> bool {
    match (op, value) {
        (Op::Eq, Value::Bool(b)) => actual == Some(*b),
        (Op::Neq, Value::Bool(b)) => actual != Some(*b),
        (Op::Eq, Value::Null) => actual.is_none(),
        (Op::Neq, Value::Null) => actual.is_some(),
        _ => false,
    }
}

// ── Convenience: filter a reader ────────────────────────────

/// Filter episodes from a KdbReader by a KQL query string.
/// Returns the positions of matching episodes.
pub fn filter_reader(
    reader: &crate::KdbReader,
    query_str: &str,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let query = parse(query_str)?;
    let mut matches = Vec::new();

    for i in 0..reader.num_episodes() {
        let meta = reader.read_meta(i)?;
        if evaluate(&query, &meta) {
            matches.push(i);
        }
    }

    Ok(matches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EpisodeId, EpisodeMeta};

    fn meta(embodiment: &str, task: &str, frames: u32, success: Option<bool>) -> EpisodeMeta {
        EpisodeMeta {
            id: EpisodeId(0),
            embodiment: embodiment.to_string(),
            language_instruction: task.to_string(),
            num_frames: frames,
            fps: 10.0,
            action_dim: 7,
            success,
            total_reward: Some(1.0),
        }
    }

    // ── Parser tests ────────────────────────────────────────

    #[test]
    fn parse_single_string_eq() {
        let q = parse("embodiment = 'franka'").unwrap();
        assert_eq!(q.conditions.len(), 1);
        assert_eq!(q.conditions[0].field, "embodiment");
        assert_eq!(q.conditions[0].op, Op::Eq);
        assert_eq!(q.conditions[0].value, Value::Str("franka".to_string()));
    }

    #[test]
    fn parse_single_bool() {
        let q = parse("success = true").unwrap();
        assert_eq!(q.conditions[0].value, Value::Bool(true));
    }

    #[test]
    fn parse_single_number() {
        let q = parse("num_frames > 50").unwrap();
        assert_eq!(q.conditions[0].op, Op::Gt);
        assert_eq!(q.conditions[0].value, Value::Int(50));
    }

    #[test]
    fn parse_float_value() {
        let q = parse("fps >= 10.5").unwrap();
        assert_eq!(q.conditions[0].op, Op::Gte);
        assert_eq!(q.conditions[0].value, Value::Float(10.5));
    }

    #[test]
    fn parse_contains() {
        let q = parse("task CONTAINS 'pick'").unwrap();
        assert_eq!(q.conditions[0].op, Op::Contains);
        assert_eq!(q.conditions[0].value, Value::Str("pick".to_string()));
    }

    #[test]
    fn parse_multiple_and() {
        let q = parse("embodiment = 'franka' AND success = true AND num_frames > 50").unwrap();
        assert_eq!(q.conditions.len(), 3);
        assert_eq!(q.conditions[0].field, "embodiment");
        assert_eq!(q.conditions[1].field, "success");
        assert_eq!(q.conditions[2].field, "num_frames");
    }

    #[test]
    fn parse_case_insensitive_and() {
        let q = parse("success = true and num_frames > 10").unwrap();
        assert_eq!(q.conditions.len(), 2);
    }

    #[test]
    fn parse_double_quoted_string() {
        let q = parse("task = \"open the drawer\"").unwrap();
        assert_eq!(
            q.conditions[0].value,
            Value::Str("open the drawer".to_string())
        );
    }

    #[test]
    fn parse_neq() {
        let q = parse("embodiment != 'franka'").unwrap();
        assert_eq!(q.conditions[0].op, Op::Neq);
    }

    #[test]
    fn parse_error_unknown_field() {
        let err = parse("bogus = 5").unwrap_err();
        assert!(matches!(err, ParseError::UnknownField { .. }));
    }

    #[test]
    fn parse_error_empty() {
        let err = parse("").unwrap_err();
        assert!(matches!(err, ParseError::EmptyQuery));
    }

    #[test]
    fn parse_error_unterminated_string() {
        let err = parse("task = 'oops").unwrap_err();
        assert!(matches!(err, ParseError::UnterminatedString { .. }));
    }

    // ── Evaluator tests ─────────────────────────────────────

    #[test]
    fn eval_embodiment_eq() {
        let q = parse("embodiment = 'franka'").unwrap();
        assert!(evaluate(&q, &meta("franka", "task", 50, Some(true))));
        assert!(!evaluate(&q, &meta("widowx", "task", 50, Some(true))));
    }

    #[test]
    fn eval_task_contains() {
        let q = parse("task CONTAINS 'pick'").unwrap();
        assert!(evaluate(&q, &meta("franka", "pick up the block", 50, None)));
        assert!(!evaluate(&q, &meta("franka", "open drawer", 50, None)));
    }

    #[test]
    fn eval_task_contains_case_insensitive() {
        let q = parse("task CONTAINS 'PICK'").unwrap();
        assert!(evaluate(&q, &meta("franka", "pick up the block", 50, None)));
    }

    #[test]
    fn eval_num_frames_comparison() {
        let q = parse("num_frames > 50").unwrap();
        assert!(evaluate(&q, &meta("franka", "t", 100, None)));
        assert!(!evaluate(&q, &meta("franka", "t", 50, None)));
        assert!(!evaluate(&q, &meta("franka", "t", 30, None)));
    }

    #[test]
    fn eval_success_bool() {
        let q = parse("success = true").unwrap();
        assert!(evaluate(&q, &meta("f", "t", 10, Some(true))));
        assert!(!evaluate(&q, &meta("f", "t", 10, Some(false))));
        assert!(!evaluate(&q, &meta("f", "t", 10, None)));
    }

    #[test]
    fn eval_success_null() {
        let q = parse("success = null").unwrap();
        assert!(evaluate(&q, &meta("f", "t", 10, None)));
        assert!(!evaluate(&q, &meta("f", "t", 10, Some(true))));
    }

    #[test]
    fn eval_compound_query() {
        let q = parse("embodiment = 'franka' AND success = true AND num_frames > 50").unwrap();

        assert!(evaluate(&q, &meta("franka", "task", 100, Some(true))));
        assert!(!evaluate(&q, &meta("franka", "task", 30, Some(true)))); // frames too low
        assert!(!evaluate(&q, &meta("widowx", "task", 100, Some(true)))); // wrong embodiment
        assert!(!evaluate(&q, &meta("franka", "task", 100, Some(false)))); // not successful
    }

    // ── filter_reader test ──────────────────────────────────

    #[test]
    fn filter_reader_integration() {
        use crate::{Episode, Frame, KdbReader, KdbWriter};

        let path = "/tmp/kinodb_kql_test.kdb";

        let mut writer = KdbWriter::create(path).unwrap();
        for i in 0..10 {
            let embodiment = if i < 5 { "franka" } else { "widowx" };
            let success = i % 2 == 0;
            let ep = Episode {
                meta: EpisodeMeta {
                    id: EpisodeId(0),
                    embodiment: embodiment.to_string(),
                    language_instruction: "task".to_string(),
                    num_frames: (i + 1) * 10,
                    fps: 10.0,
                    action_dim: 7,
                    success: Some(success),
                    total_reward: Some(if success { 1.0 } else { 0.0 }),
                },
                frames: (0..((i + 1) * 10))
                    .map(|t| Frame {
                        timestep: t as u32,
                        images: vec![],
                        state: vec![0.0; 6],
                        action: vec![0.0; 7],
                        reward: Some(0.0),
                        is_terminal: t == (i + 1) * 10 - 1,
                    })
                    .collect(),
            };
            writer.write_episode(&ep).unwrap();
        }
        writer.finish().unwrap();

        let reader = KdbReader::open(path).unwrap();

        // Filter: franka only
        let hits = filter_reader(&reader, "embodiment = 'franka'").unwrap();
        assert_eq!(hits.len(), 5);

        // Filter: successful widowx
        // i=5 (widowx, i%2=false), i=6 (widowx, true), i=7 (widowx, false),
        // i=8 (widowx, true), i=9 (widowx, false)
        let hits = filter_reader(&reader, "embodiment = 'widowx' AND success = true").unwrap();
        assert_eq!(hits.len(), 2); // i=6 and i=8

        // Filter: long episodes
        let hits = filter_reader(&reader, "num_frames >= 80").unwrap();
        assert_eq!(hits.len(), 3); // i=7(80), i=8(90), i=9(100)

        std::fs::remove_file(path).ok();
    }
}
