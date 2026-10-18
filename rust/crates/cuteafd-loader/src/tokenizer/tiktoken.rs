//! A checkpoint that ships a TikToken BPE (`tiktoken.model`) and no
//! `tokenizer.json` (Kimi K3) as an in-memory `tokenizers::Tokenizer`: the
//! ranks become a byte-level BPE vocabulary whose merges replay TikToken's
//! rank order, the official split pattern is the pre-tokenizer, and the
//! reserved special range is filled exactly as the bundled tokenizer does.
//! Every existing path (incremental decode, long-text pieces, added tokens)
//! then serves it unchanged.
use anyhow::{bail, ensure, Context, Result};
use base64::Engine;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;

/// The split pattern of the bundled `tokenization_kimi.py` (`pat_str`). It is
/// Python code, not data, so the reader requires the shipped file to contain
/// these alternatives verbatim rather than trusting a stale copy.
const KIMI_PATTERN: [&str; 8] = [
    r"[\p{Han}]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
];
/// `num_reserved_special_tokens` in the bundled tokenizer.
const KIMI_RESERVED: usize = 256;

/// Whether `snapshot` carries a TikToken tokenizer this reader understands.
pub fn present(snapshot: &Path) -> bool {
    snapshot.join("tiktoken.model").is_file() && snapshot.join("tokenization_kimi.py").is_file()
}

/// The `tokenizer.json` document for the snapshot's TikToken BPE.
pub fn tokenizer_json(snapshot: &Path) -> Result<Value> {
    let source = std::fs::read_to_string(snapshot.join("tokenization_kimi.py"))
        .context("reading tokenization_kimi.py")?;
    for alternative in KIMI_PATTERN {
        ensure!(source.contains(alternative),
            "tokenization_kimi.py pat_str differs from the Kimi split pattern this build reads (missing {alternative})");
    }
    ensure!(source.contains(&format!("num_reserved_special_tokens = {KIMI_RESERVED}")),
        "tokenization_kimi.py reserves a different special-token count");
    let ranks = read_ranks(&std::fs::read(snapshot.join("tiktoken.model")).context("reading tiktoken.model")?)?;
    let config: Value = serde_json::from_slice(&std::fs::read(snapshot.join("tokenizer_config.json"))
        .context("reading tokenizer_config.json")?)?;
    let named: HashMap<usize, String> = config["added_tokens_decoder"].as_object()
        .context("tokenizer_config.json lacks added_tokens_decoder")?.iter()
        .map(|(id, entry)| Ok((id.parse::<usize>()?, entry["content"].as_str().context("added token content")?.to_owned())))
        .collect::<Result<_>>()?;
    let base = ranks.len();
    ensure!(named.keys().all(|id| (base..base + KIMI_RESERVED).contains(id)),
        "named special tokens must lie in the reserved range {base}..{}", base + KIMI_RESERVED);
    Ok(build(&ranks, (base..base + KIMI_RESERVED)
        .map(|id| (id, named.get(&id).cloned().unwrap_or_else(|| format!("<|reserved_token_{id}|>"))))
        .collect(), &KIMI_PATTERN.join("|")))
}

/// `base64(token) rank` lines; ranks must be exactly 0..n.
fn read_ranks(file: &[u8]) -> Result<Vec<Vec<u8>>> {
    let mut ranks: Vec<Option<Vec<u8>>> = Vec::new();
    for (line_no, line) in file.split(|b| *b == b'\n').enumerate() {
        let line = std::str::from_utf8(line).context("tiktoken.model is not UTF-8")?.trim();
        if line.is_empty() { continue; }
        let (token, rank) = line.split_once(' ').with_context(|| format!("tiktoken.model line {}", line_no + 1))?;
        let rank: usize = rank.trim().parse().with_context(|| format!("tiktoken.model line {} rank", line_no + 1))?;
        let token = base64::engine::general_purpose::STANDARD.decode(token)
            .with_context(|| format!("tiktoken.model line {} token", line_no + 1))?;
        if ranks.len() <= rank { ranks.resize(rank + 1, None); }
        ensure!(ranks[rank].replace(token).is_none(), "tiktoken.model repeats rank {rank}");
    }
    ranks.into_iter().enumerate().map(|(rank, token)| token.with_context(|| format!("tiktoken.model lacks rank {rank}"))).collect()
}

/// GPT-2's byte-to-printable mapping used by byte-level BPE vocabularies.
fn byte_chars() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next = 256u32;
    for byte in 0..=255u8 {
        let printable = matches!(byte, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
        table[byte as usize] = if printable { char::from(byte) } else { let c = char::from_u32(next).unwrap(); next += 1; c };
    }
    table
}

/// Byte-level BPE whose merges reproduce TikToken: each multi-byte token's
/// merge is the pair its own bytes reach by merging lowest ranks first,
/// restricted to ranks below its own.
fn build(ranks: &[Vec<u8>], specials: Vec<(usize, String)>, pattern: &str) -> Value {
    let chars = byte_chars();
    let text = |bytes: &[u8]| bytes.iter().map(|b| chars[*b as usize]).collect::<String>();
    let index: HashMap<&[u8], usize> = ranks.iter().enumerate().map(|(rank, token)| (token.as_slice(), rank)).collect();
    let mut vocab = serde_json::Map::with_capacity(ranks.len());
    let mut merges = Vec::with_capacity(ranks.len());
    for (rank, token) in ranks.iter().enumerate() {
        vocab.insert(text(token), json!(rank));
        if token.len() < 2 { continue; }
        let mut parts: Vec<&[u8]> = (0..token.len()).map(|i| &token[i..i + 1]).collect();
        loop {
            let best = (0..parts.len() - 1).filter_map(|i| {
                let joined = &token[offset(&parts, i, token)..offset(&parts, i + 2, token)];
                index.get(joined).filter(|r| **r < rank).map(|r| (*r, i))
            }).min();
            let Some((_, i)) = best else { break };
            let start = offset(&parts, i, token);
            let end = offset(&parts, i + 2, token);
            parts.splice(i..i + 2, [&token[start..end]]);
        }
        // A well-formed TikToken vocabulary always ends in exactly two parts.
        if let [left, right] = parts[..] { merges.push(json!([text(left), text(right)])); }
    }
    let added: Vec<Value> = specials.iter().map(|(id, content)| json!({"id": id, "content": content,
        "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true})).collect();
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added, "normalizer": null,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
            "vocab": vocab, "merges": merges},
    })
}

/// Byte offset of part `i` within `token` (parts are contiguous slices of it).
fn offset(parts: &[&[u8]], i: usize, token: &[u8]) -> usize {
    if i >= parts.len() { token.len() } else { parts[i].as_ptr() as usize - token.as_ptr() as usize }
}

/// The in-memory tokenizer.
pub fn load(snapshot: &Path) -> Result<tokenizers::Tokenizer> {
    let document = tokenizer_json(snapshot)?;
    let tokenizer: tokenizers::Tokenizer = serde_json::from_value(document)
        .map_err(|error| anyhow::anyhow!("building tokenizer from tiktoken.model: {error}"))?;
    if tokenizer.get_vocab_size(true) == 0 { bail!("tiktoken.model produced an empty vocabulary"); }
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_replay_rank_order() {
        // Bytes a b c, then "ab" (rank 3), "bc" (4), "abc" (5): abc merges as ab + c.
        let ranks: Vec<Vec<u8>> = [&b"a"[..], b"b", b"c", b"ab", b"bc", b"abc"].iter().map(|t| t.to_vec()).collect();
        let doc = build(&ranks, vec![(6, "<|x|>".into())], r"\S+|\s+");
        let merges = doc["model"]["merges"].as_array().unwrap();
        assert_eq!(merges, &vec![json!(["a", "b"]), json!(["b", "c"]), json!(["ab", "c"])]);
        let tokenizer: tokenizers::Tokenizer = serde_json::from_value(doc).unwrap();
        // "abc" is one token; "cab" has no merge for "ca", so c + ab.
        assert_eq!(tokenizer.encode_fast("abc", false).unwrap().get_ids(), &[5]);
        assert_eq!(tokenizer.encode_fast("cab", false).unwrap().get_ids(), &[2, 3]);
        assert_eq!(tokenizer.decode(&[5, 6], false).unwrap(), "abc<|x|>");
    }

    /// Byte-exact against the official TikToken encoding of a real snapshot:
    /// `CUTEAFD_KIMI_TOKENIZER_ORACLE=<snapshot>:<oracle.json>` with
    /// `[{text, ids}]` from `tiktoken.Encoding(..., pat_str from the shipped
    /// tokenization_kimi.py).encode(text, allowed_special='all')`.
    #[test]
    fn official_tiktoken_oracle_when_present() {
        let Ok(spec) = std::env::var("CUTEAFD_KIMI_TOKENIZER_ORACLE") else { return };
        let (snapshot, oracle) = spec.rsplit_once(':').expect("<snapshot>:<oracle.json>");
        let loaded = crate::tokenizer::LoadedTokenizer::from_snapshot(Path::new(snapshot)).unwrap();
        let cases: Vec<Value> = serde_json::from_slice(&std::fs::read(oracle).unwrap()).unwrap();
        let mut tokens = 0;
        for case in &cases {
            let text = case["text"].as_str().unwrap();
            let expected: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let ids = loaded.encode_ids(text).unwrap();
            assert_eq!(ids, expected, "{:?}", &text[..text.len().min(80)]);
            assert_eq!(loaded.decode_ids(&ids, false).unwrap().text, text);
            tokens += ids.len();
        }
        assert!(cases.len() > 10 && tokens > 1000);
    }

    #[test]
    fn byte_table_matches_gpt2() {
        let table = byte_chars();
        assert_eq!(table[b'A' as usize], 'A');
        assert_eq!(table[b' ' as usize], 'Ġ');
        assert_eq!(table[b'\n' as usize], 'Ċ');
        assert_eq!(table.iter().collect::<std::collections::HashSet<_>>().len(), 256);
    }
}
