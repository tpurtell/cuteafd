use super::*;

fn byte_tokenizer() -> tempfile::TempDir {
    // ByteLevel's reversible byte alphabet, with IDs equal to raw bytes.
    let mut extra = 256u32;
    let mut vocab = serde_json::Map::new();
    for byte in 0..=255u32 {
        let character = if (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174
        {
            char::from_u32(byte).unwrap()
        } else {
            let character = char::from_u32(extra).unwrap();
            extra += 1;
            character
        };
        vocab.insert(character.to_string(), serde_json::json!(byte));
    }
    let dir = tempfile::tempdir().unwrap();
    let value = serde_json::json!({"version":"1.0","truncation":null,"padding":null,
        "added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,
        "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        "model":{"type":"WordLevel","vocab":vocab,"unk_token":"?"}});
    std::fs::write(dir.path().join("tokenizer.json"), value.to_string()).unwrap();
    dir
}

#[test]
fn added_markers_and_grammar_document_use_native_tiktoken_without_json_file() {
    let dir = tempfile::tempdir().unwrap();
    crate::plan::testing::write_kimi_tokenizer(dir.path());
    let tokenizer = LoadedTokenizer::from_snapshot(dir.path()).unwrap();
    assert_eq!(tokenizer.added_token_ids(&["<|end_of_msg|>", "a", "missing", "<|open|>"]), [257, 256]);
    let doc = tokenizer.grammar_tokenizer_json().unwrap();
    let grammar: tokenizers::Tokenizer = serde_json::from_str(&doc).unwrap();
    for text in ["hello", "<|open|>user<|end_of_msg|>"] {
        assert_eq!(grammar.encode_fast(text, false).unwrap().get_ids(), tokenizer.encode_ids(text).unwrap());
    }
    assert!(!dir.path().join("tokenizer.json").exists());
}

#[test]
fn added_markers_do_not_include_ordinary_json_vocabulary() {
    let dir = byte_tokenizer();
    let mut value: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.path().join("tokenizer.json")).unwrap()).unwrap();
    value["added_tokens"] = serde_json::json!([{"id":256,"content":"<|open|>","single_word":false,
        "lstrip":false,"rstrip":false,"normalized":false,"special":true}]);
    std::fs::write(dir.path().join("tokenizer.json"), value.to_string()).unwrap();
    let tokenizer = LoadedTokenizer::from_snapshot(dir.path()).unwrap();
    assert_eq!(tokenizer.added_token_ids(&["a", "missing", "<|open|>"]), [256]);
}

#[test]
fn terminal_utf8_matches_lossy_whole_decode_at_every_byte_boundary() {
    let dir = byte_tokenizer();
    for text in [
        "",
        "ASCII",
        "台北 café e\u{301}",
        "👩🏽\u{200d}💻 🦜 🇹🇼",
        "עברית العربية हिन्दी ไทย",
        "\0\r\n\t",
        "�",
        "x��",
        "�x�",
    ] {
        for end in 0..=text.len() {
            let bytes = &text.as_bytes()[..end];
            let mut decoder = streaming_token_decoder(dir.path(), false).unwrap();
            let mut result = String::new();
            for &byte in bytes {
                if let Some(piece) = decoder.step(u32::from(byte)).unwrap() {
                    result.push_str(&piece);
                }
            }
            if let Some(piece) = decoder.finish().unwrap() {
                result.push_str(&piece);
            }
            assert_eq!(
                result,
                String::from_utf8_lossy(bytes),
                "{text:?}, cut {end}"
            );
            assert_eq!(
                decoder.finish().unwrap(),
                None,
                "finish must not emit twice"
            );
        }
    }
}

#[test]
fn terminal_literal_replacement_and_partial_scalar_are_not_dropped() {
    let dir = byte_tokenizer();
    for bytes in [&b"\xef\xbf\xbd"[..], &b"\xf0\x9f\xa6"[..]] {
        let mut decoder = streaming_token_decoder(dir.path(), false).unwrap();
        for &byte in bytes {
            assert!(decoder.step(u32::from(byte)).unwrap().is_none());
        }
        assert_eq!(decoder.finish().unwrap().as_deref(), Some("�"));
        assert_eq!(decoder.step(u32::from(b'A')).unwrap().as_deref(), Some("A"));
        assert_eq!(decoder.finish().unwrap(), None);
    }
}

#[test]
#[ignore = "requires the pinned official tokenizer via CUTEAFD_UNICODE_MODEL"]
fn official_unicode_roundtrip_and_every_token_cut() {
    let snapshot = PathBuf::from(std::env::var_os("CUTEAFD_UNICODE_MODEL").expect("model path"));
    let mut cases = vec![
        "台北，世界！",
        "café e\u{301} Å A\u{30a}",
        "👩🏽\u{200d}💻 🦜 🇹🇼",
        "עברית العربية हिन्दी ไทย",
        "�",
        "x��",
        "\0\r\n\t",
        "\u{202e}abc\u{202c}",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    // Cover every valid scalar's UTF-8 lead/continuation class with a fixed
    // deterministic sample across the Unicode range, plus boundary scalars.
    cases.extend(
        (0..=0x10ffff)
            .step_by(7919)
            .filter_map(char::from_u32)
            .map(|c| format!("A{c}Z")),
    );
    cases.extend(
        [
            0x7f, 0x80, 0x7ff, 0x800, 0xd7ff, 0xe000, 0xffff, 0x10000, 0x10ffff,
        ]
        .into_iter()
        .map(|c| char::from_u32(c).unwrap().to_string()),
    );
    let mut cuts = 0;
    for text in &cases {
        let ids = encode_tokenizer_text(&snapshot, text, false)
            .unwrap()
            .token_ids;
        assert_eq!(
            decode_tokenizer_ids(&snapshot, &ids, false).unwrap().text,
            *text
        );
        for end in 0..=ids.len() {
            let expected = decode_tokenizer_ids(&snapshot, &ids[..end], false)
                .unwrap()
                .text;
            let mut decoder = streaming_token_decoder(&snapshot, false).unwrap();
            let mut result = String::new();
            for &id in &ids[..end] {
                if let Some(piece) = decoder.step(id).unwrap() {
                    result.push_str(&piece);
                }
            }
            if let Some(piece) = decoder.finish().unwrap() {
                result.push_str(&piece);
            }
            assert_eq!(result, expected, "{text:?}, cut {end}");
            cuts += 1;
        }
    }
    println!(
        "official Unicode: {} strings, {cuts} token cuts",
        cases.len()
    );
}

// ---------------------------------------------------------------- long texts by pieces

/// GLM 5.3's pre-tokenizer split.
const GLM_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The test tokenizer's added tokens: GLM-like markers, two that overlap (`<a>b` is longer than
/// `<a>`) and one that ends in a letter.
const ADDED: [&str; 8] = ["<|user|>", "<|assistant|>", "<think>", "</think>", "[gMASK]", "<a>", "<a>b", "/nothink"];

/// A byte-level BPE tokenizer shaped like GLM 5.3's (its split, `ignore_merges`, byte-level
/// pieces), with merges that build a few words and numbers, and [`ADDED`]. `lstrip`: the first
/// added token takes the spaces on its left.
fn bpe_tokenizer(lstrip: bool) -> tempfile::TempDir {
    let mut extra = 256u32;
    let chars: Vec<char> = (0..=255u32).map(|byte| {
        if (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174 {
            char::from_u32(byte).unwrap()
        } else {
            extra += 1;
            char::from_u32(extra - 1).unwrap()
        }
    }).collect();
    let mut vocab = serde_json::Map::new();
    for (id, c) in chars.iter().enumerate() {
        vocab.insert(c.to_string(), serde_json::json!(id));
    }
    let mut merges: Vec<String> = Vec::new();
    for word in ["the", " the", "amber", " amber", " vault", "code", "123", "45"] {
        let bytes: Vec<String> = word.bytes().map(|b| chars[b as usize].to_string()).collect();
        let mut built = bytes[0].clone();
        for byte in &bytes[1..] {
            let merge = format!("{built} {byte}");
            built.push_str(byte);
            if !merges.contains(&merge) {
                merges.push(merge);
            }
            if !vocab.contains_key(&built) {
                let id = vocab.len();
                vocab.insert(built.clone(), serde_json::json!(id));
            }
        }
    }
    let added: Vec<serde_json::Value> = ADDED.iter().enumerate().map(|(i, content)| serde_json::json!({
        "id": vocab.len() + i, "content": content, "single_word": false, "lstrip": lstrip && i == 0,
        "rstrip": false, "normalized": false, "special": content.starts_with("<|") || content.starts_with('[')}))
        .collect();
    let value = serde_json::json!({"version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": GLM_SPLIT}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true},
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": true,
            "vocab": vocab, "merges": merges}});
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), value.to_string()).unwrap();
    dir
}

/// Deterministic text of at least `len` bytes: words, numbers, punctuation, whitespace runs and
/// other scripts, with an added token (or a near miss of one) about every `spacing` bytes.
fn mixed_text(seed: u64, len: usize, spacing: usize) -> String {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = move |n: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % n as u64) as usize
    };
    let parts: &[&str] = &["the", " the", " amber", " basalt", " vault", "code", " 12345", " 6", "123456789", ".",
        ", ", "!?", "...", "'s", "'LL", " (", ")", "\n", "\r\n", "\t", "   ", "  \n  ", " 台北", " café", " e\u{301}",
        " 👩🏽\u{200d}💻", "über"];
    let near: &[&str] = &["<|use", "<a", "a>b", "/noth", "<think", "<|assistant|"];
    let mut text = String::with_capacity(len + 64);
    let mut since = 0;
    while text.len() < len {
        if since >= spacing && next(4) == 0 {
            text.push_str(if next(3) == 0 { near[next(near.len())] } else { ADDED[next(ADDED.len())] });
            since = 0;
        } else {
            let part = parts[next(parts.len())];
            text.push_str(part);
            since += part.len();
        }
    }
    text
}

#[test]
fn long_texts_encode_by_pieces_as_the_tokenizer_does() {
    let dir = bpe_tokenizer(false);
    let tokenizer = LoadedTokenizer::from_snapshot(dir.path()).unwrap();
    let reference = |text: &str| tokenizer.tokenizer.encode(text, false).unwrap().get_ids().to_vec();
    let pieces = tokenizer.pieces.as_ref().expect("added tokens that match by content alone");
    let long = |seed| mixed_text(seed, 6_000, usize::MAX);
    let cases = [
        mixed_text(1, pieces::LONG_TEXT + 4_000, 2_000),
        mixed_text(2, pieces::LONG_TEXT + 8_000, 9_000),
        // Added tokens first, last, adjacent and overlapping (`<a>b` over `<a>`).
        format!("[gMASK]<|user|>{}<a>b{}<a>{}/nothink{}<|assistant|><think>{}", long(3), long(4), long(5), long(6),
            mixed_text(7, pieces::LONG_TEXT, 3_000)),
        // No added token at all: one piece.
        mixed_text(8, pieces::LONG_TEXT + 1_000, usize::MAX),
    ];
    for (case, text) in cases.iter().enumerate() {
        assert!(text.len() >= pieces::LONG_TEXT);
        let expected = reference(text);
        assert_eq!(tokenizer.encode_ids(text).unwrap(), expected, "case {case}");
        assert_eq!(tokenizer.encode_ids(text).unwrap(), expected, "case {case}, kept pieces");
        // A character added in the middle never brings back kept ids.
        let middle = (text.len() / 2..).find(|&i| text.is_char_boundary(i)).unwrap();
        let mut changed = text.clone();
        changed.insert(middle, 'Z');
        assert_eq!(tokenizer.encode_ids(&changed).unwrap(), reference(&changed), "case {case}, changed");
        assert_eq!(tokenizer.encode_text(text, false).unwrap().token_ids, expected, "case {case}, encode_text");
    }
    assert!(pieces.kept_pieces() > 0);
}

#[test]
fn a_growing_conversation_encodes_only_its_new_pieces() {
    let dir = bpe_tokenizer(false);
    let tokenizer = LoadedTokenizer::from_snapshot(dir.path()).unwrap();
    let reference = |text: &str| tokenizer.tokenizer.encode(text, false).unwrap().get_ids().to_vec();
    let pieces = tokenizer.pieces.as_ref().unwrap();
    let mut text = format!("[gMASK]<|user|>{}<|assistant|><think>", mixed_text(11, pieces::LONG_TEXT, usize::MAX));
    assert_eq!(tokenizer.encode_ids(&text).unwrap(), reference(&text));
    assert_eq!(pieces.kept_pieces(), 1);
    for turn in 0..3u64 {
        text.push_str(&format!("Plan {turn}.</think>Answer {turn}.<|user|>{}<|assistant|><think>",
            mixed_text(20 + turn, 5_000, usize::MAX)));
        assert_eq!(tokenizer.encode_ids(&text).unwrap(), reference(&text), "turn {turn}");
    }
    // The first message was kept and reused; each turn's long message was kept once.
    assert_eq!(pieces.kept_pieces(), 4);
}

#[test]
fn added_tokens_that_strip_spaces_encode_whole() {
    let dir = bpe_tokenizer(true);
    let tokenizer = LoadedTokenizer::from_snapshot(dir.path()).unwrap();
    assert!(tokenizer.pieces.is_none());
    let text = mixed_text(31, pieces::LONG_TEXT + 2_000, 1_500);
    assert_eq!(tokenizer.encode_ids(&text).unwrap(), tokenizer.tokenizer.encode(text.as_str(), false).unwrap().get_ids());
}

/// A GLM-shaped conversation: `[gMASK]<sop>`, a system turn, a long first user message, then
/// `turns` assistant turns with reasoning and tool output.
fn glm_conversation(seed: u64, first: usize, turns: u64) -> String {
    let mut text = format!("[gMASK]<sop><|system|>You are terse.<|user|>{}", mixed_text(seed, first, 50_000));
    for turn in 0..turns {
        text.push_str(&format!("<|assistant|><think>Plan {turn}.</think>Calling.<tool_call>read<arg_key>path</arg_key>\
            <arg_value>a{turn}.rs</arg_value></tool_call><|observation|><tool_response>{}</tool_response>",
            mixed_text(seed + 1 + turn, 20_000, 7_000)));
    }
    text + "<|assistant|><think>"
}

#[test]
#[ignore = "requires an official tokenizer via CUTEAFD_LONG_PROMPT_MODEL"]
fn official_long_prompts_encode_by_pieces_as_the_tokenizer_does() {
    let snapshot = PathBuf::from(std::env::var_os("CUTEAFD_LONG_PROMPT_MODEL").expect("model path"));
    let tokenizer = LoadedTokenizer::from_snapshot(&snapshot).unwrap();
    assert!(tokenizer.pieces.is_some(), "this tokenizer encodes long texts whole");
    let reference = |text: &str| tokenizer.tokenizer.encode(text, false).unwrap().get_ids().to_vec();
    for turns in 0..4 {
        let text = glm_conversation(41, 300_000, turns);
        let expected = reference(&text);
        assert_eq!(tokenizer.encode_ids(&text).unwrap(), expected, "{turns} turns");
        assert_eq!(tokenizer.encode_ids(&text).unwrap(), expected, "{turns} turns, kept pieces");
    }
}

/// A 1M-token needle-in-a-haystack prompt's shape: `[salt] `, `items` words with a running
/// index, the vault code at 37% depth and the question. Its words come from a fixed generator,
/// so its tokens differ slightly from other runs'; `CUTEAFD_LONG_PROMPT_TEXT` sends a given text.
fn haystack(items: usize) -> String {
    const WORDS: [&str; 20] = ["amber", "basalt", "cedar", "delta", "ember", "fjord", "granite", "harbor", "iris",
        "juniper", "kelp", "lagoon", "meadow", "nectar", "orchid", "pebble", "quartz", "reef", "sable", "tundra"];
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut body: Vec<String> = (0..items).map(|i| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        format!("{}-{i}", WORDS[(state % 20) as usize])
    }).collect();
    body.insert(items * 37 / 100, "(The vault code is 51767.)".into());
    format!("[173313487] {}\nWhat is the vault code mentioned above? Answer with the number only.", body.join(" "))
}

/// CPU benchmark of a long prompt's encoding on the serve path: the tokenizer's whole encoding
/// (the path before pieces), then by pieces on first sight, from the kept pieces, and with a new
/// turn appended. The prompt is that needle shape (about a million GLM 5.3 tokens) in the GLM
/// chat format. Run it optimized:
/// `CUTEAFD_LONG_PROMPT_MODEL=<snapshot> cargo test --release -p cuteafd-loader --lib long_prompt_benchmark -- --ignored --nocapture`
#[test]
#[ignore = "benchmark; requires CUTEAFD_LONG_PROMPT_MODEL"]
fn long_prompt_benchmark() {
    use std::time::Instant;
    let snapshot = PathBuf::from(std::env::var_os("CUTEAFD_LONG_PROMPT_MODEL").expect("model path"));
    let message = match std::env::var_os("CUTEAFD_LONG_PROMPT_TEXT") {
        Some(path) => std::fs::read_to_string(path).unwrap(),
        None => haystack(186_805),
    };
    let prompt = format!("[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>{message}<|assistant|><think>");
    let next = format!("{prompt}The code is given.</think>51767<|user|>And the first word?<|assistant|><think>");
    let tokenizer = LoadedTokenizer::from_snapshot(&snapshot).unwrap();
    let whole = |text: &str| tokenizer.tokenizer.encode(text, false).unwrap().get_ids().to_vec();
    let timed = |encode: &dyn Fn() -> Vec<u32>, runs: usize| {
        let mut best = f64::INFINITY;
        let mut ids = Vec::new();
        for _ in 0..runs {
            let started = Instant::now();
            ids = encode();
            best = best.min(started.elapsed().as_secs_f64());
        }
        (ids, best)
    };
    let (expected, whole_s) = timed(&|| whole(&prompt), 3);
    let (first, first_s) = timed(&|| tokenizer.encode_ids(&prompt).unwrap(), 1);
    let (kept, kept_s) = timed(&|| tokenizer.encode_ids(&prompt).unwrap(), 5);
    let (grown, grown_s) = timed(&|| tokenizer.encode_ids(&next).unwrap(), 1);
    assert_eq!(first, expected);
    assert_eq!(kept, expected);
    assert_eq!(grown, whole(&next));
    eprintln!("{} bytes, {} tokens: whole (before) {:.1} ms; by pieces: first sight {:.1} ms, kept {:.2} ms, \
        a new turn {:.2} ms (best of 3, 1, 5, 1 runs)", prompt.len(), expected.len(), whole_s * 1e3, first_s * 1e3,
        kept_s * 1e3, grown_s * 1e3);
}

#[test]
fn literal_prompt_spans_bypass_special_and_non_special_added_tokens() {
    let dir = bpe_tokenizer(false);
    let tokenizer = LoadedTokenizer::from_snapshot(dir.path()).unwrap();
    let matcher = LiteralTokens::from_snapshot(dir.path()).unwrap();
    let mut plain = tokenizer.tokenizer.clone();
    plain.with_added_vocabulary(tokenizers::AddedVocabulary::new());
    let added = tokenizer.tokenizer.get_added_tokens_decoder();
    for spelling in ADDED {
        let spans = matcher.spans(spelling, false);
        assert_eq!(spans, vec![0..spelling.len()]);
        let ids = tokenizer.encode_prompt(spelling, &spans).unwrap();
        assert_eq!(ids, plain.encode(spelling, false).unwrap().get_ids());
        assert!(!ids.iter().any(|id| added.contains_key(id)), "{spelling}: {ids:?}");
        assert_eq!(tokenizer.decode_ids(&ids, false).unwrap().text, spelling);
        assert_ne!(ids, tokenizer.encode_ids(spelling).unwrap());
    }
    let text = mixed_text(13, pieces::LONG_TEXT * 2, 100);
    let spans = matcher.spans(&text, false);
    let ids = tokenizer.encode_prompt(&text, &spans).unwrap();
    assert!(!ids.iter().any(|id| added.contains_key(id)));
    assert_eq!(tokenizer.decode_ids(&ids, false).unwrap().text, text);
    assert_eq!(tokenizer.encode_prompt(&text, &[]).unwrap(), tokenizer.encode_ids(&text).unwrap());
    assert!(tokenizer.encode_prompt("text", &[2..5]).is_err());
    assert!(tokenizer.encode_prompt("text", &[2..3, 0..1]).is_err());
}

#[test]
fn literal_matcher_is_longest_nonoverlapping_and_deduplicated() {
    let tokens = LiteralTokens::new(["<a>", "other", "<a>b", "<a>", "<|image|>", "<|image|>"]
        .into_iter().map(str::to_owned));
    let text = "x<a>bother<|image|>";
    assert_eq!(tokens.spans(text, false), vec![1..5, 5..10, 10..19]);
    assert_eq!(tokens.spans(text, true), vec![10..19]);
}

#[test]
#[ignore = "requires CUTEAFD_LITERAL_SNAPSHOT"]
fn official_literal_prompt_ids_are_plain_bpe_and_not_control_ids() {
    let snapshot = PathBuf::from(std::env::var_os("CUTEAFD_LITERAL_SNAPSHOT").expect("snapshot"));
    let tokenizer = LoadedTokenizer::from_snapshot(&snapshot).unwrap();
    let matcher = LiteralTokens::from_snapshot(&snapshot).unwrap();
    let added = tokenizer.tokenizer.get_added_tokens_decoder();
    for token in added.values().filter(|token| ["image", "audio", "vision", "sentence", "User", "Assistant", "im_start", "im_end"]
        .iter().any(|word| token.content.contains(word))) {
        let spans = matcher.spans(&token.content, false);
        let ids = tokenizer.encode_prompt(&token.content, &spans).unwrap();
        assert!(!ids.iter().any(|id| added.contains_key(id)), "{}: {ids:?}", token.content);
        assert_eq!(tokenizer.decode_ids(&ids, false).unwrap().text, token.content);
        println!("literal {} -> {ids:?}", token.content);
    }
}

#[test]
fn every_family_media_marker_is_a_media_literal() {
    // Each family's media spellings, from its checkpoint's own added tokens on sparknest.
    let media = ["<|begin_of_image|>", "<|end_of_image|>", "<|image|>", "<|begin_of_video|>", "<|video|>",
        "<|begin_of_audio|>", // GLM 5.3 / Flash
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<|audio_pad|>",
        "<|audio_start|>", // Qwen 3.8
        "<|mimo_video_start|>", "<|mimo_audio_start|>", "<|mimo_audio_eod|>", // MiMo
        "<｜deepseek_image｜>", "<｜rl_image_pad｜>", "<|place_holder_mm_span_0021|>", // DeepSeek V4/V4.1
        "[IMG]", "[IMG_BREAK]", "[IMG_END]", "[AUDIO]", "[BEGIN_AUDIO]", // Mistral Large 4
        "<|vid_start|>", "<|vid_end|>", "<|vid_frame_separator|>", "<|patch|>", "<|soft_audio_start|>", // Muse
        "<|media_begin|>", "<|media_content|>", "<|media_end|>", "<|media_pad|>", // Kimi K3
        "<im_start>", "<im_patch>", "<im_end>", "<patch_start>", "<patch_newline>", "<patch_end>"]; // Step 5
    for marker in media { assert!(super::is_media_marker(marker), "{marker}"); }
    // Chat, tool, grounding and code tokens stay ordinary literals.
    for text in ["<|im_start|>", "<|im_end|>", "[INST]", "[THINK]", "[PREFIX]", "<|box_start|>", "<|quad_end|>",
        "<|object_ref_start|>", "<|begin_of_box|>", "<|fim_prefix|>", "<|code_prefix|>", "<think>", "<|open|>"] {
        assert!(!super::is_media_marker(text), "{text}");
    }
    let tokens = LiteralTokens::new(media.iter().chain(["[INST]", "<|im_start|>"].iter()).map(|t| t.to_string()));
    let text = media.join(" ") + " [INST] <|im_start|>";
    let found: Vec<_> = tokens.spans(&text, true).into_iter().map(|span| text[span].to_owned()).collect();
    assert_eq!(found, media.map(str::to_owned));
}
