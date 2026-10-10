use crate::shared::token_io::ScoreRows;
use anyhow::{ensure, Result};
use cuteafd_api::openai::{NativeConstraint, NativeFailure};
use cuteafd_ffi::{NativeLibrary, CuteafdXGrammarCompiler, CuteafdXGrammarGrammar,
    CuteafdXGrammarMatcher, CUTEAFD_XGRAMMAR_STRUCTURAL_TAG};
use std::{collections::{HashMap, VecDeque}, path::PathBuf, sync::Arc};

/// The tokenizer's stop token, the only stop id the grammar compiler is given.
const STOP_TOKEN: u32 = 1;

pub(crate) struct Compiler<'a> {
    library: &'a NativeLibrary,
    tokenizer: PathBuf,
    vocab: usize,
    /// Every stop id the grammar accepts where it may end.
    stops: Vec<u32>,
    compiler: Option<CuteafdXGrammarCompiler<'a>>,
    grammars: HashMap<NativeConstraint, Arc<CuteafdXGrammarGrammar<'a>>>,
    order: VecDeque<NativeConstraint>,
}
impl<'a> Compiler<'a> {
    /// A compiler for a DeepSeek tokenizer: `vocab` logits per row, stop id 1.
    pub fn new(library: &'a NativeLibrary, tokenizer: PathBuf, vocab: usize) -> Self {
        Self::with_vocab(library, tokenizer, vocab, vec![STOP_TOKEN])
    }

    /// A compiler for another tokenizer: `vocab` logits per row and its stop ids.
    pub fn with_vocab(library: &'a NativeLibrary, tokenizer: PathBuf, vocab: usize, stops: Vec<u32>) -> Self {
        Self { library, tokenizer, vocab, stops, compiler: None, grammars: HashMap::new(), order: VecDeque::new() }
    }
    pub fn matcher(&mut self, spec: &NativeConstraint) -> Result<State<'a>> {
        if self.compiler.is_none() {
            let stops: Vec<i32> = self.stops.iter().map(|&s| s as i32).collect();
            self.compiler = Some(self.library.xgrammar_compiler(&self.tokenizer, self.vocab, &stops)?);
        }
        let grammar = if let Some(grammar) = self.grammars.get(spec) { grammar.clone() } else {
            let grammar = Arc::new(self.compiler.as_ref().unwrap().compile(
                CUTEAFD_XGRAMMAR_STRUCTURAL_TAG, Some(&spec.0), true)
                .map_err(|error| NativeFailure::BadRequest(format!("{error:#}")))?);
            if self.grammars.len() == 64 {
                if let Some(old) = self.order.pop_front() { self.grammars.remove(&old); }
            }
            self.grammars.insert(spec.clone(), grammar.clone());
            grammar
        };
        self.order.retain(|key| key != spec);
        self.order.push_back(spec.clone());
        Ok(State { matcher: grammar.matcher()?, mask: vec![0; self.vocab.div_ceil(32)], stops: self.stops.clone(),
            terminated: false })
    }
}

/// Collect one mask per row, honouring each row's `needs_mask`.
///
/// `fill` reuses a single word buffer across rows, exactly like the XGrammar
/// matcher's own mask, so a row that needs no mask must be recorded as `None`
/// rather than inheriting the previous row's bits. Keeping that decision in one
/// place is what makes it testable without a compiled grammar.
fn collect_row_masks<F>(rows: usize, words: usize,
    mut fill: F,
) -> Result<Vec<Option<Vec<u32>>>>
where
    F: FnMut(usize, &mut [u32]) -> Result<bool>,
{
    let mut buffer = vec![0u32; words];
    let mut collected = Vec::with_capacity(rows);
    for index in 0..rows {
        let needs_mask = fill(index, &mut buffer)?;
        collected.push(needs_mask.then(|| buffer.clone()));
    }
    Ok(collected)
}

pub(crate) struct State<'a> {
    matcher: CuteafdXGrammarMatcher<'a>,
    mask: Vec<u32>,
    stops: Vec<u32>,
    /// The grammar accepted one of its stop tokens. XGrammar then refuses
    /// every further mask and token, so the request ends here: callers finish
    /// it (`terminated`), and any mask asked for meanwhile admits only stops.
    terminated: bool,
}
impl State<'_> {
    /// True once the grammar accepted a stop token: the request is complete.
    pub fn terminated(&self) -> bool {
        self.terminated
    }
    /// A mask that admits only the grammar's stop tokens.
    fn stop_only(&self) -> Vec<u32> {
        let mut mask = vec![0u32; self.mask.len()];
        for &stop in &self.stops {
            if let Some(word) = mask.get_mut(stop as usize / 32) {
                *word |= 1 << (stop % 32);
            }
        }
        mask
    }
    pub fn mask(&mut self) -> Result<Option<&[u32]>> {
        if self.terminated {
            self.mask = self.stop_only();
            return Ok(Some(&self.mask));
        }
        Ok(if self.matcher.fill_bitmask(&mut self.mask)? { Some(&self.mask) } else { None })
    }
    /// Commits an emitted token. Accepting a stop token terminates the grammar;
    /// a further stop is a no-op, any other token after it is an error.
    pub fn accept(&mut self, token: u32) -> Result<()> {
        if self.terminated {
            ensure!(self.stops.contains(&token), "token {token} after the request grammar ended");
            return Ok(());
        }
        ensure!(self.matcher.accept_token(token)?, "emitted token violates request grammar");
        self.terminated = self.stops.contains(&token);
        Ok(())
    }
    /// Keep the drafts the grammar accepts, stopping at the first it rejects.
    ///
    /// A grammar whose root could end here (`is_completed`) may still accept
    /// more tokens: free text around tool-call tags is always completable. So
    /// completion alone must not end the proposal, or every tool-enabled request
    /// loses speculation for all of its free text. When only the stop token is
    /// legal, the next non-stop draft is rejected here anyway. A stop-token draft
    /// (any of the compiler's stops) is never verified; the target emits the
    /// stop token itself.
    ///
    /// Drafts only speed decoding up: a grammar error here keeps the anchor
    /// alone instead of failing the request (or, worse, its whole batch).
    pub fn truncate_proposal(&self, input: &mut Vec<u32>) -> Result<()> {
        if self.terminated {
            input.truncate(1);
            return Ok(());
        }
        let kept = (|| -> Result<usize> {
            let mut branch = self.matcher.fork()?;
            for index in 1..input.len() {
                if self.stops.contains(&input[index]) || !branch.accept_token(input[index])? {
                    return Ok(index);
                }
            }
            Ok(input.len())
        })();
        input.truncate(kept.unwrap_or_else(|error| {
            tracing::warn!("grammar draft check failed, verifying the anchor alone: {error:#}");
            1
        }));
        Ok(())
    }
    /// The first `rows` masks along a verification round's hypothetical
    /// prefix: row 0 is the mask *before* any draft is accepted; row
    /// `index > 0` first accepts `input[index]` into a private fork. A
    /// terminated grammar (its anchor was a stop) admits only stops.
    fn branch_masks(&self, input: &[u32], rows: usize) -> Result<Vec<Option<Vec<u32>>>> {
        ensure!(!input.is_empty(), "grammar mask request has no emitted anchor");
        ensure!(rows <= input.len(), "grammar mask row is outside the verification round");
        if self.terminated {
            return Ok(vec![Some(self.stop_only()); rows]);
        }
        let mut branch = self.matcher.fork()?;
        collect_row_masks(rows, self.mask.len(), |index, mask| {
            // The authoritative state already contains input[0], the emitted
            // anchor. Each later row follows the preceding legal draft token.
            if index > 0 {
                ensure!(!self.stops.contains(&input[index]), "stop token in verification draft");
                ensure!(branch.accept_token(input[index])?, "illegal verification draft token");
            }
            branch.fill_bitmask(mask)
        })
    }
    /// The packed grammar mask of **one** row of a verification round, or
    /// `None` when the grammar allows every token at that row.
    ///
    /// Rows up to and including `row` are accepted into a private fork. Used to
    /// validate a device-selected masked row at retention time, where only the
    /// accepted frontier's mask is needed.
    pub fn prepare_verification_mask_row(&self, input: &[u32],
        row: usize,
    ) -> Result<Option<Vec<u32>>> {
        ensure!(row < input.len(), "grammar mask row is outside the verification round");
        Ok(self.branch_masks(input, row + 1)?.pop().flatten())
    }
    /// The packed grammar mask of every row of a verification round, in row
    /// order, or `None` for a row that needs no mask.
    ///
    /// Row 0 is the mask *before* any draft is accepted; row `index > 0` first
    /// accepts `input[index]` into a private fork. That is exactly the
    /// hypothetical-prefix mask `select_verification` uses, so the device reader
    /// produces the same masked argmax as the CPU.
    ///
    /// A row whose grammar allows every token must report `None`, never the
    /// previous row's bits: the matcher's mask buffer is reused across rows, so
    /// dropping `fill_bitmask`'s return value would silently apply a stale
    /// grammar to that row.
    pub fn prepare_verification_masks(&self, input: &[u32]) -> Result<Vec<Option<Vec<u32>>>> {
        self.branch_masks(input, input.len())
    }
    pub fn select_verification(&self, scores: &ScoreRows, offset: usize, input: &[u32]) -> Result<Vec<u32>> {
        self.prepare_verification_masks(input)?.iter().enumerate()
            .map(|(index, mask)| scores.select(offset + index, mask.as_deref()))
            .collect()
    }
    /// Stochastic twin of [`Self::select_verification`]. The grammar mask is
    /// applied first along the hypothetical draft prefix; the exact sampler then
    /// draws from the masked full-vocabulary row at the absolute emitted-token
    /// position `base_position + index`. The resulting tokens feed the same
    /// sample-and-match verifier, so an accepted draft never biases the target
    /// distribution.
    pub fn select_verification_sampled(
        &self,
        scores: &ScoreRows,
        offset: usize,
        input: &[u32],
        params: cuteafd_core::TargetSamplingParams,
        base_position: u64,
    ) -> Result<Vec<u32>> {
        self.prepare_verification_masks(input)?.iter().enumerate()
            .map(|(index, mask)| scores.sample(offset + index, mask.as_deref(), params, base_position + index as u64))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The XGrammar matcher's mask buffer is reused across rows and
    /// `fill_bitmask` returns whether that row needs a mask at all. A row whose
    /// grammar allows every token must be recorded as `None`; recording the
    /// buffer anyway would apply the **previous** row's grammar to it, which is
    /// a wrong-answer path (the device would either report a spurious
    /// `grammar allows no target token` or silently pick a filtered token).
    ///
    /// The closure below deliberately leaves the reused buffer untouched when
    /// no mask is needed, exactly as the matcher does.
    #[test]
    fn a_row_that_needs_no_mask_does_not_inherit_the_previous_rows_bits() {
        let words = 4;
        // Masked, unmasked, masked, unmasked: reuse is exercised both ways.
        let needs = [true, false, true, false];
        let collected = collect_row_masks(needs.len(), words, |index, mask| {
            if needs[index] {
                for word in mask.iter_mut() {
                    *word = 0xAA00 + index as u32;
                }
            }
            Ok(needs[index])
        })
        .unwrap();
        assert_eq!(collected.len(), 4);
        assert_eq!(collected[0].as_deref().unwrap()[0], 0xAA00);
        assert!(collected[1].is_none(), "row 1 must not inherit row 0's bits");
        assert_eq!(collected[2].as_deref().unwrap()[0], 0xAA02);
        assert!(collected[3].is_none(), "row 3 must not inherit row 2's bits");

        // And the reverse: an unmasked first row, then masked rows.
        let needs = [false, true, true];
        let collected = collect_row_masks(needs.len(), words, |index, mask| {
            if needs[index] {
                for word in mask.iter_mut() {
                    *word = 0xBB00 + index as u32;
                }
            }
            Ok(needs[index])
        })
        .unwrap();
        assert!(collected[0].is_none(), "an unmasked first row stays None");
        assert_eq!(collected[1].as_deref().unwrap()[0], 0xBB01);
        assert_eq!(collected[2].as_deref().unwrap()[0], 0xBB02);
    }

    /// Every row of an all-unmasked or all-masked sequence keeps its own answer.
    #[test]
    fn collect_row_masks_preserves_each_rows_decision() {
        let all_unmasked = collect_row_masks(5, 4, |_, _| Ok(false)).unwrap();
        assert!(all_unmasked.iter().all(Option::is_none));
        let all_masked = collect_row_masks(5, 4, |index, mask| {
            mask[0] = index as u32;
            Ok(true)
        })
        .unwrap();
        assert_eq!(
            all_masked.iter().map(|mask| mask.as_deref().unwrap()[0]).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
    }

    /// Chunk 4b: the **real** xgrammar matcher, forked per speculative prefix row.
    ///
    /// This is the one test that exercises `prepare_verification_masks` and
    /// `prepare_verification_mask_row` against a compiled grammar (the other tests
    /// drive `collect_row_masks` with a synthetic fill). It pins the three chunk-4b
    /// constraints on the mask-preparation half of the flow:
    /// per-row masks follow the hypothetical prefix, the single-row retention
    /// variant agrees with the batch form, and the authoritative matcher is never
    /// advanced (fork/rollback: the branches are dropped, so the committed state
    /// still governs the next authoritative `accept`).
    ///
    /// Skips without `CUTEAFD_NATIVE_LIB`; uses the committed tiny tokenizer
    /// (vocab 8) so the test does not need a model snapshot.
    #[test]
    fn verification_masks_fork_per_row_without_mutating_the_authoritative_matcher() {
        let Some(path) = std::env::var_os("CUTEAFD_NATIVE_LIB") else {
            eprintln!("skipping: CUTEAFD_NATIVE_LIB is not set");
            return;
        };
        let library = unsafe { NativeLibrary::load(path).unwrap() };
        let tokenizer = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../native/tests/fixtures/xgrammar_tiny_tokenizer.json");
        let compiler = library.xgrammar_compiler(&tokenizer, 8, &[6]).unwrap();
        let schema = r#"{"type":"object","properties":{"x":{"type":"string"}},"required":["x"],"additionalProperties":false}"#;
        let grammar = compiler
            .compile(cuteafd_ffi::CUTEAFD_XGRAMMAR_JSON_SCHEMA, Some(schema), true)
            .unwrap();
        let mut matcher = grammar.matcher().unwrap();
        // The committed anchor `{` (token 1) is already in the authoritative state.
        assert!(matcher.accept_token(1).unwrap(), "the anchor must be an allowed token");
        let mut state = State { matcher, mask: vec![0u32; 1], stops: vec![6], terminated: false };
        let before = state.mask().unwrap().map(<[u32]>::to_vec);

        // Hypothetical draft prefix: anchor, space, "x", colon.
        let input = [1u32, 7, 2, 3];
        let masks = state.prepare_verification_masks(&input).unwrap();
        assert_eq!(masks.len(), input.len());
        assert!(masks.iter().all(Option::is_some), "this grammar needs a mask on every row");
        // Per-row masks follow the prefix: after `{ "x"` the allowed set is no
        // longer the anchor's.
        assert_ne!(masks[0], masks[2], "row 2 must follow the x-token prefix, not the anchor");
        assert_ne!(masks[0], masks[3], "row 3 must follow the colon prefix");
        // The retention-time single-row form agrees with the batch form.
        for (row, mask) in masks.iter().enumerate() {
            let single = state.prepare_verification_mask_row(&input, row).unwrap();
            assert_eq!(mask, &single, "row {row}: single-row mask differs from the batch form");
        }
        // Fork/rollback: preparing (and dropping) every branch left the
        // authoritative state exactly where it was.
        let after = state.mask().unwrap().map(<[u32]>::to_vec);
        assert_eq!(before, after, "verification mask preparation mutated the authoritative matcher");

        // An illegal draft token is the same hard error the CPU path reports.
        let illegal = state.prepare_verification_masks(&[1, 5]).unwrap_err();
        assert!(illegal.to_string().contains("illegal verification draft token"),
            "unexpected error: {illegal}");
        // A row outside the round is rejected rather than silently clamped.
        assert!(state.prepare_verification_mask_row(&input, input.len()).is_err());
    }

    /// Drafts survive grammar truncation exactly as far as the grammar accepts
    /// them. Free text around a tag is always completable, which must not stop
    /// speculation; a finished JSON value admits only the stop token, which the
    /// proposal never carries into verification.
    #[test]
    fn truncation_keeps_every_draft_the_grammar_accepts() {
        let Some(path) = std::env::var_os("CUTEAFD_NATIVE_LIB") else {
            eprintln!("skipping: CUTEAFD_NATIVE_LIB is not set");
            return;
        };
        let library = unsafe { NativeLibrary::load(path).unwrap() };
        let tokenizer = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../native/tests/fixtures/xgrammar_tiny_tokenizer.json");
        // Vocabulary: 1 `{`, 2 `"x"`, 3 `:`, 4 `"a"`, 5 `}`, 6 <eos>, 7 space.
        let compiler = library.xgrammar_compiler(&tokenizer, 8, &[6]).unwrap();

        // Free text with an optional triggered tag, like tool calls without tool_choice=required.
        let tagged = r#"{"type":"structural_tag","format":{"type":"triggered_tags","triggers":["{"],"tags":[{"type":"tag","begin":"{","content":{"type":"const_string","value":"\"a\""},"end":"}"}],"at_least_one":false,"stop_after_first":true}}"#;
        let grammar = compiler.compile(CUTEAFD_XGRAMMAR_STRUCTURAL_TAG, Some(tagged), true).unwrap();
        let mut matcher = grammar.matcher().unwrap();
        assert!(matcher.accept_token(7).unwrap(), "free text accepts the anchor");
        assert!(matcher.is_completed().unwrap(), "free text is completable, the case that used to drop drafts");
        let state = State { matcher, mask: vec![0u32; 1], stops: vec![6], terminated: false };
        let truncated = |proposal: &[u32]| { let mut input = proposal.to_vec(); state.truncate_proposal(&mut input).unwrap(); input };
        assert_eq!(truncated(&[7, 2, 3, 7, 2]), [7, 2, 3, 7, 2], "free-text drafts are all kept");
        assert_eq!(truncated(&[7, 2, 1, 4, 5, 7]), [7, 2, 1, 4, 5], "drafts continue through the tag, then stop after it");
        assert_eq!(truncated(&[7, 2, 1, 2]), [7, 2, 1], "an illegal draft inside the tag ends the proposal");
        assert_eq!(truncated(&[7, 2, 6, 3]), [7, 2], "a stop-token draft is not verified");

        // A completed JSON value admits only the stop token.
        let schema = r#"{"type":"object","properties":{"x":{"type":"string"}},"required":["x"],"additionalProperties":false}"#;
        let grammar = compiler.compile(cuteafd_ffi::CUTEAFD_XGRAMMAR_JSON_SCHEMA, Some(schema), true).unwrap();
        let mut matcher = grammar.matcher().unwrap();
        // The anchor (the closing brace) is already in the authoritative state.
        for token in [1, 2, 3, 4, 5] { assert!(matcher.accept_token(token).unwrap()); }
        assert!(matcher.is_completed().unwrap());
        let state = State { matcher, mask: vec![0u32; 1], stops: vec![6], terminated: false };
        let truncated = |proposal: &[u32]| { let mut input = proposal.to_vec(); state.truncate_proposal(&mut input).unwrap(); input };
        assert_eq!(truncated(&[5, 6, 1]), [5], "after the closing brace no draft is verified");
        assert_eq!(truncated(&[5, 1, 2]), [5], "a token past the completed value is rejected");
    }

    /// A tokenizer can have several EOS/turn-end ids. Accepting any of them
    /// terminates a private matcher: asking for its next mask then fails the
    /// worker batch. Keep every compiler stop out of speculative prefixes,
    /// while allowing the target to emit the stop from the preceding mask.
    #[test]
    fn every_compiler_stop_is_truncated_before_verification() {
        let Some(path) = std::env::var_os("CUTEAFD_NATIVE_LIB") else {
            eprintln!("skipping: CUTEAFD_NATIVE_LIB is not set");
            return;
        };
        // SAFETY: the test supplies the engine's native library, retained for
        // the lifetime of every compiler, grammar and matcher below.
        let library = unsafe { NativeLibrary::load(path).unwrap() };
        let tokenizer = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../native/tests/fixtures/xgrammar_tiny_tokenizer.json");
        // Explicit compiler stops override token spelling; token 0 supplies a
        // second EOS without needing a second fixture or any model weights.
        let stops = vec![6, 0];
        let mut compiler = Compiler::with_vocab(&library, tokenizer, 8, stops.clone());
        let spec = NativeConstraint(serde_json::json!({"type":"structural_tag", "format":{
            "type":"ds41_json_schema", "strict":true,
            "json_schema":{"type":"object", "properties":{"x":{"type":"string"}},
                "required":["x"], "additionalProperties":false}
        }}).to_string());
        let mut state = compiler.matcher(&spec).unwrap();
        let mut sibling = compiler.matcher(&spec).unwrap();
        state.accept(1).unwrap();
        sibling.accept(1).unwrap();
        let before = state.mask().unwrap().map(<[u32]>::to_vec);
        let sibling_before = sibling.mask().unwrap().map(<[u32]>::to_vec);

        for stop in stops {
            let mut input = vec![1, 2, 3, 4, 5, stop, 7];
            state.truncate_proposal(&mut input).unwrap();
            assert_eq!(input, [1, 2, 3, 4, 5], "stop {stop} must never enter verification");
            let masks = state.prepare_verification_masks(&input).unwrap();
            for (row, mask) in masks.iter().enumerate() {
                assert_eq!(*mask, state.prepare_verification_mask_row(&input, row).unwrap());
            }
            let last = masks.last().unwrap().as_ref().unwrap();
            assert_ne!(last[0] & (1 << stop), 0, "the target can still select stop {stop}");

            // An accidentally untruncated caller is rejected before it can
            // terminate a branch and call fill_bitmask on it.
            let untruncated = [1, 2, 3, 4, 5, stop];
            let error = state.prepare_verification_masks(&untruncated).unwrap_err();
            assert!(error.to_string().contains("stop token in verification draft"));
            let error = state.prepare_verification_mask_row(&untruncated, 5).unwrap_err();
            assert!(error.to_string().contains("stop token in verification draft"));
            assert_eq!(state.mask().unwrap().map(<[u32]>::to_vec), before,
                "private branches must not advance the authoritative matcher");
            assert_eq!(sibling.mask().unwrap().map(<[u32]>::to_vec), sibling_before,
                "another request's matcher must remain healthy");
            assert_eq!(sibling.prepare_verification_masks(&[1, 2, 3]).unwrap().len(), 3);
        }

        for token in [2, 3, 4, 5, 0] { state.accept(token).unwrap(); }
        // The target's secondary EOS ended this request, not its sibling.
        for token in [2, 3, 4, 5, 6] { sibling.accept(token).unwrap(); }
    }

    /// The v0 smoke failure: a forced tool call's grammar accepted a stop
    /// token, decoding went on, and the next `fill_bitmask` on the terminated
    /// matcher failed the whole batch (GLM: the coordinator). Once a stop is
    /// accepted the state reports `terminated`, masks admit only stops, drafts
    /// are dropped, and no native call on the dead matcher is made.
    #[test]
    fn an_accepted_stop_terminates_the_request_without_native_errors() {
        let Some(path) = std::env::var_os("CUTEAFD_NATIVE_LIB") else {
            eprintln!("skipping: CUTEAFD_NATIVE_LIB is not set");
            return;
        };
        // SAFETY: the test supplies the engine's native library, retained for
        // the lifetime of every compiler, grammar and matcher below.
        let library = unsafe { NativeLibrary::load(path).unwrap() };
        let tokenizer = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../native/tests/fixtures/xgrammar_tiny_tokenizer.json");
        let mut compiler = Compiler::with_vocab(&library, tokenizer, 8, vec![6, 0]);
        let spec = NativeConstraint(serde_json::json!({"type":"structural_tag", "format":{
            "type":"ds41_json_schema", "strict":true,
            "json_schema":{"type":"object", "properties":{"x":{"type":"string"}},
                "required":["x"], "additionalProperties":false}
        }}).to_string());
        for stop in [6u32, 0] {
            let mut state = compiler.matcher(&spec).unwrap();
            for token in [1, 2, 3, 4, 5] { state.accept(token).unwrap(); }
            assert!(!state.terminated());
            state.accept(stop).unwrap();
            assert!(state.terminated(), "stop {stop} ends the grammar");
            let only_stops = 1u32 << 6 | 1;
            assert_eq!(state.mask().unwrap().map(<[u32]>::to_vec), Some(vec![only_stops]));
            let mut proposal = vec![stop, 7, 1];
            state.truncate_proposal(&mut proposal).unwrap();
            assert_eq!(proposal, [stop], "no draft follows a terminated grammar");
            assert_eq!(state.prepare_verification_masks(&proposal).unwrap(), vec![Some(vec![only_stops])]);
            assert_eq!(state.prepare_verification_mask_row(&proposal, 0).unwrap(), Some(vec![only_stops]));
            // Another stop is harmless; anything else is a per-request error.
            state.accept(6).unwrap();
            assert!(state.accept(1).unwrap_err().to_string().contains("after the request grammar ended"));
        }
    }

    /// A grammar error in one sequence of a verify batch fails that sequence
    /// alone: its rows are still in the batch (unmasked, so row offsets hold)
    /// and its error comes back to be sent to it; the sibling keeps its masks.
    #[test]
    fn a_grammar_error_fails_one_sequence_not_the_batch() {
        let Some(path) = std::env::var_os("CUTEAFD_NATIVE_LIB") else {
            eprintln!("skipping: CUTEAFD_NATIVE_LIB is not set");
            return;
        };
        // SAFETY: the test supplies the engine's native library, retained for
        // the lifetime of every compiler, grammar and matcher below.
        let library = unsafe { NativeLibrary::load(path).unwrap() };
        let tokenizer = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../native/tests/fixtures/xgrammar_tiny_tokenizer.json");
        let mut compiler = Compiler::with_vocab(&library, tokenizer, 8, vec![6]);
        let spec = NativeConstraint(serde_json::json!({"type":"structural_tag", "format":{
            "type":"ds41_json_schema", "strict":true,
            "json_schema":{"type":"object", "properties":{"x":{"type":"string"}},
                "required":["x"], "additionalProperties":false}
        }}).to_string());
        let mut broken = compiler.matcher(&spec).unwrap();
        let mut healthy = compiler.matcher(&spec).unwrap();
        broken.accept(1).unwrap();
        healthy.accept(1).unwrap();
        let sampling = cuteafd_core::TargetSamplingParams::greedy();
        let mut batch = crate::shared::token_io::SelectBatch::default();
        // `5` after `{` is illegal: an untruncated draft, as a buggy caller would send.
        let error = batch.push_sequence_isolated(sampling, Some(&broken), &[1, 5], 10);
        assert!(error.unwrap().contains("illegal verification draft token"));
        assert_eq!(batch.rows.len(), 2, "the failed sequence keeps its rows");
        assert!(batch.rows.iter().all(|row| row.mask.is_none()));
        assert!(batch.push_sequence_isolated(sampling, Some(&healthy), &[1, 2], 20).is_none());
        assert_eq!(batch.rows.len(), 4);
        assert!(batch.rows[2..].iter().all(|row| row.mask.is_some()), "the sibling keeps its grammar");
        assert_eq!(batch.rows.iter().map(|row| row.position).collect::<Vec<_>>(), [10, 11, 20, 21]);
    }
}
