//! Checkpoint-specific V4.1 history rendering around `deepseek-recipe-encoding`.
//!
//! `deepseek-ai/DeepSeek-V4.1-Flash-1017` ships an `encoding/encoding.py` whose
//! thinking-mode history drops the reasoning of completed turns when no tools
//! are rendered: an assistant message before the last user turn renders as
//! `<｜Assistant｜></think>{content}`, and a user message that holds only tool
//! results does not start a new turn (`find_last_user_index`). The pinned
//! `deepseek-recipe-encoding 0.1.0` keeps every turn's reasoning, which is what
//! the original `DeepSeek-V4.1-Flash` checkpoint has always been served with.
//! Each checkpoint renders as its own reference does, so the 1017 rule is
//! selected from the checkpoint's identity (its repository id or `base_model`
//! chain), never by a launch flag.
use deepseek_recipe_core::conversation::Conversation;
use deepseek_recipe_core::messages::InputMessage;
use deepseek_recipe_core::tools::ToolChoice;
use deepseek_recipe_encoding::{v4::dsv41::DeepseekV41Encoding, PromptEncoding, RenderedPrompt};

/// The checkpoint whose own encoding introduced the turn-scoped rule.
pub const V41_1017_MODEL_ID: &str = "deepseek-ai/DeepSeek-V4.1-Flash-1017";

/// How a V4.1 checkpoint renders thinking-mode history.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum V41Turns {
    /// Every assistant turn keeps its reasoning (the original checkpoint).
    #[default]
    KeepReasoning,
    /// 1017: completed turns drop their reasoning unless tools are rendered;
    /// tool-result-only user turns do not start a new turn.
    DropCompleted,
}

impl V41Turns {
    /// The rule for a checkpoint whose lineage (its own id first, then every
    /// `base_model` ancestor) is `lineage`.
    pub fn for_lineage<'a>(lineage: impl IntoIterator<Item = &'a str>) -> Self {
        if lineage.into_iter().any(|id| id == V41_1017_MODEL_ID) { Self::DropCompleted } else { Self::KeepReasoning }
    }
}

// Private-use code points around a fixed tag; a request containing this exact
// string is refused rather than rendered ambiguously.
const DROPPED: &str = "\u{e000}\u{e001}cuteafd-v41-dropped-turn\u{e002}";

/// Render `conversation` for a V4.1 checkpoint with the given turn rule.
pub fn render(conversation: &Conversation, turns: V41Turns) -> Result<RenderedPrompt, String> {
    let encoding = DeepseekV41Encoding::new();
    // `encoding.py` keeps every turn's reasoning when tools are rendered.
    let tools_rendered = conversation.tool_choice != ToolChoice::None && !conversation.tools.is_empty();
    let boundary = match turns {
        V41Turns::DropCompleted if conversation.thinking_mode && !tools_rendered => last_turn_start(&conversation.messages),
        _ => None,
    };
    let Some(boundary) = boundary.filter(|&b| conversation.messages[..b].iter()
        .any(|m| matches!(m, InputMessage::Assistant { .. }))) else {
        return Ok(encoding.render_conversation(conversation));
    };
    // The crate renders an assistant at index > 0 in thinking mode as
    // `<｜Assistant｜><think>{reasoning}</think>{content}`; a dropped turn is
    // `<｜Assistant｜></think>{content}`. Mark each dropped turn's reasoning
    // with a sentinel the request cannot contain and rewrite exactly those.
    if contains_sentinel(conversation) {
        return Err("request text contains a reserved private-use sequence".into());
    }
    let mut marked = conversation.clone();
    for message in &mut marked.messages[..boundary] {
        if let InputMessage::Assistant { reasoning_content, .. } = message {
            *reasoning_content = Some(DROPPED.to_owned());
        }
    }
    let mut rendered = encoding.render_conversation(&marked);
    let marker = format!("<｜Assistant｜><think>{DROPPED}</think>");
    rendered.prompt = rendered.prompt.replace(&marker, "<｜Assistant｜></think>");
    if rendered.prompt.contains(DROPPED) {
        return Err("V4.1 history rendering left a dropped-turn marker".into());
    }
    Ok(rendered)
}

/// The index of the first message of the last run that starts a user turn:
/// a mid-conversation system message, or a run of user/tool messages (one
/// rendered user block) that holds at least one user message. A run of tool
/// results alone continues the current turn.
fn last_turn_start(messages: &[InputMessage]) -> Option<usize> {
    let mut last = None;
    let mut index = 0;
    while index < messages.len() {
        match &messages[index] {
            InputMessage::System { .. } if index > 0 => { last = Some(index); index += 1; }
            InputMessage::User { .. } | InputMessage::Tool { .. } => {
                let start = index;
                let mut user = false;
                while index < messages.len() && matches!(messages[index], InputMessage::User { .. } | InputMessage::Tool { .. }) {
                    user |= matches!(messages[index], InputMessage::User { .. });
                    index += 1;
                }
                if user { last = Some(start); }
            }
            _ => index += 1,
        }
    }
    last
}

fn contains_sentinel(conversation: &Conversation) -> bool {
    conversation.messages.iter().any(|message| match message {
        InputMessage::System { content } | InputMessage::User { content, .. }
        | InputMessage::Tool { content, .. } | InputMessage::LatestReminder { content } => content.contains(DROPPED),
        InputMessage::Assistant { content, reasoning_content, tool_calls } => content.contains(DROPPED)
            || reasoning_content.as_deref().is_some_and(|r| r.contains(DROPPED))
            || tool_calls.iter().flatten().any(|c| c.name.contains(DROPPED) || c.arguments.contains(DROPPED)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::{build, ModelEncoding, ModelProfile, NativeLimits};
    use serde_json::{json, Value};

    fn cases(name: &str) -> Vec<Value> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v41-encoding").join(name);
        let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        fixture["cases"].as_array().unwrap().clone()
    }

    fn rendered(profile: &ModelProfile, request: &Value) -> String {
        let mut body = request.clone();
        body["model"] = json!(profile.id);
        // The served path expands the adapter's image marker to the checkpoint's
        // placeholder before tokenizing, as encoding.py's IMAGE_PLACEHOLDER.
        build(profile, NativeLimits::default(), &Default::default(), body)
            .unwrap_or_else(|r| panic!("{}", r.message)).prompt().replace("<｜image｜>", "<｜deepseek_image｜>")
    }

    /// DeepSeek-V4.1-Flash-1017's own encoding.py, case for case
    /// (scripts/qualify/deepseek_v41/encoding-conformance.py writes the fixture).
    #[test]
    fn checkpoint_1017_renders_as_its_own_encoding() {
        let profile = ModelProfile::new(V41_1017_MODEL_ID, ModelEncoding::DeepseekV41)
            .with_v41_turns(V41Turns::for_lineage([V41_1017_MODEL_ID]));
        assert_eq!(profile.v41_turns, V41Turns::DropCompleted);
        let cases = cases("deepseek-v4.1-flash-1017.json");
        assert!(cases.len() >= 17);
        let failed: Vec<_> = cases.iter().filter(|case| {
            let ours = rendered(&profile, &case["request"]);
            let differs = ours != case["prompt"].as_str().unwrap();
            if differs && std::env::var_os("V41_CONFORMANCE_DIFF").is_some() {
                eprintln!("== {}\nours: {ours:?}\nref:  {:?}", case["id"], case["prompt"].as_str().unwrap());
            }
            differs
        }).map(|case| case["id"].as_str().unwrap().to_owned()).collect();
        assert!(failed.is_empty(), "differs from the checkpoint's encoding.py: {failed:?}");
    }

    /// The original checkpoint keeps the rendering it has always been served
    /// with (deepseek-recipe-encoding 0.1.0 as is), on every conformance case.
    #[test]
    fn original_checkpoint_rendering_is_unchanged() {
        use deepseek_recipe::{openai::ChatCompletionRequest, request::{ConversionOptions, ProtocolRequest}};
        let profile = ModelProfile::new(crate::openai::MODEL, ModelEncoding::DeepseekV41);
        assert_eq!(profile.v41_turns, V41Turns::KeepReasoning);
        for name in ["deepseek-v4.1-flash.json", "deepseek-v4.1-flash-1017.json"] {
            for case in cases(name) {
                let mut body = case["request"].clone();
                body["model"] = json!(crate::openai::MODEL);
                let budget = body.as_object_mut().unwrap().remove("reasoning_effort");
                let parsed: ChatCompletionRequest = serde_json::from_value(body).unwrap();
                let converted = parsed.convert(ConversionOptions::default().with_default_thinking_mode(true)).unwrap();
                let mut expected = DeepseekV41Encoding::new().render_conversation(&converted.conversation).prompt
                    .replace("<｜image｜>", "<｜deepseek_image｜>");
                if let Some(budget) = budget.and_then(|b| b.as_u64()) {
                    expected = crate::openai::v41_set_effort_budget(expected, budget as u8);
                }
                assert_eq!(rendered(&profile, &case["request"]), expected, "{name} {}", case["id"]);
            }
        }
    }

    #[test]
    fn lineage_selects_the_rule() {
        assert_eq!(V41Turns::for_lineage(["deepseek-ai/DeepSeek-V4.1-Flash"]), V41Turns::KeepReasoning);
        assert_eq!(V41Turns::for_lineage(["org/V41-1017-EXL3", V41_1017_MODEL_ID]), V41Turns::DropCompleted);
        assert_eq!(V41Turns::for_lineage(["nvidia/DeepSeek-V4.1-Flash-NVFP4", "deepseek-ai/DeepSeek-V4.1-Flash"]),
            V41Turns::KeepReasoning);
    }

    #[test]
    fn sentinel_in_a_request_is_refused_not_rewritten() {
        let profile = ModelProfile::new(V41_1017_MODEL_ID, ModelEncoding::DeepseekV41).with_v41_turns(V41Turns::DropCompleted);
        let body = json!({"model": V41_1017_MODEL_ID, "reasoning_effort": 50, "messages": [
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": format!("a{DROPPED}"), "reasoning_content": "r1"},
            {"role": "user", "content": "q2"}]});
        assert!(build(&profile, NativeLimits::default(), &Default::default(), body).is_err());
    }
}
