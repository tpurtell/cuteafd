//! The session layer: conversation state shared by every front end, and the
//! typed operations on it.
//!
//! History operations (append, insert, edit, delete, truncate, fork) are real
//! here: they change the item list and report what the engine would have to
//! redo ([`EngineEffect`]). Steer, compact, splice and the KV hooks are typed
//! and validated but answer `Unsupported` until phase B wires them to the
//! engine (PLAN.md "v3 API gateway and sessions (design)").
//!
//! Responses `previous_response_id` and Realtime conversations both live on
//! this store: a stored response is an immutable snapshot node chained to its
//! parent, so continuing from any earlier response is a cheap virtual fork.
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::error::{ErrorKind, GatewayError};
use super::turn::Item;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(pub String);

impl SessionId {
    pub fn fresh(prefix: &str) -> Self { Self(format!("{prefix}_{}", uuid::Uuid::new_v4().simple())) }
}

/// A history item with the id clients address it by (Realtime `item_id`,
/// Responses output item ids).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredItem {
    pub id: String,
    pub item: Item,
}

/// Where an inserted item goes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "at", rename_all = "snake_case")]
pub enum Position {
    End,
    Start,
    /// Directly after this item (Realtime `previous_item_id`).
    After { item_id: String },
}

/// Server-side compaction policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case")]
pub enum CompactPolicy {
    /// Drop the oldest whole turns until the history fits `keep_tokens`.
    TruncateOldest { keep_tokens: u32 },
    /// Replace items before `keep_last` with a model-written summary.
    Summarize { keep_last: usize, target_tokens: u32 },
}

/// A mid-request steer: new guidance for a turn that is already running.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Steer {
    /// Append input that the running turn should see at its next safe point
    /// (between decode steps, or between prefill chunks).
    Inject { item: Item },
    /// Replace an item the running turn already consumed; implies recompute
    /// from that item.
    Replace { item_id: String, item: Item },
}

/// KV-cache treatment hooks. Ranges are item ids, resolved to token spans by
/// the engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum KvOp {
    /// Keep the KV for items up to and including `through` resident.
    Pin { through: String },
    /// Release the KV for the session (or from `from` onward).
    Evict { from: Option<String> },
    /// Name the prefix through `through` so other sessions can fork from it.
    Mark { through: String, label: String },
}

/// Every operation the session layer accepts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum SessionOp {
    Append { items: Vec<Item> },
    Insert { position: Position, item: Item },
    Edit { item_id: String, item: Item },
    Delete { item_id: String },
    /// Cut an item's content short (Realtime `conversation.item.truncate`):
    /// assistant text keeps `keep_chars`, audio keeps `audio_end_ms`.
    Truncate { item_id: String, content_index: usize, keep_chars: Option<usize>, audio_end_ms: Option<u32> },
    Steer { steer: Steer },
    /// Cancel the running turn, keeping what was produced.
    Cancel,
    Compact { policy: CompactPolicy },
    /// A new session sharing this one's history through `through` (or all).
    Fork { through: Option<String> },
    /// Replace items `[from, to]` (inclusive) with `items`.
    Splice { from: String, to: String, items: Vec<Item> },
    Kv { kv: KvOp },
}

/// What an applied operation costs the engine (phase B), so callers and logs
/// are honest about it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "effect", rename_all = "snake_case")]
pub enum EngineEffect {
    /// Nothing cached is invalidated (append: the next turn extends the
    /// cached prefix).
    PrefixKept,
    /// KV from item index `index` onward is stale; the next turn recomputes
    /// from there (positions and attention depend on every earlier token).
    RecomputeFrom { index: usize },
    /// A new session shares the parent's KV pages through `shared_items`
    /// items (prefix-cache marks and page sharing).
    SharedPrefix { session: SessionId, shared_items: usize },
    /// The running turn was signalled.
    TurnSignalled,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpOutcome {
    pub revision: u64,
    pub effect: EngineEffect,
    /// Ids of items the operation created.
    pub created: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub id: SessionId,
    pub system: Option<String>,
    pub items: Vec<StoredItem>,
    /// Bumped by every mutation; front ends use it to detect races.
    pub revision: u64,
    /// The running turn's cancel handle, if any.
    pub running: Option<tokio::sync::watch::Sender<bool>>,
}

impl Session {
    pub fn new(id: SessionId) -> Self { Self { id, system: None, items: Vec::new(), revision: 0, running: None } }

    pub fn index_of(&self, item_id: &str) -> Result<usize, GatewayError> {
        self.items.iter().position(|item| item.id == item_id)
            .ok_or_else(|| GatewayError::not_found(format!("no item '{item_id}' in session")).with_param("item_id"))
    }

    pub fn items(&self) -> Vec<Item> { self.items.iter().map(|stored| stored.item.clone()).collect() }

    /// Apply one operation. `Fork` is handled by [`SessionStore::fork`].
    pub fn apply(&mut self, op: SessionOp) -> Result<OpOutcome, GatewayError> {
        let (effect, created) = match op {
            SessionOp::Append { items } => {
                let created = items.into_iter().map(|item| self.push(item)).collect();
                (EngineEffect::PrefixKept, created)
            }
            SessionOp::Insert { position, item } => {
                let index = match position {
                    Position::End => self.items.len(),
                    Position::Start => 0,
                    Position::After { item_id } => self.index_of(&item_id)? + 1,
                };
                let id = new_item_id();
                self.items.insert(index, StoredItem { id: id.clone(), item });
                (recompute(index, self.items.len()), vec![id])
            }
            SessionOp::Edit { item_id, item } => {
                let index = self.index_of(&item_id)?;
                self.items[index].item = item;
                (EngineEffect::RecomputeFrom { index }, Vec::new())
            }
            SessionOp::Delete { item_id } => {
                let index = self.index_of(&item_id)?;
                self.items.remove(index);
                (recompute(index, self.items.len()), Vec::new())
            }
            SessionOp::Truncate { item_id, content_index, keep_chars, audio_end_ms } => {
                let index = self.index_of(&item_id)?;
                truncate_item(&mut self.items[index].item, content_index, keep_chars, audio_end_ms)?;
                (EngineEffect::RecomputeFrom { index }, Vec::new())
            }
            SessionOp::Cancel => {
                let Some(running) = &self.running else {
                    return Err(GatewayError::invalid("no response is in progress"));
                };
                let _ = running.send(true);
                (EngineEffect::TurnSignalled, Vec::new())
            }
            SessionOp::Steer { .. } => return Err(phase_b("steer")),
            SessionOp::Compact { .. } => return Err(phase_b("compact")),
            SessionOp::Splice { .. } => return Err(phase_b("splice")),
            SessionOp::Kv { .. } => return Err(phase_b("KV hooks")),
            SessionOp::Fork { .. } => return Err(GatewayError::internal("fork goes through SessionStore::fork")),
        };
        self.revision += 1;
        Ok(OpOutcome { revision: self.revision, effect, created })
    }

    fn push(&mut self, item: Item) -> String {
        let id = new_item_id();
        self.items.push(StoredItem { id: id.clone(), item });
        id
    }
}

impl Session {
    /// Realtime uses client-chosen ids and announces output ids before completion.
    /// Validate first, then retain the normal operation's effects and revision.
    pub fn apply_with_item_id(&mut self, op: SessionOp, id: String) -> Result<OpOutcome, GatewayError> {
        if id.is_empty() || id.len() > 128 || !id.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-') {
            return Err(GatewayError::invalid("item id must be 1-128 alphanumeric, underscore or hyphen characters").with_param("item.id"));
        }
        if self.items.iter().any(|item| item.id == id) {
            return Err(GatewayError::invalid("item id already exists").with_param("item.id"));
        }
        let creates_one = match &op {
            SessionOp::Insert { .. } => true,
            SessionOp::Append { items } => items.len() == 1,
            _ => false,
        };
        if !creates_one {
            return Err(GatewayError::invalid("apply_with_item_id requires insert or single-item append"));
        }
        let mut outcome = self.apply(op)?;
        let generated = outcome.created[0].clone();
        let index = self.index_of(&generated)?;
        self.items[index].id = id.clone();
        outcome.created = vec![id];
        Ok(outcome)
    }
}

fn recompute(index: usize, len: usize) -> EngineEffect {
    if index >= len { EngineEffect::PrefixKept } else { EngineEffect::RecomputeFrom { index } }
}

fn phase_b(what: &str) -> GatewayError {
    GatewayError::new(ErrorKind::Unsupported, format!("session {what} needs the engine backend (phase B)"))
}

pub fn new_item_id() -> String { format!("item_{}", uuid::Uuid::new_v4().simple()) }

fn truncate_item(item: &mut Item, content_index: usize, keep_chars: Option<usize>, audio_end_ms: Option<u32>)
    -> Result<(), GatewayError> {
    use super::turn::{Part, Role};
    let Item::Message { role: Role::Assistant, content } = item else {
        return Err(GatewayError::invalid("only assistant messages can be truncated").with_param("item_id"));
    };
    let part = content.get_mut(content_index)
        .ok_or_else(|| GatewayError::invalid("content_index out of range").with_param("content_index"))?;
    match part {
        Part::Text { text } => {
            if let Some(keep) = keep_chars {
                if let Some((byte, _)) = text.char_indices().nth(keep) { text.truncate(byte); }
            }
        }
        // Audio truncation needs the PCM timeline; the text transcript is cut
        // proportionally by the caller (Realtime front end) via keep_chars.
        Part::Audio { .. } if audio_end_ms.is_some() => {}
        _ => return Err(GatewayError::invalid("content part cannot be truncated").with_param("content_index")),
    }
    Ok(())
}

/// An immutable snapshot of a finished response's history: `items` appended
/// after the parent's. Chained so continuing from any response is O(new items).
#[derive(Debug)]
pub struct Snapshot {
    pub parent: Option<Arc<Snapshot>>,
    pub system: Option<String>,
    pub items: Vec<StoredItem>,
    /// The response id at the root of this chain (this response's own id
    /// when it has no parent): the session id usage tracking groups a
    /// `previous_response_id` conversation under.
    pub root_response_id: String,
}

impl Snapshot {
    /// The full item list, oldest first.
    pub fn history(&self) -> Vec<StoredItem> {
        let mut chain = Vec::new();
        let mut node = Some(self);
        while let Some(current) = node {
            chain.push(current);
            node = current.parent.as_deref();
        }
        chain.into_iter().rev().flat_map(|node| node.items.iter().cloned()).collect()
    }
}

/// Live sessions plus stored response snapshots, both bounded.
#[derive(Clone)]
pub struct SessionStore {
    inner: Arc<Mutex<StoreInner>>,
}

struct StoreInner {
    sessions: HashMap<SessionId, Arc<tokio::sync::Mutex<Session>>>,
    responses: HashMap<String, Arc<Snapshot>>,
    response_order: VecDeque<String>,
    max_responses: usize,
}

impl Default for SessionStore {
    fn default() -> Self { Self::new(4096) }
}

impl SessionStore {
    pub fn new(max_responses: usize) -> Self {
        Self { inner: Arc::new(Mutex::new(StoreInner {
            sessions: HashMap::new(), responses: HashMap::new(), response_order: VecDeque::new(), max_responses,
        })) }
    }

    pub fn create(&self, prefix: &str) -> Arc<tokio::sync::Mutex<Session>> {
        let id = SessionId::fresh(prefix);
        let session = Arc::new(tokio::sync::Mutex::new(Session::new(id.clone())));
        self.inner.lock().unwrap().sessions.insert(id, session.clone());
        session
    }

    pub fn get(&self, id: &SessionId) -> Option<Arc<tokio::sync::Mutex<Session>>> {
        self.inner.lock().unwrap().sessions.get(id).cloned()
    }

    pub fn close(&self, id: &SessionId) { self.inner.lock().unwrap().sessions.remove(id); }

    /// A new live session sharing `parent`'s history through `through`.
    pub async fn fork(&self, parent: &Session, through: Option<&str>) -> Result<(Arc<tokio::sync::Mutex<Session>>, OpOutcome), GatewayError> {
        let end = match through { Some(id) => parent.index_of(id)? + 1, None => parent.items.len() };
        let child = self.create("sess");
        let id = {
            let mut session = child.lock().await;
            session.system = parent.system.clone();
            session.items = parent.items[..end].to_vec();
            session.id.clone()
        };
        Ok((child, OpOutcome { revision: 0, effect: EngineEffect::SharedPrefix { session: id, shared_items: end }, created: Vec::new() }))
    }

    /// Store a finished response's snapshot under `response_id`.
    pub fn put_response(&self, response_id: String, snapshot: Arc<Snapshot>) {
        let mut inner = self.inner.lock().unwrap();
        if inner.responses.insert(response_id.clone(), snapshot).is_none() {
            inner.response_order.push_back(response_id);
        }
        while inner.response_order.len() > inner.max_responses {
            if let Some(old) = inner.response_order.pop_front() { inner.responses.remove(&old); }
        }
    }

    pub fn response(&self, response_id: &str) -> Option<Arc<Snapshot>> {
        self.inner.lock().unwrap().responses.get(response_id).cloned()
    }

    pub fn delete_response(&self, response_id: &str) -> bool {
        let mut inner = self.inner.lock().unwrap();
        inner.response_order.retain(|id| id != response_id);
        inner.responses.remove(response_id).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::turn::{Part, Role};

    fn user(text: &str) -> Item { Item::Message { role: Role::User, content: vec![Part::text(text)] } }
    fn assistant(text: &str) -> Item { Item::Message { role: Role::Assistant, content: vec![Part::text(text)] } }

    #[test]
    fn history_ops_report_engine_effects() {
        let mut session = Session::new(SessionId("s".into()));
        let out = session.apply(SessionOp::Append { items: vec![user("a"), assistant("b")] }).unwrap();
        assert_eq!(out.effect, EngineEffect::PrefixKept);
        let (first, second) = (out.created[0].clone(), out.created[1].clone());
        let out = session.apply(SessionOp::Insert { position: Position::After { item_id: first.clone() }, item: user("x") }).unwrap();
        assert_eq!(out.effect, EngineEffect::RecomputeFrom { index: 1 });
        session.apply(SessionOp::Truncate { item_id: second.clone(), content_index: 0, keep_chars: Some(0), audio_end_ms: None }).unwrap();
        assert_eq!(session.items[2].item, assistant(""));
        assert!(session.apply(SessionOp::Truncate { item_id: first.clone(), content_index: 0, keep_chars: Some(0), audio_end_ms: None }).is_err());
        assert_eq!(session.apply(SessionOp::Delete { item_id: first }).unwrap().effect, EngineEffect::RecomputeFrom { index: 0 });
        assert_eq!(session.revision, 4);
        let err = session.apply(SessionOp::Compact { policy: CompactPolicy::TruncateOldest { keep_tokens: 10 } }).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Unsupported);
        assert!(session.apply(SessionOp::Cancel).is_err());
    }

    #[tokio::test]
    async fn snapshots_chain_and_evict() {
        let store = SessionStore::new(2);
        let first = Arc::new(Snapshot { parent: None, system: None, items: vec![StoredItem { id: "1".into(), item: user("a") }], root_response_id: "r1".into() });
        let second = Arc::new(Snapshot { parent: Some(first.clone()), system: None, items: vec![StoredItem { id: "2".into(), item: assistant("b") }], root_response_id: "r1".into() });
        store.put_response("r1".into(), first);
        store.put_response("r2".into(), second);
        assert_eq!(store.response("r2").unwrap().history().len(), 2);
        store.put_response("r3".into(), Arc::new(Snapshot { parent: None, system: None, items: vec![], root_response_id: "r3".into() }));
        assert!(store.response("r1").is_none());
        assert_eq!(store.response("r2").unwrap().history().len(), 2, "children keep evicted parents alive");
        let parent = store.create("sess");
        parent.lock().await.apply(SessionOp::Append { items: vec![user("a"), assistant("b")] }).unwrap();
        let guard = parent.lock().await;
        let first_id = guard.items[0].id.clone();
        let (child, out) = store.fork(&guard, Some(&first_id)).await.unwrap();
        assert_eq!(child.lock().await.items.len(), 1);
        assert!(matches!(out.effect, EngineEffect::SharedPrefix { shared_items: 1, .. }));
    }
}
