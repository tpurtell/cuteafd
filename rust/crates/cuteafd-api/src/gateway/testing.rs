//! A scripted backend for front-end tests: each `start` pops the next script
//! of events and records the turn it was given.
use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;

use super::backend::{Backend, BackendCapabilities, ModelInfo, TurnStream};
use super::error::GatewayError;
use super::turn::{TurnEvent, TurnRequest};

#[derive(Default, Clone)]
pub(crate) struct Scripted {
    pub scripts: Arc<Mutex<Vec<Vec<Result<TurnEvent, GatewayError>>>>>,
    pub seen: Arc<Mutex<Vec<TurnRequest>>>,
    pub capabilities: BackendCapabilities,
}

impl Scripted {
    pub fn new(scripts: Vec<Vec<TurnEvent>>) -> Self {
        let scripts = scripts.into_iter().rev().map(|s| s.into_iter().map(Ok).collect()).collect();
        Self { scripts: Arc::new(Mutex::new(scripts)), ..Self::default() }
    }
    pub fn turns(&self) -> Vec<TurnRequest> { self.seen.lock().unwrap().clone() }
}

impl Backend for Scripted {
    fn name(&self) -> &str { "scripted" }
    fn capabilities(&self) -> BackendCapabilities { self.capabilities }
    fn models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo { id: "served-model".into(), context_tokens: Some(131072), max_output_tokens: Some(32768), owned_by: "cuteafd".into() }]
    }
    fn start(&self, turn: TurnRequest) -> BoxFuture<'static, Result<TurnStream, GatewayError>> {
        self.seen.lock().unwrap().push(turn);
        let script = self.scripts.lock().unwrap().pop();
        Box::pin(async move {
            let script = script.ok_or_else(|| GatewayError::internal("scripted backend has no more turns"))?;
            Ok(Box::pin(futures::stream::iter(script)) as TurnStream)
        })
    }
    fn count_tokens(&self, turn: TurnRequest) -> BoxFuture<'static, Result<u32, GatewayError>> {
        Box::pin(async move { Ok(super::backend::estimate_tokens(&turn)) })
    }
}
