//! Preserve the fatal scheduler cause for admitted and queued clients.
use cuteafd_api::openai::{InferenceChunk, NativeFailure, NativeRequest};
use tokio::sync::mpsc;

type Event = Result<InferenceChunk, NativeFailure>;

/// Weak senders do not keep finished or cancelled request streams open. The
/// bounded scheduler owns the strong senders until its failure is reported.
#[derive(Default)]
pub(super) struct FailureRecipients {
    events: Vec<mpsc::WeakUnboundedSender<Event>>,
}

impl FailureRecipients {
    pub(super) fn watch(&mut self, events: &mpsc::UnboundedSender<Event>) {
        self.events.retain(|sender| sender.strong_count() != 0);
        // A KV admission retry watches the same request again.
        if !self.events.iter().any(|sender| sender.upgrade().is_some_and(|s| s.same_channel(events))) {
            self.events.push(events.downgrade());
        }
    }

    /// Finish may have been sent while the request still owns turn-cache data.
    /// That client must not receive another terminal event on a later failure.
    pub(super) fn finished(&mut self, events: &mpsc::UnboundedSender<Event>) {
        self.events.retain(|sender| sender.upgrade().is_some_and(|s| !s.same_channel(events)));
    }

    pub(super) fn fail_and_close(&self, error: &anyhow::Error, receive: &mut mpsc::Receiver<NativeRequest>) {
        // Close new queue reservations before draining jobs already queued.
        receive.close();
        let cause = format!("{error:#}");
        for events in self.events.iter().filter_map(mpsc::WeakUnboundedSender::upgrade) {
            let _ = events.send(Err(NativeFailure::Worker(cause.clone())));
        }
        while let Ok(job) = receive.try_recv() {
            let _ = job.events.send(Err(NativeFailure::Worker(cause.clone())));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(events: mpsc::UnboundedSender<Event>) -> NativeRequest {
        NativeRequest {
            prompt: "queued".into(), constraint: None, images: vec![], media: Vec::new(), audio: Vec::new(), max_tokens: 1,
            sampling: Default::default(), stop_token_ids: vec![], events, usage: None, probe: None,
        }
    }

    #[test]
    fn fatal_cause_reaches_admitted_waiting_and_queued_requests_once() {
        let mut recipients = FailureRecipients::default();
        let mut senders = Vec::new();
        let mut receivers = Vec::new();
        // Active decode, pending prefill and deferred KV admission.
        for _ in 0..3 {
            let (tx, rx) = mpsc::unbounded_channel();
            recipients.watch(&tx);
            recipients.watch(&tx);
            senders.push(tx);
            receivers.push(rx);
        }
        let (queue, mut receive) = mpsc::channel(2);
        let (tx, rx) = mpsc::unbounded_channel();
        assert!(queue.try_send(job(tx)).is_ok());
        receivers.push(rx);
        let error = anyhow::anyhow!("real backend failure").context("target submission");
        recipients.fail_and_close(&error, &mut receive);
        assert!(queue.is_closed());
        assert!(receive.is_empty());
        for mut events in receivers {
            assert!(matches!(events.try_recv(), Ok(Err(NativeFailure::Worker(cause)))
                if cause == "target submission: real backend failure"));
            assert!(events.try_recv().is_err(), "duplicate terminal event");
        }
    }

    #[test]
    fn a_finished_response_does_not_receive_a_later_scheduler_error() {
        let mut recipients = FailureRecipients::default();
        let (tx, mut rx) = mpsc::unbounded_channel();
        recipients.watch(&tx);
        assert!(tx.send(Ok(InferenceChunk::Finish {
            finish_reason: cuteafd_api::openai::InferenceFinishReason::Stop,
        })).is_ok());
        recipients.finished(&tx);
        let (_queue, mut receive) = mpsc::channel(1);
        recipients.fail_and_close(&anyhow::anyhow!("later failure"), &mut receive);
        assert!(matches!(rx.try_recv(), Ok(Ok(InferenceChunk::Finish { .. }))));
        assert!(matches!(rx.try_recv(), Err(mpsc::error::TryRecvError::Empty)));
    }

    #[test]
    fn finished_requests_do_not_keep_the_stream_or_registry_alive() {
        let mut recipients = FailureRecipients::default();
        let (tx, mut rx) = mpsc::unbounded_channel();
        recipients.watch(&tx);
        drop(tx);
        assert!(matches!(rx.try_recv(), Err(mpsc::error::TryRecvError::Disconnected)));
        let (next, mut events) = mpsc::unbounded_channel();
        recipients.watch(&next);
        assert_eq!(recipients.events.len(), 1);
        let (_queue, mut receive) = mpsc::channel(1);
        recipients.fail_and_close(&anyhow::anyhow!("later failure"), &mut receive);
        assert!(matches!(events.try_recv(), Ok(Err(NativeFailure::Worker(cause))) if cause == "later failure"));
        assert!(matches!(rx.try_recv(), Err(mpsc::error::TryRecvError::Disconnected)));
    }
}
