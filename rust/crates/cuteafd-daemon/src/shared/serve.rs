//! Serve-loop parts a family scheduler composes: independent decode lanes
//! ([`lanes`]) and the two-part lane commit ([`lanes::LaneCommit`]).
pub(crate) mod lanes;

/// A host-side validation failure attributed to one request, before executing
/// its round. Device/transport errors must never be wrapped in this type.
#[derive(Debug, Clone, thiserror::Error)]
#[error("request {id}: {message}; retry the request")]
pub(crate) struct RequestFailure {
    pub id: u64,
    pub message: String,
}
/// Two independently rejected requests at the same drained lane boundary.
#[derive(Debug, thiserror::Error)]
#[error("both lanes rejected requests: {}; {}", .0[0], .0[1])]
pub(crate) struct RequestFailures(pub [RequestFailure; 2]);

impl RequestFailure {
    pub fn new(id: u64, message: impl ToString) -> Self { Self { id, message: message.to_string() } }
    pub fn retry(&self) -> cuteafd_api::openai::NativeFailure {
        cuteafd_api::openai::NativeFailure::Retry(self.to_string())
    }
    pub fn check_lane(id: u64, occupied: usize, capacity: usize) -> Result<(), Self> {
        if occupied < capacity { Ok(()) } else {
            Err(Self::new(id, "decode lane request capacity exhausted"))
        }
    }
}

/// Forward the original startup error, never its Display text. After readiness,
/// the scheduler keeps ownership of its normal completion result.
pub(crate) fn forward_initialization(result: anyhow::Result<()>, ready: bool,
    context: &'static str, send: impl FnOnce(anyhow::Error) -> Result<(), anyhow::Error>) -> anyhow::Result<()> {
    if !ready { return result; }
    let error = match result {
        Err(error) => error.context(context),
        Ok(()) => anyhow::anyhow!("{context}: worker exited before readiness"),
    };
    match send(error) { Ok(()) => Ok(()), Err(error) => Err(error) }
}

#[cfg(test)]
mod readiness_tests {
    #[test]
    fn all_startup_channels_use_typed_forwarding() {
        for source in [
            include_str!("../families/deepseek_v4/serve.rs"),
            include_str!("../families/glm5/serve.rs"), include_str!("../families/glm5_flash/serve.rs"),
            include_str!("../families/kimi_k3/serve.rs"), include_str!("../families/mimo_v2/serve.rs"),
            include_str!("../families/muse_spark/serve.rs"), include_str!("../families/qwen3_5/serve.rs"),
            include_str!("../families/qwen4/serve.rs"), include_str!("../families/step5/serve.rs"),
            include_str!("../families/hy4_flash/serve.rs"),
            include_str!("../families/hy4/serve.rs"),
            include_str!("../families/deepseek_v41/v41_native_serve.rs"),
            include_str!("../families/deepseek_v41/v41_vision_encoder.rs"),
        ] { assert!(source.contains("shared::serve::forward_initialization(")); }
        let error = super::forward_initialization(Ok(()), true, "fixture", Err).unwrap_err();
        assert!(error.to_string().contains("exited before readiness"));
        let result = super::forward_initialization(Ok(()), false, "fixture", |_| panic!("already ready"));
        assert!(result.is_ok());
    }


}
