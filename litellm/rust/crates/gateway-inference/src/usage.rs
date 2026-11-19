use std::sync::Arc;

use litellm_llms_types::formats::chat_completions::ChatCompletionsUsage;

/// What the call a request was served on used.
#[derive(Clone, Debug, PartialEq)]
pub struct ServedUsage {
    /// The model the client asked for.
    pub model: String,
    /// The model group that served it.
    pub model_group: String,
    /// As Python's litellm `Usage` counts it.
    pub usage: ChatCompletionsUsage,
}

/// A request extension the mounting server sets to hear what the served call used: a complete
/// answer's usage before it goes out, a stream's when it ends. A stream that fails, breaks or is
/// dropped before its end tells nothing. Only `/v1/messages` tells it yet.
#[derive(Clone)]
pub struct UsageSink(Arc<dyn Fn(ServedUsage) + Send + Sync>);

impl UsageSink {
    pub fn new(hear: impl Fn(ServedUsage) + Send + Sync + 'static) -> Self {
        Self(Arc::new(hear))
    }

    pub fn tell(&self, served: ServedUsage) {
        (self.0)(served);
    }
}
