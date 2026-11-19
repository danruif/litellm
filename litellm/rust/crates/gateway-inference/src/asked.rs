//! What a call asks of the routes, told to the mounting server's rate limiter before the route
//! runs: the proxy's call type for the matched route, and the model the body names.

use serde_json::{Map, Value};

use crate::{Gateway, request};

/// The proxy's call type for a Responses call, whose output cap is its own field.
pub const RESPONSES_CALL_TYPE: &str = "aresponses";

/// The call type the proxy gives the route matched at `path`, which decides what a token estimate
/// reads from the body. A route the proxy estimates by no call type of its own is `""`.
pub fn call_type(path: &str) -> &'static str {
    if path.ends_with("/messages") {
        "anthropic_messages"
    } else if path.ends_with("/chat/completions") {
        "acompletion"
    } else if path.ends_with("/responses") {
        RESPONSES_CALL_TYPE
    } else {
        ""
    }
}

/// The model a body asks for and the model group `model_group_alias` gives that name, so a limit
/// keyed by either is found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelAsked {
    pub requested: String,
    pub group: String,
}

/// Tells the model a body asks for.
pub trait AskedModels: Send + Sync {
    /// `None` for a body that names no model.
    fn model_asked(&self, body: &Map<String, Value>) -> Option<ModelAsked>;
}

impl<H: Send + Sync> AskedModels for Gateway<H> {
    fn model_asked(&self, body: &Map<String, Value>) -> Option<ModelAsked> {
        let requested = request::model(body)
            .ok()
            .filter(|model| !model.is_empty())?;
        let registry = self.router.registry();
        let group = registry
            .alias_of(requested)
            .filter(|group| !group.is_empty())
            .unwrap_or(requested);
        Some(ModelAsked {
            requested: requested.to_owned(),
            group: group.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::messages("/v1/messages", "anthropic_messages")]
    #[case::chat("/v1/chat/completions", "acompletion")]
    #[case::a_deployment_path("/openai/deployments/{model}/chat/completions", "acompletion")]
    #[case::responses("/responses", "aresponses")]
    #[case::any_other("/v1/audio/transcriptions", "")]
    fn each_route_is_estimated_as_the_proxys_call_type(#[case] path: &str, #[case] expected: &str) {
        assert_eq!(call_type(path), expected);
    }
}
