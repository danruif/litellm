use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    body::{Body, to_bytes},
    extract::{MatchedPath, Request, State},
    http::{HeaderName, HeaderValue, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use litellm_callbacks::Abort;
use litellm_callbacks_parallel_request_limiter::{
    AuthMetadata, CapSettings, Descriptor, DescriptorRequest, Engine, EstimateRequest,
    LimiterState, Lookups, OutputCapFacts, ParallelRequestLimiter, PostCallFailureFacts,
    PostCallSuccessFacts, PreCallFacts, PythonServed, RATE_LIMITED, RateLimitedModel, RequestFacts,
    Settings as EngineSettings, SuccessFacts, TargetFacts, TokenRateLimitType, UsageObject,
    aggregate_only, charged_tokens, explicit_output_cap, header_map, output_cap,
    tokens_for_request, write_output_cap,
};
use litellm_config::{Config, Environment};
use litellm_gateway_auth::{AuthenticatedRequest, AuthenticationMethod, keys::KeyLimits};
use litellm_gateway_inference::{
    AskedModels, JsonObject, MAX_BODY_BYTES, ModelAsked, RESPONSES_CALL_TYPE, RateLimited, Refused,
    ServedUsage, UsageSink, call_type,
};
use litellm_redis_client::RedisClient;
use litellm_redis_store::{Remote, rate_limits::RateLimits};
use parking_lot::Mutex;
use serde_json::{Map, Value, json};

/// `RedisCache.default_ttl` when `default_redis_ttl` is unset.
pub(crate) const REDIS_DEFAULT_TTL: Duration = Duration::from_secs(60);
/// The `default_in_memory_ttl` of the proxy's `internal_usage_cache`, the limiter's memory tier.
const MEMORY_TTL: Duration = Duration::from_secs(1);
/// The api key and the user Python names a master key caller by, `LITELLM_PROXY_MASTER_KEY_ALIAS`
/// and `LITELLM_PROXY_ADMIN_NAME`.
const MASTER_KEY_ALIAS: &str = "litellm_proxy_master_key";
const PROXY_ADMIN_NAME: &str = "default_user_id";
const KEY_RPM_LIMIT: HeaderName = HeaderName::from_static("x-litellm-key-rpm-limit");
const KEY_TPM_LIMIT: HeaderName = HeaderName::from_static("x-litellm-key-tpm-limit");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settings {
    window_size: i64,
    tpm_reservation_enabled: bool,
    token_type: TokenRateLimitType,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            window_size: 60,
            tpm_reservation_enabled: true,
            token_type: TokenRateLimitType::Total,
        }
    }
}

impl Settings {
    /// The window and the reservation as the environment sets them, and
    /// `general_settings.token_rate_limit_type`.
    pub fn read(config: &Config, environment: &Environment) -> Self {
        let defaults = Self::default();
        Self {
            window_size: environment
                .litellm_rate_limit_window_size
                .unwrap_or(defaults.window_size),
            tpm_reservation_enabled: environment
                .litellm_tpm_token_reservation_enabled
                .map_or(defaults.tpm_reservation_enabled, |flag| flag.0),
            token_type: TokenRateLimitType::from_setting(&json!(
                config.general_settings.token_rate_limit_type
            )),
        }
    }
}

/// The proxy's `parallel_request_limiter_v3`, over the coordination Redis when there is one and
/// over this process's memory otherwise, as the proxy's `internal_usage_cache` is.
pub struct RateLimit {
    limiter: ParallelRequestLimiter,
    token_type: TokenRateLimitType,
    /// The inference routes, which tell the model a body asks for.
    models: Arc<dyn AskedModels>,
}

/// The tokens a call asks for, input and output, as the proxy estimates them before it runs. `Err`
/// for a body only Python's estimator reads, one that carries media.
fn estimated(
    body: &Value,
    call_type: &str,
    min_tpm_limit: Option<i64>,
    cap: &CapSettings,
) -> Result<i64, PythonServed> {
    let tokens = tokens_for_request(
        EstimateRequest {
            data: body,
            call_type,
            min_configured_tpm_limit: min_tpm_limit,
            configured_output_tokens: None,
            include_media: true,
        },
        cap,
    )?;
    Ok(i64::try_from(tokens).unwrap_or(i64::MAX))
}

/// Reads `body` as the JSON value the estimator takes, without copying it.
fn as_value<T>(body: &mut Map<String, Value>, read: impl FnOnce(&mut Value) -> T) -> T {
    let mut sent = Value::Object(std::mem::take(body));
    let found = read(&mut sent);
    if let Value::Object(sent) = sent {
        *body = sent;
    }
    found
}

/// Whether only Python's estimator tells the tokens the body sent to `path` asks for.
pub(crate) fn python_estimates(body: &mut Map<String, Value>, path: &str) -> bool {
    as_value(body, |sent| {
        estimated(sent, call_type(path), None, &CapSettings::default()).is_err()
    })
}

/// What a call reserves from a token limit, after the output cap the proxy writes into a body
/// that sets none, so one unbounded answer cannot overshoot the smallest limit.
fn reserved(
    body: &mut Value,
    call_type: &str,
    min_tpm_limit: i64,
    cap: &CapSettings,
) -> Result<i64, PythonServed> {
    let explicit = match &*body {
        Value::Object(request) => explicit_output_cap(request, call_type, cap)?,
        _ => None,
    };
    let implicit = output_cap(&OutputCapFacts {
        min_configured_limit: Some(min_tpm_limit),
        responses: call_type == RESPONSES_CALL_TYPE,
        exempt: explicit.is_some(),
        configured_output_tokens: None,
        fraction: cap.fraction,
        default_max_tokens_estimate: cap.default_max_tokens_estimate,
        responses_min_output_tokens: cap.responses_min_output_tokens,
    });
    if let Some(implicit) = implicit {
        write_output_cap(body, call_type, implicit);
    }
    estimated(body, call_type, Some(min_tpm_limit), cap)
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

/// Who Python says the caller is: its `api_key` and its `user_id`.
struct Caller {
    api_key: String,
    user_id: Option<&'static str>,
    limits: KeyLimits,
}

impl Caller {
    fn of(identity: &AuthenticatedRequest) -> Self {
        let authentication = identity.caller().authentication();
        match authentication.method {
            AuthenticationMethod::MasterKey => Self {
                api_key: MASTER_KEY_ALIAS.to_owned(),
                user_id: Some(PROXY_ADMIN_NAME),
                limits: authentication.limits.clone(),
            },
            AuthenticationMethod::VirtualKey | AuthenticationMethod::External(_) => Self {
                api_key: authentication.credential_id.clone(),
                user_id: None,
                limits: authentication.limits.clone(),
            },
        }
    }

    fn targets(&self) -> TargetFacts {
        TargetFacts {
            user_api_key_hash: Some(self.api_key.clone()),
            user_api_key_user_id: self.user_id.map(str::to_owned),
            ..TargetFacts::default()
        }
    }
}

/// The tokens Python charges a served call: those `token_type` counts, less the prompt tokens read
/// from cache.
fn charged(served: &ServedUsage, token_type: TokenRateLimitType) -> i64 {
    let usage = &served.usage;
    let count = |tokens: u64| Value::from(i64::try_from(tokens).unwrap_or(i64::MAX));
    let usage = UsageObject::Chat {
        members: Map::from_iter([
            ("prompt_tokens".to_owned(), count(usage.prompt_tokens)),
            (
                "completion_tokens".to_owned(),
                count(usage.completion_tokens),
            ),
            ("total_tokens".to_owned(), count(usage.total_tokens)),
            (
                "cached_tokens".to_owned(),
                count(usage.prompt_tokens_details.cached_tokens),
            ),
        ]),
    };
    match charged_tokens(Some(&usage), token_type) {
        0 => aggregate_only(Some(&usage)),
        charged => charged,
    }
}

impl RateLimit {
    pub fn new(
        settings: Settings,
        redis: Option<RedisClient>,
        models: Arc<dyn AskedModels>,
    ) -> Self {
        let remote = redis.map(|client| Remote {
            client,
            default_ttl: REDIS_DEFAULT_TTL,
        });
        let engine = Engine::new(
            Arc::new(RateLimits::new(remote, Some(MEMORY_TTL))),
            EngineSettings {
                window_size: settings.window_size,
                tpm_reservation_enabled: settings.tpm_reservation_enabled,
            },
            Arc::new(unix_now),
        );
        Self {
            limiter: ParallelRequestLimiter::new(engine),
            token_type: settings.token_type,
            models,
        }
    }

    /// The limits the caller's key holds, as the proxy describes them: its `api_key`, and its
    /// `model_per_key` for the model asked for.
    fn descriptors(&self, caller: &Caller, model: Option<&RateLimitedModel>) -> Vec<Descriptor> {
        let KeyLimits {
            rpm,
            tpm,
            max_parallel_requests,
            model_rpm,
            model_tpm,
        } = &caller.limits;
        let auth = Map::from_iter([
            ("api_key".to_owned(), json!(caller.api_key)),
            ("rpm_limit".to_owned(), json!(rpm)),
            ("tpm_limit".to_owned(), json!(tpm)),
            (
                "max_parallel_requests".to_owned(),
                json!(max_parallel_requests),
            ),
        ]);
        let metadata = AuthMetadata {
            key_model_rpm_limit: json!(**model_rpm),
            key_model_tpm_limit: json!(**model_tpm),
            ..AuthMetadata::default()
        };
        let lookups = Lookups {
            candidates: model
                .map(|model| model.requested.clone())
                .into_iter()
                .collect(),
            groups: model
                .map(|model| (model.requested.clone(), Some(model.group.clone())))
                .into_iter()
                .collect(),
            ..Lookups::default()
        };
        DescriptorRequest {
            auth: &auth,
            rpm_limit_type: None,
            tpm_limit_type: None,
            metadata: &metadata,
            lookups: &lookups,
            request: &RequestFacts::default(),
            requested_model: model.map(|model| model.requested.as_str()),
            window_size: self.limiter.engine().settings().window_size,
        }
        .build()
        .map(|built| built.descriptors)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|descriptor| serde_json::from_value(descriptor).ok())
        .collect()
    }

    /// What the pre-call hook is told of a call to `path`: the caller's limits and, when one counts
    /// tokens, what the body asks for. A body nobody here estimates reserves the least.
    fn pre_call_facts(
        &self,
        caller: &Caller,
        path: &str,
        body: Option<&mut Map<String, Value>>,
    ) -> PreCallFacts {
        let model = body
            .as_deref()
            .and_then(|body| self.models.model_asked(body))
            .map(|ModelAsked { requested, group }| RateLimitedModel { requested, group });
        let descriptors = self.descriptors(caller, model.as_ref());
        let min_tpm_limit = descriptors
            .iter()
            .filter_map(|descriptor| descriptor.limits().tokens_per_unit)
            .min();
        let reserves = self.limiter.engine().settings().tpm_reservation_enabled;
        let estimated_tokens = match (min_tpm_limit, body) {
            (Some(min_tpm_limit), Some(body)) if reserves => as_value(body, |sent| {
                reserved(
                    sent,
                    call_type(path),
                    min_tpm_limit,
                    &self.limiter.cap_settings(),
                )
            })
            .ok(),
            _ => None,
        };
        PreCallFacts {
            descriptors,
            requested_model: model.as_ref().map(|model| model.requested.clone()),
            estimated_tokens,
            reserved_model: model,
            ..PreCallFacts::default()
        }
    }

    /// `async_log_success_event`: gives back what the call still holds and charges the tokens it
    /// used to the caller's counters, which a TPM limit reads.
    async fn settle(&self, state: &LimiterState, targets: TargetFacts, served: ServedUsage) {
        let facts = SuccessFacts {
            total_tokens: charged(&served, self.token_type),
            targets,
            served_model: Some(RateLimitedModel {
                requested: served.model,
                group: served.model_group,
            }),
            ..SuccessFacts::default()
        };
        self.limiter.settle_success(state, facts, None).await;
    }
}

/// One admitted call: its limiter state, who it is charged to, and what the route says it used.
struct Admitted {
    rate_limit: Arc<RateLimit>,
    state: LimiterState,
    targets: TargetFacts,
    served: Arc<Mutex<Option<ServedUsage>>>,
}

impl Admitted {
    /// Settles off the request, as the proxy's success log runs after the answer: a call the route
    /// told the usage of is charged, and any other only gives its slot back.
    fn settle_later(self) {
        tokio::spawn(async move {
            let Self {
                rate_limit,
                state,
                targets,
                served,
            } = self;
            let served = served.lock().take();
            match served {
                Some(served) => rate_limit.settle(&state, targets, served).await,
                None => rate_limit.limiter.release_on_disconnect(&state).await,
            }
        });
    }
}

/// Settles a streamed call once its body is done or dropped.
struct SettledWhenDropped(Option<Admitted>);

impl Drop for SettledWhenDropped {
    fn drop(&mut self) {
        if let Some(admitted) = self.0.take() {
            admitted.settle_later();
        }
    }
}

fn streamed(response: &Response) -> bool {
    response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"))
}

/// Where Python refuses the call `error` describes: its authentication reads the limits that need
/// no body before the body, and leaves a limit on the model, and the tokens a reserving limiter
/// counts, to the handler.
fn refused_where(error: &litellm_callbacks::AbortError, reserves_tokens: bool) -> Refused {
    let field = |name: &str| error.fields.get(name).and_then(Value::as_str);
    let per_model = field("descriptor_key").is_some_and(|key| key.starts_with("model_per_"));
    let tokens = field("rate_limit_type") == Some("tokens");
    if per_model || (tokens && reserves_tokens) {
        Refused::InHandler
    } else {
        Refused::BeforeBody
    }
}

/// What a refusal says, with its headers in the order the proxy sends them. The handler's own
/// refusal names the key's limits as its other answers do.
fn refused(abort: Abort, path: &str, caller: &Caller, reserves_tokens: bool) -> Response {
    let Some(error) = abort.error.filter(|error| error.kind == RATE_LIMITED) else {
        let status = StatusCode::from_u16(abort.status).unwrap_or(StatusCode::SERVICE_UNAVAILABLE);
        return (status, abort.message).into_response();
    };
    let headers = error
        .fields
        .get("headers")
        .and_then(|headers| serde_json::from_value::<Vec<(String, String)>>(headers.clone()).ok())
        .unwrap_or_else(|| abort.headers.into_iter().collect());
    let refused = refused_where(&error, reserves_tokens);
    let response = RateLimited {
        message: abort.message,
        headers,
        refused,
    }
    .response(path);
    match refused {
        Refused::InHandler => with_key_limits(response, caller),
        Refused::BeforeBody => response,
    }
}

fn with_statuses(response: Response, state: &LimiterState) -> Response {
    let statuses = state
        .stash()
        .rate_limit_response
        .as_ref()
        .map(|response| header_map(&response.statuses))
        .unwrap_or_default();
    let (mut parts, body) = response.into_parts();
    for (name, value) in statuses {
        let value = match value {
            Value::String(text) => text,
            other => other.to_string(),
        };
        if let (Ok(name), Ok(value)) = (HeaderName::try_from(name), HeaderValue::try_from(value)) {
            parts.headers.insert(name, value);
        }
    }
    Response::from_parts(parts, body)
}

/// `x-litellm-key-rpm-limit` and `x-litellm-key-tpm-limit`, which the proxy's custom headers
/// carry on every answer to a call its pre-call hook admitted, a failed one included.
fn with_key_limits(mut response: Response, caller: &Caller) -> Response {
    for (name, limit) in [
        (KEY_RPM_LIMIT, caller.limits.rpm),
        (KEY_TPM_LIMIT, caller.limits.tpm),
    ] {
        if let Some(limit) = limit {
            response
                .headers_mut()
                .insert(name, HeaderValue::from(limit));
        }
    }
    response
}

fn multipart(request: &Request) -> bool {
    request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.to_ascii_lowercase().starts_with("multipart/"))
}

/// Reads a JSON body ahead of the route, which then takes it parsed from the extensions. A body
/// that is no JSON object is left for the route to refuse, and `None` is one too large to read.
async fn parsed(request: Request) -> Option<Request> {
    if request.extensions().get::<JsonObject>().is_some() || multipart(&request) {
        return Some(request);
    }
    let (mut parts, body) = request.into_parts();
    let sent = to_bytes(body, MAX_BODY_BYTES).await.ok()?;
    if let Ok(fields) = serde_json::from_slice::<Map<String, Value>>(&sent) {
        parts.extensions.insert(JsonObject(fields));
    }
    Some(Request::from_parts(parts, Body::from(sent)))
}

/// `parallel_request_limiter_v3` for a key's own `rpm_limit`, `tpm_limit`, `max_parallel_requests`
/// and per-model limits: the pre-call hook counts the request, reserves the tokens its body asks
/// for and takes a parallel slot, or refuses it with the proxy's `ProxyRateLimitError` in the
/// route's envelope. Only a key that holds a token or per-model limit has its body read here. A complete answer gives the slot back before it
/// goes out and carries the `x-ratelimit-*` headers on success, and a streamed one gives it back
/// when its body is done or dropped. After either, the tokens the route says the call used are
/// charged to the caller's counters off the request, as the proxy's success log does, for every
/// caller, the master key included.
///
/// The refusal takes the route's envelope, as Python's route maps `ProxyRateLimitError`, not the
/// OpenAI body the other middleware answers with. A key holding any other limit never gets here,
/// because the key lookup does not report it active. Accepted difference: a route that tells no
/// usage, every one but `/v1/messages`, charges no tokens.
pub(super) async fn limit(
    State(rate_limit): State<Arc<RateLimit>>,
    identity: AuthenticatedRequest,
    path: MatchedPath,
    request: Request,
    next: Next,
) -> Response {
    let caller = Caller::of(&identity);
    let reads_body = caller.limits.reads_body();
    let request = match reads_body {
        true => match parsed(request).await {
            Some(request) => request,
            None => return StatusCode::PAYLOAD_TOO_LARGE.into_response(),
        },
        false => request,
    };
    let (mut parts, body) = request.into_parts();
    let sent = parts
        .extensions
        .get_mut::<JsonObject>()
        .map(|JsonObject(fields)| fields);
    let facts = rate_limit.pre_call_facts(&caller, path.as_str(), sent);
    let state = LimiterState::default();
    if let Some(abort) = rate_limit.limiter.pre_call(&state, facts, None).await {
        let reserves_tokens = rate_limit
            .limiter
            .engine()
            .settings()
            .tpm_reservation_enabled;
        return refused(abort, path.as_str(), &caller, reserves_tokens);
    }
    let served = Arc::new(Mutex::new(None));
    let heard = served.clone();
    parts
        .extensions
        .insert(UsageSink::new(move |usage| *heard.lock() = Some(usage)));
    let response = with_key_limits(next.run(Request::from_parts(parts, body)).await, &caller);
    let admitted = Admitted {
        rate_limit: rate_limit.clone(),
        state,
        targets: caller.targets(),
        served,
    };
    if streamed(&response) {
        let held = SettledWhenDropped(Some(admitted));
        let (parts, body) = response.into_parts();
        let body = body.into_data_stream().map(move |chunk| {
            let _ = &held;
            chunk
        });
        return Response::from_parts(parts, Body::from_stream(body));
    }
    if !response.status().is_success() {
        rate_limit
            .limiter
            .post_call_failure(&admitted.state, PostCallFailureFacts::default())
            .await;
        return response;
    }
    rate_limit
        .limiter
        .post_call_success(&admitted.state, PostCallSuccessFacts::default())
        .await;
    let response = with_statuses(response, &admitted.state);
    admitted.settle_later();
    response
}

#[cfg(test)]
mod tests {
    use axum::{
        Extension, Json, Router,
        body::{Bytes, to_bytes},
        middleware::from_fn_with_state,
        routing::post,
    };
    use futures_util::stream;
    use litellm_core_utils::core_helpers::usage_from_parts;
    use litellm_gateway_auth::{
        Auth, KeyAuthenticator, KeyHolder, NoAdditionalPolicy, SystemClock, authenticate,
        hash_token,
        keys::{KeyHash, KeyLookup, KeyStatus, LookupFuture},
    };
    use litellm_redis_store::{
        Number,
        rate_limits::{Entry, Local, Scope},
    };
    use litellm_secret_value::SecretValue;
    use rstest::rstest;
    use tower::ServiceExt;

    use super::*;

    const MASTER_KEY: &str = "sk-master";
    const KEY: &str = "sk-limited";

    struct Holding(KeyLimits);

    impl KeyLookup for Holding {
        fn lookup<'a>(&'a self, _: &'a KeyHash) -> LookupFuture<'a> {
            Box::pin(async move {
                Ok(Some(KeyStatus::Active {
                    expires_at: None,
                    models: Arc::from([]),
                    limits: self.0.clone(),
                }))
            })
        }
    }

    /// 8 input, 14 output, 100 read from cache and 5 written to it, which Python charges as 27.
    fn used() -> ServedUsage {
        ServedUsage {
            model: "claude".into(),
            model_group: "claude-group".into(),
            usage: usage_from_parts(8, 14, 100, 5),
        }
    }

    fn app(limits: KeyLimits) -> Router {
        served_app(limits).0
    }

    fn served_app(limits: KeyLimits) -> (Router, Arc<RateLimit>) {
        mounted(
            limits,
            RateLimit::new(Settings::default(), None, Arc::new(Named)),
        )
    }

    /// The model a body names, with `sonnet` an alias of the group `claude`.
    struct Named;

    impl AskedModels for Named {
        fn model_asked(&self, body: &Map<String, Value>) -> Option<ModelAsked> {
            let requested = body.get("model")?.as_str()?;
            let group = if requested == "sonnet" {
                "claude"
            } else {
                requested
            };
            Some(ModelAsked {
                requested: requested.to_owned(),
                group: group.to_owned(),
            })
        }
    }

    /// The limiter's layer inside the caller's authentication, over routes that answer what the
    /// upstream would: `/v1/messages` tells its usage and says the `max_tokens` it was sent.
    fn mounted(limits: KeyLimits, rate_limit: RateLimit) -> (Router, Arc<RateLimit>) {
        let auth = Auth::new(
            Arc::new(KeyAuthenticator::new(
                Some(SecretValue::new(MASTER_KEY)),
                Arc::new(Holding(limits)),
            )),
            Arc::new(KeyHolder),
            Arc::new(NoAdditionalPolicy),
            Arc::new(SystemClock),
        );
        let rate_limit = Arc::new(rate_limit);
        let router =
            Router::new()
                .route(
                    "/v1/messages",
                    post(
                        |Extension(sink): Extension<UsageSink>,
                         sent: Option<Extension<JsonObject>>| async move {
                            sink.tell(used());
                            let max_tokens = sent.and_then(|Extension(JsonObject(sent))| {
                                sent.get("max_tokens").cloned()
                            });
                            Json(json!({"type": "message", "max_tokens": max_tokens}))
                        },
                    ),
                )
                .route(
                    "/v1/messages/stream",
                    post(|Extension(sink): Extension<UsageSink>| async move {
                        let events = stream::iter([Ok::<_, std::io::Error>(Bytes::from(
                            "event: message_stop\ndata: {}\n\n",
                        ))])
                        .chain(
                            stream::once(async move { sink.tell(used()) })
                                .filter_map(|()| async { None }),
                        );
                        (
                            [(header::CONTENT_TYPE, "text/event-stream")],
                            Body::from_stream(events),
                        )
                    }),
                )
                .route(
                    "/v1/chat/completions",
                    post(|| async { (StatusCode::BAD_REQUEST, "refused upstream") }),
                )
                .route_layer(from_fn_with_state(rate_limit.clone(), limit))
                .route_layer(from_fn_with_state(auth, authenticate));
        (router, rate_limit)
    }

    fn charged_to(rate_limit: &RateLimit, descriptor: &str, value: &str) -> Option<f64> {
        let entry = Entry::new(
            "tokens",
            Scope {
                descriptor: descriptor.into(),
                value: value.into(),
            },
        )
        .unwrap();
        match rate_limit
            .limiter
            .engine()
            .store()
            .get_local(&entry)
            .unwrap()
        {
            Some(Local::Number(number)) => Some(Number::as_f64(number)),
            _ => None,
        }
    }

    /// What `descriptor`'s token counter holds once it reads `expected`, or what it last held.
    async fn settled(
        rate_limit: &RateLimit,
        descriptor: &str,
        value: &str,
        expected: f64,
    ) -> Option<f64> {
        for _ in 0..100 {
            if charged_to(rate_limit, descriptor, value) == Some(expected) {
                break;
            }
            tokio::task::yield_now().await;
        }
        charged_to(rate_limit, descriptor, value)
    }

    async fn call(app: &Router, path: &str, token: &str) -> Response {
        app.clone()
            .oneshot(
                Request::post(path)
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn call_sending(app: &Router, path: &str, body: Value) -> Response {
        app.clone()
            .oneshot(
                Request::post(path)
                    .header("authorization", format!("Bearer {KEY}"))
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    /// A body asking `model` for at most `max_tokens`, whose prompt is estimated at one token.
    fn asking(model: &str, max_tokens: Option<u64>) -> Value {
        let body = json!({"model": model, "messages": [{"role": "user", "content": "hi"}]});
        match (body, max_tokens) {
            (Value::Object(mut body), Some(max_tokens)) => {
                body.insert("max_tokens".to_owned(), json!(max_tokens));
                Value::Object(body)
            }
            (body, _) => body,
        }
    }

    fn by_model(limits: &[(&str, i64)]) -> litellm_gateway_auth::keys::ModelLimits {
        Arc::new(
            limits
                .iter()
                .map(|(model, limit)| ((*model).to_owned(), *limit))
                .collect(),
        )
    }

    async fn json_body(response: Response) -> Value {
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap()
    }

    fn ratelimit_headers(response: &Response) -> Vec<(String, String)> {
        let mut found: Vec<(String, String)> = response
            .headers()
            .iter()
            .filter(|(name, _)| name.as_str().starts_with("x-ratelimit-"))
            .map(|(name, value)| (name.to_string(), value.to_str().unwrap().to_owned()))
            .collect();
        found.sort();
        found
    }

    fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[tokio::test]
    async fn a_keys_rpm_limit_counts_each_call_and_refuses_the_one_past_it() {
        let app = app(KeyLimits {
            rpm: Some(2),
            max_parallel_requests: None,
            ..KeyLimits::default()
        });

        let first = call(&app, "/v1/messages", KEY).await;
        let second = call(&app, "/v1/messages", KEY).await;
        let third = call(&app, "/v1/messages", KEY).await;

        assert_eq!(
            ratelimit_headers(&first),
            headers(&[
                ("x-ratelimit-api_key-limit-requests", "2"),
                ("x-ratelimit-api_key-remaining-requests", "1"),
            ])
        );
        assert_eq!(second.status(), StatusCode::OK);
        assert_eq!(
            ratelimit_headers(&second),
            headers(&[
                ("x-ratelimit-api_key-limit-requests", "2"),
                ("x-ratelimit-api_key-remaining-requests", "0"),
            ])
        );
        assert_eq!(third.status(), StatusCode::TOO_MANY_REQUESTS);
        let refusal: Vec<(&str, &str)> = third
            .headers()
            .iter()
            .filter(|(name, _)| *name != header::CONTENT_TYPE && *name != header::CONTENT_LENGTH)
            .map(|(name, value)| (name.as_str(), value.to_str().unwrap()))
            .collect();
        assert_eq!(
            refusal.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
            [
                "retry-after",
                "rate_limit_type",
                "rate_limited_by",
                "reset_at"
            ]
        );
        assert_eq!(
            &refusal[..3],
            [
                ("retry-after", "60"),
                ("rate_limit_type", "requests"),
                ("rate_limited_by", "api_key"),
            ]
        );
        let body = json_body(third).await;
        assert_eq!(body["error"]["type"], "rate_limit_error");
        let message = body["error"]["message"].as_str().unwrap();
        let expected = format!(
            "Rate limit exceeded for api_key: {}. Limit type: requests. Current limit: 2, Remaining: 0. Limit resets at: ",
            hash_token(KEY)
        );
        assert!(message.starts_with(&expected), "{message}");
    }

    #[tokio::test]
    async fn a_complete_answer_gives_its_parallel_slot_back_before_it_goes_out() {
        let app = app(KeyLimits {
            rpm: None,
            max_parallel_requests: Some(1),
            ..KeyLimits::default()
        });

        for _ in 0..3 {
            let response = call(&app, "/v1/messages", KEY).await;
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                ratelimit_headers(&response),
                headers(&[
                    ("x-ratelimit-api_key-limit-max_parallel_requests", "1"),
                    ("x-ratelimit-api_key-remaining-max_parallel_requests", "0"),
                ])
            );
        }
        let failed = call(&app, "/v1/chat/completions", KEY).await;
        assert_eq!(failed.status(), StatusCode::BAD_REQUEST);
        assert_eq!(ratelimit_headers(&failed), []);
        assert_eq!(
            call(&app, "/v1/messages", KEY).await.status(),
            StatusCode::OK
        );
    }

    async fn read_to_its_end(response: Response) {
        to_bytes(response.into_body(), usize::MAX).await.unwrap();
    }

    async fn dropped(response: Response) {
        drop(response);
    }

    async fn served_once_released(app: &Router) -> bool {
        for _ in 0..100 {
            if call(app, "/v1/messages", KEY).await.status() == StatusCode::OK {
                return true;
            }
            tokio::task::yield_now().await;
        }
        false
    }

    #[rstest]
    #[case::read(read_to_its_end)]
    #[case::dropped(dropped)]
    #[tokio::test]
    async fn a_streamed_answer_holds_its_parallel_slot_until_its_body_is_done<F>(
        #[case] finish: fn(Response) -> F,
    ) where
        F: Future<Output = ()>,
    {
        let app = app(KeyLimits {
            rpm: None,
            max_parallel_requests: Some(1),
            ..KeyLimits::default()
        });

        let streaming = call(&app, "/v1/messages/stream", KEY).await;
        assert_eq!(streaming.status(), StatusCode::OK);
        assert_eq!(ratelimit_headers(&streaming), []);
        let refused = call(&app, "/v1/messages", KEY).await;
        assert_eq!(refused.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            refused.headers()["rate_limit_type"],
            "max_parallel_requests"
        );

        finish(streaming).await;
        assert!(served_once_released(&app).await);
    }

    #[rstest]
    #[case::the_master_key(MASTER_KEY, KeyLimits { rpm: Some(0), max_parallel_requests: Some(0), ..KeyLimits::default() })]
    #[case::a_key_without_limits(KEY, KeyLimits::default())]
    #[tokio::test]
    async fn a_caller_without_limits_of_its_own_is_never_refused(
        #[case] token: &str,
        #[case] key_limits: KeyLimits,
    ) {
        let app = app(key_limits);

        for _ in 0..3 {
            let response = call(&app, "/v1/messages", token).await;
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(ratelimit_headers(&response), []);
        }
    }

    #[rstest]
    #[case::complete("/v1/messages", read_to_its_end)]
    #[case::streamed_to_its_end("/v1/messages/stream", read_to_its_end)]
    #[tokio::test]
    async fn a_served_call_charges_its_tokens_less_the_cached_reads_to_its_key_and_model<F>(
        #[case] path: &str,
        #[case] finish: fn(Response) -> F,
    ) where
        F: Future<Output = ()>,
    {
        let (app, rate_limit) = served_app(KeyLimits {
            rpm: Some(5),
            max_parallel_requests: None,
            ..KeyLimits::default()
        });
        let hash = hash_token(KEY);

        finish(call(&app, path, KEY).await).await;

        assert_eq!(
            settled(&rate_limit, "api_key", &hash, 27.0).await,
            Some(27.0)
        );
        assert_eq!(
            settled(
                &rate_limit,
                "model_per_key",
                &format!("{hash}:claude-group"),
                27.0
            )
            .await,
            Some(27.0)
        );
    }

    #[tokio::test]
    async fn the_master_key_is_charged_as_the_proxy_names_it() {
        let (app, rate_limit) = served_app(KeyLimits::default());

        call(&app, "/v1/messages", MASTER_KEY).await;

        for (descriptor, value) in [
            ("api_key", "litellm_proxy_master_key"),
            ("user", "default_user_id"),
            ("model_per_key", "litellm_proxy_master_key:claude-group"),
        ] {
            assert_eq!(
                settled(&rate_limit, descriptor, value, 27.0).await,
                Some(27.0),
                "{descriptor}"
            );
        }
    }

    #[rstest]
    #[case::a_dropped_stream("/v1/messages/stream")]
    #[case::a_failed_call("/v1/chat/completions")]
    #[tokio::test]
    async fn a_call_that_tells_no_usage_charges_nothing(#[case] path: &str) {
        let (app, rate_limit) = served_app(KeyLimits {
            rpm: Some(5),
            max_parallel_requests: Some(1),
            ..KeyLimits::default()
        });

        drop(call(&app, path, KEY).await);

        assert!(served_once_released(&app).await);
        assert_eq!(
            settled(
                &rate_limit,
                "model_per_key",
                &format!("{}:claude-group", hash_token(KEY)),
                27.0
            )
            .await,
            Some(27.0),
            "the call that told no usage was charged too"
        );
    }

    #[rstest]
    #[case::complete("/v1/messages", StatusCode::OK)]
    #[case::streamed("/v1/messages/stream", StatusCode::OK)]
    #[case::failed("/v1/chat/completions", StatusCode::BAD_REQUEST)]
    #[tokio::test]
    async fn every_answer_to_an_admitted_call_names_the_keys_rpm_limit(
        #[case] path: &str,
        #[case] status: StatusCode,
    ) {
        let app = app(KeyLimits {
            rpm: Some(1),
            max_parallel_requests: None,
            ..KeyLimits::default()
        });

        let admitted = call(&app, path, KEY).await;
        let refused = call(&app, path, KEY).await;

        assert_eq!(admitted.status(), status);
        assert_eq!(admitted.headers()["x-litellm-key-rpm-limit"], "1");
        assert_eq!(refused.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(!refused.headers().contains_key("x-litellm-key-rpm-limit"));
        let unlimited = call(&self::app(KeyLimits::default()), path, KEY).await;
        assert!(!unlimited.headers().contains_key("x-litellm-key-rpm-limit"));
    }

    #[tokio::test]
    async fn a_keys_tpm_limit_reserves_what_the_body_asks_for_and_settles_what_was_used() {
        let (app, rate_limit) = served_app(KeyLimits {
            tpm: Some(60),
            ..KeyLimits::default()
        });
        let hash = hash_token(KEY);

        let admitted = call_sending(&app, "/v1/messages", asking("claude", Some(50))).await;

        assert_eq!(admitted.status(), StatusCode::OK);
        assert_eq!(admitted.headers()["x-litellm-key-tpm-limit"], "60");
        assert_eq!(
            ratelimit_headers(&admitted),
            headers(&[
                ("x-ratelimit-api_key-limit-tokens", "60"),
                ("x-ratelimit-api_key-remaining-tokens", "9"),
            ])
        );
        assert_eq!(
            settled(&rate_limit, "api_key", &hash, 27.0).await,
            Some(27.0),
            "the 51 reserved were not settled to the 27 used"
        );
        let over = call_sending(&app, "/v1/messages", asking("claude", Some(50))).await;
        assert_eq!(over.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(over.headers()["x-litellm-key-tpm-limit"], "60");
        assert!(!over.headers().contains_key("retry-after"));
        let error = json_body(over).await["error"].clone();
        assert_eq!(error["type"], "rate_limit_error");
        assert_eq!(error.get("litellm_error_type"), None);
        assert!(
            error["message"]
                .as_str()
                .unwrap()
                .starts_with(&format!("Rate limit exceeded for api_key: {hash}. Limit type: tokens. Current limit: 60, Remaining: 33."))
        );
        assert_eq!(
            call_sending(&app, "/v1/messages", asking("claude", Some(20)))
                .await
                .status(),
            StatusCode::OK,
            "the refused call kept its reservation"
        );
    }

    #[tokio::test]
    async fn a_failed_call_gives_its_reserved_tokens_back() {
        let app = app(KeyLimits {
            tpm: Some(60),
            ..KeyLimits::default()
        });

        let failed = call_sending(&app, "/v1/chat/completions", asking("claude", Some(50))).await;
        let next = call_sending(&app, "/v1/messages", asking("claude", Some(50))).await;

        assert_eq!(failed.status(), StatusCode::BAD_REQUEST);
        assert_eq!(next.status(), StatusCode::OK);
    }

    #[rstest]
    #[case::none_sent_under_a_token_limit(Some(100), None, json!(25))]
    #[case::one_sent(Some(100), Some(50), json!(50))]
    #[case::none_sent_without_a_token_limit(None, None, json!(null))]
    #[tokio::test]
    async fn a_body_that_sets_no_output_cap_is_capped_at_a_quarter_of_the_token_limit(
        #[case] tpm: Option<i64>,
        #[case] sent: Option<u64>,
        #[case] reached: Value,
    ) {
        let app = app(KeyLimits {
            tpm,
            rpm: Some(5),
            ..KeyLimits::default()
        });

        let response = call_sending(&app, "/v1/messages", asking("claude", sent)).await;

        assert_eq!(json_body(response).await["max_tokens"], reached);
    }

    #[tokio::test]
    async fn a_body_parsed_before_the_layer_is_the_one_estimated_and_capped() {
        let app = app(KeyLimits {
            tpm: Some(100),
            ..KeyLimits::default()
        });
        let Value::Object(parsed) = asking("claude", None) else {
            unreachable!()
        };

        let response = app
            .oneshot(
                Request::post("/v1/messages")
                    .header("authorization", format!("Bearer {KEY}"))
                    .extension(JsonObject(parsed))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(json_body(response).await["max_tokens"], 25);
    }

    #[tokio::test]
    async fn a_body_only_python_estimates_reserves_the_least() {
        let app = app(KeyLimits {
            tpm: Some(60),
            ..KeyLimits::default()
        });
        let body = json!({"model": "claude", "max_tokens": 1000, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}},
        ]}]});

        assert_eq!(
            call_sending(&app, "/v1/messages", body).await.status(),
            StatusCode::OK
        );
    }

    #[rstest]
    #[case::requests(
        KeyLimits { model_rpm: by_model(&[("claude", 1)]), ..KeyLimits::default() },
        "requests"
    )]
    #[case::tokens(
        KeyLimits { model_tpm: by_model(&[("claude", 60)]), ..KeyLimits::default() },
        "tokens"
    )]
    #[tokio::test]
    async fn a_keys_model_limit_counts_only_the_model_it_names(
        #[case] limits: KeyLimits,
        #[case] limit_type: &str,
    ) {
        let (app, rate_limit) = served_app(limits);

        let admitted = call_sending(&app, "/v1/messages", asking("claude", Some(50))).await;
        assert_eq!(admitted.status(), StatusCode::OK);
        settled(
            &rate_limit,
            "model_per_key",
            &format!("{}:claude", hash_token(KEY)),
            27.0,
        )
        .await;
        let refused = call_sending(&app, "/v1/messages", asking("claude", Some(50))).await;
        let other = call_sending(&app, "/v1/messages", asking("gpt", Some(50))).await;

        assert_eq!(refused.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(!refused.headers().contains_key("retry-after"));
        let error = json_body(refused).await["error"].clone();
        assert_eq!(error.get("litellm_error_type"), None);
        assert!(error["message"].as_str().unwrap().starts_with(&format!(
            "Rate limit exceeded for model_per_key: {}:claude. Limit type: {limit_type}.",
            hash_token(KEY)
        )));
        assert_eq!(other.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn a_model_limit_is_counted_under_the_models_group() {
        let app = app(KeyLimits {
            model_rpm: by_model(&[("claude", 1)]),
            ..KeyLimits::default()
        });

        let by_alias = call_sending(&app, "/v1/messages", asking("sonnet", Some(50))).await;
        let by_group = call_sending(&app, "/v1/messages", asking("claude", Some(50))).await;

        assert_eq!(by_alias.status(), StatusCode::OK);
        assert_eq!(by_group.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[rstest]
    #[case::output("output", 14.0)]
    #[case::input("input", 13.0)]
    #[case::anything_else("tokens", 27.0)]
    #[tokio::test]
    async fn a_served_call_is_charged_the_tokens_the_configured_type_counts(
        #[case] token_rate_limit_type: &str,
        #[case] charged: f64,
    ) {
        let config = Config::from_yaml(&format!(
            "general_settings:\n  token_rate_limit_type: {token_rate_limit_type}\n"
        ))
        .unwrap();
        let settings = Settings::read(&config, &Environment::default());
        let (app, rate_limit) = mounted(
            KeyLimits::default(),
            RateLimit::new(settings, None, Arc::new(Named)),
        );

        call(&app, "/v1/messages", KEY).await;

        assert_eq!(
            settled(&rate_limit, "api_key", &hash_token(KEY), charged).await,
            Some(charged)
        );
    }

    #[rstest]
    fn the_window_and_the_reservation_come_from_the_environment() {
        let environment = Environment {
            litellm_rate_limit_window_size: Some(30),
            litellm_tpm_token_reservation_enabled: Some(litellm_config::Flag(false)),
            ..Environment::default()
        };

        assert_eq!(
            Settings::read(&Config::default(), &environment),
            Settings {
                window_size: 30,
                tpm_reservation_enabled: false,
                token_type: TokenRateLimitType::Total,
            }
        );
        assert_eq!(
            Settings::read(&Config::default(), &Environment::default()),
            Settings::default()
        );
    }
}
