use crate::error::Result;
use opentelemetry::{trace::TracerProvider as _, KeyValue};
use opentelemetry_sdk::Resource;
use std::env;
use std::sync::mpsc::Sender;
use tracing::{Event, Subscriber};
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

pub struct OtelGuard {
    tracer_provider: Option<opentelemetry_sdk::trace::SdkTracerProvider>,
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.tracer_provider.take() {
            // flush remaining traces on shutdown
            if let Err(e) = provider.shutdown() {
                eprintln!("error shutting down tracer provider: {}", e);
            }
        }
    }
}

struct LogLineLayer {
    sender: Sender<String>,
}

impl LogLineLayer {
    fn new(sender: Sender<String>) -> Self {
        Self { sender }
    }
}

#[derive(Default)]
struct MessageVisitor {
    message: Option<String>,
}

impl Visit for MessageVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = Some(value.to_string());
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" && self.message.is_none() {
            self.message = Some(format!("{value:?}"));
        }
    }
}

impl<S> Layer<S> for LogLineLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);

        let level = event.metadata().level().as_str().to_lowercase();
        let message = visitor
            .message
            .unwrap_or_else(|| event.metadata().target().to_string());
        let message = message.trim_matches('"');

        let line = if message.is_empty() {
            level
        } else {
            format!("{level} {message}")
        };

        let _ = self.sender.send(line);
    }
}

fn read_bool_env(keys: &[&str], default: bool) -> bool {
    for key in keys {
        if let Ok(value) = env::var(key) {
            let value = value.to_lowercase();
            return value == "1" || value == "true" || value == "yes";
        }
    }
    default
}

pub fn init_tracing(service_name: &str, log_sender: Option<Sender<String>>) -> Result<OtelGuard> {
    let enabled = read_bool_env(&["ENABLE_TRACING"], false);

    let endpoint = env::var("PHOENIX_COLLECTOR_ENDPOINT")
        .or_else(|_| env::var("OTEL_EXPORTER_OTLP_ENDPOINT"))
        .or_else(|_| env::var("OTEL_ENDPOINT"))
        .ok();

    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into());
    if !enabled || endpoint.is_none() {
        // tracing not enabled, just set up basic logging
        if let Some(sender) = log_sender {
            let subscriber = tracing_subscriber::registry()
                .with(LogLineLayer::new(sender))
                .with(env_filter);
            subscriber.init();
        } else {
            let subscriber = tracing_subscriber::registry()
                .with(tracing_subscriber::fmt::layer().with_writer(std::io::stdout))
                .with(env_filter);
            subscriber.init();
        }

        tracing::info!("basic logging initialized (service={})", service_name);

        return Ok(OtelGuard {
            tracer_provider: None,
        });
    }

    let endpoint_url = endpoint.unwrap();

    // initialize otlp exporter with tonic (grpc)
    use opentelemetry_otlp::WithExportConfig;

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint_url)
        .build()
        .map_err(|e| crate::error::BbtError::Tracing(format!("exporter build failed: {}", e)))?;

    let resource = Resource::builder_empty()
        .with_attribute(KeyValue::new("service.name", service_name.to_string()))
        .build();

    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build();

    // bridge tracing to opentelemetry
    let telemetry = tracing_opentelemetry::layer().with_tracer(provider.tracer(service_name.to_string()));

    // combine telemetry layer with fmt layer for console output
    if let Some(sender) = log_sender {
        let subscriber = tracing_subscriber::registry()
            .with(telemetry)
            .with(LogLineLayer::new(sender))
            .with(env_filter);
        subscriber.init();
    } else {
        let subscriber = tracing_subscriber::registry()
            .with(telemetry)
            .with(tracing_subscriber::fmt::layer().with_writer(std::io::stdout))
            .with(env_filter);
        subscriber.init();
    }

    tracing::info!(
        "opentelemetry tracing initialized for {} (endpoint: {})",
        service_name,
        endpoint_url
    );

    Ok(OtelGuard {
        tracer_provider: Some(provider),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_tracing_without_endpoint() {
        // should succeed but not enable otel
        let guard = init_tracing("test", None);
        assert!(guard.is_ok());
    }
}
