use crate::error::Result;
use opentelemetry::{trace::TracerProvider as _, KeyValue};
use opentelemetry_sdk::Resource;
use std::env;
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

pub fn init_tracing(service_name: &str) -> Result<OtelGuard> {
    let enabled = env::var("BOOKMARKS_ENABLE_TRACING")
        .map(|v| {
            let v = v.to_lowercase();
            v == "1" || v == "true" || v == "yes"
        })
        .unwrap_or(false);

    let endpoint = env::var("PHOENIX_COLLECTOR_ENDPOINT")
        .or_else(|_| env::var("OTEL_EXPORTER_OTLP_ENDPOINT"))
        .ok();

    if !enabled || endpoint.is_none() {
        // tracing not enabled, just set up basic logging
        let subscriber = tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
            )
            .finish();

        subscriber.init();

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
    let subscriber = tracing_subscriber::registry()
        .with(telemetry)
        .with(tracing_subscriber::fmt::layer())
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()));

    subscriber.init();

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
        let guard = init_tracing("test");
        assert!(guard.is_ok());
    }
}
