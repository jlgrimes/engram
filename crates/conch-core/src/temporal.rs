use chrono::{DateTime, Duration, FixedOffset, Utc};

use crate::memory::TemporalMetadata;

/// Minimal temporal extractor for relative intents like "in 2 days".
pub fn extract_temporal_metadata(
    text: &str,
    anchor: DateTime<FixedOffset>,
) -> Option<TemporalMetadata> {
    let lower = text.to_ascii_lowercase();
    let units = [
        ("day", 1_i64),
        ("days", 1_i64),
        ("week", 7_i64),
        ("weeks", 7_i64),
        ("hour", 0_i64),
        ("hours", 0_i64),
    ];

    let tokens: Vec<&str> = lower.split_whitespace().collect();
    for i in 0..tokens.len().saturating_sub(1) {
        if let Ok(n) = tokens[i].parse::<i64>() {
            let unit = tokens[i + 1].trim_matches(|c: char| !c.is_alphabetic());
            if let Some((_, day_mult)) = units.iter().find(|(u, _)| *u == unit) {
                let resolved = if *day_mult == 0 {
                    anchor + Duration::hours(n.max(1))
                } else {
                    anchor + Duration::days((n * day_mult).max(1))
                };
                let utterance = anchor.with_timezone(&Utc);
                return Some(TemporalMetadata {
                    raw_text: text.to_string(),
                    utterance_at: utterance,
                    timezone: anchor.offset().to_string(),
                    resolved_at: resolved.with_timezone(&Utc),
                    resolved_end_at: None,
                    temporal_kind: "deadline".to_string(),
                    status: "pending".to_string(),
                });
            }
        }
    }

    None
}
