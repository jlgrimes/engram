use clap::{Parser, Subcommand};
use conch_core::{ConchDB, memory::{MemoryKind, RememberResult}};
use std::io;

#[derive(Parser)]
#[command(name = "conch", about = "Biological memory for AI agents")]
struct Cli {
    #[arg(long, default_value_t = default_db_path())]
    db: String,
    #[arg(long, global = true)]
    json: bool,
    #[arg(long, global = true)]
    quiet: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Store a fact (subject-relation-object triple)
    Remember {
        subject: String,
        relation: String,
        object: String,
        /// Comma-separated tags (e.g. "preference,technical")
        #[arg(long)]
        tags: Option<String>,
        /// Source of this memory (e.g. "cli", "discord", "cron"). Defaults to "cli".
        #[arg(long)]
        source: Option<String>,
        /// Session identifier for grouping related memories
        #[arg(long)]
        session_id: Option<String>,
        /// Channel or context within the source
        #[arg(long)]
        channel: Option<String>,
    },
    /// Store an episode (free-text event)
    RememberEpisode {
        text: String,
        /// Comma-separated tags (e.g. "decision,project")
        #[arg(long)]
        tags: Option<String>,
        /// Source of this memory (e.g. "cli", "discord", "cron"). Defaults to "cli".
        #[arg(long)]
        source: Option<String>,
        /// Session identifier for grouping related memories
        #[arg(long)]
        session_id: Option<String>,
        /// Channel or context within the source
        #[arg(long)]
        channel: Option<String>,
    },
    /// Semantic search for memories
    Recall {
        query: String,
        #[arg(long, default_value_t = 5)]
        limit: usize,
        /// Filter results to only memories with this tag
        #[arg(long)]
        tag: Option<String>,
    },
    /// Delete memories
    Forget {
        #[arg(long)]
        id: Option<String>,
        #[arg(long)]
        subject: Option<String>,
        #[arg(long)]
        older_than: Option<String>,
    },
    /// Run temporal decay pass
    Decay,
    /// Show database statistics
    Stats,
    /// Generate embeddings for all memories missing them
    Embed,
    /// Export all memories as JSON to stdout
    Export,
    /// Import memories from JSON on stdin
    Import,
    /// Consolidate related memories (sleep-like memory consolidation)
    Consolidate {
        /// Preview what would be consolidated without making changes
        #[arg(long)]
        dry_run: bool,
    },
    /// Show or set importance scores for memories
    Importance {
        /// Set importance for a specific memory ID
        #[arg(long)]
        id: Option<i64>,
        /// Importance value to set (0.0-1.0)
        #[arg(long)]
        set: Option<f64>,
        /// Recompute importance scores for all memories
        #[arg(long)]
        score: bool,
    },
}

fn default_db_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    format!("{home}/.conch/default.db")
}

fn parse_duration_secs(s: &str) -> Result<i64, String> {
    let s = s.trim();
    if s.is_empty() { return Err("empty duration".to_string()); }
    let (num_str, suffix) = s.split_at(s.len() - 1);
    let num: i64 = num_str.parse().map_err(|e| format!("invalid number: {e}"))?;
    match suffix {
        "s" => Ok(num),
        "m" => Ok(num * 60),
        "h" => Ok(num * 3600),
        "d" => Ok(num * 86400),
        "w" => Ok(num * 604800),
        _ => Err(format!("unknown suffix: {suffix} (use s/m/h/d/w)")),
    }
}

fn parse_tags(tags: Option<&str>) -> Vec<String> {
    match tags {
        Some(s) if !s.is_empty() => s.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect(),
        _ => vec![],
    }
}

fn main() {
    let cli = Cli::parse();
    if let Some(parent) = std::path::Path::new(&cli.db).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let db = match ConchDB::open(&cli.db) {
        Ok(db) => db,
        Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
    };
    if let Err(e) = run(&cli, &db) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run(cli: &Cli, db: &ConchDB) -> Result<(), Box<dyn std::error::Error>> {
    match &cli.command {
        Command::Remember { subject, relation, object, tags, source, session_id, channel } => {
            let tag_list = parse_tags(tags.as_deref());
            let src = Some(source.as_deref().unwrap_or("cli"));
            let result = db.remember_fact_dedup_full(subject, relation, object, &tag_list, src, session_id.as_deref(), channel.as_deref())?;
            if cli.json { println!("{}", serde_json::to_string_pretty(&result)?); }
            else if !cli.quiet {
                let tag_suffix = if tag_list.is_empty() { String::new() } else { format!(" [{}]", tag_list.join(", ")) };
                match &result {
                    RememberResult::Created(_) => println!("Remembered: {subject} {relation} {object}{tag_suffix}"),
                    RememberResult::Updated(mem) => {
                        println!("Updated existing fact (id: {}): {subject} {relation} {object}{tag_suffix}", mem.id);
                    }
                    RememberResult::Duplicate { existing, similarity } => {
                        println!("Duplicate detected (similarity: {similarity:.3}), reinforced existing memory (id: {}, strength: {:.2})", existing.id, existing.strength);
                    }
                }
            }
        }
        Command::RememberEpisode { text, tags, source, session_id, channel } => {
            let tag_list = parse_tags(tags.as_deref());
            let src = Some(source.as_deref().unwrap_or("cli"));
            let result = db.remember_episode_dedup_full(text, &tag_list, src, session_id.as_deref(), channel.as_deref())?;
            if cli.json { println!("{}", serde_json::to_string_pretty(&result)?); }
            else if !cli.quiet {
                let tag_suffix = if tag_list.is_empty() { String::new() } else { format!(" [{}]", tag_list.join(", ")) };
                match &result {
                    RememberResult::Created(_) => println!("Remembered episode: {text}{tag_suffix}"),
                    RememberResult::Updated(mem) => println!("Updated existing memory (id: {}){tag_suffix}", mem.id),
                    RememberResult::Duplicate { existing, similarity } => {
                        println!("Duplicate detected (similarity: {similarity:.3}), reinforced existing memory (id: {}, strength: {:.2})", existing.id, existing.strength);
                    }
                }
            }
        }
        Command::Recall { query, limit, tag } => {
            let results = db.recall_with_tag(query, *limit, tag.as_deref())?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else if !cli.quiet {
                if results.is_empty() { println!("No memories found."); }
                for r in &results {
                    let tag_suffix = if r.memory.tags.is_empty() { String::new() } else { format!(" [{}]", r.memory.tags.join(", ")) };
                    match &r.memory.kind {
                        MemoryKind::Fact(f) => println!("[fact] {} {} {} (str: {:.2}, score: {:.3}){tag_suffix}", f.subject, f.relation, f.object, r.memory.strength, r.score),
                        MemoryKind::Episode(e) => println!("[episode] {} (str: {:.2}, score: {:.3}){tag_suffix}", e.text, r.memory.strength, r.score),
                    }
                }
            }
        }
        Command::Forget { id, subject, older_than } => {
            if id.is_none() && subject.is_none() && older_than.is_none() {
                return Err("specify --id, --subject, or --older-than".into());
            }
            let mut deleted = 0;
            if let Some(mid) = id { deleted += db.forget_by_id(mid)?; }
            if let Some(subj) = subject { deleted += db.forget_by_subject(subj)?; }
            if let Some(dur) = older_than { deleted += db.forget_older_than(parse_duration_secs(dur)?)?; }
            if cli.json { println!("{}", serde_json::json!({ "deleted": deleted })); }
            else if !cli.quiet { println!("Deleted {deleted} memories."); }
        }
        Command::Decay => {
            let result = db.decay()?;
            if cli.json { println!("{}", serde_json::to_string_pretty(&result)?); }
            else if !cli.quiet { println!("Decayed {} memories.", result.decayed); }
        }
        Command::Stats => {
            let stats = db.stats()?;
            if cli.json { println!("{}", serde_json::to_string_pretty(&stats)?); }
            else if !cli.quiet {
                println!("Memories: {} ({} facts, {} episodes)", stats.total_memories, stats.total_facts, stats.total_episodes);
                println!("Avg strength: {:.3}", stats.avg_strength);
            }
        }
        Command::Embed => {
            let count = db.embed_all()?;
            if cli.json { println!("{}", serde_json::json!({ "embedded": count })); }
            else if !cli.quiet {
                if count == 0 { println!("All memories have embeddings."); }
                else { println!("Embedded {count} memories."); }
            }
        }
        Command::Export => {
            let data = db.export()?;
            println!("{}", serde_json::to_string_pretty(&data)?);
        }
        Command::Import => {
            let input = std::io::read_to_string(io::stdin())?;
            let data: conch_core::ExportData = serde_json::from_str(&input)?;
            let count = db.import(&data)?;
            if cli.json { println!("{}", serde_json::json!({ "imported": count })); }
            else if !cli.quiet { println!("Imported {count} memories."); }
        }
        Command::Consolidate { dry_run } => {
            if *dry_run {
                let clusters = db.consolidate_clusters()?;
                if cli.json {
                    println!("{}", serde_json::to_string_pretty(&clusters)?);
                } else if !cli.quiet {
                    if clusters.is_empty() {
                        println!("No clusters found for consolidation.");
                    } else {
                        println!("Found {} cluster(s) (dry run — no changes made):", clusters.len());
                        for (i, cluster) in clusters.iter().enumerate() {
                            println!("\n  Cluster {}:", i + 1);
                            println!("    Canonical: [id:{}] {} (str: {:.2})", cluster.canonical.id, cluster.canonical.text_for_embedding(), cluster.canonical.strength);
                            for dup in &cluster.duplicates {
                                println!("    Duplicate: [id:{}] {} (str: {:.2})", dup.id, dup.text_for_embedding(), dup.strength);
                            }
                        }
                    }
                }
            } else {
                let result = db.consolidate(false)?;
                if cli.json { println!("{}", serde_json::to_string_pretty(&result)?); }
                else if !cli.quiet {
                    println!("Consolidated {} cluster(s): {} memories archived, {} canonical memories boosted.", result.clusters, result.archived, result.boosted);
                }
            }
        }
        Command::Importance { id, set, score } => {
            if let (Some(mem_id), Some(value)) = (id, set) {
                db.set_importance(*mem_id, *value)?;
                if cli.json { println!("{}", serde_json::json!({ "id": mem_id, "importance": value })); }
                else if !cli.quiet { println!("Set importance for memory {mem_id} to {value:.2}"); }
            } else if *score {
                let count = db.score_importance()?;
                if cli.json { println!("{}", serde_json::json!({ "scored": count })); }
                else if !cli.quiet { println!("Recomputed importance for {count} memories."); }
            } else {
                let infos = db.list_importance()?;
                if cli.json {
                    println!("{}", serde_json::to_string_pretty(&infos)?);
                } else if !cli.quiet {
                    if infos.is_empty() { println!("No memories found."); }
                    for info in &infos {
                        println!("[id:{}] importance: {:.3} | accesses: {} | tags: {} | source: {} | {}",
                            info.id, info.importance, info.access_count, info.tag_count,
                            if info.has_source { "yes" } else { "no" },
                            truncate(&info.content, 60));
                    }
                }
            }
        }
    }
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() } else { format!("{}...", &s[..max]) }
}
