use clap::{Parser, Subcommand};
use conch_core::{ConchDB, memory::MemoryKind};

#[derive(Parser)]
#[command(name = "conch", about = "Biological memory for AI agents")]
struct Cli {
    /// Path to the database file
    #[arg(long, default_value_t = default_db_path())]
    db: String,

    /// Output as JSON
    #[arg(long, global = true)]
    json: bool,

    /// Minimal output
    #[arg(long, global = true)]
    quiet: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Store a fact (subject-relation-object triple)
    Remember {
        /// Subject of the fact
        subject: String,
        /// Relation
        relation: String,
        /// Object of the fact
        object: String,
    },
    /// Store an episode (free-text event)
    RememberEpisode {
        /// Episode text
        text: String,
    },
    /// Semantic search for memories
    Recall {
        /// Search query
        query: String,
        /// Maximum number of results
        #[arg(long, default_value_t = 5)]
        limit: usize,
    },
    /// Create an association between two entities
    Relate {
        /// First entity
        entity_a: String,
        /// Relation name
        relation: String,
        /// Second entity
        entity_b: String,
    },
    /// Delete memories
    Forget {
        /// Delete by subject
        #[arg(long)]
        subject: Option<String>,
        /// Delete memories older than duration (e.g., "30d", "1h")
        #[arg(long)]
        older_than: Option<String>,
    },
    /// Run temporal decay pass
    Decay,
    /// Show database statistics
    Stats,
    /// Generate embeddings for all memories missing them
    Embed,
    /// Export all memories and associations as JSON to stdout
    Export,
    /// Import memories and associations from JSON on stdin
    Import,
}

fn default_db_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    format!("{home}/.conch/default.db")
}

fn parse_duration_secs(s: &str) -> Result<i64, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty duration".to_string());
    }
    let (num_str, suffix) = s.split_at(s.len() - 1);
    let num: i64 = num_str.parse().map_err(|e| format!("invalid number: {e}"))?;
    match suffix {
        "s" => Ok(num),
        "m" => Ok(num * 60),
        "h" => Ok(num * 3600),
        "d" => Ok(num * 86400),
        "w" => Ok(num * 604800),
        _ => Err(format!("unknown duration suffix: {suffix} (use s/m/h/d/w)")),
    }
}

fn main() {
    let cli = Cli::parse();

    // Ensure the database directory exists
    if let Some(parent) = std::path::Path::new(&cli.db).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let db = match ConchDB::open(&cli.db) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("Error opening database: {e}");
            std::process::exit(1);
        }
    };

    let result = run_command(&cli, &db);
    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run_command(cli: &Cli, db: &ConchDB) -> Result<(), Box<dyn std::error::Error>> {
    match &cli.command {
        Command::Remember { subject, relation, object } => {
            let record = db.remember_fact(subject, relation, object)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&record)?);
            } else if !cli.quiet {
                println!("Remembered: {} {} {}", subject, relation, object);
            }
        }
        Command::RememberEpisode { text } => {
            let record = db.remember_episode(text)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&record)?);
            } else if !cli.quiet {
                println!("Remembered episode: {}", text);
            }
        }
        Command::Recall { query, limit } => {
            let results = db.recall(query, *limit)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else if !cli.quiet {
                if results.is_empty() {
                    println!("No memories found.");
                } else {
                    for result in &results {
                        let m = &result.memory;
                        match &m.kind {
                            MemoryKind::Fact(f) => {
                                println!(
                                    "[fact] {} {} {} (strength: {:.2}, score: {:.3})",
                                    f.subject, f.relation, f.object, m.strength, result.score
                                );
                            }
                            MemoryKind::Episode(e) => {
                                println!(
                                    "[episode] {} (strength: {:.2}, score: {:.3})",
                                    e.text, m.strength, result.score
                                );
                            }
                        }
                    }
                }
            }
        }
        Command::Relate { entity_a, relation, entity_b } => {
            let id = db.relate(entity_a, relation, entity_b)?;
            if cli.json {
                println!("{}", serde_json::json!({
                    "id": id,
                    "entity_a": entity_a,
                    "relation": relation,
                    "entity_b": entity_b
                }));
            } else if !cli.quiet {
                println!("Related: {} --[{}]--> {}", entity_a, relation, entity_b);
            }
        }
        Command::Forget { subject, older_than } => {
            if subject.is_none() && older_than.is_none() {
                return Err("specify --subject or --older-than".into());
            }
            let mut deleted = 0;
            if let Some(subj) = subject {
                deleted += db.forget_by_subject(subj)?;
            }
            if let Some(dur) = older_than {
                let secs = parse_duration_secs(dur)?;
                deleted += db.forget_older_than(secs)?;
            }
            if cli.json {
                println!("{}", serde_json::json!({ "deleted": deleted }));
            } else if !cli.quiet {
                println!("Deleted {deleted} memories.");
            }
        }
        Command::Decay => {
            let result = db.decay()?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else if !cli.quiet {
                println!("Decayed {} memories.", result.decayed);
            }
        }
        Command::Stats => {
            let stats = db.stats()?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else if !cli.quiet {
                println!("Total memories: {}", stats.total_memories);
                println!("  Facts:    {}", stats.total_facts);
                println!("  Episodes: {}", stats.total_episodes);
                println!("Associations: {}", stats.total_associations);
                println!("Avg strength: {:.3}", stats.avg_strength);
            }
        }
        Command::Embed => {
            let count = db.embed_all()?;
            if cli.json {
                println!("{}", serde_json::json!({ "embedded": count }));
            } else if !cli.quiet {
                if count == 0 {
                    println!("All memories already have embeddings.");
                } else {
                    println!("Generated embeddings for {count} memories.");
                }
            }
        }
        Command::Export => {
            let data = db.export()?;
            println!("{}", serde_json::to_string_pretty(&data)?);
        }
        Command::Import => {
            let input = std::io::read_to_string(std::io::stdin())?;
            let data: conch_core::ExportData = serde_json::from_str(&input)?;
            let (memories, associations) = db.import(&data)?;
            if cli.json {
                println!("{}", serde_json::json!({
                    "memories_imported": memories,
                    "associations_imported": associations
                }));
            } else if !cli.quiet {
                println!("Imported {memories} memories and {associations} associations.");
            }
        }
    }
    Ok(())
}
