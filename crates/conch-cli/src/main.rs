use clap::{Parser, Subcommand, ValueEnum};
use conch_core::{memory::MemoryKind, ConchDB, RecallKindFilter};
use std::io::{self, Read};

#[derive(Parser)]
#[command(name = "conch", about = "Biological memory for AI agents")]
struct Cli {
    #[arg(long, default_value_t = default_db_path())]
    db: String,
    #[arg(long, global = true)]
    json: bool,
    #[arg(long, global = true)]
    quiet: bool,
    #[arg(long, global = true, default_value = "default")]
    namespace: String,
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
    },
    /// Store an episode (free-text event)
    RememberEpisode { text: String },
    /// Semantic search for memories
    Recall {
        query: String,
        #[arg(long, default_value_t = 5)]
        limit: usize,
        #[arg(long, value_enum, default_value_t = RecallKindArg::All)]
        kind: RecallKindArg,
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
    /// Export memories in the selected namespace as JSON to stdout
    Export,
    /// Import memories from JSON on stdin into the selected namespace
    Import,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RecallKindArg {
    All,
    Fact,
    Episode,
}

impl From<RecallKindArg> for RecallKindFilter {
    fn from(value: RecallKindArg) -> Self {
        match value {
            RecallKindArg::All => RecallKindFilter::All,
            RecallKindArg::Fact => RecallKindFilter::Facts,
            RecallKindArg::Episode => RecallKindFilter::Episodes,
        }
    }
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
    let num: i64 = num_str
        .parse()
        .map_err(|e| format!("invalid number: {e}"))?;
    match suffix {
        "s" => Ok(num),
        "m" => Ok(num * 60),
        "h" => Ok(num * 3600),
        "d" => Ok(num * 86400),
        "w" => Ok(num * 604800),
        _ => Err(format!("unknown suffix: {suffix} (use s/m/h/d/w)")),
    }
}

fn main() {
    let cli = Cli::parse();
    if let Some(parent) = std::path::Path::new(&cli.db).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let db = match ConchDB::open(&cli.db) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
    if let Err(e) = run(&cli, &db) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn import_from_reader(
    db: &ConchDB,
    namespace: &str,
    mut reader: impl Read,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut input = String::new();
    reader.read_to_string(&mut input)?;
    let data: conch_core::ExportData = serde_json::from_str(&input)?;
    Ok(db.import_into(namespace, &data)?)
}

fn run(cli: &Cli, db: &ConchDB) -> Result<(), Box<dyn std::error::Error>> {
    match &cli.command {
        Command::Remember {
            subject,
            relation,
            object,
        } => {
            let record = db.remember_fact_in(&cli.namespace, subject, relation, object)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&record)?);
            } else if !cli.quiet {
                println!("Remembered: {subject} {relation} {object}");
            }
        }
        Command::RememberEpisode { text } => {
            let record = db.remember_episode_in(&cli.namespace, text)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&record)?);
            } else if !cli.quiet {
                println!("Remembered episode: {text}");
            }
        }
        Command::Recall { query, limit, kind } => {
            let results = db.recall_filtered_in(&cli.namespace, query, *limit, (*kind).into())?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else if !cli.quiet {
                if results.is_empty() {
                    println!("No memories found.");
                }
                for r in &results {
                    match &r.memory.kind {
                        MemoryKind::Fact(f) => println!(
                            "[fact] {} {} {} (str: {:.2}, score: {:.3})",
                            f.subject, f.relation, f.object, r.memory.strength, r.score
                        ),
                        MemoryKind::Episode(e) => println!(
                            "[episode] {} (str: {:.2}, score: {:.3})",
                            e.text, r.memory.strength, r.score
                        ),
                    }
                }
            }
        }
        Command::Forget {
            id,
            subject,
            older_than,
        } => {
            if id.is_none() && subject.is_none() && older_than.is_none() {
                return Err("specify --id, --subject, or --older-than".into());
            }
            let mut deleted = 0;
            if let Some(mid) = id {
                deleted += db.forget_by_id_in(&cli.namespace, mid)?;
            }
            if let Some(subj) = subject {
                deleted += db.forget_by_subject_in(&cli.namespace, subj)?;
            }
            if let Some(dur) = older_than {
                deleted += db.forget_older_than_in(&cli.namespace, parse_duration_secs(dur)?)?;
            }
            if cli.json {
                println!("{}", serde_json::json!({ "deleted": deleted }));
            } else if !cli.quiet {
                println!("Deleted {deleted} memories.");
            }
        }
        Command::Decay => {
            let result = db.decay_in(&cli.namespace)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else if !cli.quiet {
                println!("Decayed {} memories.", result.decayed);
            }
        }
        Command::Stats => {
            let stats = db.stats_in(&cli.namespace)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else if !cli.quiet {
                println!(
                    "Memories: {} ({} facts, {} episodes)",
                    stats.total_memories, stats.total_facts, stats.total_episodes
                );
                println!("Avg strength: {:.3}", stats.avg_strength);
            }
        }
        Command::Embed => {
            let count = db.embed_all_in(&cli.namespace)?;
            if cli.json {
                println!("{}", serde_json::json!({ "embedded": count }));
            } else if !cli.quiet {
                if count == 0 {
                    println!("All memories have embeddings.");
                } else {
                    println!("Embedded {count} memories.");
                }
            }
        }
        Command::Export => {
            let data = db.export_in(&cli.namespace)?;
            println!("{}", serde_json::to_string_pretty(&data)?);
        }
        Command::Import => {
            let count = import_from_reader(db, &cli.namespace, io::stdin())?;
            if cli.json {
                println!("{}", serde_json::json!({ "imported": count }));
            } else if !cli.quiet {
                println!("Imported {count} memories.");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use conch_core::embed::{EmbedError, Embedder, Embedding};

    struct MockEmbedder;

    impl Embedder for MockEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbedError> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }

        fn dimension(&self) -> usize {
            2
        }
    }

    fn test_db() -> ConchDB {
        ConchDB::open_in_memory_with(Box::new(MockEmbedder)).unwrap()
    }

    #[test]
    fn run_remember_and_recall_kind_respect_namespace() {
        let db = test_db();

        let remember_cli = Cli {
            db: "unused".to_string(),
            json: false,
            quiet: true,
            namespace: "team-a".to_string(),
            command: Command::Remember {
                subject: "Jared".to_string(),
                relation: "builds".to_string(),
                object: "Conch".to_string(),
            },
        };
        run(&remember_cli, &db).unwrap();

        let remember_episode_cli = Cli {
            db: "unused".to_string(),
            json: false,
            quiet: true,
            namespace: "team-a".to_string(),
            command: Command::RememberEpisode {
                text: "Daily standup happened".to_string(),
            },
        };
        run(&remember_episode_cli, &db).unwrap();

        let recall_fact_cli = Cli {
            db: "unused".to_string(),
            json: false,
            quiet: true,
            namespace: "team-a".to_string(),
            command: Command::Recall {
                query: "Jared".to_string(),
                limit: 10,
                kind: RecallKindArg::Fact,
            },
        };
        run(&recall_fact_cli, &db).unwrap();

        let recall_fact_results = db
            .recall_filtered_in("team-a", "Jared", 10, RecallKindFilter::Facts)
            .unwrap();
        assert!(!recall_fact_results.is_empty());
        assert!(recall_fact_results
            .iter()
            .all(|r| matches!(r.memory.kind, MemoryKind::Fact(_))));

        let other_namespace_results = db
            .recall_filtered_in("team-b", "Jared", 10, RecallKindFilter::Facts)
            .unwrap();
        assert!(other_namespace_results.is_empty());
    }

    #[test]
    fn export_import_namespace_roundtrip_via_command_level_helpers() {
        let source = test_db();
        source
            .remember_fact_in("team-a", "Rust", "is", "great")
            .unwrap();
        source
            .remember_episode_in("team-a", "incident retrospective")
            .unwrap();
        source
            .remember_fact_in("team-b", "Go", "is", "fast")
            .unwrap();

        let exported = source.export_in("team-a").unwrap();
        let json = serde_json::to_string(&exported).unwrap();

        let dest = test_db();
        let imported = import_from_reader(&dest, "team-c", std::io::Cursor::new(json)).unwrap();
        assert_eq!(imported, 2);

        let team_c = dest.export_in("team-c").unwrap();
        assert_eq!(team_c.memories.len(), 2);
        assert!(team_c.memories.iter().all(|m| m.namespace == "team-c"));

        let team_a = dest.export_in("team-a").unwrap();
        assert!(team_a.memories.is_empty());
    }
}
