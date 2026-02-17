# Search & Ranking Research for Conch Memory Engine

## Current Approach Analysis

### How Recall Works Now

The `recall()` method in `storage.rs` does:

1. **Vector search first**: If embeddings are available (fastembed all-MiniLM-L6-v2, 384-dim), scans ALL rows with embeddings, computes cosine similarity, returns anything > 0.0
2. **FTS5 fallback**: If no vector results, falls back to SQLite FTS5 (which uses BM25 internally via `rank`)
3. **Relations search**: Always runs, uses `LIKE '%query%'` with a **hardcoded relevance of 0.5**
4. **Sort**: By relevance desc, then salience desc

### What's Wrong

**Problem 1: Vector search is OR, not AND with FTS**
- If embeddings exist, FTS is completely skipped. No hybrid scoring.
- Cosine similarity on all-MiniLM-L6-v2 for short texts like "Jared prefers Rust" produces high similarity for any query mentioning "Jared" because the entity name dominates the embedding space for short strings.

**Problem 2: Brute-force scan with threshold of 0.0**
- Every fact/episode with an embedding that has *any* positive cosine similarity is returned. For a 384-dim model, almost everything will have sim > 0.0.
- No minimum threshold, no normalization.

**Problem 3: Relations always get relevance 0.5**
- `search_relations()` returns hardcoded `relevance: 0.5` regardless of how well the relation matches the query.
- A relation "Jared -[likes]-> pizza" gets the same relevance as "Jared -[works_on]-> engram" when querying "what are Jared's side projects".

**Problem 4: No query understanding**
- "What are Jared's side projects" should match on semantic concepts (projects, building, working on) not just entity name overlap.
- Short fact strings like "Jared prefers Rust" embed poorly — the entity name overwhelms semantic content.

**Problem 5: No IDF-like weighting**
- "Jared" appears in many facts. A good ranking system would downweight common terms. BM25 does this via IDF; pure cosine similarity on embeddings does not.

## What the Best Projects Do

### 1. `bm25` crate — Lightweight BM25 Search Engine
- **GitHub**: https://github.com/Michael-JB/bm25
- **Crate**: https://crates.io/crates/bm25
- **Algorithm**: Full Okapi BM25 with configurable k1, b, avgdl parameters
- **Features**: 
  - Multilingual tokenizer with stemming and stop word removal
  - Language detection
  - In-memory search engine, embedder, and scorer — three levels of abstraction
  - Parallelism for batch fitting
  - WebAssembly compatible
- **Adaptation**: Can replace FTS5 entirely or be used alongside it. The `SearchEngine` API is simple: fit to corpus, search with query, get scored results. The `Scorer` API lets us score individual query-document pairs.
- **Rust crate**: `bm25` — direct dependency, no server needed
- **Why it's ideal**: Lightweight, no external process, proper IDF weighting, handles short documents well with configurable `b` parameter (set lower for short texts)

### 2. Tantivy — Full-Text Search Engine
- **GitHub**: https://github.com/quickwit-oss/tantivy (~13k ⭐)
- **Algorithm**: BM25 (same as Lucene), with k1=1.2, b=0.75
- **Features**: Inverted index, phrase queries, incremental indexing, fast startup (<10ms)
- **Adaptation**: Could replace SQLite FTS5 entirely with a tantivy index
- **Rust crate**: `tantivy`
- **Tradeoff**: Heavy dependency (~50 transitive crates). Overkill for conch's use case (small corpus, in-process). Better for large corpora. The `bm25` crate is much lighter.

### 3. Qdrant — Vector Database
- **GitHub**: https://github.com/qdrant/qdrant (~22k ⭐)
- **Algorithm**: HNSW for approximate nearest neighbor, cosine/dot/euclidean distance
- **Key insight**: Qdrant supports **hybrid search** — combining dense vectors with sparse BM25 vectors using Reciprocal Rank Fusion (RRF)
- **Adaptation**: Way too heavy (it's a server). But their *approach* — hybrid search with RRF — is exactly what conch needs.

### 4. Mem0 — AI Memory Layer
- **GitHub**: https://github.com/mem0ai/mem0 (~25k ⭐)
- **Algorithm**: Vector similarity search (via external vector DB) + graph relations returned in parallel
- **Key insight**: Mem0 does NOT try to rank graph results against vector results. It returns them as separate `results` and `relations` arrays. This avoids the mixed-ranking problem conch has.
- **Adaptation**: Consider separating relation results from scored results, OR apply proper scoring to relations too.

### 5. Reciprocal Rank Fusion (RRF) — The Standard Hybrid Approach
- **Used by**: Weaviate, OpenSearch, Qdrant, Elasticsearch
- **Formula**: `RRF_score(d) = Σ 1 / (k + rank_i(d))` where k=60 (standard constant)
- **Why it works**: Combines rankings from different scoring methods without needing to normalize raw scores. BM25 scores and cosine similarity scores are on completely different scales — RRF sidesteps this by using ranks instead.
- **No crate needed**: ~10 lines of code to implement.

### 6. `strsim` — String Similarity
- **Crate**: https://crates.io/crates/strsim
- **Algorithms**: Levenshtein, Jaro-Winkler, etc.
- **Not useful here**: Character-level similarity isn't what we need. We need semantic/term-level scoring.

## Recommended Approach

### Strategy: Hybrid Search with RRF (BM25 + Vector + Relation Scoring)

Replace the current "vector OR fts" with "vector AND bm25, fused via RRF."

### Dependencies to Add

```toml
[dependencies]
bm25 = "1"  # Lightweight BM25 scoring — no server, no index files
```

That's it. One new dependency. The `bm25` crate gives us proper BM25 with IDF weighting, which is the biggest missing piece.

### Implementation Plan

#### Step 1: Always run both BM25 and vector search (not OR)

```rust
pub fn recall(&self, query: &str, opts: &RecallOpts) -> SqlResult<Vec<Memory>> {
    // Always run BM25 search
    let mut bm25_results = self.bm25_search(query, opts)?;
    
    // Always run vector search if available
    let mut vector_results = Vec::new();
    if let Some(embedder) = self.embedder.as_ref() {
        if let Ok(query_vecs) = embedder.embed(&[query]) {
            if let Some(qv) = query_vecs.into_iter().next() {
                vector_results.extend(self.vector_search_facts(&qv, opts)?);
                vector_results.extend(self.vector_search_episodes(&qv, opts)?);
            }
        }
    }
    
    // Fuse with RRF
    let mut fused = reciprocal_rank_fusion(vec![bm25_results, vector_results], 60.0);
    
    // Add relation results (scored separately)
    fused.extend(self.search_relations_scored(query, opts)?);
    
    fused.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal));
    
    let limit = opts.limit.unwrap_or(10);
    fused.truncate(limit);
    Ok(fused)
}
```

#### Step 2: BM25 scoring with the `bm25` crate

```rust
use bm25::{SearchEngine, SearchEngineBuilder, Language};

fn bm25_search(&self, query: &str, opts: &RecallOpts) -> SqlResult<Vec<Memory>> {
    // Load all non-forgotten content
    let docs = self.load_all_content(opts)?; // Vec<(Memory, String)>
    
    if docs.is_empty() {
        return Ok(vec![]);
    }
    
    let corpus: Vec<&str> = docs.iter().map(|(_, text)| text.as_str()).collect();
    
    let engine: SearchEngine<usize> = SearchEngineBuilder::with_corpus(
        Language::English,
        corpus.iter().enumerate().map(|(i, &text)| (i, text))
    ).build();
    
    let results = engine.search(query, opts.limit.unwrap_or(10));
    
    Ok(results.into_iter().map(|(idx, score)| {
        let mut mem = docs[idx].0.clone();
        mem.relevance = score as f64;
        mem
    }).collect())
}
```

**Performance note**: For small corpora (<10k memories), rebuilding the BM25 index per query is fine (~1ms). For larger stores, cache the `SearchEngine` and rebuild on writes.

#### Step 3: Reciprocal Rank Fusion

```rust
fn reciprocal_rank_fusion(ranked_lists: Vec<Vec<Memory>>, k: f64) -> Vec<Memory> {
    use std::collections::HashMap;
    
    let mut scores: HashMap<String, (f64, Memory)> = HashMap::new();
    
    for list in ranked_lists {
        for (rank, mem) in list.into_iter().enumerate() {
            let rrf_score = 1.0 / (k + rank as f64 + 1.0);
            scores.entry(mem.id.clone())
                .and_modify(|(s, _)| *s += rrf_score)
                .or_insert((rrf_score, mem));
        }
    }
    
    let mut results: Vec<Memory> = scores.into_values()
        .map(|(score, mut mem)| {
            mem.relevance = score;
            mem
        })
        .collect();
    
    results.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal));
    results
}
```

#### Step 4: Fix relation scoring

Instead of hardcoded 0.5, score relations using BM25 against the concatenated text:

```rust
fn search_relations_scored(&self, query: &str, opts: &RecallOpts) -> SqlResult<Vec<Memory>> {
    // Load all relations
    let relations = self.load_all_relations()?;
    if relations.is_empty() {
        return Ok(vec![]);
    }
    
    let corpus: Vec<String> = relations.iter()
        .map(|r| format!("{} {} {}", r.from, r.relation, r.to))
        .collect();
    
    let engine: SearchEngine<usize> = SearchEngineBuilder::with_corpus(
        Language::English,
        corpus.iter().enumerate().map(|(i, text)| (i, text.as_str()))
    ).build();
    
    let results = engine.search(query, opts.limit.unwrap_or(5));
    
    Ok(results.into_iter().map(|(idx, score)| {
        let r = &relations[idx];
        Memory {
            relevance: score as f64 * 0.8, // slight discount vs facts/episodes
            kind: MemoryKind::Relation,
            // ... rest of fields from r
        }
    }).collect())
}
```

#### Step 5: Raise vector search threshold

```rust
// In vector_search_facts / vector_search_episodes:
if sim > 0.3 {  // Was: > 0.0. Threshold filters noise.
    scored.push(/* ... */);
}
```

### Why This Fixes the Problem

For `conch recall "what are Jared's side projects"`:

**Before**: "Jared prefers dark mode" scores high because cosine similarity on "Jared" entity overlap gives ~0.6, and since vector search finds results, FTS (which would properly downweight "Jared" via IDF) is never consulted.

**After**:
1. BM25 properly downweights "Jared" (appears in many docs → low IDF) and upweights "projects" (rare term → high IDF)
2. Vector search still runs but with a 0.3 threshold, filtering noise
3. RRF combines both rankings — a memory must rank well in BOTH methods to score high
4. "Jared works_on engram" ranks high in BM25 (matches "projects" semantically less, but "works_on" relation type is relevant) AND vector search
5. "Jared prefers dark mode" might rank ok in vector search but poorly in BM25, so its RRF score is lower

### Alternative: Just Use BM25 (Simplest Fix)

If you want the minimal change: **swap FTS5 for the `bm25` crate and always run it, even when embeddings exist.** The `bm25` crate's IDF weighting alone would fix most of the ranking issues. You could add RRF later.

```rust
// Minimal fix in recall():
// 1. Always run BM25 (not just as fallback)
let mut results = self.bm25_search(query, opts)?;

// 2. If vector search available, boost results that also score well there
if let Some(embedder) = self.embedder.as_ref() {
    // ... compute vector scores, use them as a boost multiplier
    for mem in &mut results {
        if let Some(vec_score) = vector_scores.get(&mem.id) {
            mem.relevance *= 1.0 + vec_score; // boost, don't replace
        }
    }
}
```

## Summary

| Approach | Complexity | Effectiveness | New Dependencies |
|----------|-----------|---------------|-----------------|
| Just add `bm25` crate as primary scorer | Low | High | `bm25` |
| Hybrid BM25 + Vector with RRF | Medium | Very High | `bm25` |
| Replace with tantivy | High | High | `tantivy` (heavy) |
| Add qdrant | Very High | Very High | External server |

**Recommendation**: Add the `bm25` crate, implement hybrid search with RRF. One small dependency, ~100 lines of new code, fixes the core ranking problem.
