# Retrieval Evaluation

How well the hybrid search finds the right moment in the audio, measured against 25 labeled queries, and where it is expected to fail.

**Current result (experiment 1):** hybrid recall@5 **0.811** and recall@10 **0.917**, up from 0.727 and 0.826 in the baseline; all success criteria pass. See [Experiment 1](#experiment-1-bge-query-instruction--keyword-weight-05) and the caveat there.

## Method

- **Query set:** [data/queries.json](../data/queries.json), 25 queries across all 7 files: 6 keyword (names, rare terms), 3 exact phrase, 10 semantic (paraphrases that avoid the speaker's own words), 3 speaker ("what did the host say about…"), 3 negative (topics no file covers).
- **Labels:** each query is labeled with the file and time range where the answer is said. Labels use segment timestamps, not chunk IDs, so they stay valid when chunking changes.
- **Match rule:** a result counts only if it is from the labeled file and overlaps the labeled range by at least 0.5 s (or half the result, for very short turns). There is no "within N seconds" tolerance, because turns sit back to back and it would accept the other speaker's adjacent turn.
- **Retrievers compared:** every query is scored for keyword-only (Postgres full-text), semantic-only (pgvector) and hybrid (Reciprocal Rank Fusion), so the gain from fusion is measured, not assumed.
- **Metrics:** recall@1/3/5/10 (share of labeled answer locations found in the top k), MRR, recall by query type, speaker accuracy, and top semantic score for real vs negative queries.

Reproduce:

```bash
python -m src.evaluation.evaluate --out output/eval/<experiment>.json
python -m src.evaluation.report output/eval/<experiment>.json --compare output/eval/baseline.json
```

## Baseline

Saved as [output/eval/baseline.json](../output/eval/baseline.json) (retrieval code at commit `1065d45`).

| Setting | Value |
|---|---|
| Transcription / diarization | WhisperX `small` (int8, CPU) + pyannote `speaker-diarization-community-1`, 2 speakers |
| Chunk | one speaker turn, split at segment boundaries after 80 words |
| Embedded text | the turn plus one neighbouring turn on each side (turns under 8 words: the turn alone) |
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim), HNSW cosine index |
| Keyword search | Postgres `english` full-text search, any query word (OR), `ts_rank` without length normalization; quoted text = exact phrase |
| Semantic search | cosine similarity; turns under 4 words excluded |
| Fusion | Reciprocal Rank Fusion, k = 60, top 20 candidates from each retriever |

#### Overall

| Method | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| Keyword (text) | 0.424 | 0.500 | 0.500 | 0.523 | 0.505 |
| Semantic | 0.265 | 0.462 | 0.629 | 0.848 | 0.524 |
| Hybrid (RRF) | 0.402 | 0.659 | 0.727 | 0.826 | 0.611 |

#### Recall by query type

| Type (n) | Key R@1 | Key R@3 | Key R@5 | Key R@10 | Sem R@1 | Sem R@3 | Sem R@5 | Sem R@10 | Hyb R@1 | Hyb R@3 | Hyb R@5 | Hyb R@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| keyword (6) | 0.81 | 1.00 | 1.00 | 1.00 | 0.22 | 0.56 | 0.69 | 1.00 | 0.81 | 1.00 | 1.00 | 1.00 |
| phrase (3) | 0.83 | 1.00 | 1.00 | 1.00 | 0.50 | 0.67 | 1.00 | 1.00 | 0.83 | 1.00 | 1.00 | 1.00 |
| semantic (10) | 0.00 | 0.00 | 0.00 | 0.05 | 0.20 | 0.38 | 0.47 | 0.67 | 0.05 | 0.35 | 0.50 | 0.62 |
| speaker (3) | 0.67 | 0.67 | 0.67 | 0.67 | 0.33 | 0.33 | 0.67 | 1.00 | 0.33 | 0.67 | 0.67 | 1.00 |
| **all (22)** | 0.42 | 0.50 | 0.50 | 0.52 | 0.27 | 0.46 | 0.63 | 0.85 | 0.40 | 0.66 | 0.73 | 0.83 |

#### Success criteria

| Criterion | Target | Result | Status |
|---|---|---|---|
| hybrid_recall@5 | ≥ 0.80 | 0.727 | FAIL |
| hybrid_recall@10 | ≥ 0.90 | 0.826 | FAIL |
| speaker_accuracy | ≥ 0.80 | 1.000 | PASS |
| hybrid recall@5 ≥ each retriever alone | holds | 0.727 vs 0.500 / 0.629 | PASS |

#### Per query

| id | type | query | Key R@1 | Key R@3 | Key R@5 | Key R@10 | Sem R@1 | Sem R@3 | Sem R@5 | Sem R@10 | Hyb R@1 | Hyb R@3 | Hyb R@5 | Hyb R@10 | Hybrid rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q01 | keyword | Barbara Wilde | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1 |
| q02 | semantic | believing others are upset with you when the… | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | >10 |
| q03 | speaker | what did the host say about people who look … | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 2 |
| q04 | keyword | Getty and Shutterstock | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1 |
| q05 | phrase | "infinite ambition" | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1 |
| q06 | semantic | will smarter computers leave lots of people … | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 1 |
| q07 | keyword | Jillian Armstrong | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1 |
| q08 | semantic | feeling isolated among male co-stars in a mo… | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 | 0.50 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 2 |
| q09 | semantic | being timid about bossing performers around … | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 2 |
| q10 | keyword | Barry Marshall | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1 |
| q11 | semantic | what do gut microbes make that nourishes the… | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 4 |
| q12 | speaker | what did the host say about stress and stoma… | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1 |
| q13 | phrase | "euglycemic clamp" | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1 |
| q14 | semantic | a short experiment that made fit young adult… | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 | 0.00 | 0.50 | 1.00 | 1.00 | 3 |
| q15 | semantic | why shots given under the skin don't mimic t… | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | >10 |
| q16 | keyword | NanoChat | 0.50 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.50 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 1 |
| q17 | semantic | why automated programming helpers struggled … | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | >10 |
| q18 | semantic | has software been gradually taking over code… | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | 0.67 | 0.67 | 0.00 | 0.00 | 0.00 | 0.67 | 6 |
| q19 | keyword | oDesk | 0.33 | 1.00 | 1.00 | 1.00 | 0.33 | 0.33 | 0.67 | 1.00 | 0.33 | 1.00 | 1.00 | 1.00 | 1 |
| q20 | phrase | "passive data collection" | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 1 |
| q21 | speaker | what did the host say about cleaning service… | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 10 |
| q22 | semantic | why do startups that connect buyers and sell… | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 | 0.00 | 0.00 | 0.00 | 0.50 | 10 |

### What the baseline shows

- **Exact words work.** Every keyword and phrase query is found in the hybrid top 3, and names are at rank 1. Keyword search carries these; semantic search rarely ranks names first.
- **Paraphrases are the weak spot.** Semantic queries reach only 0.50 recall@5 in hybrid. Semantic search finds 0.67 of them by rank 10, but late.
- **Fusion helps at the top.** Hybrid has the best R@3, R@5 and MRR, and lifts paraphrases that semantic search ranked low (q06, q08, q09, q14 all reach the top 3–5). It beats each retriever alone at recall@5 (0.727 vs 0.500 / 0.629).
- **Fusion also hurts some queries.** Because any query word can match, keyword search returns chunks that share only common words ("face", "other"). RRF weights those like a strong semantic match, so on q02, q18, q21 and q22 hybrid ranks the answer lower than semantic search alone. This is why semantic edges hybrid at R@10 (0.848 vs 0.826).
- **Speaker accuracy (1.00) is close to automatic.** Each chunk is one speaker turn and the labels were written from the same diarization, so it checks consistency, not whether diarization is right. The real speaker test is the speaker queries' R@5 of 0.67.
- **Negative queries score lower but overlap real ones.** Mean top semantic score is 0.56 for negatives vs 0.70 for real queries, but name queries score as low as 0.51. A semantic-score threshold alone would reject valid name searches.

## Where retrieval is expected to fail

Each item is backed by the baseline results or a direct test.

1. **Paraphrases with no words in common with the speaker.** The small embedding model misses meaning-only matches, especially for technical content. In the baseline q15 ("why shots given under the skin don't mimic the body's own delivery route", about portal-vein insulin delivery) and q17 ("why automated programming helpers struggled with his unusual project") were in no retriever's top 10. Experiment 1 recovers q15 (rank 3); q17 is still missed.
2. **Queries that share only common words with the wrong passages.** OR-ed keyword search matches any word, and fusion trusts that match as much as a semantic one. The right answer gets pushed down (q02: semantic finds it by rank 10, hybrid did not). Halving the keyword weight in experiment 1 reduces this but q02 is still only at rank 10.
3. **"Host" and "guest".** Speaker labels are anonymous (`SPEAKER_00`, `SPEAKER_01`) and search has no notion of roles, so "what did the host say about X" is answered from content only. It can return the guest's turn on the same topic first (q21: host's turn at rank 10, unchanged in experiment 1).
4. **Asking about a speaker by name.** "What did Karpathy say…" cannot filter to Karpathy's turns: nothing links names to `SPEAKER_00/01`. It only matches if the name is spoken in the audio.
5. **Names the transcription got wrong.** Whisper misspells names and full-text search needs the exact word. `DeFronzo`, `James Caan` and `Gerald Reaven` each return 0 keyword hits; they were transcribed as "DeFranco", "Jimmy Khan" and "Jerry Reven", which each return 1. Semantic search does not recover names either. The trigram index in the schema could, but search does not use it yet.
6. **Wrong speaker labels.** Retrieval inherits diarization errors. Overlapping speech and short interjections are also often attributed to the wrong person.
7. **Very short turns.** Turns under 4 words ("Look.", "What?") are excluded from semantic search (8 of 240 chunks). They can still be found by keyword, but an answer like "Yes" or "Exactly" is only findable through the question before it.
8. **Queries made only of common words.** The `english` configuration drops stopwords and stems words, so "we are the" finds nothing by keyword, and an exact quote made of common words cannot be matched literally.
9. **Off-topic questions still get answers.** There is no relevance cutoff: "who won the 2022 football world cup" returns 5 results (top semantic score 0.509). See the threshold note above for why a simple cutoff is not enough.
10. **Answers spread across many turns, or across files.** Neighbouring turns share most of their embedded text, so the top results cluster around one passage. Spots that are far apart in time, or in other files, fall below the top k (q18 and q22 have 2–3 labeled locations and only some are found). No cross-file queries were labeled, so this is untested.
11. **Sentences cut at a speaker change.** When the speaker changes mid-sentence the sentence is split across two chunks ("…And you" / "think, Ralph, that…"), so neither chunk has the full sentence.

**Limits of the evaluation itself:** 22 labeled queries is a small sample (one query is ~4.5 points of recall). The queries were drafted by the coding agent from the transcripts and then reviewed, so they may be easier than real user questions. Only English was tested.

## Experiment log

Each change is evaluated with the same queries and compared to the baseline. Results are saved as `output/eval/<experiment>.json`.

| # | Change | Hybrid R@5 | Hybrid R@10 | MRR | Kept? | Notes |
|---|---|---|---|---|---|---|
| 0 | Baseline | 0.727 | 0.826 | 0.611 | – | see above |
| 1a | Keyword weight 0.5 or 0.7 in RRF | 0.758 | 0.826 | 0.616 | – | fixes near-misses q18, q22 only partly |
| 1b | Keyword match must cover ≥ 2 query words | 0.758 | 0.826 | 0.678 | – | best MRR without the query instruction |
| 1c | BGE query instruction, plain RRF | 0.788 | 0.811 | 0.600 | – | semantic-only R@10 0.848 → 0.917 |
| **1** | **BGE query instruction + keyword weight 0.5** | **0.811** | **0.917** | **0.619** | **yes** | only variant passing both recall targets; R@1 drops 0.402 → 0.356 |
| 1d | BGE query instruction + keyword ≥ 2 words | 0.742 | 0.871 | 0.669 | no | |
| 1e | BGE query instruction + keyword ≥ half the query words | 0.697 | 0.917 | 0.716 | no | best MRR, worst R@5 |

## Experiment 1: BGE query instruction + keyword weight 0.5

Saved as [output/eval/exp1_query_instruction_kw_weight.json](../output/eval/exp1_query_instruction_kw_weight.json).

**Changes** (no re-indexing needed; chunk embeddings are unchanged):

- **Query instruction.** Queries are embedded as `"Represent this sentence for searching relevant passages: " + query`, the usage BGE documents for short queries searching longer passages. Chunks are embedded without it. This is the model's intended usage, not a tuned value.
- **Keyword weight 0.5 in fusion.** Each keyword rank contributes `0.5 / (60 + rank)` instead of `1 / (60 + rank)`. Because keyword search matches any query word, its list includes chunks sharing only common words; at full weight they pushed real answers out of the top 5. Weights 0.5 and 0.7 give identical results, so the outcome does not hinge on the exact value.

{tables}
**Reading the change:**

- **Paraphrase queries gain the most:** semantic-type hybrid recall@5 0.50 → 0.68. q15 goes from missed to rank 3, q22 from 10 to 4, q18 from 6 to 3.
- **Rank 1 gets slightly worse:** hybrid R@1 0.402 → 0.356, and q03, q09, q12 each drop one place. With the instruction, semantic search is better at getting the answer into the top few and worse at putting it first.
- **Names and phrases are unaffected:** still 1.00 at R@3.

**Caveat:** the kept variant was chosen from about a dozen scored on these same 22 queries, so the improvement is likely somewhat optimistic. It should be confirmed on queries written after this change (a held-out set).
