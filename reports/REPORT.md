# Retrieval evaluation

The three methods searched all 57,650 songs using title, artist, and lyrics. Semantic retrieval uses `all-MiniLM-L6-v2`; hybrid retrieval combines semantic and BM25 ranks with weight 0.65 and rank constant 60.

## Results

Macro averages over 25 queries, each with ten relevant songs, at K = 10:

| Method | Precision@10 | Recall@10 | MRR |
| --- | ---: | ---: | ---: |
| Keyword | 0.0080 | 0.0080 | 0.0243 |
| Semantic | 0.0240 | 0.0240 | 0.0841 |
| Hybrid | 0.0200 | 0.0200 | 0.1123 |

Precision divides relevant hits by K; recall divides hits by the number of labeled songs. MRR uses the first relevant result in the full positive-score ranking. Hybrid has the best MRR, while semantic search has the best top-ten precision and recall on these labels.

## Method tradeoffs and examples

For the query about comforting someone who feels broken, the expected list starts with Coldplay's “Fix You.”

- **Keyword (BM25):** Useful for recognizable lyric words, with scores based on word overlap. It misses paraphrases and can match words without the intended meaning. Here it ranks Frankie Valli's “What Becomes Of The Broken Hearted” first, emphasizing heartbreak rather than reassurance.
- **Semantic:** Can match themes despite different wording and has the best Precision@10 and Recall@10 here. It can still misunderstand emotional intent: its first result, “Country Death Song” by Violent Femmes, is a poor fit for comfort.
- **Hybrid:** Combines word and meaning matches and has the best MRR on this test set. Keyword overlap can also weaken the final ranking; its Precision@10 is below semantic search. For this query, it ranks Eurythmics' “Walking On Broken Glass” first, so combining methods does not resolve the mismatch.

Relevance labels are hand-selected and opinion-based. Reviewing the entire dataset for each query was impractical, so labels are imperfect and incomplete. Unlabeled songs may be relevant; these scores measure agreement with the selected references rather than exhaustive search quality. Some references are broader thematic matches or cover versions. No ranking weights were tuned on this set.

[Full results and expected songs](full-corpus-evaluation.json).
