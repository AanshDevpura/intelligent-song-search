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

## Examples and limits

For the query about accepting that a former partner has moved on, semantic search returns “Old Songs” by David Pomeranz first; the expected list starts with Adele's “Someone Like You.” For comforting someone who feels broken, it returns “Country Death Song” by Violent Femmes first, showing that shared themes do not guarantee the intended emotional meaning.

Relevance labels are hand-selected and opinion-based. Reviewing the entire dataset for each query was impractical, so labels are imperfect and incomplete. Unlabeled songs may be relevant; these scores measure agreement with the selected references rather than exhaustive search quality. Some references are broader thematic matches or cover versions. No ranking weights were tuned on this set.

[Full results and expected songs](full-corpus-evaluation.json).
