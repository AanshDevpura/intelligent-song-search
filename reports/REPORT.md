# Full-corpus retrieval evaluation

## Data and setup

57,650 songs with full source lyrics; 643 artist names; 100,334 tokenizer-bounded chunks. All source rows were retained. Artist, title, and lyrics were indexed without generated descriptions. Missing genre/year metadata was not invented. The source revision and checksums are recorded in the corpus manifest.

Configurations: BM25 (`k1=1.2`, `b=0.75`), sentence-transformer cosine retrieval (`all-MiniLM-L6-v2`, pinned revision), and reciprocal-rank fusion (alpha 0.65, rank constant 60). Evaluation used the complete corpus for every query; there was no small candidate-only benchmark. The 25 queries now have ten assistant-curated expected songs each (250 query–song judgments). The original single-reference benchmark was inspected before expanding these labels; this is not a blind held-out study. The additional selections were not copied from search output. No ranking weights were tuned on the expanded set.

## Results

K = 10. Macro averages over the 25 queries:

| Method | Precision@10 | Recall@10 | MRR |
| --- | ---: | ---: | ---: |
| Keyword | 0.0080 | 0.0080 | 0.0243 |
| Semantic | 0.0240 | 0.0240 | 0.0841 |
| Hybrid | 0.0200 | 0.0200 | 0.1123 |

The labels are editorial expectations, not human-validated or exhaustive ground truth. Narrow queries include broader thematic matches; some entries are cover versions counted as separate corpus records. The order expresses an editorial preference, but these binary metrics treat all ten references equally. Unlabeled relevant songs count as nonrelevant. These scores measure agreement with the reference lists and cannot establish overall relevance precision. No statistical significance claim is made.

## Qualitative examples

For “A sad song about accepting that a former partner has moved on,” semantic retrieval returns “Old Songs” by David Pomeranz, “For Herself” by Reba Mcentire, “Long Gone And Moved On” by The Script. The reference list begins with “Someone Like You” by Adele.

For “Offering comfort to someone who feels broken and has lost something important,” semantic retrieval returns “Country Death Song” by Violent Femmes, “Last Christmas” by Christmas Songs, “All She Knows” by Bruno Mars. The reference list begins with “Fix You” by Coldplay.

These examples expose alternate interpretations and failures to capture narrative stance. A song about distress may share vocabulary with a request for comfort while offering a very different message. A stronger study needs pooled human judgments and separate development and held-out query sets.

The raw report contains every expected song's ID/title/artist, per-query scores, aggregate metrics, and the top ten results for each method. MRR is calculated over the full positive-score ranking. Catalogue and judgment hashes are included for reproducibility. The UI shows the expected lists before evaluation and supports direct evaluation of browser-collected labels.

## Practical limits

This is a complete, runnable retrieval/evaluation implementation, not a validated high-accuracy song-identification service. Broad descriptions have many reasonable answers; metaphors and narrative stance are difficult for a general sentence encoder. Chunk-max pooling can favor long songs, and lexical fusion can promote weak shared-word matches. Corpus artist credits and transcriptions can be incorrect. The dataset is not a complete inventory of all released music.

Useful next experiments are domain-specific embeddings and reranking, evaluated against human-reviewed labels. They are not added as dependencies or presented as completed validation in this minimal project.
