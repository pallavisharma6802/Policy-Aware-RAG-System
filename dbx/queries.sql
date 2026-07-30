-- Example Databricks SQL after sync

-- Latency / refusal overview (last 7 days of query events)
SELECT
  date(ts) AS day,
  count(*) AS queries,
  avg(latency_ms) AS avg_latency_ms,
  approx_percentile(latency_ms, 0.5) AS p50_ms,
  approx_percentile(latency_ms, 0.95) AS p95_ms,
  avg(CASE WHEN refused THEN 1 ELSE 0 END) AS refusal_rate
FROM main.policy_rag.query_events
GROUP BY 1
ORDER BY 1 DESC;

-- Latest eval runs
SELECT ran_at, mode, model, num_items, hit_at_3, answer_accuracy, refusal_f1, latency_p50_ms
FROM main.policy_rag.eval_runs
ORDER BY ran_at DESC
LIMIT 20;

-- Chunk inventory
SELECT policy_source, count(*) AS chunks
FROM main.policy_rag.policy_chunks
GROUP BY 1;
