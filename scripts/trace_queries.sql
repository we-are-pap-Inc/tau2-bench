-- DuckDB queries for StageGate JSONL traces.
-- Replace the glob path with your artifact location.

CREATE OR REPLACE VIEW trace_events AS
SELECT * FROM read_json_auto('artifacts/**/trace_events.jsonl');

-- Event counts by condition/domain.
SELECT condition, domain, event_type, count(*) AS n
FROM trace_events
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;

-- Validator blocks by reason.
SELECT condition, domain, validator_reason, count(*) AS n
FROM trace_events
WHERE event_type = 'validator_block'
GROUP BY 1, 2, 3
ORDER BY n DESC;

-- Stage calls per benchmark task.
SELECT condition, domain, benchmark_task_id, count(*) AS stage_calls
FROM trace_events
WHERE event_type = 'advance_stage_call'
GROUP BY 1, 2, 3
ORDER BY stage_calls DESC;

-- Ledger updates per benchmark task.
SELECT condition, domain, benchmark_task_id, count(*) AS ledger_updates
FROM trace_events
WHERE event_type = 'ledger_update'
GROUP BY 1, 2, 3
ORDER BY ledger_updates DESC;
