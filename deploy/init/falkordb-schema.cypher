// FalkorDB temporal schema — executed once at startup.
//
// Index strategy (realtime-temporal-spec.py §H):
//   - (id, valid_from): primary temporal index for time-travel queries
//   - (valid_until):    current-state index for IS NULL filtering
//   - (observed_at):    change-query index used by get_changes_since (critical perf path)
//   - edge (valid_from, valid_until): edge versioning queries
//
// Note: FalkorDB requires unnamed index syntax — "CREATE INDEX FOR ... ON ..."
// without an explicit index name (named indexes are not supported).

CREATE INDEX FOR (e:Entity) ON (e.id, e.valid_from);
CREATE INDEX FOR (e:Entity) ON (e.valid_until);
CREATE INDEX FOR (e:Entity) ON (e.observed_at);
CREATE INDEX FOR ()-[r:RELATES]-() ON (r.valid_from, r.valid_until);
