# PostgreSQL Senior DBA / Data Engineer — Interview Preparation Guide

---

**Note:** The PDF placeholder `[PASTE PDF CONTENT HERE]` was left blank, so the interview plot below is constructed from the technical scope you described (PostgreSQL, replication, connectivity, DBLink, CDC, search_path, BARMan). Replace or extend the plot section once you have the actual PDF content.

---

## Interview Plot Summary

The interview is for a **Senior PostgreSQL Database Administrator / Data Engineer** role at a mid-to-large enterprise that operates a multi-tenant, high-availability PostgreSQL environment. The company runs multiple PostgreSQL clusters across data centers, uses logical and physical replication for DR and reporting, and relies on cross-database queries, change tracking pipelines, and automated backup strategies. The candidate is expected to demonstrate deep knowledge of PostgreSQL internals, replication architectures, backup and recovery tooling (BARMan), CDC patterns, cross-database connectivity (DBLink), schema search behavior (search_path), and overall database lifecycle management. The panel typically includes a Principal DBA, a Data Platform Engineer, and a DevOps/Infrastructure lead.

---

## Mock Interview Questions and Answers

**1. Walk me through your experience managing high-availability PostgreSQL clusters in production.**

I have designed and maintained streaming replication clusters with hot-standby replicas, automated failover using Patroni, and monitoring via pg_stat_replication. I have handled failover drills, switchover procedures, and replica lag tuning in environments serving thousands of transactions per second.

---

**2. What is the difference between physical replication and logical replication in PostgreSQL, and when would you choose one over the other?**

Physical replication copies the raw WAL byte-stream and replicates the entire cluster state, making it ideal for DR standbys. Logical replication decodes WAL into row-level changes and replicates selected tables, making it ideal for cross-version upgrades, selective sync, or feeding downstream systems.

---

**3. Explain the role of the WAL (Write-Ahead Log) in PostgreSQL and how it supports replication and recovery.**

WAL records every data modification before it is applied to heap files, guaranteeing durability and enabling crash recovery. It is also the backbone of both streaming and logical replication, as standbys or subscribers consume WAL segments to replay changes in order.

---

**4. How does PostgreSQL handle MVCC (Multi-Version Concurrency Control), and what are common performance pitfalls?**

PostgreSQL stores multiple row versions in the heap to serve concurrent readers without blocking writers. The main pitfall is table bloat from dead tuples; this is managed by autovacuum, but high-churn tables may need aggressive vacuum tuning or manual intervention to reclaim space.

---

**5. What strategies do you use to diagnose and resolve slow queries in PostgreSQL?**

I start with `EXPLAIN (ANALYZE, BUFFERS)` to identify sequential scans, bad estimates, or join strategy mismatches, then check `pg_stat_statements` for aggregate runtime patterns. Common fixes include adding partial or composite indexes, updating statistics with `ANALYZE`, and rewriting correlated subqueries as CTEs or lateral joins.

---

**6. Describe how you would set up a logical replication slot and what monitoring is critical around it.**

You create a slot with `pg_create_logical_replication_slot('slot_name', 'pgoutput')` and attach a publication/subscription. Critical monitoring is `pg_replication_slots.confirmed_flush_lsn` versus current WAL position; an inactive or lagging slot will cause unbounded WAL retention and can fill the disk if not caught early.

---

**7. What is the significance of `wal_level`, `max_wal_senders`, and `max_replication_slots` parameters?**

`wal_level = logical` is required to enable logical decoding on top of replication; `replica` suffices for streaming replication only. `max_wal_senders` caps the number of concurrent replication connections, and `max_replication_slots` must be at least equal to the number of active slots; undersizing either will silently block new subscribers.

---

**8. How would you perform a zero-downtime major version upgrade of PostgreSQL?**

The standard approach uses `pg_upgrade --check` first, then the actual upgrade with minimal downtime by keeping the old cluster alive as a physical standby is not possible across major versions — so the preferred production path is logical replication between old and new versions, cutting over application connections once the new cluster is caught up.

---

**9. Explain table partitioning in PostgreSQL — types, use cases, and common mistakes.**

PostgreSQL supports range, list, and hash partitioning via declarative syntax. Range partitioning is ideal for time-series data with partition pruning on date ranges. A common mistake is forgetting to create indexes on each partition individually (prior to PG 11) or using a partition key that doesn't match the most common WHERE clause, defeating pruning.

---

**10. What tools and approaches do you use for PostgreSQL connection pooling, and why is it important?**

PgBouncer is the standard lightweight pooler, supporting transaction-mode pooling to multiplex many application connections over few server connections. Without pooling, each new connection spawns a backend process with ~5–10 MB overhead; at hundreds of concurrent connections this degrades performance and can exhaust `max_connections`.

---

**11. How do you approach backup validation? It is not enough to just take backups.**

After every backup I run a restore-verify job to a separate standby or ephemeral instance, apply WAL to a recent point, and confirm table counts and critical checksums match production. BARMan's `barman check` and `barman verify-backup` commands automate much of this as part of the workflow.

---

**12. What is the difference between `pg_dump` and a continuous archiving (PITR) backup, and when do you use each?**

`pg_dump` produces a logical, schema-level snapshot useful for cross-version migrations, selective restores, or development copies. PITR via WAL archiving records every change and allows restoring to any point in time, which is mandatory for production DR where RPO must be measured in seconds, not hours.

---

**13. How does PostgreSQL's autovacuum work, and how do you tune it for a write-heavy OLTP table?**

Autovacuum wakes up periodically, scans tables that have exceeded dead-tuple thresholds (`autovacuum_vacuum_scale_factor`, `autovacuum_vacuum_threshold`), and removes dead versions plus updates the visibility map. For write-heavy tables I lower per-table `autovacuum_vacuum_scale_factor` to 0.01 or even 0, set an absolute `autovacuum_vacuum_threshold`, and increase `autovacuum_vacuum_cost_delay` to reduce I/O impact.

---

**14. Describe your experience with Patroni or similar HA orchestration tools.**

Patroni manages the leader election, fencing, and configuration management for a PostgreSQL HA cluster using a distributed consensus store (etcd, Consul, or ZooKeeper). I have configured Patroni with watchdog fencing to prevent split-brain, tuned `ttl` and `loop_wait` for sub-30-second failover, and scripted pre-promote and on-role-change hooks for DNS cutover.

---

**15. What is the purpose of `pg_hba.conf`, and what are common security misconfigurations?**

`pg_hba.conf` controls client authentication — it maps connection source, database, user, and authentication method (md5, scram-sha-256, cert, peer). Common misconfigurations include using `trust` for local connections in multi-tenant environments, overly broad CIDR ranges like `0.0.0.0/0`, or using md5 instead of the stronger scram-sha-256 method available since PG 10.

---

**16. How do you monitor replication lag, and what is your alerting threshold?**

I monitor `pg_stat_replication.write_lag`, `flush_lag`, and `replay_lag` on the primary, plus query `pg_last_wal_receive_lsn()` vs `pg_last_wal_replay_lsn()` on standbys. Alert thresholds are workload-dependent, but I typically page at >30 seconds of replay lag and set a critical threshold at >2 minutes, combined with disk space monitoring for WAL accumulation.

---

**17. Explain the concept of a tablespace in PostgreSQL and a practical use case.**

A tablespace maps a logical name to a filesystem directory, allowing specific tables or indexes to be placed on different storage volumes. A practical use case is placing a large, read-heavy index tablespace on NVMe SSD while keeping archival partition data on cheaper spinning disks, balancing cost and I/O performance.

---

**18. How would you handle a scenario where `pg_stat_activity` shows many queries waiting on `Lock`?**

I first query `pg_locks` joined with `pg_stat_activity` to identify the blocking PID and the resource being locked. I then investigate whether the blocker is a long-running transaction, an idle-in-transaction session, or a DDL operation; the fix is either killing the blocking session with `pg_terminate_backend()`, setting `idle_in_transaction_session_timeout`, or refactoring application locking order.

---

**19. What is the difference between `TRUNCATE` and `DELETE` in the context of MVCC and WAL?**

`DELETE` marks individual rows as dead tuples, generating WAL records per row and respecting MVCC; existing snapshots still see the deleted rows. `TRUNCATE` is a DDL operation that instantly drops and recreates the storage file, generates a single WAL record regardless of table size, and is not visible to concurrent transactions using older snapshots.

---

**20. Describe a disaster recovery drill you have led and what you learned from it.**

In a quarterly DR drill I restored a BARMan base backup to an isolated environment, applied WAL up to a target recovery point, verified row counts and application connectivity, and measured RTO at approximately 22 minutes for a 500 GB database. The key learning was that `restore_command` misconfiguration is the most common failure point and must be tested independently before every drill.

---

## Technical Deep Dive: DBLink, CDC, search_path, BARMan

---

**21. What is DBLink in PostgreSQL, and how does it differ from Foreign Data Wrappers (FDW)?**

DBLink is a contrib extension (`CREATE EXTENSION dblink`) that allows ad-hoc SQL queries against remote PostgreSQL databases within a single session connection. FDW (`postgres_fdw`) is the modern successor — it integrates remote tables as first-class foreign tables with planner-level pushdown, better connection reuse, and support for non-PostgreSQL sources, making it preferred for recurring cross-database access.

---

**22. You are tasked with setting up cross-database queries in a PostgreSQL environment using DBLink. How would you implement it, and what are the security considerations?**

You load the extension, then call `dblink('host=remote_host dbname=db user=usr password=pwd', 'SELECT id, name FROM orders')` returning typed rows. Security considerations include never hardcoding credentials in SQL — instead store them in `pg_catalog.pg_user_mappings` via a named server, restrict the DBLink user to read-only privileges on the remote, and use SSL (`sslmode=require`) to prevent credential interception.

---

**23. What are the performance implications of using DBLink for frequent cross-database joins?**

Each `dblink()` call opens a new connection to the remote (unless you use a persistent named connection with `dblink_connect`), and the entire remote result set is transferred to the local planner before joining — there is no predicate pushdown. For high-frequency or large-dataset joins, `postgres_fdw` with `use_remote_estimate=true` is significantly more efficient as it pushes WHERE clauses and aggregations to the remote server.

---

**24. What is Change Data Capture (CDC), and what are the two main approaches used in PostgreSQL?**

CDC is the process of tracking row-level data changes (INSERT, UPDATE, DELETE) in near-real-time for streaming to downstream systems. In PostgreSQL the two main approaches are: (1) **logical decoding** via the `pgoutput` or `wal2json` output plugin, which reads WAL and emits structured change events without touching production tables, and (2) **trigger-based CDC**, which writes change records to audit tables — simpler to set up but adds write overhead and contention.

---

**25. How would you set up a CDC pipeline from PostgreSQL to an Apache Kafka topic using Debezium?**

You configure PostgreSQL with `wal_level=logical`, create a replication slot (`pgoutput` plugin), grant `REPLICATION` privilege to the Debezium connector user, and deploy the Debezium PostgreSQL connector pointed at the slot. Debezium reads WAL change events and publishes them as JSON messages to Kafka topics partitioned by table; downstream consumers process the change stream for ETL, search index updates, or cache invalidation.

---

**26. What are common pitfalls with CDC using logical replication slots and how do you mitigate them?**

The most critical pitfall is slot lag: if the CDC consumer stalls or disconnects, the slot prevents WAL cleanup, leading to disk exhaustion. Mitigation includes setting `max_slot_wal_keep_size` (PG 13+) to cap WAL retention, alerting on `pg_replication_slots.confirmed_flush_lsn` drift, and implementing a dead-slot cleanup policy to drop unused slots automatically after a configurable inactivity window.

---

**27. Explain `search_path` in PostgreSQL — what it controls, its default value, and security implications.**

`search_path` is a session-level ordered list of schemas that PostgreSQL searches when resolving an unqualified object name (e.g., `SELECT * FROM orders` without a schema prefix). The default is `"$user", public`, meaning it first looks in the schema matching the current user's name, then falls back to `public`. The security implication is that a malicious or accidental object in an earlier schema can shadow a trusted object in a later one — known as a search_path injection attack.

---

**28. You discover that a scheduled job runs as a superuser and the `search_path` is set to include a schema owned by an untrusted user. What is the risk and how do you fix it?**

The risk is search_path injection: the untrusted schema owner can create a function or table with the same name as one the job relies on, hijacking its execution context with superuser privileges. The fix is to always qualify object names with explicit schemas in the job's SQL, set `search_path = pg_catalog, public` (or schema-lock the job's role with `ALTER ROLE job_user SET search_path = trusted_schema`), and revoke CREATE on public from untrusted roles.

---

**29. How do you set `search_path` at different levels — session, role, and database — and which takes precedence?**

You can set it at session level with `SET search_path TO myschema, public` (highest runtime precedence), at role level with `ALTER ROLE myuser SET search_path = myschema, public` (persists across logins), and at database level with `ALTER DATABASE mydb SET search_path = myschema, public` (applies to all connections to that database). Session-level overrides role-level, which overrides database-level; they all override the compiled-in default.

---

**30. What is BARMan (Backup and Recovery Manager), and what problem does it solve for PostgreSQL DBAs?**

BARMan (by 2ndQuadrant/EDB) is an open-source tool that centralizes backup, retention management, WAL archiving, and restore orchestration for multiple PostgreSQL servers from a single backup catalog server. It solves the operational complexity of managing WAL segments, base backups, retention policies, and PITR restores at scale without writing custom shell scripts.

---

**31. Explain the difference between BARMan's `rsync` backup method and the streaming backup method.**

The `rsync` method SSH-copies data files from the PostgreSQL data directory to the BARMan catalog using pg_start_backup/pg_stop_backup markers; it is I/O-efficient via incremental file sync but requires SSH access. The `streaming` method uses `pg_basebackup` over the replication protocol, requiring only `REPLICATION` privilege and no SSH, making it more firewall-friendly and the preferred method in cloud and containerized environments.

---

**32. How do you configure WAL archiving with BARMan, and what is the role of `archive_command`?**

On the PostgreSQL side you set `archive_mode = on` and `archive_command = 'barman-wal-archive backup-server pg-server %p'` in `postgresql.conf`; this calls the BARMan client script for every completed WAL segment. BARMan receives and catalogs WAL files on the backup server, linking them to base backups to enable PITR; you verify WAL continuity with `barman switch-wal` and `barman check`.

---

**33. Describe a scenario where you would use `barman recover` and what parameters are critical.**

After a catastrophic primary failure I would run `barman recover pg-server latest /var/lib/postgresql/data --target-time "2026-06-17 08:00:00" --remote-ssh-command "ssh postgres@dbhost"` to restore the latest base backup and apply WAL up to the target time on the remote host. Critical parameters are `--target-time` or `--target-xid` for PITR precision, `--remote-ssh-command` for direct remote restore, and post-recover verification of `recovery.conf`/`postgresql.auto.conf` before starting the instance.

---

**34. How does BARMan handle retention policies, and what are the two retention policy models?**

BARMan supports two models: **recovery window** (e.g., `retention_policy = RECOVERY WINDOW OF 7 DAYS`), which keeps enough backups and WAL to restore to any point within the past 7 days, and **redundancy** (e.g., `retention_policy = REDUNDANCY 3`), which keeps a fixed number of base backups. `barman cron` enforces the policy by running `barman delete` on backups that fall outside the window; WAL is pruned automatically once no backup requires it.

---

**35. What monitoring checks does `barman check` perform, and what does a failed check indicate in production?**

`barman check` validates SSH connectivity, replication slot health (if used), WAL archiving continuity, minimum redundancy thresholds, disk space, and backup catalog integrity. A failed check (e.g., `WAL archive: FAILED`) in production means the backup chain is broken — new backups cannot be taken or restored reliably — and should trigger an immediate page, as the system is operating without a valid recovery path.

---

**Summary of Key Concepts**

- **DBLink**: Extension for ad-hoc remote queries; superseded by `postgres_fdw` for production use; always use named connections and SSL.
- **CDC**: Logical decoding (WAL-based, low overhead) is preferred over trigger-based CDC; monitor replication slot lag vigilantly.
- **search_path**: Controls schema resolution order; misconfiguration enables injection attacks; always qualify critical objects explicitly.
- **BARMan**: Centralizes PostgreSQL backup/PITR; supports rsync and streaming methods; retention policies and `barman check` are critical operational controls.

---

*Guide version: June 2026. Answers reflect PostgreSQL 14–16 behavior and BARMan 3.x. Verify against your specific version's release notes before the interview.*
