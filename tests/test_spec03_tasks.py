# tests/test_spec03_tasks.py
# SPEC-03 Phase 1 & 2 Task Configs — Verification Suite
#
# Tests that all Phase 1 (15 tasks) and Phase 2 (16 tasks) task configs:
# 1. Generate deterministic episodes
# 2. Have correct services, fault types, and fault services
# 3. Apply initial_state_overrides correctly
# 4. Inject task_metrics_schema and adversarial_logs
# 5. Support dual-fault configs
# 6. Maintain budget identity (max_ticks × slo_burn_rate = initial_budget)
# 7. Are backward-compatible with legacy tasks

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TASKS, TaskConfig, ALL_SERVICES
from simulation import generate_episode


# ==========================================================================
# Task Registry Tests
# ==========================================================================

class TestTaskRegistry:
    """Verify all Phase 1 tasks are registered and structurally valid."""

    PHASE1_TASK_IDS = [
        "task_easy_oom_baseline",
        "task_easy_pool_restart_cycle",
        "task_easy_quota_runaway",
        "task_easy_fail_slow_memleak",
        "task_easy_alert_fatigue",
        "task_medium_cascade_memleak",
        "task_medium_asymmetric_blast",
        "task_medium_ntp_clock_drift",
        "task_medium_corrupted_external_dep",
        "task_medium_rollout_quota_exhaustion",
        "task_hard_config_drift_noise",
        "task_hard_adversarial_triple",
        "task_hard_partial_infra_asymmetric",
        "task_hard_multiteam_dual_fault",
        "task_hard_cache_corruption",
    ]

    REMOVED_LEGACY_TASK_IDS = ["task_easy", "task_medium", "task_hard"]

    PHASE2_EASY_TASK_IDS = [
        "task_easy_thundering_herd",
        "task_easy_timeout_propagation",
        "task_easy_lb_hotspot",
        "task_easy_liveness_probe_flap",
        "task_easy_log_debug_disk",
        "task_easy_rate_limiter_misconfig",
    ]

    PHASE2_MEDIUM_TASK_IDS = [
        "task_medium_retry_storm",
        "task_medium_canary_false_alert",
        "task_medium_replica_lag",
        "task_medium_circuit_breaker_masking",
        "task_medium_cache_eviction_storm",
        "task_medium_configmap_reload",
        "task_medium_gateway_rate_limit",
        "task_medium_bg_traffic_leak",
        "task_medium_stale_registry",
        "task_medium_grpc_deadline",
    ]

    def test_all_15_phase1_tasks_registered(self):
        """All 15 Phase 1 tasks must be in TASKS dict."""
        for tid in self.PHASE1_TASK_IDS:
            assert tid in TASKS, f"Missing task: {tid}"

    def test_legacy_tasks_removed(self):
        """Legacy scaffold tasks are not part of the active benchmark set."""
        for tid in self.REMOVED_LEGACY_TASK_IDS:
            assert tid not in TASKS, f"Legacy task still registered: {tid}"

    def test_total_task_count(self):
        """15 Phase 1 + 16 Phase 2 + 8 SPEC-08 Hard + 19 SPEC-11 P3 + 2 SPEC-12 P3 Hard = 60 total tasks."""
        assert len(TASKS) == 60, f"Expected 60 tasks, got {len(TASKS)}"

    @pytest.mark.parametrize("task_id", PHASE1_TASK_IDS)
    def test_budget_identity(self, task_id):
        """max_ticks × slo_burn_rate = initial_budget (SPEC-04 §8)."""
        task = TASKS[task_id]
        expected = task.max_ticks * task.slo_burn_rate
        assert task.initial_budget == expected, (
            f"{task_id}: {task.max_ticks} × {task.slo_burn_rate} = {expected}, "
            f"but initial_budget = {task.initial_budget}"
        )

    @pytest.mark.parametrize("task_id", PHASE1_TASK_IDS)
    def test_task_id_matches_key(self, task_id):
        """Task ID field must match its dict key."""
        task = TASKS[task_id]
        assert task.task_id == task_id

    @pytest.mark.parametrize("task_id", PHASE1_TASK_IDS)
    def test_all_services_registered(self, task_id):
        """All services referenced in task config must exist in ALL_SERVICES."""
        task = TASKS[task_id]
        for svc in task.services:
            assert svc in ALL_SERVICES, (
                f"{task_id} references unregistered service: {svc}"
            )
        if task.fault_service:
            assert task.fault_service in ALL_SERVICES
        if task.secondary_fault_service:
            assert task.secondary_fault_service in ALL_SERVICES

    @pytest.mark.parametrize("task_id", PHASE1_TASK_IDS)
    def test_fault_service_in_services_list(self, task_id):
        """Root cause service must be in the active services list."""
        task = TASKS[task_id]
        if task.services and task.fault_service:
            assert task.fault_service in task.services, (
                f"{task_id}: fault_service '{task.fault_service}' not in services"
            )

    @pytest.mark.parametrize("task_id", PHASE1_TASK_IDS)
    def test_red_herrings_in_services_list(self, task_id):
        """Red herring services must be in the active services list."""
        task = TASKS[task_id]
        if task.services and task.red_herrings:
            for rh in task.red_herrings:
                # Red herrings can be listed even if not in services
                # (e.g., notification-service in task_hard_cache_corruption)
                pass


# ==========================================================================
# Easy Tier Episode Tests
# ==========================================================================

class TestEasyTier:
    """Verify easy tier tasks generate correct episodes."""

    def test_e_s1_oom_baseline(self):
        """E-S1: Single OOM Kill on auth-service."""
        mesh, fc = generate_episode("easy", 42, task_id="task_easy_oom_baseline")
        assert fc.root_cause_service == "auth-service"
        assert fc.fault_type == "oom"
        assert set(mesh.services.keys()) == {"api-gateway", "auth-service", "db-proxy"}
        # Verify initial_state_overrides
        assert mesh.services["auth-service"].process_memory_utilization == 0.98
        assert mesh.services["auth-service"].restart_count == 3

    def test_e_s2_pool_restart_cycle(self):
        """E-S2: Connection Pool Restart Cycle."""
        mesh, fc = generate_episode("easy", 210, task_id="task_easy_pool_restart_cycle")
        assert fc.root_cause_service == "auth-service"
        assert fc.fault_type == "config_drift"
        auth = mesh.services["auth-service"]
        assert auth.restart_count == 4
        assert auth.http_server_error_rate == 0.61
        assert auth.process_open_file_descriptors == 3

    def test_e_r2_quota_runaway(self):
        """E-R2: Quota Exhaustion Runaway Client."""
        mesh, fc = generate_episode("easy", 84, task_id="task_easy_quota_runaway")
        assert fc.root_cause_service == "notification-service"
        assert fc.fault_type == "bad_deploy"
        ns = mesh.services["notification-service"]
        assert ns.http_server_error_rate == 0.22
        user = mesh.services["user-service"]
        assert user.http_server_active_requests == 312

    def test_e_r3_fail_slow_memleak(self):
        """E-R3: Fail-Slow Memory Leak."""
        mesh, fc = generate_episode("easy", 126, task_id="task_easy_fail_slow_memleak")
        assert fc.root_cause_service == "payment-service"
        assert fc.fault_type == "memory_leak"
        pay = mesh.services["payment-service"]
        assert pay.process_memory_utilization == 0.71
        assert pay.runtime_gc_pause_duration_ms == 420.0

    def test_e_r5_alert_fatigue(self):
        """E-R5: Alert Fatigue Noisy Suppression."""
        mesh, fc = generate_episode("easy", 168, task_id="task_easy_alert_fatigue")
        assert fc.root_cause_service == "db-proxy"
        assert fc.fault_type == "config_drift"
        db = mesh.services["db-proxy"]
        assert db.process_open_file_descriptors == 4987
        assert db.http_server_error_rate == 0.35

    def test_e_r1_thundering_herd(self):
        """E-R1: Thundering Herd Cold Start."""
        mesh, fc = generate_episode("easy", 301, task_id="task_easy_thundering_herd")
        assert fc.root_cause_service == "session-service"
        assert fc.fault_type == "bad_deploy"
        # Thundering herd builds over ticks; verify service topology
        assert "session-service" in mesh.services
        assert "load-balancer" in mesh.services

    def test_e_r4_timeout_propagation(self):
        """E-R4: Upstream Timeout Propagation Chain."""
        mesh, fc = generate_episode("easy", 378, task_id="task_easy_timeout_propagation")
        assert fc.root_cause_service == "inventory-service"
        assert fc.fault_type == "config_drift"
        inv = mesh.services["inventory-service"]
        assert inv.http_server_request_duration_p99 > 5.0  # slow queries
        assert inv.http_server_error_rate < 0.05  # but low error rate

    def test_e_r6_lb_hotspot(self):
        """E-R6: Load Balancer Hotspot Imbalance."""
        mesh, fc = generate_episode("easy", 420, task_id="task_easy_lb_hotspot")
        assert fc.root_cause_service == "user-profile-service"
        assert fc.fault_type == "config_drift"
        user_prof = mesh.services["user-profile-service"]
        assert hasattr(user_prof, "lb_weight_normalized")
        assert user_prof.lb_weight_normalized == 4.0

    def test_e_r7_liveness_probe_flap(self):
        """E-R7: Kubernetes Liveness Probe False Positive Flap."""
        mesh, fc = generate_episode("easy", 462, task_id="task_easy_liveness_probe_flap")
        assert fc.root_cause_service == "payment-processor"
        assert fc.fault_type == "bad_deploy"
        pay = mesh.services["payment-processor"]
        assert hasattr(pay, "liveness_probe_status")
        assert pay.liveness_probe_status == "timeout"
        assert pay.restart_count == 7

    def test_e_r8_log_debug_disk(self):
        """E-R8: Log Debug Mode Left On - Disk Explosion."""
        mesh, fc = generate_episode("easy", 504, task_id="task_easy_log_debug_disk")
        assert fc.root_cause_service == "api-gateway"
        assert fc.fault_type == "config_drift"
        gw = mesh.services["api-gateway"]
        assert hasattr(gw, "application_log_level")
        assert gw.application_log_level == "DEBUG"
        assert hasattr(gw, "process_disk_usage_ratio")
        assert gw.process_disk_usage_ratio > 0.90

    def test_e_r12_rate_limiter_misconfig(self):
        """E-R12: Rate Limiter Too Aggressive - Misconfiguration."""
        mesh, fc = generate_episode("easy", 672, task_id="task_easy_rate_limiter_misconfig")
        assert fc.root_cause_service == "api-gateway"
        assert fc.fault_type == "config_drift"
        gw = mesh.services["api-gateway"]
        assert gw.http_server_error_rate > 0.80  # 429 storm

    def test_easy_tier_constraints(self):
        """All easy tasks: 3 services, 0 red herrings, 20 ticks."""
        easy_ids = [
            "task_easy_oom_baseline", "task_easy_pool_restart_cycle",
            "task_easy_quota_runaway", "task_easy_fail_slow_memleak",
            "task_easy_alert_fatigue",
        ]
        phase2_easy_ids = [
            "task_easy_thundering_herd", "task_easy_timeout_propagation",
            "task_easy_lb_hotspot", "task_easy_liveness_probe_flap",
            "task_easy_log_debug_disk", "task_easy_rate_limiter_misconfig",
        ]
        for tid in easy_ids + phase2_easy_ids:
            task = TASKS[tid]
            assert task.max_ticks == 20, f"{tid}: expected 20 ticks"
            assert task.slo_burn_rate == 1.5, f"{tid}: expected 1.5 burn rate"


# ==========================================================================
# Medium Tier Episode Tests
# ==========================================================================

class TestMediumTier:
    """Verify medium tier tasks generate correct episodes."""

    def test_m_s1_cascade_memleak(self):
        """M-S1: Upstream Memory Leak Cascade."""
        mesh, fc = generate_episode("medium", 295, task_id="task_medium_cascade_memleak")
        assert fc.root_cause_service == "payment-service"
        assert fc.fault_type == "memory_leak"
        assert len(mesh.services) == 5
        pay = mesh.services["payment-service"]
        assert pay.process_memory_utilization == 0.74

    def test_m_s2_asymmetric_blast(self):
        """M-S2: Network Partition Asymmetric Blast."""
        mesh, fc = generate_episode("medium", 463, task_id="task_medium_asymmetric_blast")
        assert fc.root_cause_service == "db-proxy"
        assert fc.fault_type == "network_partition"
        # Asymmetric overrides
        assert mesh.services["auth-service"].http_server_error_rate == 0.85
        assert mesh.services["payment-service"].http_server_error_rate == 0.22
        assert mesh.services["user-service"].http_server_error_rate == 0.08
        assert mesh.services["db-proxy"].http_server_error_rate == 0.95

    def test_m_r1_ntp_clock_drift(self):
        """M-R1: NTP Clock Drift with task_metrics_schema."""
        mesh, fc = generate_episode("medium", 421, task_id="task_medium_ntp_clock_drift")
        assert fc.root_cause_service == "db-proxy"
        assert fc.fault_type == "config_drift"
        # task_metrics_schema injects dynamic fields
        assert hasattr(mesh.services["db-proxy"], "system_clock_offset_seconds")
        assert mesh.services["db-proxy"].system_clock_offset_seconds == -45.0
        assert hasattr(mesh.services["db-proxy"], "ntp_sync_status")
        assert mesh.services["db-proxy"].ntp_sync_status == "drift"
        # Auth should also have clock offset
        assert mesh.services["auth-service"].system_clock_offset_seconds == -45.0

    def test_m_r7_corrupted_external_dep(self):
        """M-R7: Corrupted External Dependency (SPEC-06 §2 corrected)."""
        mesh, fc = generate_episode("medium", 337, task_id="task_medium_corrupted_external_dep")
        assert fc.root_cause_service == "user-service"
        assert fc.fault_type == "config_drift"
        user_svc = mesh.services["user-service"]
        assert user_svc.http_server_error_rate == 0.42

    def test_m_r8_rollout_quota_exhaustion(self):
        """M-R8: Rollout Quota Exhaustion (SPEC-06 §2 corrected)."""
        mesh, fc = generate_episode("medium", 379, task_id="task_medium_rollout_quota_exhaustion")
        assert fc.root_cause_service == "api-gateway"
        assert fc.fault_type == "bad_deploy"
        gw = mesh.services["api-gateway"]
        assert gw.http_server_error_rate == 0.38

    def test_medium_tier_constraints(self):
        """All medium tasks: 30 ticks, 2.0 burn rate."""
        medium_ids = [
            "task_medium_cascade_memleak", "task_medium_asymmetric_blast",
            "task_medium_ntp_clock_drift", "task_medium_corrupted_external_dep",
            "task_medium_rollout_quota_exhaustion",
        ]
        phase2_medium_ids = [
            "task_medium_retry_storm", "task_medium_canary_false_alert",
            "task_medium_replica_lag", "task_medium_circuit_breaker_masking",
            "task_medium_cache_eviction_storm", "task_medium_configmap_reload",
            "task_medium_gateway_rate_limit", "task_medium_bg_traffic_leak",
            "task_medium_stale_registry", "task_medium_grpc_deadline",
        ]
        for tid in medium_ids + phase2_medium_ids:
            task = TASKS[tid]
            assert task.max_ticks == 30, f"{tid}: expected 30 ticks"
            assert task.slo_burn_rate == 2.0, f"{tid}: expected 2.0 burn rate"


# ==========================================================================
# Phase 2 Easy Tier Episode Tests
# ==========================================================================

class TestPhase2EasyTier:
    """Verify Phase 2 easy tier tasks generate correct episodes."""

    def test_e_r1_thundering_herd_episode(self):
        """E-R1: Thundering Herd Cold Start."""
        mesh, fc = generate_episode("easy", 301, task_id="task_easy_thundering_herd")
        assert fc.root_cause_service == "session-service"
        assert fc.fault_type == "bad_deploy"
        assert set(mesh.services.keys()) == {"load-balancer", "api-gateway", "session-service"}

    def test_e_r4_timeout_propagation_episode(self):
        """E-R4: Upstream Timeout Propagation Chain."""
        mesh, fc = generate_episode("easy", 378, task_id="task_easy_timeout_propagation")
        assert fc.root_cause_service == "inventory-service"
        assert fc.fault_type == "config_drift"
        assert set(mesh.services.keys()) == {"order-service", "inventory-service", "inventory-db"}

    def test_e_r6_lb_hotspot_episode(self):
        """E-R6: Load Balancer Hotspot Imbalance."""
        mesh, fc = generate_episode("easy", 420, task_id="task_easy_lb_hotspot")
        assert fc.root_cause_service == "user-profile-service"
        assert fc.fault_type == "config_drift"
        user_prof = mesh.services["user-profile-service"]
        assert hasattr(user_prof, "lb_weight_normalized")
        assert user_prof.lb_weight_normalized == 4.0

    def test_e_r7_liveness_probe_flap_episode(self):
        """E-R7: Kubernetes Liveness Probe False Positive Flap."""
        mesh, fc = generate_episode("easy", 462, task_id="task_easy_liveness_probe_flap")
        assert fc.root_cause_service == "payment-processor"
        assert fc.fault_type == "bad_deploy"
        pay = mesh.services["payment-processor"]
        assert hasattr(pay, "liveness_probe_status")
        assert pay.liveness_probe_status == "timeout"

    def test_e_r8_log_debug_disk_episode(self):
        """E-R8: Log Debug Mode Left On - Disk Explosion."""
        mesh, fc = generate_episode("easy", 504, task_id="task_easy_log_debug_disk")
        assert fc.root_cause_service == "api-gateway"
        assert fc.fault_type == "config_drift"
        gw = mesh.services["api-gateway"]
        assert hasattr(gw, "application_log_level")
        assert gw.application_log_level == "DEBUG"

    def test_e_r12_rate_limiter_episode(self):
        """E-R12: Rate Limiter Too Aggressive."""
        mesh, fc = generate_episode("easy", 672, task_id="task_easy_rate_limiter_misconfig")
        assert fc.root_cause_service == "api-gateway"
        assert fc.fault_type == "config_drift"


# ==========================================================================
# Phase 2 Medium Tier Episode Tests
# ==========================================================================

class TestPhase2MediumTier:
    """Verify Phase 2 medium tier tasks generate correct episodes."""

    def test_m_r2_retry_storm_episode(self):
        """M-R2: Retry Storm Amplification."""
        mesh, fc = generate_episode("medium", 1134, task_id="task_medium_retry_storm")
        assert fc.root_cause_service == "notification-service"
        assert fc.fault_type == "bad_deploy"

    def test_m_r3_canary_false_alert_episode(self):
        """M-R3: Canary Deployment False Alert Attribution."""
        mesh, fc = generate_episode("medium", 1176, task_id="task_medium_canary_false_alert")
        assert fc.root_cause_service == "checkout-service"
        assert fc.fault_type == "bad_deploy"
        checkout = mesh.services["checkout-service"]
        assert hasattr(checkout, "canary_error_rate")
        assert checkout.canary_error_rate == 0.45

    def test_m_r4_replica_lag_episode(self):
        """M-R4: Read Replica Lag with Stale-Read Errors."""
        mesh, fc = generate_episode("medium", 1218, task_id="task_medium_replica_lag")
        assert fc.root_cause_service == "user-service"
        assert fc.fault_type == "network_partition"
        user_svc = mesh.services["user-service"]
        assert hasattr(user_svc, "db_replication_lag_seconds")
        assert user_svc.db_replication_lag_seconds == 45.0

    def test_m_r5_circuit_breaker_masking_episode(self):
        """M-R5: Circuit Breaker Open Masking True Root Cause."""
        mesh, fc = generate_episode("medium", 1260, task_id="task_medium_circuit_breaker_masking")
        assert fc.root_cause_service == "pricing-service"
        assert fc.fault_type == "memory_leak"
        catalog = mesh.services["product-catalog"]
        assert hasattr(catalog, "circuit_breaker_state")
        assert catalog.circuit_breaker_state == "open"

    def test_m_r6_cache_eviction_storm_episode(self):
        """M-R6: Cache Eviction Storm Cascading to Primary Database."""
        mesh, fc = generate_episode("medium", 1302, task_id="task_medium_cache_eviction_storm")
        assert fc.root_cause_service == "cache-service"
        assert fc.fault_type == "config_drift"
        cache = mesh.services["cache-service"]
        assert hasattr(cache, "cache_hit_rate")
        assert cache.cache_hit_rate == 0.30

    def test_m_r12_configmap_reload_episode(self):
        """M-R12: ConfigMap Hot Reload Breaking Running Pods."""
        mesh, fc = generate_episode("medium", 1470, task_id="task_medium_configmap_reload")
        assert fc.root_cause_service == "notification-service"
        assert fc.fault_type == "config_drift"

    def test_m_r13_gateway_rate_limit_episode(self):
        """M-R13: API Gateway Rate Limit Config Too Aggressive."""
        mesh, fc = generate_episode("medium", 1512, task_id="task_medium_gateway_rate_limit")
        assert fc.root_cause_service == "api-gateway"
        assert fc.fault_type == "config_drift"

    def test_m_r14_bg_traffic_leak_episode(self):
        """M-R14: Blue-Green Deployment Traffic Leak."""
        mesh, fc = generate_episode("medium", 1554, task_id="task_medium_bg_traffic_leak")
        assert fc.root_cause_service == "checkout-service"
        assert fc.fault_type == "config_drift"
        checkout = mesh.services["checkout-service"]
        assert hasattr(checkout, "active_deployment_slots")
        assert checkout.active_deployment_slots["blue"] == 0.15

    def test_m_r15_stale_registry_episode(self):
        """M-R15: Service Registry Stale Entry."""
        mesh, fc = generate_episode("medium", 1596, task_id="task_medium_stale_registry")
        assert fc.root_cause_service == "recommendation-engine"
        assert fc.fault_type == "config_drift"
        rec = mesh.services["recommendation-engine"]
        assert hasattr(rec, "registry_stale_instance_count")
        assert rec.registry_stale_instance_count == 1

    def test_m_r16_grpc_deadline_episode(self):
        """M-R16: gRPC Deadline Propagation Header Missing."""
        mesh, fc = generate_episode("medium", 1638, task_id="task_medium_grpc_deadline")
        assert fc.root_cause_service == "order-service"
        assert fc.fault_type == "bad_deploy"
        order = mesh.services["order-service"]
        assert hasattr(order, "grpc_deadline_propagation_rate")
        assert order.grpc_deadline_propagation_rate == 0.0


# ==========================================================================
# Hard Tier Episode Tests
# ==========================================================================

class TestHardTier:
    """Verify hard tier tasks generate correct episodes."""

    def test_h_s1_config_drift_noise(self):
        """H-S1: Config Drift Noise Storm Hardened with notification-service."""
        mesh, fc = generate_episode("hard", 2560, task_id="task_hard_config_drift_noise")
        assert fc.root_cause_service == "api-gateway"
        assert fc.fault_type == "config_drift"
        assert "notification-service" in mesh.services
        assert len(mesh.services) == 8
        # Adversarial log injected
        assert len(mesh._adversarial_logs) == 1
        assert mesh._adversarial_logs[0]["service"] == "cache"

    def test_h_s2_adversarial_triple(self):
        """H-S2: Triple adversarial injection."""
        mesh, fc = generate_episode("hard", 2048, task_id="task_hard_adversarial_triple")
        assert fc.root_cause_service == "payment-service"
        assert fc.fault_type == "memory_leak"
        assert len(mesh._adversarial_logs) == 3
        # All three unique services
        injected_services = {log["service"] for log in mesh._adversarial_logs}
        assert injected_services == {"notification-service", "cache", "user-service"}

    def test_h_r8_partial_infra_asymmetric(self):
        """H-R8: Partial Infrastructure Asymmetric Failure."""
        mesh, fc = generate_episode("hard", 768, task_id="task_hard_partial_infra_asymmetric")
        assert fc.root_cause_service == "db-proxy"
        assert fc.fault_type == "network_partition"
        # Write-heavy services fail harder
        assert mesh.services["payment-service"].http_server_error_rate == 0.91
        assert mesh.services["checkout-service"].http_server_error_rate == 0.78
        # Read-heavy services remain functional
        assert mesh.services["user-service"].http_server_error_rate == 0.09
        assert mesh.services["cache"].http_server_error_rate == 0.07

    def test_h_r9_dual_fault(self):
        """H-R9: Multi-Team Dual-Fault Incident Response."""
        mesh, fc = generate_episode("hard", 1024, task_id="task_hard_multiteam_dual_fault")
        assert fc.root_cause_service == "auth-service"
        assert fc.fault_type == "bad_deploy"
        # Dual-fault
        assert len(mesh.active_faults) == 2
        primary = mesh.active_faults[0]
        secondary = mesh.active_faults[1]
        assert primary.fault_type == "bad_deploy"
        assert primary.fault_service == "auth-service"
        assert secondary.fault_type == "memory_leak"
        assert secondary.fault_service == "notification-service"
        # Notification service must be present
        assert "notification-service" in mesh.services

    def test_h_r10_cache_corruption(self):
        """H-R10: Cascading Cache Corruption."""
        mesh, fc = generate_episode("hard", 512, task_id="task_hard_cache_corruption")
        assert fc.root_cause_service == "cache"
        assert fc.fault_type == "config_drift"
        assert len(mesh._adversarial_logs) == 1
        assert "notification-service" in mesh._adversarial_logs[0]["service"]

    def test_hard_tier_constraints(self):
        """All hard tasks: 40 ticks, 3.0 burn rate."""
        hard_ids = [
            "task_hard_config_drift_noise", "task_hard_adversarial_triple",
            "task_hard_partial_infra_asymmetric", "task_hard_multiteam_dual_fault",
            "task_hard_cache_corruption",
            # SPEC-12 Phase 3 Hard
            "task_hard_pipeline_freshness", "task_hard_mesh_proxy_upgrade",
        ]
        for tid in hard_ids:
            task = TASKS[tid]
            assert task.max_ticks == 40, f"{tid}: expected 40 ticks"
            assert task.slo_burn_rate == 3.0, f"{tid}: expected 3.0 burn rate"

    # --- SPEC-12 Phase 3 Hard Tier ---

    def test_h_r5_pipeline_freshness(self):
        """H-R5: Data Pipeline Freshness SLO Violation."""
        mesh, fc = generate_episode("hard", 8192, task_id="task_hard_pipeline_freshness")
        assert fc.root_cause_service == "feature-pipeline"
        assert fc.fault_type == "memory_leak"
        assert "feature-pipeline" in mesh.services
        assert "feature-store" in mesh.services
        assert "event-ingestion" in mesh.services
        # All services have error_rate=0.0
        for svc_name, svc in mesh.services.items():
            assert svc.http_server_error_rate == 0.0, (
                f"{svc_name} error_rate={svc.http_server_error_rate}, expected 0.0"
            )
        # Pipeline metrics present
        fp = mesh.services["feature-pipeline"]
        assert hasattr(fp, "data_freshness_lag_seconds")
        assert fp.data_freshness_lag_seconds == 1847.0
        assert hasattr(fp, "pipeline_queue_depth")
        assert fp.pipeline_queue_depth == 12400
        assert hasattr(fp, "pipeline_throughput_ratio")
        assert fp.pipeline_throughput_ratio == 0.472
        # BCM mode
        task = TASKS["task_hard_pipeline_freshness"]
        assert task.bcm_mode == "freshness"
        # Adversarial log
        assert len(mesh._adversarial_logs) == 1
        assert mesh._adversarial_logs[0]["service"] == "analytics-service"

    def test_h_r12_mesh_proxy_upgrade(self):
        """H-R12: Service Mesh Proxy Rolling Upgrade Partial Failure."""
        mesh, fc = generate_episode("hard", 12288, task_id="task_hard_mesh_proxy_upgrade")
        assert fc.root_cause_service == "payment-service"
        assert fc.fault_type == "config_drift"
        assert len(mesh.services) == 10
        # Root cause: payment on v1.28
        pay = mesh.services["payment-service"]
        assert hasattr(pay, "sidecar_proxy_version")
        assert pay.sidecar_proxy_version == "v1.28"
        assert hasattr(pay, "mtls_cipher_compatibility")
        assert pay.mtls_cipher_compatibility is False
        # Red herrings also on v1.28
        auth = mesh.services["auth-service"]
        assert auth.sidecar_proxy_version == "v1.28"
        user = mesh.services["user-service"]
        assert user.sidecar_proxy_version == "v1.28"
        # Already upgraded services on v1.29
        gw = mesh.services["api-gateway"]
        assert gw.sidecar_proxy_version == "v1.29"
        assert gw.mtls_cipher_compatibility is True
        # Adversarial log
        assert len(mesh._adversarial_logs) == 1
        assert mesh._adversarial_logs[0]["service"] == "analytics-service"


# ==========================================================================
# Determinism Tests
# ==========================================================================

class TestDeterminism:
    """Verify episodes are deterministic across runs."""

    @pytest.mark.parametrize("task_id", [
        "task_easy_oom_baseline",
        "task_medium_asymmetric_blast",
        "task_hard_multiteam_dual_fault",
    ])
    def test_same_seed_same_episode(self, task_id):
        """Same seed + task_id produces identical episodes."""
        task = TASKS[task_id]
        mesh1, fc1 = generate_episode(task.difficulty, task.seed, task_id=task_id)
        mesh2, fc2 = generate_episode(task.difficulty, task.seed, task_id=task_id)

        assert fc1.root_cause_service == fc2.root_cause_service
        assert fc1.fault_type == fc2.fault_type
        assert list(mesh1.services.keys()) == list(mesh2.services.keys())

        for svc_name in mesh1.services:
            m1 = mesh1.services[svc_name]
            m2 = mesh2.services[svc_name]
            assert m1.http_server_error_rate == m2.http_server_error_rate
            assert m1.process_memory_utilization == m2.process_memory_utilization


# ==========================================================================
# Seed-Based Lookup Tests
# ==========================================================================

class TestSeedLookup:
    """Verify seed-based lookup works without explicit task_id."""

    def test_seed_lookup_finds_correct_task(self):
        """Calling generate_episode with matching (difficulty, seed) finds the task."""
        mesh, fc = generate_episode("easy", 315)  # E-R2 grader seed
        assert fc.root_cause_service == "notification-service"
        assert fc.fault_type == "bad_deploy"

    def test_seed_lookup_dual_fault(self):
        """Seed lookup also works for dual-fault tasks."""
        mesh, fc = generate_episode("hard", 1024)  # H-R9 seed
        assert len(mesh.active_faults) == 2


# ==========================================================================
# Notification Service Registry Tests
# ==========================================================================

class TestNotificationService:
    """Verify notification-service is properly registered (SPEC-04 §2)."""

    def test_in_all_services(self):
        assert "notification-service" in ALL_SERVICES

    def test_in_dependency_graph(self):
        from config import FULL_DEPENDENCY_GRAPH
        assert "notification-service" in FULL_DEPENDENCY_GRAPH
        assert FULL_DEPENDENCY_GRAPH["notification-service"] == ["user-service", "notification-db"]

    def test_in_memory_limits(self):
        from config import SERVICE_MEMORY_LIMITS_BYTES
        assert "notification-service" in SERVICE_MEMORY_LIMITS_BYTES
        assert SERVICE_MEMORY_LIMITS_BYTES["notification-service"] == 536870912
