"""T21en_scheduled_task_management grader — English variant."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T41zh_scheduled_task_management")


class ScheduledTaskManagementGraderEN(_Base):
    """English variant — overrides language-specific class attributes only."""

    COMM_ENTITIES = [
        "JOB-001", "JOB-002", "JOB-003", "JOB-004", "JOB-005",
        "JOB-006", "JOB-007", "JOB-008",
        "daily_backup", "weekly_report", "log_cleanup",
        "data_sync_primary", "data_sync_secondary",
        "certificate_renewal", "temp_migration_job",
    ]

    OPS_EMAIL = "ops@company.com"
