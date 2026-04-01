from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from core.alerts import AlertManager
from core.config import RetrainingSettings, get_settings
from core.entities import AlertEvent, FeatureDriftReport, RetrainingTriggerResult
from utils.logging import get_logger, log_event

logger = get_logger(__name__)

RetrainingPipeline = Callable[[FeatureDriftReport, list[AlertEvent]], dict[str, Any]]


def placeholder_retraining_pipeline(
    report: FeatureDriftReport,
    alerts: list[AlertEvent],
) -> dict[str, Any]:
    affected_features = sorted({alert.feature_name for alert in alerts})
    return {
        "status": "accepted",
        "pipeline": "placeholder_retraining_pipeline",
        "dataset_name": report.dataset_name,
        "qualifying_alerts": len(alerts),
        "affected_features": affected_features,
        "message": "Placeholder retraining pipeline invoked. Replace with real ML orchestration.",
    }


class RetrainingManager:
    def __init__(
        self,
        settings: RetrainingSettings | None = None,
        alert_manager: AlertManager | None = None,
        pipeline: RetrainingPipeline | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        self.settings = settings or get_settings().retraining
        self.alert_manager = alert_manager or AlertManager()
        self.pipeline = pipeline or placeholder_retraining_pipeline
        self.logger = logger_ or logger

    def evaluate(self, report: FeatureDriftReport) -> RetrainingTriggerResult:
        log_event(
            self.logger,
            logging.INFO,
            "Evaluating retraining trigger.",
            event="retraining_evaluation_started",
            dataset_name=report.dataset_name,
            trigger_severity=self.settings.trigger_severity,
            min_alert_count=self.settings.min_alert_count,
            pipeline_name=self.settings.pipeline_name,
        )

        if not self.settings.enabled:
            result = RetrainingTriggerResult(
                triggered=False,
                reason="Retraining trigger is disabled in configuration.",
                severity_threshold=self.settings.trigger_severity,
                qualifying_alert_count=0,
                pipeline_name=self.settings.pipeline_name,
            )
            self._log_result(report, result)
            return result

        alerts = self.alert_manager.build_alerts(report)
        qualifying_alerts = [
            alert
            for alert in alerts
            if self._severity_rank(alert.severity) >= self._severity_rank(self.settings.trigger_severity)
        ]

        if len(qualifying_alerts) < self.settings.min_alert_count:
            result = RetrainingTriggerResult(
                triggered=False,
                reason=(
                    "Retraining conditions not met. "
                    f"Found {len(qualifying_alerts)} qualifying alerts, "
                    f"requires at least {self.settings.min_alert_count}."
                ),
                severity_threshold=self.settings.trigger_severity,
                qualifying_alert_count=len(qualifying_alerts),
                pipeline_name=self.settings.pipeline_name,
                affected_features=sorted({alert.feature_name for alert in qualifying_alerts}),
            )
            self._log_result(report, result)
            return result

        pipeline_response = self.pipeline(report, qualifying_alerts)
        affected_features = sorted({alert.feature_name for alert in qualifying_alerts})
        result = RetrainingTriggerResult(
            triggered=True,
            reason="Retraining trigger activated because critical drift threshold was exceeded.",
            severity_threshold=self.settings.trigger_severity,
            qualifying_alert_count=len(qualifying_alerts),
            pipeline_name=self.settings.pipeline_name,
            affected_features=affected_features,
            pipeline_response=pipeline_response,
        )
        self._log_result(report, result)
        return result

    def _log_result(
        self,
        report: FeatureDriftReport,
        result: RetrainingTriggerResult,
    ) -> None:
        level = logging.WARNING if result.triggered else logging.INFO
        log_event(
            self.logger,
            level,
            "Retraining trigger evaluation completed.",
            event="retraining_evaluation_completed",
            dataset_name=report.dataset_name,
            triggered=result.triggered,
            reason=result.reason,
            severity_threshold=result.severity_threshold,
            qualifying_alert_count=result.qualifying_alert_count,
            pipeline_name=result.pipeline_name,
            affected_features=result.affected_features,
            pipeline_response=result.pipeline_response,
        )

    @staticmethod
    def _severity_rank(severity: str) -> int:
        return {"warning": 1, "critical": 2}[severity]
