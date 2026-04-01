from __future__ import annotations

import json
import logging
from dataclasses import asdict
from urllib.error import URLError
from urllib.request import Request, urlopen

from core.config import AlertSettings, get_settings
from core.entities import AlertDispatchReport, AlertEvent, DriftMetricResult, FeatureDriftReport
from utils.logging import log_event


class AlertManager:
    def __init__(
        self,
        settings: AlertSettings | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings or get_settings().alerts
        self.logger = logger or logging.getLogger(__name__)

    def dispatch(self, report: FeatureDriftReport) -> AlertDispatchReport:
        log_event(
            self.logger,
            logging.INFO,
            "Starting alert dispatch.",
            event="alert_dispatch_started",
            dataset_name=report.dataset_name,
            generated_at=report.generated_at,
        )
        alerts = self.build_alerts(report)
        dispatch_report = AlertDispatchReport(
            total_alerts=len(alerts),
            alerts=alerts,
        )

        if not self.settings.enabled or not alerts:
            log_event(
                self.logger,
                logging.INFO,
                "Alert dispatch skipped.",
                event="alert_dispatch_skipped",
                enabled=self.settings.enabled,
                total_alerts=len(alerts),
            )
            return dispatch_report

        if self.settings.log_alerts:
            self._log_alerts(alerts)
            dispatch_report.logged_alerts = len(alerts)

        if self.settings.webhook_url:
            dispatch_report.webhook_sent = self._send_webhook(
                url=self.settings.webhook_url,
                payload=self._build_webhook_payload(alerts, report),
            )

        if self.settings.slack_webhook_url:
            dispatch_report.slack_sent = self._send_webhook(
                url=self.settings.slack_webhook_url,
                payload=self._build_slack_payload(alerts, report),
            )

        log_event(
            self.logger,
            logging.INFO,
            "Completed alert dispatch.",
            event="alert_dispatch_completed",
            total_alerts=dispatch_report.total_alerts,
            logged_alerts=dispatch_report.logged_alerts,
            webhook_sent=dispatch_report.webhook_sent,
            slack_sent=dispatch_report.slack_sent,
        )
        return dispatch_report

    def build_alerts(self, report: FeatureDriftReport) -> list[AlertEvent]:
        alerts: list[AlertEvent] = []

        for feature in report.features:
            if not feature.supported:
                continue

            for metric in self._iter_metrics(feature.metrics):
                if metric is None or not metric.drift_detected:
                    continue

                severity = self._determine_severity(metric)
                if not self._meets_minimum_severity(severity):
                    continue

                alerts.append(
                    AlertEvent(
                        dataset_name=report.dataset_name,
                        feature_name=feature.feature_name,
                        feature_type=feature.feature_type,
                        metric_name=metric.metric_name,
                        metric_value=metric.value,
                        threshold=metric.threshold,
                        severity=severity,
                        p_value=metric.p_value,
                        generated_at=report.generated_at,
                        message=self._format_alert_message(
                            feature_name=feature.feature_name,
                            metric=metric,
                            severity=severity,
                        ),
                    ),
                )

        log_event(
            self.logger,
            logging.INFO,
            "Built drift alerts from report.",
            event="alerts_built",
            dataset_name=report.dataset_name,
            total_alerts=len(alerts),
        )
        return alerts

    def _log_alerts(self, alerts: list[AlertEvent]) -> None:
        for alert in alerts:
            log_method = self.logger.critical if alert.severity == "critical" else self.logger.warning
            log_method(
                alert.message,
                extra={
                    "event": "alert_triggered",
                    "dataset_name": alert.dataset_name,
                    "feature_name": alert.feature_name,
                    "feature_type": alert.feature_type,
                    "metric_name": alert.metric_name,
                    "metric_value": alert.metric_value,
                    "threshold": alert.threshold,
                    "severity": alert.severity,
                    "p_value": alert.p_value,
                    "generated_at": alert.generated_at,
                },
            )

    def _determine_severity(self, metric: DriftMetricResult) -> str:
        if metric.p_value is not None:
            critical_boundary = metric.threshold * self.settings.critical_p_value_ratio
            return "critical" if metric.p_value <= critical_boundary else "warning"

        if metric.value is not None and metric.value >= metric.threshold * self.settings.critical_excess_ratio:
            return "critical"

        return "warning"

    def _meets_minimum_severity(self, severity: str) -> bool:
        severity_order = {"warning": 1, "critical": 2}
        return severity_order[severity] >= severity_order[self.settings.minimum_severity]

    def _format_alert_message(
        self,
        feature_name: str,
        metric: DriftMetricResult,
        severity: str,
    ) -> str:
        metric_value = f"{metric.value:.6f}" if metric.value is not None else "n/a"
        p_value = f", p_value={metric.p_value:.6f}" if metric.p_value is not None else ""
        return (
            f"[{severity.upper()}] Drift detected for feature '{feature_name}' via "
            f"{metric.metric_name}: value={metric_value}, threshold={metric.threshold:.6f}"
            f"{p_value}"
        )

    def _build_webhook_payload(
        self,
        alerts: list[AlertEvent],
        report: FeatureDriftReport,
    ) -> dict:
        return {
            "dataset_name": report.dataset_name,
            "generated_at": report.generated_at,
            "total_alerts": len(alerts),
            "alerts": [asdict(alert) for alert in alerts],
        }

    def _build_slack_payload(
        self,
        alerts: list[AlertEvent],
        report: FeatureDriftReport,
    ) -> dict:
        lines = [
            f"*Drift alerts for `{report.dataset_name}`*",
            f"Generated at: {report.generated_at}",
            f"Total alerts: {len(alerts)}",
        ]

        for alert in alerts[:10]:
            metric_value = f"{alert.metric_value:.6f}" if alert.metric_value is not None else "n/a"
            line = (
                f"- `{alert.severity.upper()}` feature `{alert.feature_name}` "
                f"metric `{alert.metric_name}` value `{metric_value}` threshold `{alert.threshold:.6f}`"
            )
            if alert.p_value is not None:
                line += f" p-value `{alert.p_value:.6f}`"
            lines.append(line)

        if len(alerts) > 10:
            lines.append(f"- ...and {len(alerts) - 10} more alerts")

        return {"text": "\n".join(lines)}

    def _send_webhook(self, url: str, payload: dict) -> bool:
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                log_event(
                    self.logger,
                    logging.INFO,
                    "Successfully sent alert webhook.",
                    event="alert_webhook_sent",
                    url=url,
                    status_code=response.status,
                )
                return 200 <= response.status < 300
        except URLError as exc:
            log_event(
                self.logger,
                logging.WARNING,
                "Failed to send alert webhook.",
                event="alert_webhook_failed",
                url=url,
                error=str(exc),
            )
            return False

    @staticmethod
    def _iter_metrics(metrics) -> tuple[DriftMetricResult | None, ...]:
        return (
            metrics.psi,
            metrics.ks,
            metrics.kl_divergence,
            metrics.chi_square,
            metrics.distribution_difference,
        )
