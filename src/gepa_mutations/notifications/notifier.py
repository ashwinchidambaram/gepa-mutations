"""Notification system for experiment status updates (SNS / Telegram)."""

from __future__ import annotations

import logging
from typing import Any

from gepa_mutations.config import Settings

logger = logging.getLogger(__name__)


class Notifier:
    """Sends experiment notifications via Telegram and/or AWS SNS."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._telegram_bot = None
        self._sns_client = None

    def _get_telegram(self):
        if self._telegram_bot is None and self.settings.telegram_bot_token:
            import telegram
            self._telegram_bot = telegram.Bot(token=self.settings.telegram_bot_token)
        return self._telegram_bot

    def _get_sns(self):
        if self._sns_client is None:
            import boto3
            self._sns_client = boto3.client("sns")
        return self._sns_client

    async def send_telegram(self, message: str) -> None:
        """Send a message via Telegram bot."""
        bot = self._get_telegram()
        if bot and self.settings.telegram_chat_id:
            try:
                await bot.send_message(
                    chat_id=self.settings.telegram_chat_id,
                    text=message,
                    parse_mode="Markdown",
                )
            except Exception as e:
                logger.warning(f"Telegram notification failed: {e}")

    def send_sns(self, subject: str, message: str, topic_arn: str | None = None) -> None:
        """Send a message via AWS SNS."""
        sns = self._get_sns()
        if sns and topic_arn:
            try:
                sns.publish(
                    TopicArn=topic_arn,
                    Subject=subject[:100],  # SNS subject limit
                    Message=message,
                )
            except Exception as e:
                logger.warning(f"SNS notification failed: {e}")

    def notify_start(self, benchmark: str, seed: int, config: dict[str, Any]) -> None:
        """Notify that an experiment has started."""
        msg = (
            f"*GEPA Experiment Started*\n"
            f"Benchmark: `{benchmark}`\n"
            f"Seed: `{seed}`\n"
            f"Model: `{config.get('model', 'unknown')}`\n"
            f"Rollout budget: `{config.get('rollout_budget', 'unknown')}`"
        )
        logger.info(msg)

    def notify_progress(
        self, benchmark: str, seed: int, iteration: int, score: float
    ) -> None:
        """Notify progress update."""
        msg = (
            f"*GEPA Progress*\n"
            f"Benchmark: `{benchmark}` | Seed: `{seed}`\n"
            f"Iteration: `{iteration}` | Best score: `{score:.4f}`"
        )
        logger.info(msg)

    def notify_complete(
        self, benchmark: str, seed: int, test_score: float, wall_clock: float
    ) -> None:
        """Notify that an experiment has completed."""
        msg = (
            f"*GEPA Experiment Complete*\n"
            f"Benchmark: `{benchmark}` | Seed: `{seed}`\n"
            f"Test score: `{test_score * 100:.2f}%`\n"
            f"Wall clock: `{wall_clock:.0f}s`"
        )
        logger.info(msg)

    def notify_error(self, benchmark: str, seed: int, error: str) -> None:
        """Notify that an experiment has failed."""
        msg = (
            f"*GEPA Experiment FAILED*\n"
            f"Benchmark: `{benchmark}` | Seed: `{seed}`\n"
            f"Error: `{error[:200]}`"
        )
        logger.error(msg)
