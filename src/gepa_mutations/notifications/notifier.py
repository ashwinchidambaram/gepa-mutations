"""Notification system for experiment status updates (SNS / Telegram)."""

from __future__ import annotations

import asyncio
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

    def send_telegram(self, message: str) -> None:
        """Send an HTML-formatted message via Telegram bot (sync wrapper)."""
        bot = self._get_telegram()
        if bot and self.settings.telegram_chat_id:
            try:
                asyncio.run(bot.send_message(
                    chat_id=self.settings.telegram_chat_id,
                    text=message,
                    parse_mode="HTML",
                ))
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

    def _send(self, msg: str) -> None:
        """Send via all configured channels."""
        self.send_telegram(msg)

    def notify_start(self, benchmark: str, seed: int, config: dict[str, Any]) -> None:
        """Notify that an experiment has started."""
        msg = (
            f"🚀 <b>GEPA Experiment Started</b>\n"
            f"Benchmark: <code>{benchmark}</code>  ·  Seed: <code>{seed}</code>\n"
            f"Model: <code>{config.get('model', 'unknown')}</code>\n"
            f"Rollout budget: <code>{config.get('rollout_budget', 'unknown')}</code>"
        )
        logger.info(msg)
        self._send(msg)

    def notify_progress(
        self, benchmark: str, seed: int, iteration: int, score: float
    ) -> None:
        """Notify progress update."""
        msg = (
            f"📈 <b>GEPA Progress</b>\n"
            f"Benchmark: <code>{benchmark}</code>  ·  Seed: <code>{seed}</code>\n"
            f"Iteration: <code>{iteration}</code>  ·  Best score: <code>{score:.4f}</code>"
        )
        logger.info(msg)
        self._send(msg)

    def notify_complete(
        self, benchmark: str, seed: int, test_score: float, wall_clock: float
    ) -> None:
        """Notify that an experiment has completed."""
        msg = (
            f"✅ <b>GEPA Experiment Complete</b>\n"
            f"Benchmark: <code>{benchmark}</code>  ·  Seed: <code>{seed}</code>\n"
            f"Test score: <code>{test_score * 100:.2f}%</code>\n"
            f"Wall clock: <code>{wall_clock:.0f}s</code>"
        )
        logger.info(msg)
        self._send(msg)

    def notify_error(self, benchmark: str, seed: int, error: str) -> None:
        """Notify that an experiment has failed."""
        msg = (
            f"❌ <b>GEPA Experiment FAILED</b>\n"
            f"Benchmark: <code>{benchmark}</code>  ·  Seed: <code>{seed}</code>\n"
            f"Error: <code>{error[:200]}</code>"
        )
        logger.error(msg)
        self._send(msg)
