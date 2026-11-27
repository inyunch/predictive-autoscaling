"""Shared Kafka settings (override with environment variables)."""
import os

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_INVOCATIONS = os.environ.get("TOPIC_INVOCATIONS", "invocations")
TOPIC_DECISIONS = os.environ.get("TOPIC_DECISIONS", "scaling-decisions")
GROUP_ID = os.environ.get("SCALER_GROUP", "autoscaler")
PARTITIONS = int(os.environ.get("TOPIC_PARTITIONS", "3"))

PRODUCER = {
    "bootstrap.servers": BOOTSTRAP,
    "enable.idempotence": True,     # no duplicates from producer retries
    "acks": "all",
    "linger.ms": 20,
    "compression.type": "lz4",
}
