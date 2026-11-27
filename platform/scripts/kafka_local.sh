#!/usr/bin/env bash
# Start a single-node Kafka broker (KRaft mode, no ZooKeeper) from a local Kafka
# install and create the topics. For Docker use platform/docker-compose.yml instead.
#   KAFKA_HOME=/opt/kafka platform/scripts/kafka_local.sh start|topics|stop
set -euo pipefail
KAFKA_HOME="${KAFKA_HOME:-/opt/kafka}"
DATA="${KAFKA_DATA:-/tmp/kraft-combined-logs}"
BASE_CONF="$KAFKA_HOME/config/kraft/server.properties"
CONF="${DATA}.properties"
BOOT="${KAFKA_BOOTSTRAP:-localhost:9092}"
PARTS="${TOPIC_PARTITIONS:-3}"

case "${1:-start}" in
  start)
    sed "s|^log.dirs=.*|log.dirs=$DATA|" "$BASE_CONF" > "$CONF"
    if [ ! -f "$DATA/meta.properties" ]; then
      "$KAFKA_HOME/bin/kafka-storage.sh" format -t "$("$KAFKA_HOME/bin/kafka-storage.sh" random-uuid)" \
        -c "$CONF" >/dev/null
    fi
    "$KAFKA_HOME/bin/kafka-server-start.sh" -daemon "$CONF"
    for i in $(seq 1 30); do
      "$KAFKA_HOME/bin/kafka-topics.sh" --bootstrap-server "$BOOT" --list >/dev/null 2>&1 && break
      sleep 1
    done
    echo "broker up on $BOOT" ;;
  topics)
    for t in invocations scaling-decisions; do
      "$KAFKA_HOME/bin/kafka-topics.sh" --bootstrap-server "$BOOT" --create --if-not-exists \
        --topic "$t" --partitions "$PARTS" --replication-factor 1
    done ;;
  reset)   # delete and recreate topics (demo re-runs)
    for t in invocations scaling-decisions; do
      "$KAFKA_HOME/bin/kafka-topics.sh" --bootstrap-server "$BOOT" --delete --topic "$t" 2>/dev/null || true
    done
    sleep 3; "$0" topics ;;
  stop)
    "$KAFKA_HOME/bin/kafka-server-stop.sh" ;;
esac
