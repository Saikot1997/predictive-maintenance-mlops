"""
Kafka Producer — real-time sensor data stream simulate করে।
confluent-kafka library ব্যবহার করা হয়েছে (maintained, Python 3.11+ compatible)।
"""

import json
import logging
import os
import random
import time
from datetime import datetime, timezone

from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def delivery_report(err, msg):
    """Kafka delivery callback — deliver হলে কি না জানায়।"""
    if err is not None:
        logger.error(f"Delivery failed for {msg.key()}: {err}")


def create_producer() -> Producer:
    """confluent-kafka Producer তৈরি করো।"""
    return Producer(
        {
            "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            "acks": "all",
            "retries": 3,
            "linger.ms": 5,
        }
    )


def generate_sensor_reading(machine_id: str) -> dict:
    """Realistic sensor data generate করো।
    Normal operation এর মধ্যে ৩% failure scenario simulate।
    """
    is_failure = random.random() < 0.03

    if is_failure:
        return {
            "machine_id": machine_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": random.choice(["L", "M", "H"]),
            "air_temperature": round(random.uniform(300.0, 310.0), 1),
            "process_temperature": round(random.uniform(315.0, 320.0), 1),
            "rotational_speed": round(random.uniform(1200.0, 1600.0), 0),
            "torque": round(random.uniform(60.0, 80.0), 1),
            "tool_wear": round(random.uniform(200.0, 300.0), 0),
        }
    return {
        "machine_id": machine_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": random.choice(["L", "M", "H"]),
        "air_temperature": round(random.uniform(295.0, 302.0), 1),
        "process_temperature": round(random.uniform(305.0, 312.0), 1),
        "rotational_speed": round(random.uniform(1400.0, 2000.0), 0),
        "torque": round(random.uniform(25.0, 55.0), 1),
        "tool_wear": round(random.uniform(0.0, 150.0), 0),
    }


def run_producer(num_machines: int = 5, interval_sec: float = 1.0):
    """Producer চালাও — প্রতি interval_sec সেকেন্ডে reading পাঠাও।"""
    topic = os.getenv("KAFKA_TOPIC_SENSOR", "sensor-data")
    machines = [f"MACHINE_{i:03d}" for i in range(1, num_machines + 1)]
    producer = create_producer()
    sent_count = 0

    logger.info(f"Producer started | Topic: {topic} | Machines: {num_machines}")
    try:
        while True:
            for machine_id in machines:
                reading = generate_sensor_reading(machine_id)
                producer.produce(
                    topic=topic,
                    key=machine_id.encode("utf-8"),
                    value=json.dumps(reading).encode("utf-8"),
                    callback=delivery_report,
                )
                sent_count += 1

            producer.poll(0)  # Non-blocking — delivery callbacks trigger করে
            if sent_count % 50 == 0:
                logger.info(f"Sent {sent_count} readings")
            time.sleep(interval_sec)

    except KeyboardInterrupt:
        logger.info("Producer stopping...")
    finally:
        producer.flush(timeout=10)
        logger.info("Producer stopped.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_producer()
