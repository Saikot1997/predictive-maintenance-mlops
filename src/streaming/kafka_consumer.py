"""
Kafka Consumer — sensor data receive করে real-time prediction করে।
confluent-kafka library ব্যবহার করা হয়েছে।
"""

import json
import logging
import os

import requests
from confluent_kafka import Consumer, KafkaError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def run_consumer():
    """Consumer চালাও — Kafka থেকে data নিয়ে predict করো।"""
    topic = os.getenv("KAFKA_TOPIC_SENSOR", "sensor-data")
    # Use KAFKA_CONSUMER_API_HOST to avoid collision with API_HOST=0.0.0.0 (bind address)
    api_host = os.getenv("KAFKA_CONSUMER_API_HOST", os.getenv("API_HOST", "localhost"))
    api_port = os.getenv("API_PORT", "8000")
    api_url = f"http://{api_host}:{api_port}/predict"

    consumer = Consumer(
        {
            "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            "group.id": "ml-prediction-group",
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,  # Fix: manual commit prevents message loss on API failure
        }
    )
    consumer.subscribe([topic])
    logger.info(f"Consumer listening on: {topic}")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error(f"Kafka error: {msg.error()}")
                continue

            # Message decode করো
            try:
                data_msg = json.loads(msg.value().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.error(f"Decode error: {e}")
                continue

            machine_id = data_msg.get("machine_id", "UNKNOWN")

            # API কে prediction request পাঠাও
            try:
                payload = {
                    k: v
                    for k, v in data_msg.items()
                    if k not in ["machine_id", "timestamp"]
                }
                response = requests.post(api_url, json=payload, timeout=5)
                response.raise_for_status()
                result = response.json()

                if result.get("risk_level") == "HIGH":
                    logger.warning(
                        f"HIGH RISK | Machine: {machine_id} | "
                        f"Prob: {result['probability']:.3f} | "
                        f"Failure: {result['failure_type']}"
                    )
                else:
                    logger.debug(
                        f"OK | Machine: {machine_id} | Risk: {result.get('risk_level')}"
                    )

                # Manual commit — only after successful prediction
                consumer.commit(asynchronous=False)

            except requests.RequestException as e:
                logger.error(f"Prediction failed for {machine_id}: {e}")
                # Do NOT commit — message will be retried on restart

    except KeyboardInterrupt:
        logger.info("Consumer stopping...")
    finally:
        consumer.close()
        logger.info("Consumer closed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_consumer()
