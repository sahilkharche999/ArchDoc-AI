import redis

from src.logger import setup_logger

redis_client = None
logger = setup_logger(__name__)


def connect_redis(host, port):
    global redis_client

    redis_client = redis.StrictRedis(
        host=host,
        port=port,
        decode_responses=True
    )

    try:
        redis_client.ping()
        logger.info("Connected to Redis!")
    except redis.ConnectionError as e:
        logger.error("Unable to connect to Redis.")
        raise

    return redis_client
