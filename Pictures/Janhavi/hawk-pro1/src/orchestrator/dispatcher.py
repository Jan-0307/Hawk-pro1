# minimal dispatcher using MQTT (no signing yet)
import json
import paho.mqtt.publish as publish

MQTT_HOST = "localhost"

def send_command(topic, payload):
    publish.single(topic, payload=json.dumps(payload), hostname=MQTT_HOST)
    return True