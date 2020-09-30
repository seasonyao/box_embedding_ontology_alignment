import ntplib
import time

def get_timestamp_from_ntp(server = "pool.ntp.org"):
    try:
        client = ntplib.NTPClient()
        response = client.request(server, version=3)
        return response.tx_timestamp
    except Exception as e:
        print(e)


def get_timestamp():
    return time.time()