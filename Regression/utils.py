import json


def save_json_data(path, data):
    with open(path, "w") as f:
        json.dump(data, f)