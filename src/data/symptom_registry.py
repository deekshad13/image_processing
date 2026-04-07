from src.config import get_config


def get_symptom_map():
    """Returns dict: symptom_id -> {raw_folder, display_name, embedding_folder}"""
    cfg = get_config()
    result = {}
    for symptom_id, info in cfg["data"]["symptoms"].items():
        result[symptom_id] = {
            "raw_folder": info["raw_folder"],
            "display_name": info["display_name"],
            "embedding_folder": info["raw_folder"].replace(" ", "_"),
        }
    return result


def get_symptom_ids():
    return list(get_symptom_map().keys())


def resolve_symptom_id(symptom_id):
    return get_symptom_map().get(symptom_id.lower())
