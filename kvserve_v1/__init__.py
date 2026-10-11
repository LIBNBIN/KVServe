"""KVServe compression library and optional vLLM connector."""

__all__ = ["CompressedKVConnector"]

def __getattr__(name):
    if name == "CompressedKVConnector":
        from kvserve_v1.connector import CompressedKVConnector
        return CompressedKVConnector
    raise AttributeError(name)
