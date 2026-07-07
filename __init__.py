# ComfyUI imports custom-node folders as packages. Keeping direct-file imports
# inert also allows the pure utility tests to run outside a ComfyUI checkout.
if __package__:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:  # pragma: no cover - only used by standalone test collection
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
