import json

class Config():
    def get_dict(self):
        """
        Get all attributes of the class as a dictionary in JSON format.
        """
        attributes = {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        }
        return json.dumps(attributes, indent=4)  # Convert to JSON format
