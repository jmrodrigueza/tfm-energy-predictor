from sklearn.preprocessing import RobustScaler
import numpy as np


class RobustScalerSerializable(RobustScaler):
    """
    Class to wrap the RobustScaler from scikit-learn to make it serializable.
    """
    def __init__(self):
        """
        Constructor of the class.
        """
        super().__init__()

    def get_config(self):
        """
            Function to serialize the scaler parameters.
        Returns:
            dict: Dictionary with the serialized parameters of the scaler
        """
        return {
            "center_": self.center_.tolist() if hasattr(self, "center_") else None,
            "scale_": self.scale_.tolist() if hasattr(self, "scale_") else None,
        }

    @classmethod
    def from_config(cls, config):
        """
        Function to reconstruct the scaler from the serialized configuration.
        :param config: Configuration to reconstruct the scaler.
        :return: Instance of the class.
        """
        instance = cls()
        if config["center_"] is not None:
            instance.center_ = np.array(config["center_"])
        if config["scale_"] is not None:
            instance.scale_ = np.array(config["scale_"])
        return instance
