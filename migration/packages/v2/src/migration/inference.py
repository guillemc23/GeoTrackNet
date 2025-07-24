"""© Indra Space - 2025."""


from importlib.resources import files

from migration import saved_models
from migration.models.vrnn_fivo import VRNNboundFIVO
from migration.utils import logger, timed

_MODEL_WEIGHTS = 'vrnn.weights.h5'
class GeotrackNet:
    def __init__(self):
        self.model = self.__load_model()
        logger.info("Geotracknet initialized")

    @timed
    def __load_model(self) -> VRNNboundFIVO:
        logger.info("Loading model...")

        model_path = files(saved_models).joinpath(_MODEL_WEIGHTS)
        if model_path.exists():


            return network

        else:
            logger.error(f"Could not find model at {model_path}")
