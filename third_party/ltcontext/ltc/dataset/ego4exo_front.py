from ltc.dataset.video_dataset import VideoDataset
import ltc.utils.logging as logging

logger = logging.get_logger(__name__)


class Ego4ExoFront(VideoDataset):
    def __init__(self, cfg, mode):
        super(Ego4ExoFront, self).__init__(cfg, mode)
        logger.info(
            "Constructing Ego4Exo_front {} dataset with {} videos.".format(
                mode, self._dataset_size
            )
        )
