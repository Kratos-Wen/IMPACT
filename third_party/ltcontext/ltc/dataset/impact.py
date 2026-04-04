from ltc.dataset.video_dataset import VideoDataset
import ltc.utils.logging as logging

logger = logging.get_logger(__name__)


class Impact(VideoDataset):
    def __init__(self, cfg, mode):
        super(Impact, self).__init__(cfg, mode)
        logger.info(
            "Constructing IMPACT {} dataset with {} videos.".format(
                mode, self._dataset_size
            )
        )
