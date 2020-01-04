import mmcv
import torch
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from detector.mmdet_detector import MMDETDetector
from utils.bbox import xyxy2xywh, xywh2xyxy
from utils.viz.image import imshow_det_bboxes
import numpy as np

__all__ = ['SiamSearcher']

class SiamSearcher:
    def __init__(self, config_file, checkpoint_file, device='cuda:0'):
        self._model = self._init_model(config_file, checkpoint_file, device)

    def _init_model(self, config_file, checkpoint_file, device):
        cfg.merge_from_file(config_file)
        cfg.CUDA = torch.cuda.is_available()
        device = torch.device(device)
        # create model
        model = ModelBuilder()
        # load model
        model.load_state_dict(torch.load(checkpoint_file,
                                         map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(device)
        searcher = build_tracker(model)
        return searcher

    def search(self, img):
        outputs = self._model.track(img)
        bbox = list(map(int, outputs['bbox']))
        score = outputs['best_score']
        bbox = np.array(bbox)[np.newaxis, :]
        score = np.array([score])[np.newaxis, :]
        return xywh2xyxy(bbox), score

    def reset(self, img, bbox):
        self._model.init(img, bbox)

    @staticmethod
    def bbox_to_siam(bboxes, mode='mmdet'):
        if mode == 'mmdet':
            bboxes = xyxy2xywh(bboxes)
            return bboxes
        else:
            raise NotImplementedError

    def siam_to_det(self, bboxes, mode='mmdet'):
        if mode == 'mmdet':
            bboxes = xyxy2xywh(bboxes)
            return bboxes
        else:
            raise NotImplementedError


if __name__ == '__main__':
    det_array = mmcv.imread('../demo/video/v_Surfing_g25_c02/frame000001.jpg')
    search_array = mmcv.imread('../demo/video/v_Surfing_g25_c02/frame000002.jpg')
    detector = MMDETDetector(config_file='../configs/detector/configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e.py',
                             checkpoint_file='../configs/detector/checkpoints/hrnet/cascade_rcnn_hrnetv2p_w32_20e_20190522-55bec4ee.pth',
                             score_thr=0.4,
                             det_person=True)
    bboxes, scores, class_ids = detector(det_array)
    imshow_det_bboxes(
        det_array,
        bboxes,
        scores,
        class_ids,
        class_names=detector.CLASSES,
        score_thr=0,
        show=False,
        wait_time=1,
        out_file='det.jpg')
    searcher = SiamSearcher(config_file='../configs/searcher/siamrpn_mobilev2_l234_dwxcorr/config.yaml',
                            checkpoint_file='../configs/searcher/siamrpn_mobilev2_l234_dwxcorr/model.pth')
    if bboxes.shape[0] > 0:
        bboxes_siam = searcher.bbox_to_siam(bboxes)
        first_bbox = bboxes_siam[0]
        first_class_id = class_ids[0:1]
        searcher.reset(det_array, first_bbox)
        first_search_bbox, first_search_score = searcher.search(search_array)
        # print(first_search_bbox, first_search_score)
        imshow_det_bboxes(
            search_array,
            first_search_bbox,
            first_search_score,
            first_class_id,
            class_names=detector.CLASSES,
            score_thr=0,
            show=False,
            wait_time=1,
            out_file='search.jpg')