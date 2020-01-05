from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
from utils.viz.image import imshow_det_bboxes
import numpy as np

__all__ = ['MMDETDetector']

class MMDETDetector:
    def __init__(self, config_file,
                 checkpoint_file,
                 device='cuda:0', score_thr=0, det_person=True):
        self._model = self._init_model(config_file, checkpoint_file, device)
        self.CLASSES  = self._model.CLASSES
        self.score_thr = score_thr
        self._det_person = det_person

    @staticmethod
    def _init_model(config_file, checkpoint_file, device):
        model = init_detector(config_file, checkpoint_file, device=device)
        return model

    def bbox_post_process(self, result):
        """
        :param result: a list 80x[N, 5]
        :return:
            numpy array [M, 5] (x1, y1, x2, y2, score)
        """
        if self._det_person:
            bboxes = result[0]
            class_ids = np.full(bboxes.shape[0], 0)
        else:
            class_ids = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(result)
            ]
            class_ids = np.concatenate(class_ids)
            bboxes = np.vstack(result)
        scores = bboxes[:, -1:]
        bboxes =  bboxes[:, :-1]
        if self.score_thr > 0:
            inds = scores[:, 0] > self.score_thr
            bboxes = bboxes[inds, :]
            scores = scores[inds, :]
            class_ids = class_ids[inds]
        return bboxes, scores, class_ids

    def __call__(self, img):
        result = inference_detector(self._model, img)
        bboxes, scores, class_ids = self.bbox_post_process(result)
        return bboxes, scores, class_ids

if __name__ == '__main__':
    img_path = '../demo/video/v_Surfing_g25_c02/frame000001.jpg'
    img_array = mmcv.imread(img_path)
    detector = MMDETDetector(config_file='../configs/detector/configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e.py',
                             checkpoint_file='../configs/detector/checkpoints/hrnet/cascade_rcnn_hrnetv2p_w32_20e_20190522-55bec4ee.pth',
                             score_thr=0.4,
                             det_person=True)
    bboxes, scores, class_ids = detector(img_array)
    imshow_det_bboxes(
        img_array,
        bboxes,
        scores,
        class_ids,
        class_names=detector.CLASSES,
        score_thr=0,
        show=False,
        wait_time=1,
        out_file='result2.jpg')