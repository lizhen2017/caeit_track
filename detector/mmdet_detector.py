from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

__all__ = ['MMDETDetector']



class MMDETDetector:
    def __init__(self, config_file = 'configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e.py',
                 checkpoint_file = 'checkpoints/cascade_rcnn_hrnetv2p_w32_20e_20190522-55bec4ee.pth',
                 device='cuda:0'):
        self._model = self._init_model(config_file, checkpoint_file, device)
        self.CLASSES  = self._model.CLASSES

    @staticmethod
    def _init_model(config_file, checkpoint_file, device):
        model = init_detector(config_file, checkpoint_file, device=device)
        return model

    def __call__(self, img):
        result = inference_detector(self._model, img)
        return result

if __name__ == '__main__':
    img_path = 'demo/coco/000000000785.jpg'
    img_array = mmcv.imread(img_path)
    detector = MMDETDetector()
    result = detector(img_array)
    show_result(img_array, result, detector.CLASSES, wait_time=1)


