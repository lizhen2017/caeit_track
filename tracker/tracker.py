from __future__ import division

import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.bbox import box_iou
from seacher.siam_searcher import SiamSearcher


class Track(object):
    def __init__(self, track_id, class_id, score, bbox):
        self.track_id = track_id
        self.skipped_frames = 0
        self.class_id = class_id
        self.score = score
        self.bbox = bbox
        self.searcher = None

    def update(self, bbox=None, score=None, smooth=False):
        if bbox is not None:
            if smooth:
                self.bbox = (self.bbox + bbox) / 2
            else:
                self.bbox = bbox
        if score is not None:
            self.score = score



class Tracker(object):
    def __init__(self, searcher_config_file, searcher_checkpoint_file, track_id_init=0, cost_thr=0.7, skip_thr=10,
                 score_thr=0.3):
        self.tracks = []
        self.track_id_counter = track_id_init
        self.cost_thr = cost_thr
        self.skip_thr = skip_thr
        self.score_thr = score_thr
        self.cur_frame = None
        self.searcher_config_file = searcher_config_file
        self.searcher_checkpoint_file = searcher_checkpoint_file
        self.track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                             (0, 255, 255), (255, 0, 255), (255, 127, 255), (127, 0, 255), (127, 0, 127)]

    def _add_one_track(self, class_id, score, bbox):
        track = Track(self.track_id_counter, class_id, score, bbox)
        self.track_id_counter += 1
        self.tracks.append(track)

    def _update_one_track(self, track_id, score, bbox):
        track = self.tracks[track_id]
        track.update(bbox, score, True)

    def _init_tracker(self, class_ids, scores, bboxes):
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            self._add_one_track(class_id, score, bbox)

    def init_searcher(self):
        for track in self.tracks:
            if track.searcher is None:
                track.searcher = SiamSearcher(self.searcher_config_file, self.searcher_checkpoint_file,
                                              score_thr=self.score_thr)
            track.searcher.reset(self.cur_frame, track.bbox)

    def _match_by_iou(self, cur_bboxes):
        # print(cur_bboxes)
        bef_bboxes = np.vstack([track.bbox for track in self.tracks])
        iou_mat = box_iou(bef_bboxes, cur_bboxes)
        cost_mat = 1 - iou_mat
        assignments = []
        for _ in range(len(self.tracks)):
            assignments.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        for i in range(len(row_ind)):
            assignments[row_ind[i]] = col_ind[i]
        # print(assignments)
        return assignments, cost_mat

    def _postprocess_iou_matching(self, assignments, cost_mat):
        for i in range(len(assignments)):
            if assignments[i] != -1:
                if cost_mat[i][assignments[i]] > self.cost_thr:
                    assignments[i] = -1
        return assignments

    def _deal_skip_too_much(self, assignments=None, mode='del'):
        tmp_tracks = []
        tmp_assignments = []
        if mode == 'del':
            for idx in range(len(self.tracks)):
                if self.tracks[idx].skipped_frames < self.skip_thr:
                    tmp_tracks.append(self.tracks[idx])
                if assignments:
                    tmp_assignments.append(assignments[idx])
            return tmp_tracks, tmp_assignments
        else:
            raise NotImplementedError

    def _deal_unassigned_cur(self, cur_class_ids, cur_scores, cur_bboxes, assignments):
        for i in range(cur_bboxes.shape[0]):
            if i not in assignments:
                self._add_one_track(cur_class_ids[i], cur_scores[i], cur_bboxes[i])

    def _search(self, track_id):
        track = self.tracks[track_id]
        searcher = track.searcher
        search_bbox, search_score = searcher.search(self.cur_frame)
        if search_bbox.shape[0] > 0:
            return search_bbox, search_score
        else:
            return search_bbox, search_score

    def _deal_unassigned_track(self, track_id):
        return self._search(track_id)

    def _track_search(self):
        for idx in range(len(self.tracks)):
            search_bbox, search_score = self._search(idx)
            if search_score.shape[0] > 0:
                self.tracks[idx].update(search_bbox[0], search_score[0])
                self.tracks[idx].skipped_frames = 0
            else:
                self.tracks[idx].skipped_frames += self.skip_thr
                self.tracks[idx].update(None, np.array([self.score_thr]))
        tmp_tracks, _ = self._deal_skip_too_much()
        self.tracks = tmp_tracks

    def track(self, cur_frame, cur_class_ids=None, cur_scores=None, cur_bboxes=None, search=True):
        self.cur_frame = cur_frame
        if search:
            self._track_search()
        else:
            if len(self.tracks) == 0:
                self._init_tracker(cur_class_ids, cur_scores, cur_bboxes)
                return cur_bboxes, cur_scores, cur_class_ids, [self.track_colors[track.track_id]
                                                               for track in self.tracks]
            self._track_search()
            assignments, cost_mat = self._match_by_iou(cur_bboxes)
            assignments = self._postprocess_iou_matching(assignments, cost_mat)
            self._deal_unassigned_cur(cur_class_ids, cur_scores, cur_bboxes, assignments)
            for idx in range(len(assignments)):
                assignment = assignments[idx]
                if assignment != -1:
                    self._update_one_track(idx, cur_scores[assignment], cur_bboxes[assignment])
                    self.tracks[idx].skipped_frames = 0
                else:
                    self._deal_unassigned_track(idx)
                    self.tracks[idx].skipped_frames += 1
            tmp_tracks, assignments = self._deal_skip_too_much()
            self.tracks = tmp_tracks
        return np.vstack([track.bbox for track in self.tracks]), np.vstack([track.score for track in self.tracks]),\
                np.array([track.class_id for track in self.tracks]), [self.track_colors[track.track_id]
                                                                      for track in self.tracks]

if __name__ == '__main__':
    import os
    import mmcv
    from utils.viz.image import imshow_det_bboxes
    from detector.mmdet_detector import MMDETDetector
    detector = MMDETDetector(config_file='../configs/detector/configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e.py',
                             checkpoint_file='../configs/detector/checkpoints/hrnet/cascade_rcnn_hrnetv2p_w32_20e_20190522-55bec4ee.pth',
                             score_thr=0.3,
                             det_person=True)
    tracker = Tracker(searcher_config_file='../configs/searcher/siamrpn_mobilev2_l234_dwxcorr/config.yaml',
                      searcher_checkpoint_file='../configs/searcher/siamrpn_mobilev2_l234_dwxcorr/model.pth',
                      score_thr=0.6)
    det_fre = 5
    # video_dir = '../demo/video/v_Surfing_g25_c02'
    # for i, img_name in enumerate(sorted(os.listdir(video_dir))):
    #     if not img_name.endswith('jpg'):
    #         continue
    #       img_path = os.path.join(video_dir, img_name)
    #       frame_array = mmcv.imread(img_path)
    video = mmcv.VideoReader('../demo/video/contest/001.mp4')
    for i, frame_array in enumerate(video):
        img_path = '../result/video/contest/001/{}.jpg'.format(str(i).zfill(5))
        if i > 30:
            break
        if i % det_fre == 0:
            bboxes, scores, class_ids = detector(frame_array)
            bboxes, scores, class_ids, track_colors = tracker.track(cur_frame=frame_array, cur_bboxes=bboxes,
                                                                    cur_scores=scores, cur_class_ids=class_ids,
                                                                    search=False)
            imshow_det_bboxes(
                frame_array,
                bboxes,
                scores,
                class_ids,
                class_names=detector.CLASSES,
                score_thr=0,
                show=False,
                wait_time=1,
                out_file='{}'.format(img_path.replace('demo', 'result')),
                bbox_colors=track_colors,
                text_colors=track_colors)
            tracker.init_searcher()
        else:
            bboxes, scores, class_ids, track_colors = tracker.track(frame_array)
            imshow_det_bboxes(
                frame_array,
                bboxes,
                scores,
                class_ids,
                class_names=detector.CLASSES,
                score_thr=0,
                show=False,
                wait_time=1,
                out_file='{}'.format(img_path.replace('demo', 'result')),
                bbox_colors=track_colors,
                text_colors=track_colors)
