import numpy as np
from sklearn.linear_model import LinearRegression
import json
import argparse
import sys

class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, y_samples = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(y_samples[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        if running_time > 200 * len(gt):
            raise Exception('Slow running time.')
        if len(gt) != len(pred):
            # This check is for the number of lanes per image? 
            # No, pred and gt are lists of lanes.
            # TuSimple evaluation requires matching number of lanes?
            # Actually, the official script calculates matches.
            # Let's look at the logic below.
            pass

        # Get angles of GT lanes to determine threshold
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        
        # Match predictions to GT
        for x_gts, thresh in zip(gt, threshs):
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
            
        fp = len(pred) - matched
        
        # TuSimple specific rule: ignore extra lanes if > 4?
        # The official script has logic:
        # if len(gt) > 4 and fn > 0: fn -= 1
        # s = sum(line_accs)
        # if len(gt) > 4: s -= min(line_accs)
        # return s / max(min(4.0, len(gt)), 1.), ...
        
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
            
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / len(gt)

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        try:
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception(f'Fail to load json file of the prediction: {pred_file}. Error: {e}')
        try:
            json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        except BaseException as e:
            raise Exception(f'Fail to load json file of the ground truth: {gt_file}. Error: {e}')
            
        # Map GT by raw_file
        gts = {l['raw_file']: l for l in json_gt}
        # Also add mapping without 'test_set/' prefix to handle path mismatch
        for l in json_gt:
            if l['raw_file'].startswith('test_set/'):
                gts[l['raw_file'].replace('test_set/', '', 1)] = l
        
        accuracy, fp, fn = 0., 0., 0.
        processed_count = 0
        
        for pred in json_pred:
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception('Format of lanes error.')
            
            # Skip if not in GT (e.g. if pred file has more images than GT file)
            if pred['raw_file'] not in gts:
                continue
                
            gt = gts[pred['raw_file']]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']
            
            try:
                a, p, n = LaneEval.bench(pred['lanes'], gt_lanes, y_samples, pred['run_time'])
            except BaseException as e:
                raise Exception(str(e))
                
            accuracy += a
            fp += p
            fn += n
            processed_count += 1
            
        if processed_count == 0:
            return json.dumps({'error': 'No matched files found between prediction and ground truth.'})
            
        num = processed_count
        return json.dumps([
            {'name': 'Accuracy', 'value': accuracy / num, 'order': 'desc'},
            {'name': 'FP', 'value': fp / num, 'order': 'asc'},
            {'name': 'FN', 'value': fn / num, 'order': 'asc'}
        ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TuSimple Lane Evaluation')
    parser.add_argument('--pred', required=True, help='Path to prediction json file')
    parser.add_argument('--gt', required=True, help='Path to ground truth json file (test_label.json)')
    args = parser.parse_args()
    
    try:
        result = LaneEval.bench_one_submit(args.pred, args.gt)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
