import torch
from collections import Counter
from utils.detection_loss import bbox_iou


def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Arguments:
        recall:    The recall curve (list).
        precision: The precision curve (list).
    Returns:
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
    mpre = torch.cat((torch.tensor([0.0]), precision, torch.tensor([0.0])))

    # compute the precision envelope
    for i in range(mpre.shape[0] - 1, 0, -1):
        mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = torch.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=80
):
    """
    Calculates mean average precision
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we have
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will have {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then convert {0:3} to {0:torch.tensor([0,0,0])}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                # box_format is x1y1x2y2 if 'corners'
                # detection and gt row: [train_idx, class, prob, x1, y1, x2, y2]
                # we pass shape (4) and (4)

                # bbox_iou expects box1 as (4,), box2 as (4, N)
                # It does box2 = box2.T -> (N, 4)

                det_t = torch.tensor(detection[3:])
                gt_t = torch.tensor(gt[3:]).unsqueeze(
                    0
                )  # (1, 4) -> bbox_iou T -> (4, 1)

                # Wait, bbox_iou implementation:
                # box2 = box2.T
                # if box2 is (4, 1), box2.T is (1, 4) -> correct

                iou = bbox_iou(det_t, gt_t, x1y1x2y2=(box_format == "corners"))

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # check if gt bbox is already covered
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return (
        sum(average_precisions) / len(average_precisions)
        if len(average_precisions) > 0
        else 0.0
    )
