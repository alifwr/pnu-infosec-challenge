import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    return iou


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.chunk(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = (
            torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        )  # shift x
        sy = (
            torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        )  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


class DetectionLoss(nn.Module):
    def __init__(self, nc=80):
        super().__init__()
        self.nc = nc  # Number of classes
        self.reg_max = 16
        self.strides = [8, 16, 32]  # Standard YOLO11/v8 strides

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.no = nc + self.reg_max * 4  # Output channels per grid

        # Loss weights
        self.box_gain = 7.5
        self.cls_gain = 0.5
        self.dfl_gain = 1.5

    def forward(self, preds, targets):
        """
        preds: List of tensors [(B, 64+nc, H, W), ...]
        targets: List of tensors [ (N, 5), ... ] where 5 is (cls, x, y, w, h)
        """
        loss = torch.zeros(3, device=preds[0].device)  # box, cls, dfl

        # 1. Preprocess predictions
        # Concatenate all scales -> (B, Total_Anchors, Channels)
        # e.g. 80x80 + 40x40 + 20x20 = 6400+1600+400 = 8400 anchors

        anchors, stride_tensor = make_anchors(preds, self.strides, 0.5)

        pred_dist_list = []
        pred_scores_list = []

        for i, pred in enumerate(preds):
            B, C, H, W = pred.shape
            # Split features: first 4*reg_max are box distribution, rest are classes
            box_ch = 4 * self.reg_max

            # Use view/permute to get (B, H*W, C)
            pred = pred.view(B, C, -1).permute(0, 2, 1).contiguous()

            pred_dist = pred[..., :box_ch].view(B, -1, 4, self.reg_max)
            pred_scores = pred[..., box_ch:]  # (B, H*W, nc)

            pred_dist_list.append(pred_dist)
            pred_scores_list.append(pred_scores)

        pred_dist = torch.cat(
            [p.view(B, -1, 4, self.reg_max) for p in pred_dist_list], 1
        )  # (B, Total_Anchors, 4, 16)
        pred_scores = torch.cat(pred_scores_list, 1)  # (B, Total_Anchors, nc)

        # Targets need to be batched and aligned
        targets_cat = []
        for i, t in enumerate(targets):
            t = t.to(preds[0].device)
            if len(t) > 0:
                # Add batch index: (idx, cls, x, y, w, h)
                idx = torch.full((len(t), 1), i, device=t.device)
                targets_cat.append(torch.cat((idx, t), 1))

        if len(targets_cat) == 0:
            return loss.sum()

        targets_cat = torch.cat(targets_cat, 0)

        # 2. Assignment (Simplified "Center" assignment for this demo)
        # Proper TAL is huge. We will use a simplified logic:
        # Assign GT to the grid cell containing its center AND correct scale.

        gt_labels = targets_cat[:, 1]
        gt_bboxes = targets_cat[:, 2:]  # xywh
        batch_idx = targets_cat[:, 0].long()

        target_bboxes = torch.zeros((B, anchors.shape[0], 4), device=anchors.device)
        target_scores = torch.zeros(
            (B, anchors.shape[0], self.nc), device=anchors.device
        )
        fg_mask = torch.zeros(
            (B, anchors.shape[0]), dtype=torch.bool, device=anchors.device
        )

        # Very simplified assignment: Nearest anchor to center
        # Expand anchors to batch
        # This is where proper TAL is usually needed.
        # For this quick fix, I will iterate targets and assign to nearest spatial anchor at the best scale
        # Or even simpler: Assign to ALL anchors inside the GT box (Center Sampling)

        for b_i in range(B):
            t_batch = targets_cat[targets_cat[:, 0] == b_i]
            if len(t_batch) == 0:
                continue

            # Filter valid classes
            t_batch = t_batch[t_batch[:, 1] < self.nc]

            if len(t_batch) == 0:
                continue

            t_cls = t_batch[:, 1].long()
            t_box = t_batch[:, 2:]  # xywh (normalized)

            # Convert normalized xywh to absolute
            # We assume images are 640x640 (stride * grid size)?
            # self.strides are [8, 16, 32]
            # anchors are in pixel coords (stride * grid_idx) + 0.5 * stride
            # We need image size. We can infer from preds.
            img_h = preds[0].shape[2] * self.strides[0]
            img_w = preds[0].shape[3] * self.strides[0]

            t_box_abs = t_box.clone()
            t_box_abs[:, 0] *= img_w
            t_box_abs[:, 1] *= img_h
            t_box_abs[:, 2] *= img_w
            t_box_abs[:, 3] *= img_h

            # Find anchors inside t_box centered
            # Let's just find the single closest anchor to center for robustness in this simple implementation

            gt_cx = t_box_abs[:, 0]
            gt_cy = t_box_abs[:, 1]

            # Dist to all anchors
            # anchors shape (N_A, 2)
            # gt shape (N_G, 2)
            dist = (anchors[:, 0:1] - gt_cx.unsqueeze(0)) ** 2 + (
                anchors[:, 1:2] - gt_cy.unsqueeze(0)
            ) ** 2

            # Find k nearest anchors (e.g. 10) or just nearest
            # Assign them
            min_dist, min_idx = torch.min(dist, 0)

            # We assign targets
            # target_scores[b_i, min_idx, t_cls] = 1.0 (Soft labels usually better but 1.0 ok)
            target_bboxes[b_i, min_idx] = t_box_abs
            target_scores[b_i, min_idx, t_cls] = 1.0
            fg_mask[b_i, min_idx] = True

        # 3. Loss Calculation

        # Box Loss (CIoU)
        # Decode pred boxes
        # pred_dist: (B, A, 4, 16)
        pred_dist_s = pred_dist.softmax(3)  # Softmax along reg_max
        pred_dist_val = pred_dist_s.matmul(
            torch.arange(16, device=anchors.device).float()
        )  # (B, A, 4) expectation

        # dist2bbox expects (B, A, 4) ltrb
        pred_bboxes = dist2bbox(
            pred_dist_val, anchors.unsqueeze(0), xywh=True
        )  # returns xywh

        if fg_mask.sum() > 0:
            # Only compute box loss on foreground
            target_bboxes_masked = target_bboxes[fg_mask]
            pred_bboxes_masked = pred_bboxes[fg_mask]

            iou = bbox_iou(
                pred_bboxes_masked.T, target_bboxes_masked, x1y1x2y2=False, CIoU=True
            )
            loss[0] = (1.0 - iou).mean() * self.box_gain

            # DFL Loss
            # Target dist
            # t_box (xywh) -> t_ltrb
            # t_ltrb = bbox2dist
            # We need to encode target box to ltrb relative to anchor
            t_xy = target_bboxes_masked[:, :2]
            t_wh = target_bboxes_masked[:, 2:]
            t_lt = t_xy - t_wh / 2
            t_rb = t_xy + t_wh / 2

            anchor_points_masked = anchors.unsqueeze(0).expand(B, -1, -1)[fg_mask]

            t_ltrb = torch.cat(
                (anchor_points_masked - t_lt, t_rb - anchor_points_masked), 1
            )
            t_ltrb = t_ltrb.clamp(
                0, self.reg_max - 1.01
            )  # Clip to reg_max - 1 to ensure tr < 16

            # Distributed Focal Loss
            # pred_dist_masked: (N_fg, 4, 16)
            pred_dist_masked = pred_dist[fg_mask]

            loss[2] = self.dfl_loss(pred_dist_masked, t_ltrb) * self.dfl_gain

        # Cls Loss (BCE)
        loss[1] = self.bce(pred_scores, target_scores).mean() * self.cls_gain

        return loss.sum()

    def dfl_loss(self, pred_dist, target):
        # Integral loss
        # target (N, 4) float
        # pred_dist (N, 4, 16) logits
        tl = target.long()  # left index
        tr = tl + 1  # right index
        wl = tr - target  # weight left
        wr = target - tl  # weight right

        return (
            F.cross_entropy(pred_dist.view(-1, 16), tl.view(-1), reduction="none").view(
                tl.shape
            )
            * wl
            + F.cross_entropy(
                pred_dist.view(-1, 16), tr.view(-1), reduction="none"
            ).view(tl.shape)
            * wr
        ).mean()
