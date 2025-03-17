from your_sam_clip_pipeline import segment_with_sam, classify_with_clip

def run_sam_clip_pipeline(image, conf_thresh=0.65):
    masks = segment_with_sam(image)
    labeled_segments = classify_with_clip(image, masks, conf_thresh=conf_thresh)
    return labeled_segments
