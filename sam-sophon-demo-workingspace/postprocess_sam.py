import numpy as np

def SamPostprocess(masks,original_sizes,reshaped_input_sizes, mask_threshold=0.0, binarize=True, pad_size=None):
    pad_size = pad_size if pad_size is not None else {"height": 1024, "width": 1024}
    target_image_size = (pad_size["height"], pad_size["width"])
    
    if isinstance(original_sizes, np.ndarray):
        original_sizes = original_sizes.tolist()
    if isinstance(reshaped_input_sizes, np.ndarray):
        reshaped_input_sizes = reshaped_input_sizes.tolist()

    output_masks = []
    for i, original_size in enumerate(original_sizes):
        if isinstance(masks[i], np.ndarray):
            masks[i] = torch.from_numpy(masks[i])
        elif not isinstance(masks[i], torch.Tensor):
            raise ValueError("Input masks should be a list of `torch.tensors` or a list of `np.ndarray`")
        interpolated_mask = F.interpolate(masks[i], target_image_size, mode="bilinear", align_corners=False)
        interpolated_mask = interpolated_mask[..., : reshaped_input_sizes[i][0], : reshaped_input_sizes[i][1]]
        interpolated_mask = F.interpolate(interpolated_mask, original_size, mode="bilinear", align_corners=False)
        if binarize:
            interpolated_mask = interpolated_mask > mask_threshold
        output_masks.append(interpolated_mask)

    return output_masks