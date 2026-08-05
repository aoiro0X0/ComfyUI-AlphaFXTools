import torch
import torch.nn.functional as F


class ImageCropByMaskTrueAlpha:
    """Crop to a mask bounding box and make pixels outside the mask transparent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 1},
                ),
                "alpha_cutoff": (
                    "FLOAT",
                    {
                        "default": 0.02,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Mask values at or below this are made fully transparent.",
                    },
                ),
                "alpha_mode": (
                    ["replace", "multiply", "preserve"],
                    {
                        "default": "replace",
                        "tooltip": (
                            "replace: use mask as alpha; multiply: source alpha × mask; "
                            "preserve: keep source alpha inside mask."
                        ),
                    },
                ),
                "binary_mask": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Hard edge. Leave disabled to preserve anti-aliased edges.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgba", "cropped_mask")
    FUNCTION = "crop"
    CATEGORY = "image/crop"
    DESCRIPTION = (
        "Crops to the mask bounding box and writes the mask into a true alpha channel. "
        "Low mask values can be cleared to remove faint transparent haze."
    )

    @staticmethod
    def _match_mask(mask, height, width):
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[-2:] != (height, width):
            mask = F.interpolate(
                mask.unsqueeze(1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        return mask.clamp(0.0, 1.0)

    def crop(self, image, mask, padding, alpha_cutoff, alpha_mode, binary_mask):
        if image.ndim != 4 or image.shape[-1] not in (3, 4):
            raise ValueError("image must be BHWC RGB or RGBA")

        batch, height, width, channels = image.shape
        mask = self._match_mask(mask, height, width).to(
            device=image.device, dtype=image.dtype
        )

        cropped_images = []
        cropped_masks = []

        for index in range(batch):
            current_mask = mask[min(index, mask.shape[0] - 1)]
            foreground = current_mask > alpha_cutoff

            if not torch.any(foreground):
                raise ValueError(
                    f"mask for batch item {index} is empty at alpha_cutoff={alpha_cutoff}"
                )

            coordinates = torch.nonzero(foreground, as_tuple=False)
            y_min = max(0, int(coordinates[:, 0].min().item()) - padding)
            y_max = min(height, int(coordinates[:, 0].max().item()) + 1 + padding)
            x_min = max(0, int(coordinates[:, 1].min().item()) - padding)
            x_max = min(width, int(coordinates[:, 1].max().item()) + 1 + padding)

            cropped_image = image[index, y_min:y_max, x_min:x_max]
            cropped_mask = current_mask[y_min:y_max, x_min:x_max]

            if binary_mask:
                cropped_mask = (cropped_mask > alpha_cutoff).to(image.dtype)
            else:
                cropped_mask = torch.where(
                    cropped_mask > alpha_cutoff,
                    cropped_mask,
                    torch.zeros_like(cropped_mask),
                )

            rgb = cropped_image[..., :3]
            if channels == 4:
                source_alpha = cropped_image[..., 3]
            else:
                source_alpha = torch.ones_like(cropped_mask)

            if alpha_mode == "replace":
                alpha = cropped_mask
            elif alpha_mode == "multiply":
                alpha = source_alpha * cropped_mask
            else:
                alpha = torch.where(
                    cropped_mask > 0,
                    source_alpha,
                    torch.zeros_like(source_alpha),
                )

            alpha = alpha.clamp(0.0, 1.0)
            # Clear hidden RGB where alpha is zero so later resize/composite nodes
            # cannot spread invisible black/grey pixels into the visible edge.
            rgb = torch.where(alpha.unsqueeze(-1) > 0, rgb, torch.zeros_like(rgb))
            cropped_images.append(torch.cat((rgb, alpha.unsqueeze(-1)), dim=-1))
            cropped_masks.append(alpha)

        # ComfyUI batches must share one tensor size. Single-image batches are
        # already tight; multi-image batches are zero-padded to the largest crop.
        max_height = max(item.shape[0] for item in cropped_images)
        max_width = max(item.shape[1] for item in cropped_images)
        output_images = []
        output_masks = []
        for rgba, alpha in zip(cropped_images, cropped_masks):
            pad_bottom = max_height - rgba.shape[0]
            pad_right = max_width - rgba.shape[1]
            output_images.append(
                F.pad(rgba, (0, 0, 0, pad_right, 0, pad_bottom), value=0.0)
            )
            output_masks.append(
                F.pad(alpha, (0, pad_right, 0, pad_bottom), value=0.0)
            )

        return (torch.stack(output_images), torch.stack(output_masks))


NODE_CLASS_MAPPINGS = {
    "ImageCropByMaskTrueAlpha": ImageCropByMaskTrueAlpha,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropByMaskTrueAlpha": "Image Crop By Mask (True Alpha)",
}
