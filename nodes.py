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
    CATEGORY = "AlphaFXTools"
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


class GlowRestoreAndCropSimple:
    """Recover glow from a black-background source, composite it, and crop once."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject_rgba": (
                    "IMAGE",
                    {"tooltip": "Klein + Keylight output. Keep it at the original canvas size."},
                ),
                "subject_mask": (
                    "MASK",
                    {"tooltip": "The subject mask from Keylight."},
                ),
                "original_black_image": (
                    "IMAGE",
                    {"tooltip": "Original image containing subject + glow on black."},
                ),
                "black_level": (
                    "FLOAT",
                    {
                        "default": 0.03,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.005,
                        "tooltip": "Suppress black-background noise. Raise it if speckles remain.",
                    },
                ),
                "edge_overlap": (
                    "INT",
                    {
                        "default": 6,
                        "min": 0,
                        "max": 128,
                        "step": 1,
                        "tooltip": "Allow restored glow to overlap the subject edge by this many pixels.",
                    },
                ),
                "effect_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
                ),
                "blend_mode": (["screen", "normal"], {"default": "screen"}),
                "crop_threshold": (
                    "FLOAT",
                    {
                        "default": 0.02,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.005,
                        "tooltip": "Alpha threshold used only to determine the final crop box.",
                    },
                ),
                "padding": (
                    "INT",
                    {"default": 8, "min": 0, "max": 4096, "step": 1},
                ),
            },
            "optional": {
                "effect_area_mask": (
                    "MASK",
                    {"tooltip": "Optional circular mask limiting where glow may be restored."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("rgba", "restored_effect", "final_alpha")
    FUNCTION = "restore"
    CATEGORY = "AlphaFXTools"
    DESCRIPTION = (
        "One-step node for restoring effects lost by background replacement: extracts glow "
        "from the original black-background image, removes the duplicate subject, composites "
        "the glow over the keyed subject, and crops the final RGBA."
    )

    @staticmethod
    def _resize_image(image, height, width):
        if image.shape[1:3] == (height, width):
            return image
        return F.interpolate(
            image.movedim(-1, 1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).movedim(1, -1)

    @staticmethod
    def _resize_mask(mask, height, width):
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[-2:] == (height, width):
            return mask
        return F.interpolate(
            mask.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    @staticmethod
    def _erode(mask, radius):
        if radius <= 0:
            return mask
        kernel = radius * 2 + 1
        return 1.0 - F.max_pool2d(
            (1.0 - mask).unsqueeze(1),
            kernel_size=kernel,
            stride=1,
            padding=radius,
        ).squeeze(1)

    @staticmethod
    def _over(subject_rgb, subject_alpha, effect_rgb, effect_alpha, blend_mode):
        subject_alpha_1 = subject_alpha.unsqueeze(-1)
        effect_alpha_1 = effect_alpha.unsqueeze(-1)
        output_alpha = (
            effect_alpha_1 + subject_alpha_1 - effect_alpha_1 * subject_alpha_1
        )

        if blend_mode == "screen":
            blended = 1.0 - (1.0 - subject_rgb) * (1.0 - effect_rgb)
            output_premult = (
                (1.0 - effect_alpha_1) * subject_rgb * subject_alpha_1
                + (1.0 - subject_alpha_1) * effect_rgb * effect_alpha_1
                + subject_alpha_1 * effect_alpha_1 * blended
            )
        else:
            output_premult = (
                effect_rgb * effect_alpha_1
                + subject_rgb * subject_alpha_1 * (1.0 - effect_alpha_1)
            )

        output_rgb = torch.where(
            output_alpha > 1e-6,
            output_premult / output_alpha.clamp_min(1e-6),
            torch.zeros_like(output_premult),
        )
        return output_rgb.clamp(0.0, 1.0), output_alpha.squeeze(-1).clamp(0.0, 1.0)

    def restore(
        self,
        subject_rgba,
        subject_mask,
        original_black_image,
        black_level,
        edge_overlap,
        effect_strength,
        blend_mode,
        crop_threshold,
        padding,
        effect_area_mask=None,
    ):
        if subject_rgba.ndim != 4 or subject_rgba.shape[-1] not in (3, 4):
            raise ValueError("subject_rgba must be BHWC RGB or RGBA")
        if original_black_image.ndim != 4 or original_black_image.shape[-1] < 3:
            raise ValueError("original_black_image must be BHWC RGB/RGBA")

        batch, height, width, subject_channels = subject_rgba.shape
        dtype = subject_rgba.dtype
        device = subject_rgba.device

        original_black_image = self._resize_image(
            original_black_image.to(device=device, dtype=dtype), height, width
        )
        subject_mask = self._resize_mask(
            subject_mask.to(device=device, dtype=dtype), height, width
        ).clamp(0.0, 1.0)

        if effect_area_mask is None:
            effect_area_mask = torch.ones(
                (1, height, width), device=device, dtype=dtype
            )
        else:
            effect_area_mask = self._resize_mask(
                effect_area_mask.to(device=device, dtype=dtype), height, width
            ).clamp(0.0, 1.0)

        outputs = []
        effects = []
        alphas = []

        for index in range(batch):
            source = original_black_image[min(index, original_black_image.shape[0] - 1), ..., :3]
            subject = subject_rgba[index, ..., :3].clamp(0.0, 1.0)
            mask = subject_mask[min(index, subject_mask.shape[0] - 1)]
            area = effect_area_mask[min(index, effect_area_mask.shape[0] - 1)]

            if subject_channels == 4:
                subject_alpha = subject_rgba[index, ..., 3].clamp(0.0, 1.0)
                # Keylight's explicit mask is authoritative while existing alpha
                # preserves any already-feathered edge details.
                subject_alpha = torch.minimum(subject_alpha, mask)
            else:
                subject_alpha = mask

            # Max-RGB unmult is appropriate for emissive effects rendered on black.
            raw_alpha = source.amax(dim=-1)
            extracted_alpha = (
                (raw_alpha - black_level) / max(1.0 - black_level, 1e-6)
            ).clamp(0.0, 1.0)
            effect_rgb = torch.where(
                raw_alpha.unsqueeze(-1) > 1e-6,
                source / raw_alpha.unsqueeze(-1).clamp_min(1e-6),
                torch.zeros_like(source),
            ).clamp(0.0, 1.0)

            # Shrinking the subject exclusion leaves a narrow overlap band so
            # glow touches the edge without copying the whole subject back in.
            eroded_subject = self._erode(mask.unsqueeze(0), edge_overlap).squeeze(0)
            outside_with_overlap = 1.0 - eroded_subject
            effect_alpha = (
                extracted_alpha * outside_with_overlap * area * effect_strength
            ).clamp(0.0, 1.0)

            output_rgb, output_alpha = self._over(
                subject, subject_alpha, effect_rgb, effect_alpha, blend_mode
            )
            output_rgba = torch.cat((output_rgb, output_alpha.unsqueeze(-1)), dim=-1)
            effect_rgba = torch.cat((effect_rgb, effect_alpha.unsqueeze(-1)), dim=-1)

            crop_foreground = output_alpha > crop_threshold
            if not torch.any(crop_foreground):
                raise ValueError(
                    f"final alpha for batch item {index} is empty at crop_threshold={crop_threshold}"
                )
            coordinates = torch.nonzero(crop_foreground, as_tuple=False)
            y_min = max(0, int(coordinates[:, 0].min().item()) - padding)
            y_max = min(height, int(coordinates[:, 0].max().item()) + 1 + padding)
            x_min = max(0, int(coordinates[:, 1].min().item()) - padding)
            x_max = min(width, int(coordinates[:, 1].max().item()) + 1 + padding)

            outputs.append(output_rgba[y_min:y_max, x_min:x_max])
            effects.append(effect_rgba[y_min:y_max, x_min:x_max])
            alphas.append(output_alpha[y_min:y_max, x_min:x_max])

        max_height = max(item.shape[0] for item in outputs)
        max_width = max(item.shape[1] for item in outputs)

        def pad_image(item):
            return F.pad(
                item,
                (0, 0, 0, max_width - item.shape[1], 0, max_height - item.shape[0]),
                value=0.0,
            )

        def pad_mask(item):
            return F.pad(
                item,
                (0, max_width - item.shape[1], 0, max_height - item.shape[0]),
                value=0.0,
            )

        return (
            torch.stack([pad_image(item) for item in outputs]),
            torch.stack([pad_image(item) for item in effects]),
            torch.stack([pad_mask(item) for item in alphas]),
        )


NODE_CLASS_MAPPINGS = {
    "ImageCropByMaskTrueAlpha": ImageCropByMaskTrueAlpha,
    "GlowRestoreAndCropSimple": GlowRestoreAndCropSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropByMaskTrueAlpha": "Image Crop By Mask (True Alpha)",
    "GlowRestoreAndCropSimple": "Glow Restore & Crop (Simple)",
}
