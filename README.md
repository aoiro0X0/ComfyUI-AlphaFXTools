# ComfyUI AlphaFXTools

用于真透明裁剪、Alpha 清理和黑底辉光恢复的 ComfyUI 节点工具集。

当前包含：

- `Image Crop By Mask (True Alpha)`
- `Glow Restore & Crop (Simple)`

它会同时完成两件事：

1. 按 Mask 的非透明区域计算最小外接矩形并裁剪画布；
2. 把 Mask 写入输出图片的 Alpha 通道，使 Mask 外真正透明。

## 安装

把整个 `ComfyUI-AlphaFXTools` 文件夹复制到：

```text
ComfyUI/custom_nodes/
```

重启 ComfyUI，然后搜索：

```text
Image Crop By Mask (True Alpha)
```

## 你的工作流推荐参数

```text
padding: 0
alpha_cutoff: 0.02
alpha_mode: replace
binary_mask: false
```

- `alpha_cutoff`：清除背景中残留的极低透明度。仍有暗雾时可提高到 `0.03～0.05`。
- `alpha_mode = replace`：直接使用 Keylight 输出的 Mask 作为 Alpha，避免重复乘 Alpha 导致边缘变暗。
- `binary_mask = false`：保留流苏和轮廓的抗锯齿、半透明细节。
- `padding`：在 Mask 外接矩形之外保留的像素边距。

## 接线

将原来进入 KJNodes `Image Crop By Mask` 的 `image` 和 `mask` 原样接入新节点。

输出：

- `rgba`：裁剪后的真透明 RGBA 图片；
- `cropped_mask`：同步裁剪后的 Mask。

图片文件本身仍然必须是矩形；所谓“沿物体轮廓裁剪”是通过矩形画布加 Alpha 透明通道实现的。

## 一步恢复辉光并裁剪

节点名称：

```text
Glow Restore & Crop (Simple)
```

它把“黑底 Unmult、排除重复主体、圆形 Mask、辉光合成、最终裁剪”合并为一个节点。

只需要连接：

```text
subject_rgba         ← Klein 换绿底后经过 Keylight 的 image_rgba
subject_mask         ← Keylight 的主体 mask
original_black_image ← 换绿底前的原始黑底特效图
effect_area_mask     ← 可选的圆形 Mask
```

推荐初始参数：

```text
black_level: 0.03
edge_overlap: 6
effect_strength: 1.0
blend_mode: screen
crop_threshold: 0.02
padding: 8
```

输出：

- `rgba`：恢复辉光、重新合成并裁剪后的最终图片；
- `restored_effect`：节点自动从黑底原图中提取的辉光层，方便检查；
- `final_alpha`：最终合成图片的透明度。

调整建议：

- 黑底噪点多：提高 `black_level`；
- 主体边缘出现重复影像：降低 `edge_overlap`；
- 辉光太弱或太强：调整 `effect_strength`；
- 辉光裁不完整：降低 `crop_threshold` 或提高 `padding`。
