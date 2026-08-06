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

它把“完整 AE Unmult、可选圆形范围 Mask、辉光合成、最终裁剪”合并为一个节点。

只需要连接：

```text
subject_rgba         ← Klein 换绿底后经过 Keylight 的 image_rgba
subject_mask         ← Keylight 的主体 mask
original_black_image ← 换绿底前的原始黑底特效图
```

推荐初始参数：

```text
black_level: 0.0
edge_overlap: 6
effect_strength: 1.0
blend_mode: screen
crop_threshold: 0.02
padding: 8
remove_duplicate_subject: false
use_internal_circle: true
circle_size: 0.90
circle_feather: 48
```

输出：

- `rgba`：恢复辉光、重新合成并裁剪后的最终图片；
- `restored_effect`：完整 AE Unmult 层，默认同时保留主体以及覆盖主体、向外延伸的辉光；
- `final_alpha`：最终合成图片的透明度。

调整建议：

- 黑底噪点多：提高 `black_level`；
- 默认保持 `remove_duplicate_subject = false`，这样不会误删覆盖在主体上的辉光；
- 只有确实需要排除重复主体时才开启 `remove_duplicate_subject`，此时用 `edge_overlap` 控制保留的边缘重叠宽度；
- `circle_size`：内置圆形直径相对于画布短边的比例；小于 `1.0` 会在四周留下安全距离，避免圆形被画布裁成直线；
- `circle_feather`：圆形边缘向内羽化的像素宽度，数值越大过渡越柔和；
- 不需要圆形限制时关闭 `use_internal_circle`；节点不再需要外接圆形 Mask；
- 辉光太弱或太强：调整 `effect_strength`；
- 辉光裁不完整：降低 `crop_threshold` 或提高 `padding`。

内置圆形只限制允许恢复特效的区域；最终裁剪框始终根据合成后的 `final_alpha`
计算，因此会同时包含主体和向外延伸的辉光。
