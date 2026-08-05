# Image Crop By Mask (True Alpha)

这是一个独立的 ComfyUI 自定义节点，用来替换 KJNodes 的 `Image Crop By Mask`。

它会同时完成两件事：

1. 按 Mask 的非透明区域计算最小外接矩形并裁剪画布；
2. 把 Mask 写入输出图片的 Alpha 通道，使 Mask 外真正透明。

## 安装

把整个 `ComfyUI-CropByMaskTrueAlpha` 文件夹复制到：

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
