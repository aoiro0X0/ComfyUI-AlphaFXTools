# ComfyUI AlphaFXTools

用于真透明裁剪、Alpha 清理和黑底辉光恢复的 ComfyUI 节点工具集。

当前包含：

- `Image Crop By Mask (True Alpha)`
- `Glow Restore & Crop (Auto)`

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
Glow Restore & Crop (Auto)
```

这是为无人值守自动管线设计的版本。它不使用固定圆形，也不需要针对每张图调参数。

只需要连接：

```text
subject_rgba         ← Klein 换绿底后经过 Keylight 的 image_rgba
subject_mask         ← Keylight 的主体 mask
original_black_image ← 换绿底前的原始黑底特效图
```

节点会自动：

1. 从画布边缘估算每张图的真实黑底亮度；
2. 根据 `subject_mask` 的大小和形状生成自适应特效范围；
3. 完整保留主体附近的弱辉光，对远处像素使用亮度和颜色置信度过滤；
4. 将原始画布四边平滑归零，避免上下或左右直线；
5. 限制特效最多把裁剪框扩张到主体外约 25%，让主体保持靠近输出边缘；
6. 根据主体尺寸自动计算少量透明安全边距。

输出：

- `rgba`：恢复辉光、重新合成并裁剪后的最终图片；
- `restored_effect`：经过自动黑底估算、内容过滤和自适应范围限制后的特效层；
- `final_alpha`：最终合成图片的透明度。

自动策略偏向干净、稳定的批量输出。距离主体非常远且亮度很弱的像素可能被舍弃，
这是为了避免模板辅助线、灰边和背景噪声进入最终结果。
