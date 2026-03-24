import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):

    warped_image = np.array(image).copy()
    h, w = warped_image.shape[:2]

    n = min(len(source_pts), len(target_pts))
    if n == 0:
        return warped_image

    source_pts = np.asarray(source_pts[:n], dtype=np.float32)
    target_pts = np.asarray(target_pts[:n], dtype=np.float32)

    # 逆向映射：输出图像 target 位置，从输入图像 source 位置取样
    disp_all = source_pts - target_pts  
    move_norm = np.linalg.norm(disp_all, axis=1)

    # 把“几乎不动”的点对当作限制区域边界点
    anchor_thresh = 2.0
    anchor_mask = move_norm < anchor_thresh
    deform_mask = ~anchor_mask

    # 真正用于驱动形变的点
    deform_src = source_pts[deform_mask]
    deform_tgt = target_pts[deform_mask]
    deform_disp = disp_all[deform_mask]

    # 限制区域点（source≈target）
    region_pts = target_pts[anchor_mask]

    # 如果没有真正的形变点，直接返回
    if len(deform_tgt) == 0:
        return warped_image

    # 为了让位移场在边界区域稳定，把限制区域点也作为 0 位移控制点加入
    if len(region_pts) > 0:
        ctrl_pts = np.concatenate([deform_tgt, region_pts], axis=0)
        ctrl_disp = np.concatenate(
            [deform_disp, np.zeros((len(region_pts), 2), dtype=np.float32)],
            axis=0
        )
    else:
        ctrl_pts = deform_tgt
        ctrl_disp = deform_disp

    # Gaussian RBF: phi(r) = exp(-r^2 / (2 sigma^2))
    # sigma 根据嘴部局部区域大小来定
    if len(region_pts) >= 3:
        bbox_min = region_pts.min(axis=0)
        bbox_max = region_pts.max(axis=0)
    else:
        all_pts = deform_tgt
        bbox_min = all_pts.min(axis=0)
        bbox_max = all_pts.max(axis=0)

    region_size = np.linalg.norm(bbox_max - bbox_min)
    sigma = max(region_size * 0.35, 12.0)

    # 控制点核矩阵
    diff = ctrl_pts[:, None, :] - ctrl_pts[None, :, :]      # [N, N, 2]
    dist2 = np.sum(diff ** 2, axis=2)                       # [N, N]
    K = np.exp(-dist2 / (2 * sigma * sigma)).astype(np.float32)

    # 稳定项
    K += 1e-5 * np.eye(K.shape[0], dtype=np.float32)

    coeff_x = np.linalg.solve(K, ctrl_disp[:, 0])
    coeff_y = np.linalg.solve(K, ctrl_disp[:, 1])

    # 稠密位移场 
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )
    query = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # [H*W, 2]

    diff_q = query[:, None, :] - ctrl_pts[None, :, :]           # [HW, N, 2]
    dist2_q = np.sum(diff_q ** 2, axis=2)                       # [HW, N]
    Phi = np.exp(-dist2_q / (2 * sigma * sigma)).astype(np.float32)

    disp_x = (Phi @ coeff_x).reshape(h, w)
    disp_y = (Phi @ coeff_y).reshape(h, w)

    # 如果给了“围嘴巴一圈”的限制点，就只在该区域内应用位移
    if len(region_pts) >= 3:
        hull = cv2.convexHull(region_pts.astype(np.int32))
        local_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(local_mask, hull, 255)

        # 让边缘过渡更自然：先膨胀一点，再高斯模糊
        k = max(int(region_size * 0.08), 5)
        if k % 2 == 0:
            k += 1

        kernel = np.ones((k, k), np.uint8)
        local_mask = cv2.dilate(local_mask, kernel, iterations=1)
        local_mask = cv2.GaussianBlur(local_mask, (k, k), 0)
        local_mask = local_mask.astype(np.float32) / 255.0
    else:
        # 没有圈限制区域时，退化为基于控制点的局部影响
        # 距离控制点越远，影响越小
        local_mask = np.zeros((h, w), dtype=np.float32)
        for pt in deform_tgt:
            dx = grid_x - pt[0]
            dy = grid_y - pt[1]
            d2 = dx * dx + dy * dy
            local_mask += np.exp(-d2 / (2 * (1.8 * sigma) * (1.8 * sigma)))
        local_mask = np.clip(local_mask, 0.0, 1.0)

    # 只在局部区域应用位移
    map_x = grid_x + alpha * local_mask * disp_x
    map_y = grid_y + alpha * local_mask * disp_y

    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, h - 1).astype(np.float32)

    warped = cv2.remap(
        warped_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )

    # 再和原图按 mask 混合，保证区域外完全不变
    local_mask_3 = local_mask[..., None]
    warped_image = (local_mask_3 * warped + (1.0 - local_mask_3) * warped_image).astype(np.uint8)

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
