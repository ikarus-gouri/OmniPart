import gradio as gr
import spaces
import os
import numpy as np
import trimesh
import time
import traceback
import torch
from PIL import Image
import cv2
import shutil
from segment_anything import SamAutomaticMaskGenerator, build_sam
from omegaconf import OmegaConf

from modules.bbox_gen.models.autogressive_bbox_gen import BboxGen
from modules.part_synthesis.process_utils import save_parts_outputs
from modules.inference_utils import load_img_mask, prepare_bbox_gen_input, prepare_part_synthesis_input, gen_mesh_from_bounds, vis_voxel_coords, merge_parts
from modules.part_synthesis.pipelines import OmniPartImageTo3DPipeline
from modules.label_2d_mask.visualizer import Visualizer

from modules.label_2d_mask.label_parts import (
    get_sam_mask, 
    get_mask, 
    clean_segment_edges,
    resize_and_pad_to_square,
    size_th as DEFAULT_SIZE_TH
)

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
MAX_SEED = np.iinfo(np.int32).max
TMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(TMP_ROOT, exist_ok=True)

# Keep SAM lightweight on low-VRAM GPUs; can be overridden via env vars.
SAM_MIN_CUDA_VRAM_GB = float(os.getenv("OMNIPART_SAM_MIN_CUDA_VRAM_GB", "10"))
SAM_GENERATOR_CONFIG = {
    "points_per_side": int(os.getenv("OMNIPART_SAM_POINTS_PER_SIDE", "16")),
    "points_per_batch": int(os.getenv("OMNIPART_SAM_POINTS_PER_BATCH", "32")),
    "crop_n_layers": int(os.getenv("OMNIPART_SAM_CROP_N_LAYERS", "0")),
    "crop_n_points_downscale_factor": int(os.getenv("OMNIPART_SAM_CROP_DOWNSCALE", "2")),
}

sam_mask_generator = None
sam_device = None
bbox_gen_model = None
part_synthesis_pipeline = None
bbox_gen_ckpt_path_cached = None
partfield_ckpt_path_cached = None

size_th = DEFAULT_SIZE_TH


def prepare_models(sam_ckpt_path, partfield_ckpt_path, bbox_gen_ckpt_path):
    global sam_mask_generator, sam_device
    global bbox_gen_ckpt_path_cached, partfield_ckpt_path_cached

    bbox_gen_ckpt_path_cached = bbox_gen_ckpt_path
    partfield_ckpt_path_cached = partfield_ckpt_path

    if sam_mask_generator is None:
        print("Loading SAM model...")
        forced_sam_device = os.getenv("OMNIPART_SAM_DEVICE")
        if forced_sam_device is not None:
            sam_device = forced_sam_device
        elif DEVICE != "cuda":
            sam_device = DEVICE
        else:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            sam_device = "cuda" if total_vram_gb >= SAM_MIN_CUDA_VRAM_GB else "cpu"
            if sam_device == "cpu":
                print(
                    f"Low VRAM detected ({total_vram_gb:.1f} GB). "
                    "Running SAM on CPU to avoid CUDA OOM."
                )

        sam_model = build_sam(checkpoint=sam_ckpt_path).to(device=sam_device)
        sam_model.eval()
        sam_mask_generator = SamAutomaticMaskGenerator(sam_model, **SAM_GENERATOR_CONFIG)

    print("Core models ready (SAM). Generation models load on demand.")


def _generate_sam_masks_with_fallback(image):
    global sam_device
    try:
        return sam_mask_generator.generate(image)
    except torch.OutOfMemoryError:
        if sam_device == "cuda":
            print("SAM hit CUDA OOM. Retrying SAM mask generation on CPU...")
            torch.cuda.empty_cache()
            sam_mask_generator.predictor.model.to("cpu")
            sam_device = "cpu"
            return sam_mask_generator.generate(image)
        raise


def _prepare_image_with_sam_alpha(image):
    """Build an RGBA image by using SAM masks as foreground alpha."""
    rgb_square = resize_and_pad_to_square(image.convert("RGB"))
    rgb_np = np.array(rgb_square)

    sam_masks = _generate_sam_masks_with_fallback(rgb_np)
    min_area = int(os.getenv("OMNIPART_SAM_FG_MIN_AREA", "256"))
    max_area_ratio = float(os.getenv("OMNIPART_SAM_FG_MAX_AREA_RATIO", "0.85"))
    max_border_touch_ratio = float(os.getenv("OMNIPART_SAM_FG_MAX_BORDER_TOUCH_RATIO", "0.12"))
    keep_components = int(os.getenv("OMNIPART_SAM_FG_KEEP_COMPONENTS", "3"))

    h, w = rgb_np.shape[:2]
    frame_area = float(h * w)
    border_len = float(max(1, (2 * h + 2 * w - 4)))

    alpha_mask = np.zeros(rgb_np.shape[:2], dtype=np.uint8)
    area_sorted_masks = sorted(sam_masks, key=lambda x: x["area"], reverse=True)
    selected_masks = []

    for mask_data in area_sorted_masks:
        area = float(mask_data["area"])
        if area < min_area:
            continue

        area_ratio = area / frame_area
        mask = mask_data["segmentation"]
        border_touch = (
            np.sum(mask[0, :])
            + np.sum(mask[-1, :])
            + np.sum(mask[1:-1, 0])
            + np.sum(mask[1:-1, -1])
        )
        border_touch_ratio = float(border_touch) / border_len

        # Skip likely background masks: too large or too strongly attached to image borders.
        if area_ratio > max_area_ratio or border_touch_ratio > max_border_touch_ratio:
            continue

        selected_masks.append(mask)

    if not selected_masks and len(area_sorted_masks) > 0:
        # Fallback: choose masks that are smallest-border and moderate-area rather than full-frame masks.
        scored_masks = []
        for mask_data in area_sorted_masks:
            area = float(mask_data["area"])
            if area < min_area:
                continue
            area_ratio = area / frame_area
            mask = mask_data["segmentation"]
            border_touch = (
                np.sum(mask[0, :])
                + np.sum(mask[-1, :])
                + np.sum(mask[1:-1, 0])
                + np.sum(mask[1:-1, -1])
            )
            border_touch_ratio = float(border_touch) / border_len
            score = (1.0 - border_touch_ratio) - abs(area_ratio - 0.35)
            scored_masks.append((score, mask))
        scored_masks.sort(key=lambda x: x[0], reverse=True)
        selected_masks = [m for _, m in scored_masks[:5]]

    for mask in selected_masks:
        alpha_mask[mask] = 255

    if np.any(alpha_mask):
        kernel = np.ones((3, 3), dtype=np.uint8)
        alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Keep only the largest connected components to suppress isolated background islands.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((alpha_mask > 0).astype(np.uint8), connectivity=8)
        if num_labels > 1:
            comp_indices = list(range(1, num_labels))
            comp_indices.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
            kept = comp_indices[:max(1, keep_components)]
            cleaned = np.zeros_like(alpha_mask)
            for comp_id in kept:
                cleaned[labels == comp_id] = 255
            alpha_mask = cleaned

    if not np.any(alpha_mask) and len(area_sorted_masks) > 0:
        alpha_mask[area_sorted_masks[0]["segmentation"]] = 255

    rgba_np = np.dstack([rgb_np, alpha_mask])
    return Image.fromarray(rgba_np, mode="RGBA"), sam_masks


def _ensure_generation_models_loaded():
    global bbox_gen_model, part_synthesis_pipeline

    if part_synthesis_pipeline is None:
        print("Loading PartSynthesis model...")
        part_synthesis_pipeline = OmniPartImageTo3DPipeline.from_pretrained('omnipart/OmniPart')
        part_synthesis_pipeline.to(DEVICE)

    if bbox_gen_model is None:
        if partfield_ckpt_path_cached is None or bbox_gen_ckpt_path_cached is None:
            raise RuntimeError("Model checkpoints were not initialized. Call prepare_models(...) first.")
        print("Loading BboxGen model...")
        bbox_gen_config = OmegaConf.load("configs/bbox_gen.yaml").model.args
        bbox_gen_config.partfield_encoder_path = partfield_ckpt_path_cached
        bbox_gen_model = BboxGen(bbox_gen_config)
        bbox_gen_model.load_state_dict(torch.load(bbox_gen_ckpt_path_cached), strict=False)
        bbox_gen_model.to(DEVICE)
        bbox_gen_model.eval().half()

    print("Generation models ready")


@spaces.GPU
def process_image(image_path, threshold, req: gr.Request):
    """Process image and generate initial segmentation"""
    global size_th, sam_device

    user_dir = os.path.join(TMP_ROOT, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    img_name = os.path.basename(image_path).split(".")[0]
    
    size_th = threshold
    
    img = Image.open(image_path).convert("RGB")
    processed_image, sam_masks = _prepare_image_with_sam_alpha(img)

    white_bg = Image.new("RGBA", processed_image.size, (255, 255, 255, 255))
    white_bg_img = Image.alpha_composite(white_bg, processed_image.convert("RGBA"))
    image = np.array(white_bg_img.convert('RGB'))
    
    rgba_path = os.path.join(user_dir, f"{img_name}_processed.png")
    processed_image.save(rgba_path)
    
    visual = Visualizer(image)

    group_ids, pre_merge_im = get_sam_mask(
        image,
        sam_mask_generator,
        visual,
        merge_groups=None,
        rgba_image=processed_image,
        img_name=img_name,
        save_dir=user_dir,
        size_threshold=size_th,
        precomputed_masks=sam_masks,
    )
    
    pre_merge_path = os.path.join(user_dir, f"{img_name}_mask_pre_merge.png")
    Image.fromarray(pre_merge_im).save(pre_merge_path)
    pre_split_vis = np.ones_like(image) * 255  
    
    unique_ids = np.unique(group_ids)
    unique_ids = unique_ids[unique_ids >= 0]  
    
    for i, unique_id in enumerate(unique_ids):
        color_r = (i * 50 + 80) % 256
        color_g = (i * 120 + 40) % 256
        color_b = (i * 180 + 20) % 256
        color = np.array([color_r, color_g, color_b])
        
        mask = (group_ids == unique_id)
        pre_split_vis[mask] = color
        
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0:
            center_y = int(np.mean(y_indices))
            center_x = int(np.mean(x_indices))
            cv2.putText(pre_split_vis, str(unique_id), 
                        (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    pre_split_path = os.path.join(user_dir, f"{img_name}_pre_split.png")
    Image.fromarray(pre_split_vis).save(pre_split_path)
    print(f"Pre-split segmentation (before disconnected parts handling) saved to {pre_split_path}")
    
    get_mask(group_ids, image, ids=2, img_name=img_name, save_dir=user_dir)
    
    init_seg_path = os.path.join(user_dir, f"{img_name}_mask_segments_2.png")
    
    seg_img = Image.open(init_seg_path)
    if seg_img.mode == 'RGBA':
        white_bg = Image.new('RGBA', seg_img.size, (255, 255, 255, 255))
        seg_img = Image.alpha_composite(white_bg, seg_img)
        seg_img.save(init_seg_path)
    
    state = {
        "image": image.tolist(),
        "processed_image": rgba_path,
        "group_ids": group_ids.tolist() if isinstance(group_ids, np.ndarray) else group_ids,
        "original_group_ids": group_ids.tolist() if isinstance(group_ids, np.ndarray) else group_ids,
        "img_name": img_name,
        "pre_split_path": pre_split_path, 
    }
    
    return init_seg_path, pre_merge_path, state


def apply_merge(merge_input, state, req: gr.Request):
    """Apply merge parameters and generate merged segmentation"""
    global sam_mask_generator
    
    if not state:
        return None, None, state

    user_dir = os.path.join(TMP_ROOT, str(req.session_hash))
        
    # Convert back from list to numpy array
    image = np.array(state["image"])
    # Use original group IDs instead of the most recent ones
    group_ids = np.array(state["original_group_ids"])
    img_name = state["img_name"]
    
    # Load processed image from path
    processed_image = Image.open(state["processed_image"])
    
    # Display the original IDs before merging, SORTED for easier reading
    unique_ids = np.unique(group_ids)
    unique_ids = unique_ids[unique_ids >= 0]  # Exclude background
    print(f"Original segment IDs (used for merging): {sorted(unique_ids.tolist())}")
    
    # Parse merge groups
    merge_groups = None
    try:
        if merge_input:
            merge_groups = []
            group_sets = merge_input.split(';')
            for group_set in group_sets:
                ids = [int(x) for x in group_set.split(',')]
                if ids:
                    # Validate if these IDs exist in the segmentation
                    existing_ids = [id for id in ids if id in unique_ids]
                    missing_ids = [id for id in ids if id not in unique_ids]
                    
                    if missing_ids:
                        print(f"Warning: These IDs don't exist in the segmentation: {missing_ids}")
                    
                    # Only add group if it has valid IDs
                    if existing_ids:
                        merge_groups.append(ids)
                        print(f"Valid merge group: {ids} (missing: {missing_ids if missing_ids else 'none'})")
                    else:
                        print(f"Skipping merge group with no valid IDs: {ids}")
            
            print(f"Using merge groups: {merge_groups}")
    except Exception as e:
        print(f"Error parsing merge groups: {e}")
        return None, None, state
    
    # Initialize visualizer
    visual = Visualizer(image)
    
    # Generate merged segmentation starting from original IDs
    # Add skip_split=True to prevent splitting after merging
    new_group_ids, merged_im = get_sam_mask(
        image, 
        sam_mask_generator, 
        visual, 
        merge_groups=merge_groups, 
        existing_group_ids=group_ids,
        rgba_image=processed_image,
        skip_split=True, 
        img_name=img_name,
        save_dir=user_dir,
        size_threshold=size_th
    )
    
    # Display the new IDs after merging for future reference
    new_unique_ids = np.unique(new_group_ids)
    new_unique_ids = new_unique_ids[new_unique_ids >= 0]  # Exclude background
    print(f"New segment IDs (after merging): {new_unique_ids.tolist()}")
    
    # Clean edges
    new_group_ids = clean_segment_edges(new_group_ids)
    
    # Save merged segmentation visualization
    get_mask(new_group_ids, image, ids=3, img_name=img_name, save_dir=user_dir)
    
    # Path to merged segmentation
    merged_seg_path = os.path.join(user_dir, f"{img_name}_mask_segments_3.png")

    save_mask = new_group_ids + 1
    save_mask = save_mask.reshape(518, 518, 1).repeat(3, axis=-1)
    cv2.imwrite(os.path.join(user_dir, f"{img_name}_mask.exr"), save_mask.astype(np.float32))
    
    # Update state with the new group IDs but keep original IDs unchanged
    state["group_ids"] = new_group_ids.tolist() if isinstance(new_group_ids, np.ndarray) else new_group_ids
    state["save_mask_path"] = os.path.join(user_dir, f"{img_name}_mask.exr")
    
    return merged_seg_path, state


def explode_mesh(mesh, explosion_scale=0.4):    

    if isinstance(mesh, trimesh.Scene):
        scene = mesh
    elif isinstance(mesh, trimesh.Trimesh):
        print("Warning: Single mesh provided, can't create exploded view")
        scene = trimesh.Scene(mesh)
        return scene
    else:
        print(f"Warning: Unexpected mesh type: {type(mesh)}")
        scene = mesh

    if len(scene.geometry) <= 1:
        print("Only one geometry found - nothing to explode")
        return scene
    
    print(f"[EXPLODE_MESH] Starting mesh explosion with scale {explosion_scale}")
    print(f"[EXPLODE_MESH] Processing {len(scene.geometry)} parts")
    
    exploded_scene = trimesh.Scene()
    
    part_centers = []
    geometry_names = []
    
    for geometry_name, geometry in scene.geometry.items():
        if hasattr(geometry, 'vertices'):
            transform = scene.graph[geometry_name][0]
            vertices_global = trimesh.transformations.transform_points(
                geometry.vertices, transform)
            center = np.mean(vertices_global, axis=0)
            part_centers.append(center)
            geometry_names.append(geometry_name)
            print(f"[EXPLODE_MESH] Part {geometry_name}: center = {center}")
    
    if not part_centers:
        print("No valid geometries with vertices found")
        return scene
    
    part_centers = np.array(part_centers)
    global_center = np.mean(part_centers, axis=0)
    
    print(f"[EXPLODE_MESH] Global center: {global_center}")
    
    for i, (geometry_name, geometry) in enumerate(scene.geometry.items()):
        if hasattr(geometry, 'vertices'):
            if i < len(part_centers):
                part_center = part_centers[i]
                direction = part_center - global_center
                
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-6:
                    direction = direction / direction_norm
                else:
                    direction = np.random.randn(3)
                    direction = direction / np.linalg.norm(direction)
                
                offset = direction * explosion_scale
            else:
                offset = np.zeros(3)
            
            original_transform = scene.graph[geometry_name][0].copy()
            
            new_transform = original_transform.copy()
            new_transform[:3, 3] = new_transform[:3, 3] + offset
            
            exploded_scene.add_geometry(
                geometry, 
                transform=new_transform, 
                geom_name=geometry_name
            )
            
            print(f"[EXPLODE_MESH] Part {geometry_name}: moved by {np.linalg.norm(offset):.4f}")
    
    print("[EXPLODE_MESH] Mesh explosion complete")
    return exploded_scene
    
@spaces.GPU(duration=90)
def generate_parts(state, seed, cfg_strength, req: gr.Request):
    _ensure_generation_models_loaded()
    explode_factor=0.3
    img_path = state["processed_image"]
    mask_path = state["save_mask_path"]
    user_dir = os.path.join(TMP_ROOT, str(req.session_hash))
    img_white_bg, img_black_bg, ordered_mask_input, img_mask_vis = load_img_mask(img_path, mask_path)
    img_mask_vis.save(os.path.join(user_dir, "img_mask_vis.png"))

    voxel_coords = part_synthesis_pipeline.get_coords(img_black_bg, num_samples=1, seed=seed, sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5})
    voxel_coords = voxel_coords.cpu().numpy()
    np.save(os.path.join(user_dir, "voxel_coords.npy"), voxel_coords)
    voxel_coords_ply = vis_voxel_coords(voxel_coords)
    voxel_coords_ply.export(os.path.join(user_dir, "voxel_coords_vis.ply"))
    print("[INFO] Voxel coordinates saved")

    bbox_gen_input = prepare_bbox_gen_input(os.path.join(user_dir, "voxel_coords.npy"), img_white_bg, ordered_mask_input)
    bbox_gen_output = bbox_gen_model.generate(bbox_gen_input)
    np.save(os.path.join(user_dir, "bboxes.npy"), bbox_gen_output['bboxes'][0])
    bboxes_vis = gen_mesh_from_bounds(bbox_gen_output['bboxes'][0])
    bboxes_vis.export(os.path.join(user_dir, "bboxes_vis.glb"))
    print("[INFO] BboxGen output saved")


    part_synthesis_input = prepare_part_synthesis_input(os.path.join(user_dir, "voxel_coords.npy"), os.path.join(user_dir, "bboxes.npy"), ordered_mask_input)

    torch.cuda.empty_cache()

    part_synthesis_output = part_synthesis_pipeline.get_slat(
        img_black_bg, 
        part_synthesis_input['coords'], 
        [part_synthesis_input['part_layouts']], 
        part_synthesis_input['masks'],
        seed=seed,
        slat_sampler_params={"steps": 25, "cfg_strength": cfg_strength},
        formats=['mesh', 'gaussian'],
        preprocess_image=False,
    )
    save_parts_outputs(
        part_synthesis_output, 
        output_dir=user_dir, 
        simplify_ratio=0.0, 
        save_video=False,
        save_glb=True,
        textured=False,
    )
    merge_parts(user_dir)
    print("[INFO] PartSynthesis output saved")

    bbox_mesh_path = os.path.join(user_dir, "bboxes_vis.glb")
    whole_mesh_path = os.path.join(user_dir, "mesh_segment.glb")

    combined_mesh = trimesh.load(whole_mesh_path)
    exploded_mesh_result = explode_mesh(combined_mesh, explosion_scale=explode_factor)
    exploded_mesh_result.export(os.path.join(user_dir, "exploded_parts.glb"))

    exploded_mesh_path = os.path.join(user_dir, "exploded_parts.glb")
    combined_gs_path = os.path.join(user_dir, "merged_gs.ply")
    exploded_gs_path = os.path.join(user_dir, "exploded_gs.ply")
    
    return bbox_mesh_path, whole_mesh_path, exploded_mesh_path, combined_gs_path, exploded_gs_path
