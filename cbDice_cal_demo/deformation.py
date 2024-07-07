
import numpy as np
from skimage.morphology import skeletonize, skeletonize_3d
from scipy import ndimage

def Dice(predicted_mask, ground_truth_mask):
    intersection = np.logical_and(predicted_mask, ground_truth_mask)
    return 2. * intersection.sum() / (predicted_mask.sum() + ground_truth_mask.sum())

from scipy.ndimage import convolve

def adaptive_size(score, target):
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    padding_out = np.zeros((target.shape[0] + 2, target.shape[1] + 2))
    padding_out[1:-1, 1:-1] = target
    h, w = 3, 3

    Y = np.zeros((target.shape[0]- h + 1, target.shape[1]- w + 1))
    target_expand = np.expand_dims(np.expand_dims(target, axis=0), axis=0)
    kernel_expand = np.expand_dims(np.expand_dims(kernel, axis=0), axis=0)
    Y = convolve(target_expand, kernel_expand, mode='constant')
    Y = Y * target
    Y[Y == 5] = 0

    C = np.count_nonzero(Y)
    S = np.count_nonzero(target)
    smooth = 1e-5
    alpha = 1 - (C + smooth) / (S + smooth)
    alpha = 2 * alpha - 1

    intersect = np.sum(score * target)
    y_sum = np.sum(target * target)
    z_sum = np.sum(score * score)
    alpha = min(alpha, 0.8)  # Truncated alpha
    loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

    return loss

def boundary_DoU(y_pred, y_true):
    loss = adaptive_size(y_pred, y_true)
    return 1-loss

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]
    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]
    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)

def clDice(v_p, v_l):
    """[this function computes the cldice metric]
    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]
    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

def combine_arrays_test(A, B, C):
    A_C = A * C
    B_C = B * C
    D = B_C.copy()
    mask_AC = (A != 0) & (B == 0)
    D[mask_AC] = A_C[mask_AC]
    return D

def get_weights_2d(mask, skel):
    # For CPU:
    dist_map = ndimage.distance_transform_edt(mask)

    dist_map[mask == 0] = 0
    skel_radius = np.zeros_like(skel, dtype=np.float32)
    skel_radius[skel == 1] = dist_map[skel == 1]

    if skel_radius.max() == 0 or skel_radius.min() == skel_radius.max():
        return mask, skel.clone(), skel.clone(), skel.clone(), skel.clone(), skel.clone()

    smooth = 1e-7
    skel_radius_max = skel_radius.max()
    dist_map[dist_map > skel_radius_max] = skel_radius_max

    skel_R = skel_radius
    skel_R_norm = skel_radius / skel_radius_max

    skel_1_R = np.zeros_like(skel, dtype=np.float32)
    skel_1_R_norm = np.zeros_like(skel, dtype=np.float32)
    skel_1_R[skel == 1] = (1 + smooth) / (skel_R[skel == 1] + smooth)
    skel_1_R_norm[skel == 1] = (1 + smooth) / (skel_R_norm[skel == 1] + smooth)
    dist_map_norm = dist_map / skel_radius_max

    return dist_map, dist_map_norm, skel_R, skel_R_norm, skel_1_R, skel_1_R_norm

def clDice_test(vp, vl):
    smooth = 1e-3
    if len(vp.shape)==2:
        sp = skeletonize(vp)
        sl = skeletonize(vl)
        vl_dist_map, vl_dist_map_norm, sl_R, sl_R_norm, sl_1_R, sl_1_R_norm = get_weights_2d(vl, sl)
        vp_dist_map, vp_dist_map_norm, sp_R, sp_R_norm, sp_1_R, sp_1_R_norm = get_weights_2d(vp, sp)
        q_sl = sl
        q_sp = sp
        q_vl = vl
        q_vp = vp
        q_slvl = sl
        q_spvp = sp
        weighted_tprec = (np.sum(q_sp*q_vl)+smooth)/(np.sum(combine_arrays_test(q_spvp, q_slvl, q_sp))+smooth)
        weighted_tsens = (np.sum(q_sl*q_vp)+smooth)/(np.sum(combine_arrays_test(q_slvl, q_spvp, q_sl))+smooth) 
        
    return 2.0 * (weighted_tprec * weighted_tsens) / (weighted_tprec + weighted_tsens)

def cbDice_test_sr(vp, vl):
    smooth = 1e-3
    if len(vp.shape)==2:
        sp = skeletonize(vp)
        sl = skeletonize(vl)
        vl_dist_map, vl_dist_map_norm, sl_R, sl_R_norm, sl_1_R, sl_1_R_norm = get_weights_2d(vl, sl)
        vp_dist_map, vp_dist_map_norm, sp_R, sp_R_norm, sp_1_R, sp_1_R_norm = get_weights_2d(vp, sp)
        q_sl = sl_R
        q_sp = sp_R
        q_vl = vl
        q_vp = vp
        q_slvl = sl
        q_spvp = sp
        weighted_tprec = (np.sum(q_sp*q_vl)+smooth)/(np.sum(combine_arrays_test(q_spvp, q_slvl, q_sp))+smooth)
        weighted_tsens = (np.sum(q_sl*q_vp)+smooth)/(np.sum(combine_arrays_test(q_slvl, q_spvp, q_sl))+smooth) 
        
    return 2.0 * (weighted_tprec * weighted_tsens) / (weighted_tprec + weighted_tsens)

def cbDice_test_mb(vp, vl):
    smooth = 1e-3
    if len(vp.shape)==2:
        sp = skeletonize(vp)
        sl = skeletonize(vl)
        vl_dist_map, vl_dist_map_norm, sl_R, sl_R_norm, sl_1_R, sl_1_R_norm = get_weights_2d(vl, sl)
        vp_dist_map, vp_dist_map_norm, sp_R, sp_R_norm, sp_1_R, sp_1_R_norm = get_weights_2d(vp, sp)
        q_sl = sl
        q_sp = sp
        q_vl = vl_dist_map
        q_vp = vp_dist_map
        q_slvl = sl_R
        q_spvp = sp_R
        weighted_tprec = (np.sum(q_sp*q_vl)+smooth)/(np.sum(combine_arrays_test(q_spvp, q_slvl, q_sp))+smooth)
        weighted_tsens = (np.sum(q_sl*q_vp)+smooth)/(np.sum(combine_arrays_test(q_slvl, q_spvp, q_sl))+smooth) 
        
    return 2.0 * (weighted_tprec * weighted_tsens) / (weighted_tprec + weighted_tsens)

def cbDice_test_srmb(vp, vl):
    smooth = 1e-3
    if len(vp.shape)==2:
        sp = skeletonize(vp)
        sl = skeletonize(vl)
        vl_dist_map, vl_dist_map_norm, sl_R, sl_R_norm, sl_1_R, sl_1_R_norm = get_weights_2d(vl, sl)
        vp_dist_map, vp_dist_map_norm, sp_R, sp_R_norm, sp_1_R, sp_1_R_norm = get_weights_2d(vp, sp)
        q_sl = sl_R
        q_sp = sp_R
        q_vl = vl_dist_map
        q_vp = vp_dist_map
        q_slvl = sl_R
        q_spvp = sp_R
        weighted_tprec = (np.sum(q_sp*q_vl)+smooth)/(np.sum(combine_arrays_test(q_spvp, q_slvl, q_sp))+smooth)
        weighted_tsens = (np.sum(q_sl*q_vp)+smooth)/(np.sum(combine_arrays_test(q_slvl, q_spvp, q_sl))+smooth) 
        
    return 2.0 * (weighted_tprec * weighted_tsens) / (weighted_tprec + weighted_tsens)

def cbDice_test_srimb(vp, vl):
    smooth = 1e-3
    if len(vp.shape)==2:
        sp = skeletonize(vp)
        sl = skeletonize(vl)
        vl_dist_map, vl_dist_map_norm, sl_R, sl_R_norm, sl_1_R, sl_1_R_norm = get_weights_2d(vl, sl)
        vp_dist_map, vp_dist_map_norm, sp_R, sp_R_norm, sp_1_R, sp_1_R_norm = get_weights_2d(vp, sp)
        q_sl = sl_1_R
        q_sp = sp_1_R
        q_vl = vl_dist_map
        q_vp = vp_dist_map
        q_slvl = sl_R
        q_spvp = sp_R
        weighted_tprec = (np.sum(q_sp*q_vl)+smooth)/(np.sum(combine_arrays_test(q_spvp, q_slvl, q_sp))+smooth)
        weighted_tsens = (np.sum(q_sl*q_vp)+smooth)/(np.sum(combine_arrays_test(q_slvl, q_spvp, q_sl))+smooth) 
        
    return 2.0 * (weighted_tprec * weighted_tsens) / (weighted_tprec + weighted_tsens)

def cbDice_test_sr_norm(vp, vl):
    smooth = 1e-3
    if len(vp.shape)==2:
        sp = skeletonize(vp)
        sl = skeletonize(vl)
        vl_dist_map, vl_dist_map_norm, sl_R, sl_R_norm, sl_1_R, sl_1_R_norm = get_weights_2d(vl, sl)
        vp_dist_map, vp_dist_map_norm, sp_R, sp_R_norm, sp_1_R, sp_1_R_norm = get_weights_2d(vp, sp)
        q_sl = sl_R_norm
        q_sp = sp_R_norm
        q_vl = vl
        q_vp = vp
        q_slvl = sl
        q_spvp = sp
        weighted_tprec = (np.sum(q_sp*q_vl)+smooth)/(np.sum(combine_arrays_test(q_spvp, q_slvl, q_sp))+smooth)
        weighted_tsens = (np.sum(q_sl*q_vp)+smooth)/(np.sum(combine_arrays_test(q_slvl, q_spvp, q_sl))+smooth) 
        
    return 2.0 * (weighted_tprec * weighted_tsens) / (weighted_tprec + weighted_tsens)

def cbDice_test_mb_norm(vp, vl):
    smooth = 1e-3
    if len(vp.shape)==2:
        sp = skeletonize(vp)
        sl = skeletonize(vl)
        vl_dist_map, vl_dist_map_norm, sl_R, sl_R_norm, sl_1_R, sl_1_R_norm = get_weights_2d(vl, sl)
        vp_dist_map, vp_dist_map_norm, sp_R, sp_R_norm, sp_1_R, sp_1_R_norm = get_weights_2d(vp, sp)
        q_sl = sl
        q_sp = sp
        q_vl = vl_dist_map_norm
        q_vp = vp_dist_map_norm
        q_slvl = sl_R_norm
        q_spvp = sp_R_norm
        weighted_tprec = (np.sum(q_sp*q_vl)+smooth)/(np.sum(combine_arrays_test(q_spvp, q_slvl, q_sp))+smooth)
        weighted_tsens = (np.sum(q_sl*q_vp)+smooth)/(np.sum(combine_arrays_test(q_slvl, q_spvp, q_sl))+smooth) 
        
    return 2.0 * (weighted_tprec * weighted_tsens) / (weighted_tprec + weighted_tsens)

def cbDice_test_srmb_norm(vp, vl):
    smooth = 1e-3
    if len(vp.shape)==2:
        sp = skeletonize(vp)
        sl = skeletonize(vl)
        vl_dist_map, vl_dist_map_norm, sl_R, sl_R_norm, sl_1_R, sl_1_R_norm = get_weights_2d(vl, sl)
        vp_dist_map, vp_dist_map_norm, sp_R, sp_R_norm, sp_1_R, sp_1_R_norm = get_weights_2d(vp, sp)
        q_sl = sl_R_norm
        q_sp = sp_R_norm
        q_vl = vl_dist_map_norm
        q_vp = vp_dist_map_norm
        q_slvl = sl_R_norm
        q_spvp = sp_R_norm
        weighted_tprec = (np.sum(q_sp*q_vl)+smooth)/(np.sum(combine_arrays_test(q_spvp, q_slvl, q_sp))+smooth)
        weighted_tsens = (np.sum(q_sl*q_vp)+smooth)/(np.sum(combine_arrays_test(q_slvl, q_spvp, q_sl))+smooth) 
        
    return 2.0 * (weighted_tprec * weighted_tsens) / (weighted_tprec + weighted_tsens)

def cbDice_test_srimb_norm(vp, vl):
    smooth = 1e-3
    if len(vp.shape)==2:
        sp = skeletonize(vp)
        sl = skeletonize(vl)
        vl_dist_map, vl_dist_map_norm, sl_R, sl_R_norm, sl_1_R, sl_1_R_norm = get_weights_2d(vl, sl)
        vp_dist_map, vp_dist_map_norm, sp_R, sp_R_norm, sp_1_R, sp_1_R_norm = get_weights_2d(vp, sp)
        q_sl = sl_1_R_norm
        q_sp = sp_1_R_norm
        q_vl = vl_dist_map_norm
        q_vp = vp_dist_map_norm
        q_slvl = sl_R_norm
        q_spvp = sp_R_norm
        weighted_tprec = (np.sum(q_sp*q_vl)+smooth)/(np.sum(combine_arrays_test(q_spvp, q_slvl, q_sp))+smooth)
        weighted_tsens = (np.sum(q_sl*q_vp)+smooth)/(np.sum(combine_arrays_test(q_slvl, q_spvp, q_sl))+smooth) 
        
    return 2.0 * (weighted_tprec * weighted_tsens) / (weighted_tprec + weighted_tsens)


reference_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

prediction_arr_1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


prediction_arr_2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


dice_1 = Dice(prediction_arr_1, reference_arr)
dice_2 = Dice(prediction_arr_2, reference_arr)
print("dice_1:", dice_1)
print("dice_2:", dice_2)

cldice_1 = clDice(prediction_arr_1, reference_arr)
cldice_2 = clDice(prediction_arr_2, reference_arr)
print("cldice_1:", cldice_1)
print("cldice_2:", cldice_2)

dice_cldice_mean_1 = (dice_1 + cldice_1) / 2
dice_cldice_mean_2 = (dice_2 + cldice_2) / 2
print("dice_cldice_mean_1:", dice_cldice_mean_1)
print("dice_cldice_mean_2:", dice_cldice_mean_2)

cldice_test_1 = clDice_test(prediction_arr_1, reference_arr)
cldice_test_2 = clDice_test(prediction_arr_2, reference_arr)
print("cldice_test_1:", cldice_test_1)
print("cldice_test_2:", cldice_test_2)

cbdice_test_sr_1 = cbDice_test_sr(prediction_arr_1, reference_arr)
cbdice_test_sr_2 = cbDice_test_sr(prediction_arr_2, reference_arr)
print("cbdice_test_sr_1:", cbdice_test_sr_1)
print("cbdice_test_sr_2:", cbdice_test_sr_2)

cbdice_test_mb_1 = cbDice_test_mb(prediction_arr_1, reference_arr)
cbdice_test_mb_2 = cbDice_test_mb(prediction_arr_2, reference_arr)
print("cbdice_test_mb_1:", cbdice_test_mb_1)
print("cbdice_test_mb_2:", cbdice_test_mb_2)

cbdice_test_srmb_1 = cbDice_test_srmb(prediction_arr_1, reference_arr)
cbdice_test_srmb_2 = cbDice_test_srmb(prediction_arr_2, reference_arr)
print("cbdice_test_srmb_1:", cbdice_test_srmb_1)
print("cbdice_test_srmb_2:", cbdice_test_srmb_2)

cbdice_test_srimb_1 = cbDice_test_srimb(prediction_arr_1, reference_arr)
cbdice_test_srimb_2 = cbDice_test_srimb(prediction_arr_2, reference_arr)
print("cbdice_test_srimb_1:", cbdice_test_srimb_1)
print("cbdice_test_srimb_2:", cbdice_test_srimb_2)

cbdice_test_sr_norm_1 = cbDice_test_sr_norm(prediction_arr_1, reference_arr)
cbdice_test_sr_norm_2 = cbDice_test_sr_norm(prediction_arr_2, reference_arr)
print("cbdice_test_sr_norm_1:", cbdice_test_sr_norm_1)
print("cbdice_test_sr_norm_2:", cbdice_test_sr_norm_2)

cbdice_test_mb_norm_1 = cbDice_test_mb_norm(prediction_arr_1, reference_arr)
cbdice_test_mb_norm_2 = cbDice_test_mb_norm(prediction_arr_2, reference_arr)
print("cbdice_test_mb_norm_1:", cbdice_test_mb_norm_1)
print("cbdice_test_mb_norm_2:", cbdice_test_mb_norm_2)

cbdice_test_srmb_norm_1 = cbDice_test_srmb_norm(prediction_arr_1, reference_arr)
cbdice_test_srmb_norm_2 = cbDice_test_srmb_norm(prediction_arr_2, reference_arr)
print("cbdice_test_srmb_norm_1:", cbdice_test_srmb_norm_1)
print("cbdice_test_srmb_norm_2:", cbdice_test_srmb_norm_2)

cbdice_test_srimb_norm_1 = cbDice_test_srimb_norm(prediction_arr_1, reference_arr)
cbdice_test_srimb_norm_2 = cbDice_test_srimb_norm(prediction_arr_2, reference_arr)
print("cbdice_test_srimb_norm_1:", cbdice_test_srimb_norm_1)
print("cbdice_test_srimb_norm_2:", cbdice_test_srimb_norm_2)

dice_cbdice_mean_1 = (dice_1 + 2*cbdice_test_srimb_norm_1) / 3
dice_cbdice_mean_2 = (dice_2 + 2*cbdice_test_srimb_norm_2) / 3
print("dice_cbdice_mean_1(1:2):", dice_cbdice_mean_1)
print("dice_cbdice_mean_2(1:2):", dice_cbdice_mean_2)

dice_cbdice_mean_1 = (dice_1 + 1.5*cbdice_test_srimb_norm_1) / 2.5
dice_cbdice_mean_2 = (dice_2 + 1.5*cbdice_test_srimb_norm_2) / 2.5
print("dice_cbdice_mean_1(1:1.5):", dice_cbdice_mean_1)
print("dice_cbdice_mean_2(1:1.5):", dice_cbdice_mean_2)

dice_cbdice_mean_1 = (dice_1 + 1.0*cbdice_test_srimb_norm_1) / 2
dice_cbdice_mean_2 = (dice_2 + 1.0*cbdice_test_srimb_norm_2) / 2
print("dice_cbdice_mean_1(1:1):", dice_cbdice_mean_1)
print("dice_cbdice_mean_2(1:1):", dice_cbdice_mean_2)

dice_cbdice_mean_1 = (dice_1 + 0.5*cbdice_test_srimb_norm_1) / 1.5
dice_cbdice_mean_2 = (dice_2 + 0.5*cbdice_test_srimb_norm_2) / 1.5
print("dice_cbdice_mean_1(1:0.5):", dice_cbdice_mean_1)
print("dice_cbdice_mean_2(1:0.5):", dice_cbdice_mean_2)

# boundary DoU
boundary_DoU_1 = boundary_DoU(prediction_arr_1, reference_arr)
boundary_DoU_2 = boundary_DoU(prediction_arr_2, reference_arr)
print("boundary_DoU_1:", boundary_DoU_1)
print("boundary_DoU_2:", boundary_DoU_2)

from metrics.BettiMatching import *
# Betti
def compute_metrics(t, relative=False, comparison='union', filtration='superlevel', construction='V'):
    BM = BettiMatching(t[0], t[1], relative=relative, comparison=comparison, filtration=filtration, construction=construction)
    return BM.loss(dimensions=[0,1]), BM.loss(dimensions=[0]), BM.loss(dimensions=[1]), BM.Betti_number_error(threshold=0.5, dimensions=[0,1]), BM.Betti_number_error(threshold=0.5, dimensions=[0]), BM.Betti_number_error(threshold=0.5, dimensions=[1])

pair_1 = [prediction_arr_1, reference_arr]
pair_2 = [prediction_arr_2, reference_arr]
pair_list = [pair_1, pair_2]
for i in range(len(pair_list)):
    pair = pair_list[i]
    print("prediction_arr:", str(i+1))
    Betti_matching_error, Betti_matching_error_0, Betti_matching_error_1, Betti_error, Betti_0_err, Betti_1_err = compute_metrics(pair,  relative=True, comparison='union', filtration='superlevel', construction='V')
    print("Betti_matching_error:", Betti_matching_error)
    print("Betti_matching_error_0:", Betti_matching_error_0)
    print("Betti_matching_error_1:", Betti_matching_error_1)
    print("Betti_error:", Betti_error)
    print("Betti_0_err:", Betti_0_err)
    print("Betti_1_err:", Betti_1_err)
