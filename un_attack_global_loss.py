import torch.optim as optim
import numpy as np
import torch
from omegaconf import DictConfig
import hydra, os, sys
from scripts.network.dataloader import HDF5Dataset

import open3d as o3d
from pyTorchChamferDistance.chamfer_distance import ChamferDistance
from torch.utils.data import DataLoader

from scripts.network.models.basic import cal_pose0to1
from scripts.pl_model import ModelWrapper
from scripts.utils.av2_eval import CATEGORY_TO_INDEX, CRITICAL_OBJECTS, compute_accuracy
from scripts.utils.mics import flow_to_rgb
from scripts.utils.o3d_view import MyVisualizer
from un_attack import calculate_iou


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cfg):
    if not os.path.exists(cfg.checkpoint):
        print(f"Checkpoint {cfg.checkpoint} does not exist. Need checkpoints for evaluation.")
        sys.exit(1)

    checkpoint_params = DictConfig(torch.load(cfg.checkpoint)["hyper_parameters"])
    cfg.output = checkpoint_params.cfg.output + f"-{cfg.av2_mode}"
    cfg.model.update(checkpoint_params.cfg.model)
    mymodel = ModelWrapper.load_from_checkpoint(cfg.checkpoint, cfg=cfg, eval=True)
    optim_on_objects(mymodel, DataLoader(HDF5Dataset(cfg.dataset_path + f"/{cfg.av2_mode}", eval=True), batch_size=1,
                                       shuffle=False))


def get_object_mask(initial_mask, flow_cat):
    ids = []
    for cats_name in CRITICAL_OBJECTS:
        selected_classes_ids = [CATEGORY_TO_INDEX[cat] for cat in CRITICAL_OBJECTS[cats_name]]
        ids.extend(selected_classes_ids)

    for id in ids:
        initial_mask = torch.logical_or(initial_mask, flow_cat == id)

    return initial_mask


def optim_on_objects(model, data_loader, itr=100, is_reverse=False):
    torch.set_grad_enabled(True)
    print(len(data_loader.dataset))
    f = open("optim_attack_eval.txt", "a")
    results = open("optim_global_results_eval_test.txt", "a")
    chamfer_list = []
    EPE_attack = []
    EPE_attack_dynamic = []
    EPE_attack_static = []
    EPE_original = []
    EPE_original_dynamic = []
    EPE_original_static = []
    MSE_atack = []
    dynamic_accuracies_attack = []
    dynamic_accuracies_original = []
    dynamic_ious_attack = []
    dynamic_ious_original = []
    static_ious_attack = []
    static_ious_original = []
    relax_accuracies_attack = []
    relax_accuracies_original = []
    for step, batch in enumerate(data_loader):
        print(step + 1)  # Images Step = 20,70, 80, 2100, 4400, 4320, 1150, 3360
        # if (step == 50):  # Chamfer 20, 1788
        #     break
        batch_org = batch.copy()
        pc0 = batch['pc0'][0]
        pose_0to1 = cal_pose0to1(batch["pose0"][0], batch["pose1"][0])
        transform_pc0 = pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
        pose_flow = transform_pc0 - pc0

        pose_flow_from_original = pose_flow[~batch['gm0'][0]].unsqueeze(0)
        batch['pc0'] = batch['pc0'][~batch['gm0']].unsqueeze(0)
        batch['pc1'] = batch['pc1'][~batch['gm1']].unsqueeze(0)
        batch['flow'] = batch['flow'][~batch['gm0']].unsqueeze(0)
        batch['eval_mask'] = batch['eval_mask'][~batch['gm0']].unsqueeze(0)
        batch['flow_is_valid'] = batch['flow_is_valid'][~batch['gm0']].unsqueeze(0)
        batch['pc1'].requires_grad = True
        original = batch['pc1'].clone().cuda()
        optimizer = optim.Adam([batch['pc1']], lr=0.01)
        adv_loss = model.loss_fn  # f_mse
        chamfer_dist = ChamferDistance()

        batch['flow_category_indices_1'] = batch['flow_category_indices_1'][~batch['gm1']].unsqueeze(0)
        batch['flow_category_indices'] = batch['flow_category_indices'][~batch['gm0']].unsqueeze(0)

        targeted_objects_mask_1 = get_object_mask(torch.zeros_like(batch['flow_category_indices_1']),
                                                  batch['flow_category_indices_1'])
        targeted_objects_mask_0 = get_object_mask(torch.zeros_like(batch['flow_category_indices']),
                                                  batch['flow_category_indices'])

        model.model.eval()

        original_loss, est_original_flow = normal_loop(model, batch_org)

        for i in range(itr + 1):
            batch['pc1'].requires_grad = True
            model.model.cuda()

            res_dict = model.model(batch)
            gt_org = batch['flow']
            if is_reverse:
                gt_flow = batch['flow']  # torch.zeros_like(batch['flow'])

                pose_flows = pose_flow_from_original
                pc0_valid_idx = res_dict['pc0_valid_point_idxes']  # since padding
                est_flow = res_dict['flow']
                est_flow_ = est_flow[0]

                pc0_valid_from_pc2res = pc0_valid_idx[0]
                valid_flows = batch['flow_is_valid'][0][pc0_valid_from_pc2res]
                eval_mask = batch['eval_mask'][0][pc0_valid_from_pc2res].squeeze()
                valid_flows = torch.logical_and(valid_flows, eval_mask)
                pose_flow_ = pose_flows[0][pc0_valid_from_pc2res].cuda()
                gt_flow_ = gt_flow[0][pc0_valid_from_pc2res].cuda()
                gt_org_flow_ = gt_org[0][pc0_valid_from_pc2res]

                pose_flow_ = pose_flow_[valid_flows]
                est_flow_ = est_flow_[valid_flows]
                gt_flow_ = gt_flow_[valid_flows]
                gt_org_flow_ = gt_org_flow_[valid_flows]

                gt_flow_ = gt_flow_ - pose_flow_
                gt_flow_ = -gt_flow_
            else:
                gt_flow = batch['flow']
                pose_flows = pose_flow_from_original

                pc0_valid_idx = res_dict['pc0_valid_point_idxes']  # since padding
                est_flow = res_dict['flow']
                est_flow_ = est_flow[0]

                pc0_valid_from_pc2res = pc0_valid_idx[0]
                valid_flows = batch['flow_is_valid'][0][pc0_valid_from_pc2res]
                eval_mask = batch['eval_mask'][0][pc0_valid_from_pc2res].squeeze()
                valid_flows = torch.logical_and(valid_flows, eval_mask)
                pose_flow_ = pose_flows[0][pc0_valid_from_pc2res].cuda()
                gt_flow_ = gt_flow[0][pc0_valid_from_pc2res].cuda()
                gt_org_flow_ = gt_org[0][pc0_valid_from_pc2res]

                pose_flow_ = pose_flow_[valid_flows]
                est_flow_ = est_flow_[valid_flows]
                gt_flow_ = gt_flow_[valid_flows]
                gt_flow_ = gt_flow_ - pose_flow_
                gt_org_flow_ = gt_org_flow_[valid_flows]

            targeted_objects_mask_0_ = targeted_objects_mask_0[0][pc0_valid_from_pc2res]
            targeted_objects_mask_0_ = targeted_objects_mask_0_[valid_flows]

            dist1, dist2 = chamfer_dist(batch['pc1'].cuda(), original.cuda())
            chamfer_loss_pc1 = (torch.mean(dist1)) + (torch.mean(dist2))
            mse = torch.linalg.vector_norm(batch['pc1'].cpu() - original.cpu(), dim=-1).mean()

            res_dict_eval = {'est_flow': est_flow_,
                             'gt_flow': gt_flow_
                             }

            res_dict_eval_2 = {'est_flow': est_flow_ + pose_flow_,
                               'gt_flow': gt_org_flow_
                               }
            model.model.zero_grad()
            total_loss_org_back, total_loss_org = adv_loss(res_dict_eval_2, targeted_objects_mask_0_)

            total_loss = 0.5 * chamfer_loss_pc1 - 1 * total_loss_org_back.mean()
            optimizer.zero_grad()
            total_loss.backward()
            batch['pc1'].grad[0][~targeted_objects_mask_1[0]] = 0
            optimizer.step()
            total_loss_obj = total_loss_org[targeted_objects_mask_0_].mean()
            # print(total_loss_obj.item())
            batch['pc1'].requires_grad = False

            if (i == (itr - 1)):
                path = "/home/yasas/UNSW/Adv_Temporal/SceneFlow/attack_data/global_flow_craft/" + str(step) + ".pt"
                torch.save(batch['pc1'], path)

        # pc0 = original
        # pc1 = res_dict['pcd1']
        # vis_p(pc0, pc1)
        mask_is_dynamic = torch.linalg.vector_norm(gt_org_flow_.cpu() - pose_flow_.cpu(), dim=-1) >= 0.05
        mask_dynamic_objects = torch.logical_and(mask_is_dynamic, targeted_objects_mask_0_)
        mask_static_objects = torch.logical_and(~mask_is_dynamic, targeted_objects_mask_0_)
        attack_loss_dynamic = total_loss_org[mask_dynamic_objects].mean()
        attack_loss_static = total_loss_org[mask_static_objects].mean()

        mask_pred_original_is_dynamic = torch.linalg.vector_norm(est_original_flow.cpu(), dim=-1) >= 0.05
        mask_pred_attack_is_dynamic = torch.linalg.vector_norm(est_flow_.cpu(), dim=-1) >= 0.05

        dynamic_accuracy_attack = torch.sum(
            mask_pred_attack_is_dynamic[targeted_objects_mask_0_] == mask_is_dynamic[targeted_objects_mask_0_]) / len(
            mask_pred_attack_is_dynamic[targeted_objects_mask_0_])

        dynamic_accuracy_original = torch.sum(
            mask_pred_original_is_dynamic[targeted_objects_mask_0_] == mask_is_dynamic[targeted_objects_mask_0_]) / len(
            mask_pred_original_is_dynamic[targeted_objects_mask_0_])

        dynamic_iou_attack = calculate_iou(mask_pred_attack_is_dynamic[targeted_objects_mask_0_],
                                           mask_is_dynamic[targeted_objects_mask_0_])
        dynamic_iou_original = calculate_iou(mask_pred_original_is_dynamic[targeted_objects_mask_0_],
                                             mask_is_dynamic[targeted_objects_mask_0_])

        static_iou_attack = calculate_iou(~mask_pred_attack_is_dynamic[targeted_objects_mask_0_],
                                          ~mask_is_dynamic[targeted_objects_mask_0_])
        static_iou_original = calculate_iou(~mask_pred_original_is_dynamic[targeted_objects_mask_0_],
                                            ~mask_is_dynamic[targeted_objects_mask_0_])

        relax_acc_attack = compute_accuracy(est_flow_[mask_dynamic_objects].detach().cpu().numpy(),
                                            (gt_org_flow_[mask_dynamic_objects].cpu() - pose_flow_[
                                                mask_dynamic_objects].cpu()).numpy(), 0.05).mean()
        relax_acc_original = compute_accuracy(est_original_flow[mask_dynamic_objects].detach().cpu().numpy(),
                                              (gt_org_flow_[mask_dynamic_objects].cpu() - pose_flow_[
                                                  mask_dynamic_objects].cpu()).numpy(), 0.05).mean()

        original_loss_dynamic = original_loss[mask_dynamic_objects].mean()
        original_loss_static = original_loss[mask_static_objects].mean()
        original_loss = original_loss[targeted_objects_mask_0_].mean()

        if str(total_loss_obj.item()) == "nan" or gt_org_flow_[targeted_objects_mask_0_].isnan().any() or est_flow_[
            targeted_objects_mask_0_].isnan().any() or \
                pose_flow_[targeted_objects_mask_0_].isnan().any():
            print("Skipping Found Nan")
        else:
            chamfer_list.append(chamfer_loss_pc1.item())
            MSE_atack.append(mse.item())
            dynamic_accuracies_attack.append(dynamic_accuracy_attack.item())
            dynamic_accuracies_original.append(dynamic_accuracy_original.item())
            relax_accuracies_attack.append(relax_acc_attack)
            relax_accuracies_original.append(relax_acc_original)
            if (str(original_loss_dynamic.item()) != "nan"):
                dynamic_ious_attack.append(dynamic_iou_attack)
                dynamic_ious_original.append(dynamic_iou_original)
            if (str(original_loss_static.item()) != "nan"):
                static_ious_attack.append(static_iou_attack)
                static_ious_original.append(static_iou_original)
            EPE_attack.append(total_loss_obj.item())
            EPE_attack_dynamic.append(attack_loss_dynamic.item())
            EPE_attack_static.append(attack_loss_static.item())
            EPE_original.append(original_loss.item())
            EPE_original_dynamic.append(original_loss_dynamic.item())
            EPE_original_static.append(original_loss_static.item())
            print("Attack: " + str(total_loss_obj))
            print("Attack Dynamic: " + str(attack_loss_dynamic))
            print("Attack Static: " + str(attack_loss_static))
            print("Dynamic Accuracy Attack: " + str(dynamic_accuracy_attack.item()))
            print("Dynamic Iou Attack: " + str(dynamic_iou_attack))
            print("Static Iou Attack: " + str(static_iou_attack))
            print("Relax Accuracy Attack: " + str(relax_acc_attack))
            print("Original: " + str(original_loss))
            print("Original Dynamic: " + str(original_loss_dynamic))
            print("Original Static: " + str(original_loss_static))
            print("Dynamic Accuracy Original: " + str(dynamic_accuracy_original.item()))
            print("Dynamic Iou Original: " + str(dynamic_iou_original))
            print("Static Iou Original: " + str(static_iou_original))
            print("Relax Accuracy Original: " + str(relax_acc_original))
            print("Chamfer:" + str(chamfer_loss_pc1))
            print("MSE:" + str(mse))
            print(str(total_loss_obj.item()) + "," + str(attack_loss_dynamic.item()) + "," + str(
                attack_loss_static.item()) + "," + str(dynamic_accuracy_attack.item()) + "," +
                  str(dynamic_iou_attack) + "," + str(static_iou_attack) + "," + str(relax_acc_attack) + "," + str(
                original_loss.item()) + "," + str(original_loss_dynamic.item()) + "," + str(
                original_loss_static.item()) + "," + str(dynamic_accuracy_original.item()) + "," + str(
                dynamic_iou_original) + "," + str(static_iou_original) + "," + str(relax_acc_original) + "," + str(
                chamfer_loss_pc1.item()) + "," + str(mse.item()), file=results)
    print("==============Results DeFlow Optim No Objective C=0.5 T=1.0 with Penalize Global Loss===============")
    print("Attack:" + str(np.nanmean(EPE_attack)))
    print("Attack Dynamic:" + str(np.nanmean(EPE_attack_dynamic)))
    print("Attack Static:" + str(np.nanmean(EPE_attack_static)))
    print("Dynamic Accuracy Attack:" + str(np.nanmean(dynamic_accuracies_attack)))
    print("Dynamic IoU Attack:" + str(np.nanmean(dynamic_ious_attack)))
    print("Static IoU Attack:" + str(np.nanmean(static_ious_attack)))
    print("Relax Accuracy Attack:" + str(np.nanmean(relax_accuracies_attack)))
    print("Original:" + str(np.nanmean(EPE_original)))
    print("Original Dynamic:" + str(np.nanmean(EPE_original_dynamic)))
    print("Original Static:" + str(np.nanmean(EPE_original_static)))
    print("Dynamic Accuracy Original:" + str(np.nanmean(dynamic_accuracies_original)))
    print("Dynamic IoU Original:" + str(np.nanmean(dynamic_ious_original)))
    print("Static IoU Original:" + str(np.nanmean(static_ious_original)))
    print("Relax Accuracy Original:" + str(np.nanmean(relax_accuracies_original)))
    print("Chamfer:" + str(np.nanmean(chamfer_list)))
    print("MSE:" + str(np.nanmean(MSE_atack)))

    print("==============Results DeFlow Optim No Objective C=0.5 T=1.0 with Penalize Global Loss===============",
          file=f)
    print("Attack:" + str(np.nanmean(EPE_attack)), file=f)
    print("Attack Dynamic:" + str(np.nanmean(EPE_attack_dynamic)), file=f)
    print("Attack Static:" + str(np.nanmean(EPE_attack_static)), file=f)
    print("Dynamic Accuracy Attack:" + str(np.nanmean(dynamic_accuracies_attack)), file=f)
    print("Relax Accuracy Attack:" + str(np.nanmean(relax_accuracies_attack)), file=f)
    print("Dynamic IoU Attack:" + str(np.nanmean(dynamic_ious_attack)), file=f)
    print("Static IoU Attack:" + str(np.nanmean(static_ious_attack)), file=f)
    print("Original:" + str(np.nanmean(EPE_original)), file=f)
    print("Original Dynamic:" + str(np.nanmean(EPE_original_dynamic)), file=f)
    print("Original Static:" + str(np.nanmean(EPE_original_static)), file=f)
    print("Dynamic Accuracy Original:" + str(np.nanmean(dynamic_accuracies_original)), file=f)
    print("Dynamic IoU Original:" + str(np.nanmean(dynamic_ious_original)), file=f)
    print("Static IoU Original:" + str(np.nanmean(static_ious_original)), file=f)
    print("Relax Accuracy Original:" + str(np.nanmean(relax_accuracies_original)), file=f)
    print("Chamfer:" + str(np.nanmean(chamfer_list)), file=f)
    print("MSE:" + str(np.nanmean(MSE_atack)), file=f)


def pgd_on_objects(model, data_loader, random_start=False, eps=0.5, alpa=0.01, itr=100, cos_pgd=True, targeted=False):
    f = open("pgd_attack_new.txt", "a")
    results = open("cos_pgd_global_results.txt", "a")
    torch.set_grad_enabled(True)
    chamfer_dist = ChamferDistance()
    chamfer_list = []
    EPE_attack = []
    EPE_attack_dynamic = []
    EPE_attack_static = []
    EPE_original = []
    EPE_original_dynamic = []
    EPE_original_static = []
    MSE_atack = []
    dynamic_accuracies_attack = []
    dynamic_accuracies_original = []
    dynamic_ious_attack = []
    dynamic_ious_original = []
    static_ious_attack = []
    static_ious_original = []
    relax_accuracies_attack = []
    relax_accuracies_original = []
    for step, batch in enumerate(data_loader):
        torch.cuda.empty_cache()
        print(step + 1)
        # if (step == 50):
        #     break
        batch_org = batch.copy()
        pc0 = batch['pc0'][0]
        pose_0to1 = cal_pose0to1(batch["pose0"][0], batch["pose1"][0])
        transform_pc0 = pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
        pose_flow = transform_pc0 - pc0
        pose_flow_from_original = pose_flow[~batch['gm0'][0]].unsqueeze(0)
        batch['flow_is_valid'] = batch['flow_is_valid'][~batch['gm0']].unsqueeze(0)
        batch['eval_mask'] = batch['eval_mask'][~batch['gm0']].unsqueeze(0)
        batch['pc0'] = batch['pc0'][~batch['gm0']].unsqueeze(0)
        batch['pc1'] = batch['pc1'][~batch['gm1']].unsqueeze(0)
        batch['flow'] = batch['flow'][~batch['gm0']].unsqueeze(0)
        batch['flow_category_indices_1'] = batch['flow_category_indices_1'][~batch['gm1']].unsqueeze(0)
        batch['flow_category_indices'] = batch['flow_category_indices'][~batch['gm0']].unsqueeze(0)

        original = batch['pc1'].clone()

        targeted_objects_mask_1 = get_object_mask(torch.zeros_like(batch['flow_category_indices_1']),
                                                  batch['flow_category_indices_1'])
        targeted_objects_mask_0 = get_object_mask(torch.zeros_like(batch['flow_category_indices']),
                                                  batch['flow_category_indices'])

        if random_start:
            batch['pc1'][targeted_objects_mask_1] = batch['pc1'][targeted_objects_mask_1] + torch.empty_like(
                batch['pc1'][targeted_objects_mask_1]).uniform_(
                -eps, eps
            )
        model.model.eval()

        for i in range(itr + 1):
            batch['pc1'].requires_grad = True
            model.model.cuda()
            res_dict = model.model(batch)
            gt_org = batch['flow']

            gt_flow = batch['flow']

            pose_flows = pose_flow_from_original

            pc0_valid_idx = res_dict['pc0_valid_point_idxes']  # since padding
            est_flow = res_dict['flow']

            pc0_valid_from_pc2res = pc0_valid_idx[0]
            valid_flows = batch['flow_is_valid'][0][pc0_valid_from_pc2res]
            eval_mask = batch['eval_mask'][0][pc0_valid_from_pc2res].squeeze()
            valid_flows = torch.logical_and(valid_flows, eval_mask)  # should be both valid and within eval mask
            pose_flow_ = pose_flows[0][pc0_valid_from_pc2res].cuda()
            est_flow_ = est_flow[0]
            gt_flow_ = gt_flow[0][pc0_valid_from_pc2res].cuda()
            gt_org_flow_ = gt_org[0][pc0_valid_from_pc2res]
            targeted_objects_mask_0_ = targeted_objects_mask_0[0][pc0_valid_from_pc2res]

            pose_flow_ = pose_flow_[valid_flows]
            est_flow_ = est_flow_[valid_flows]
            gt_flow_ = gt_flow_[valid_flows]
            gt_org_flow_ = gt_org_flow_[valid_flows]
            targeted_objects_mask_0_ = targeted_objects_mask_0_[valid_flows]

            dist1, dist2 = chamfer_dist(batch['pc1'].cuda(), original.cuda())
            chamfer_loss_pc1 = (torch.mean(dist1)) + (torch.mean(dist2))
            mse = torch.linalg.vector_norm(batch['pc1'].cpu() - original.cpu(), dim=-1).mean()

            gt_flow_ = gt_flow_ - pose_flow_
            if targeted:
                gt_flow_ = -gt_flow_
            res_dict_eval = {'est_flow': est_flow_,
                             'gt_flow': gt_flow_
                             }

            res_dict_eval_2 = {'est_flow': est_flow_ + pose_flow_,
                               'gt_flow': gt_org_flow_
                               }
            model.model.zero_grad()
            total_loss_back, total_loss = model.loss_fn(res_dict_eval, targeted_objects_mask_0_)
            total_loss_org_back, total_loss_org = model.loss_fn(res_dict_eval_2, targeted_objects_mask_0_)

            if targeted:
                if (cos_pgd):
                    cos_reg = 1 - f_cosim(res_dict_eval)
                    total_loss_cos = cos_reg * total_loss_back
                    total_loss_obj = total_loss_org[targeted_objects_mask_0_].mean()
                    total_loss_cos.mean().backward()
                else:
                    total_loss = total_loss_back
                    total_loss_obj = total_loss_org[targeted_objects_mask_0_].mean()
                    total_loss.mean().backward()
            else:
                if (cos_pgd):
                    cos_reg = f_cosim(res_dict_eval)
                    total_loss_cos = cos_reg * total_loss_back
                    total_loss_obj = total_loss[targeted_objects_mask_0_].mean()
                    total_loss_cos.mean().backward()
                else:
                    total_loss_obj = total_loss[targeted_objects_mask_0_].mean()
                    total_loss_back.mean().backward()
                total_loss_org = total_loss

            grad = batch['pc1'].grad
            batch['pc1'].requires_grad = False
            if targeted:
                batch['pc1'][targeted_objects_mask_1] = batch['pc1'][targeted_objects_mask_1] - alpa * grad.sign()[
                    targeted_objects_mask_1]
            else:
                batch['pc1'][targeted_objects_mask_1] = batch['pc1'][targeted_objects_mask_1] + alpa * grad.sign()[
                    targeted_objects_mask_1]
            pert = torch.clamp(batch['pc1'] - original, min=-eps, max=eps)
            batch['pc1'] = original + pert

            if (i == (itr - 1)):
                path = "/home/yasas/UNSW/Adv_Temporal/SceneFlow/attack_data/global_cos_pgd/" + str(step) + ".pt"
                torch.save(batch['pc1'], path)

        # pc0 = res_dict['pcd0']
        # pc1 = res_dict['pcd1']
        # vis_p(pc0, pc1)
        mask_is_dynamic = torch.linalg.vector_norm(gt_org_flow_.cpu() - pose_flow_.cpu(), dim=-1) >= 0.05
        mask_dynamic_objects = torch.logical_and(mask_is_dynamic, targeted_objects_mask_0_)
        mask_static_objects = torch.logical_and(~mask_is_dynamic, targeted_objects_mask_0_)
        attack_loss_dynamic = total_loss_org[mask_dynamic_objects].mean()
        attack_loss_static = total_loss_org[mask_static_objects].mean()
        original_loss, est_original_flow = normal_loop(model, batch_org)

        mask_pred_original_is_dynamic = torch.linalg.vector_norm(est_original_flow.cpu(), dim=-1) >= 0.05
        mask_pred_attack_is_dynamic = torch.linalg.vector_norm(est_flow_.cpu(), dim=-1) >= 0.05

        dynamic_accuracy_attack = torch.sum(
            mask_pred_attack_is_dynamic[targeted_objects_mask_0_] == mask_is_dynamic[targeted_objects_mask_0_]) / len(
            mask_pred_attack_is_dynamic[targeted_objects_mask_0_])

        dynamic_accuracy_original = torch.sum(
            mask_pred_original_is_dynamic[targeted_objects_mask_0_] == mask_is_dynamic[targeted_objects_mask_0_]) / len(
            mask_pred_original_is_dynamic[targeted_objects_mask_0_])

        dynamic_iou_attack = calculate_iou(mask_pred_attack_is_dynamic[targeted_objects_mask_0_],
                                           mask_is_dynamic[targeted_objects_mask_0_])
        dynamic_iou_original = calculate_iou(mask_pred_original_is_dynamic[targeted_objects_mask_0_],
                                             mask_is_dynamic[targeted_objects_mask_0_])

        static_iou_attack = calculate_iou(~mask_pred_attack_is_dynamic[targeted_objects_mask_0_],
                                          ~mask_is_dynamic[targeted_objects_mask_0_])
        static_iou_original = calculate_iou(~mask_pred_original_is_dynamic[targeted_objects_mask_0_],
                                            ~mask_is_dynamic[targeted_objects_mask_0_])

        relax_acc_attack = compute_accuracy(est_flow_[mask_dynamic_objects].detach().cpu().numpy(),
                                            (gt_org_flow_[mask_dynamic_objects].cpu() - pose_flow_[
                                                mask_dynamic_objects].cpu()).numpy(), 0.05).mean()
        relax_acc_original = compute_accuracy(est_original_flow[mask_dynamic_objects].detach().cpu().numpy(),
                                              (gt_org_flow_[mask_dynamic_objects].cpu() - pose_flow_[
                                                  mask_dynamic_objects].cpu()).numpy(), 0.05).mean()

        original_loss_dynamic = original_loss[mask_dynamic_objects].mean()
        original_loss_static = original_loss[mask_static_objects].mean()
        original_loss = original_loss[targeted_objects_mask_0_].mean()

        if str(total_loss_obj.item()) == "nan" or gt_org_flow_[targeted_objects_mask_0_].isnan().any() or est_flow_[
            targeted_objects_mask_0_].isnan().any() or \
                pose_flow_[targeted_objects_mask_0_].isnan().any():
            print("Skipping Found Nan")
        else:
            chamfer_list.append(chamfer_loss_pc1.item())
            MSE_atack.append(mse.item())
            dynamic_accuracies_attack.append(dynamic_accuracy_attack.item())
            dynamic_accuracies_original.append(dynamic_accuracy_original.item())
            relax_accuracies_attack.append(relax_acc_attack)
            relax_accuracies_original.append(relax_acc_original)
            if (str(original_loss_dynamic.item()) != "nan"):
                dynamic_ious_attack.append(dynamic_iou_attack)
                dynamic_ious_original.append(dynamic_iou_original)
            if (str(original_loss_static.item()) != "nan"):
                static_ious_attack.append(static_iou_attack)
                static_ious_original.append(static_iou_original)
            EPE_attack.append(total_loss_obj.item())
            EPE_attack_dynamic.append(attack_loss_dynamic.item())
            EPE_attack_static.append(attack_loss_static.item())
            EPE_original.append(original_loss.item())
            EPE_original_dynamic.append(original_loss_dynamic.item())
            EPE_original_static.append(original_loss_static.item())
            print("Attack: " + str(total_loss_obj))
            print("Attack Dynamic: " + str(attack_loss_dynamic))
            print("Attack Static: " + str(attack_loss_static))
            print("Dynamic Accuracy Attack: " + str(dynamic_accuracy_attack.item()))
            print("Dynamic Iou Attack: " + str(dynamic_iou_attack))
            print("Static Iou Attack: " + str(static_iou_attack))
            print("Relax Accuracy Attack: " + str(relax_acc_attack))
            print("Original: " + str(original_loss))
            print("Original Dynamic: " + str(original_loss_dynamic))
            print("Original Static: " + str(original_loss_static))
            print("Dynamic Accuracy Original: " + str(dynamic_accuracy_original.item()))
            print("Dynamic Iou Original: " + str(dynamic_iou_original))
            print("Static Iou Original: " + str(static_iou_original))
            print("Relax Accuracy Original: " + str(relax_acc_original))
            print("Chamfer:" + str(chamfer_loss_pc1))
            print("MSE:" + str(mse))
            print(str(total_loss_obj.item()) + "," + str(attack_loss_dynamic.item()) + "," + str(
                attack_loss_static.item()) + "," + str(dynamic_accuracy_attack.item()) + "," +
                  str(dynamic_iou_attack) + "," + str(static_iou_attack) + "," + str(relax_acc_attack) + "," + str(
                original_loss.item()) + "," + str(original_loss_dynamic.item()) + "," + str(
                original_loss_static.item()) + "," + str(dynamic_accuracy_original.item()) + "," + str(
                dynamic_iou_original) + "," + str(static_iou_original) + "," + str(relax_acc_original) + "," + str(
                chamfer_loss_pc1.item()) + "," + str(mse.item()), file=results)
    print("==============Results Deflow PGD Global Loss Random False EPS = 0.5 Alpha= 0.01===============")
    print("Attack:" + str(np.nanmean(EPE_attack)))
    print("Attack Dynamic:" + str(np.nanmean(EPE_attack_dynamic)))
    print("Attack Static:" + str(np.nanmean(EPE_attack_static)))
    print("Dynamic Accuracy Attack:" + str(np.nanmean(dynamic_accuracies_attack)))
    print("Dynamic IoU Attack:" + str(np.nanmean(dynamic_ious_attack)))
    print("Static IoU Attack:" + str(np.nanmean(static_ious_attack)))
    print("Relax Accuracy Attack:" + str(np.nanmean(relax_accuracies_attack)))
    print("Original:" + str(np.nanmean(EPE_original)))
    print("Original Dynamic:" + str(np.nanmean(EPE_original_dynamic)))
    print("Original Static:" + str(np.nanmean(EPE_original_static)))
    print("Dynamic Accuracy Original:" + str(np.nanmean(dynamic_accuracies_original)))
    print("Dynamic IoU Original:" + str(np.nanmean(dynamic_ious_original)))
    print("Static IoU Original:" + str(np.nanmean(static_ious_original)))
    print("Relax Accuracy Original:" + str(np.nanmean(relax_accuracies_original)))
    print("Chamfer:" + str(np.nanmean(chamfer_list)))
    print("MSE:" + str(np.nanmean(MSE_atack)))

    print("==============Results Deflow PGD Global Loss Random False EPS = 0.5 Alpha= 0.01===============",
          file=f)
    print("Attack:" + str(np.nanmean(EPE_attack)), file=f)
    print("Attack Dynamic:" + str(np.nanmean(EPE_attack_dynamic)), file=f)
    print("Attack Static:" + str(np.nanmean(EPE_attack_static)), file=f)
    print("Dynamic Accuracy Attack:" + str(np.nanmean(dynamic_accuracies_attack)), file=f)
    print("Relax Accuracy Attack:" + str(np.nanmean(relax_accuracies_attack)), file=f)
    print("Dynamic IoU Attack:" + str(np.nanmean(dynamic_ious_attack)), file=f)
    print("Static IoU Attack:" + str(np.nanmean(static_ious_attack)), file=f)
    print("Original:" + str(np.nanmean(EPE_original)), file=f)
    print("Original Dynamic:" + str(np.nanmean(EPE_original_dynamic)), file=f)
    print("Original Static:" + str(np.nanmean(EPE_original_static)), file=f)
    print("Dynamic Accuracy Original:" + str(np.nanmean(dynamic_accuracies_original)), file=f)
    print("Dynamic IoU Original:" + str(np.nanmean(dynamic_ious_original)), file=f)
    print("Static IoU Original:" + str(np.nanmean(static_ious_original)), file=f)
    print("Relax Accuracy Original:" + str(np.nanmean(relax_accuracies_original)), file=f)
    print("Chamfer:" + str(np.nanmean(chamfer_list)), file=f)
    print("MSE:" + str(np.nanmean(MSE_atack)), file=f)


def avg_mse(flow1, flow2):
    return torch.mean((flow1 - flow2) ** 2)


def f_mse(res_dict):
    pred = res_dict['est_flow'].cuda()
    target = res_dict['gt_flow'].cuda()
    return avg_mse(pred, target)


def f_cosim(res_dict):
    pred = res_dict['est_flow'].cuda()
    target = res_dict['gt_flow'].cuda()
    return torch.sum(pred * target, dim=1) / torch.sqrt(torch.sum(pred * pred, dim=1)) * torch.sqrt(
        torch.sum(target * target, dim=1))


def avg_epe(res_dict):
    flow1 = res_dict['est_flow'].cuda()
    flow2 = res_dict['gt_flow'].cuda()
    diff_squared = (flow1 - flow2) ** 2
    epe = torch.mean(torch.sum(diff_squared, dim=-1).sqrt())
    return epe


def smooth(res_dict, neighbour=100):
    pred = res_dict['est_flow'].cuda()
    target = res_dict['gt_flow'].cuda()
    dist = torch.cdist(pred, target)  # [4096, 4096]
    sorted_dist, ind_dist = torch.sort(dist, dim=1)
    return sorted_dist[:, :neighbour]


def vis_p(pc0s, pc1s):
    pcd = o3d.geometry.PointCloud()
    points_transformed = pc0s.detach().cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(points_transformed[0])
    pcd.paint_uniform_color([1.0, 0, 0.0])

    pcd2 = o3d.geometry.PointCloud()
    points_2 = pc1s.detach().cpu().numpy()[0]  # [:-5000]
    pcd2.points = o3d.utility.Vector3dVector(points_2)
    pcd2.paint_uniform_color([0.0, 0.0, 1.0])

    o3d.visualization.draw_geometries([pcd2, pcd])


def vis_flow(flow_without_pose, pc0s, mask):
    VIEW_FILE = f"assets/view/av2.json"
    o3d_vis = MyVisualizer(view_file=VIEW_FILE)
    opt = o3d_vis.vis.get_render_option()
    opt.background_color = np.asarray([216, 216, 216]) / 255.0
    # opt.background_color = np.asarray([80 / 255, 90 / 255, 110 / 255])
    # opt.background_color = np.asarray([1, 1, 1])
    critical_objects = pc0s[mask]

    pc0 = pc0s.detach().cpu().numpy()
    flow_without_pose = flow_without_pose.detach().cpu().numpy()
    flow_color = flow_to_rgb(flow_without_pose) / 255.0
    flow_color[~mask] = [0, 0, 0]
    pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc0[:, :3][~gm0])
    # pcd.colors = o3d.utility.Vector3dVector(flow_color[~gm0])
    pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(flow_color)
    o3d_vis.update([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)])

    # flow_without_pose = flow_without_pose[mask].detach().cpu().numpy()
    #
    # pc0 = critical_objects.detach().cpu().numpy()
    # pc0_adversarial = pc0 + flow_without_pose
    #
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
    # pcd.paint_uniform_color([1.0, 0, 0])
    #
    # pcd_adv = o3d.geometry.PointCloud()
    # pcd_adv.points = o3d.utility.Vector3dVector(pc0_adversarial[:, :3])
    # pcd_adv.paint_uniform_color([0.0, 0.0, 1.0])
    #
    # o3d.visualization.draw_geometries([pcd, pcd_adv])


def normal_loop(model, batch):
    batch['origin_pc0'] = batch['pc0'].clone()
    pc0 = batch['pc0'][0]
    pose_0to1 = cal_pose0to1(batch["pose0"][0], batch["pose1"][0])
    transform_pc0 = pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
    pose_flow = transform_pc0 - pc0

    pose_flow_from_original = pose_flow[~batch['gm0'][0]].unsqueeze(0)
    batch['eval_mask'] = batch['eval_mask'][~batch['gm0']].unsqueeze(0)
    batch['flow_is_valid'] = batch['flow_is_valid'][~batch['gm0']].unsqueeze(0)
    batch['pc0'] = batch['pc0'][~batch['gm0']].unsqueeze(0)
    batch['pc1'] = batch['pc1'][~batch['gm1']].unsqueeze(0)
    model.model.cuda()
    res_dict = model.model(batch)

    gt_flow = batch['flow'][~batch['gm0']].unsqueeze(0)
    pose_flows = pose_flow_from_original
    pc0_valid_idx = res_dict['pc0_valid_point_idxes']  # since padding
    est_flow = res_dict['flow']
    est_flow_ = est_flow[0]

    pc0_valid_from_pc2res = pc0_valid_idx[0]
    valid_flows = batch['flow_is_valid'][0][pc0_valid_from_pc2res]
    eval_mask = batch['eval_mask'][0][pc0_valid_from_pc2res].squeeze()
    valid_flows = torch.logical_and(valid_flows, eval_mask)
    pose_flow_ = pose_flows[0][pc0_valid_from_pc2res].cuda()
    gt_flow_ = gt_flow[0][pc0_valid_from_pc2res]

    pose_flow_ = pose_flow_[valid_flows]
    est_flow_ = est_flow_[valid_flows]
    gt_flow_ = gt_flow_[valid_flows]

    # gt_flow_ = gt_flow_ - pose_flow_
    res_dict = {'est_flow': est_flow_ + pose_flow_,
                'gt_flow': gt_flow_
                }
    model.model.zero_grad()
    _, loss = model.loss_fn(res_dict)
    return loss, est_flow_


if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()
