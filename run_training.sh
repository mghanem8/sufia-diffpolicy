#!/bin/bash
export HF_HOME=/lustre/fsw/portfolios/healthcareeng/users/nigeln/hf_home
export TRANSFORMERS_CACHE=/lustre/fsw/portfolios/healthcareeng/users/nigeln/hf_home

# GPU 0
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=lift_tissue-50 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/lift_tissue-50" \
    training.seed=0 \
    training.device="cuda:0" \
    exp_name=lift_tissue-50 \
    logging.mode=online &

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=liver_needle_lift-30 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/liver_needle_lift-30" \
    training.seed=0 \
    training.device="cuda:0" \
    exp_name=liver_needle_lift-30 \
    logging.mode=online &

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=peg_transfer-20 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/peg_transfer-20" \
    training.seed=0 \
    training.device="cuda:0" \
    exp_name=peg_transfer-20 \
    logging.mode=online &

# GPU 1
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=liver_needle_lift-50 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/liver_needle_lift-50" \
    training.seed=0 \
    training.device="cuda:1" \
    exp_name=liver_needle_lift-50 \
    logging.mode=online &

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=liver_needle_handover-30 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/liver_needle_handover-30" \
    training.seed=0 \
    training.device="cuda:1" \
    exp_name=liver_needle_handover-30 \
    logging.mode=online &

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=suture_pad-20 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/suture_pad-20" \
    training.seed=0 \
    training.device="cuda:1" \
    exp_name=suture_pad-20 \
    logging.mode=online &

# GPU 2
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=liver_needle_handover-50 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/liver_needle_handover-50" \
    training.seed=0 \
    training.device="cuda:2" \
    exp_name=liver_needle_handover-50 \
    logging.mode=online &

CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=peg_transfer-30 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/peg_transfer-30" \
    training.seed=0 \
    training.device="cuda:2" \
    exp_name=peg_transfer-30 \
    logging.mode=online &

CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=lift_tissue-10 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/lift_tissue-10" \
    training.seed=0 \
    training.device="cuda:2" \
    exp_name=lift_tissue-10 \
    logging.mode=online &

CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=suture_pad-10 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/suture_pad-10" \
    training.seed=0 \
    training.device="cuda:2" \
    exp_name=suture_pad-10 \
    logging.mode=online &

# GPU 3
CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=peg_transfer-50 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/peg_transfer-50" \
    training.seed=0 \
    training.device="cuda:3" \
    exp_name=peg_transfer-50 \
    logging.mode=online &

CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=suture_pad-30 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/suture_pad-30" \
    training.seed=0 \
    training.device="cuda:3" \
    exp_name=suture_pad-30 \
    logging.mode=online &

CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=liver_needle_lift-10 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/liver_needle_lift-10" \
    training.seed=0 \
    training.device="cuda:3" \
    exp_name=liver_needle_lift-10 \
    logging.mode=online &

# GPU 4
CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=suture_pad-50 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/suture_pad-50" \
    training.seed=0 \
    training.device="cuda:4" \
    exp_name=suture_pad-50 \
    logging.mode=online &

CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=lift_tissue-20 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/lift_tissue-20" \
    training.seed=0 \
    training.device="cuda:4" \
    exp_name=lift_tissue-20 \
    logging.mode=online &

CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=liver_needle_lift-20 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/liver_needle_lift-20" \
    training.seed=0 \
    training.device="cuda:4" \
    exp_name=liver_needle_lift-20 \
    logging.mode=online &

# GPU 5
CUDA_VISIBLE_DEVICES=5 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=lift_tissue-40 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/lift_tissue-40" \
    training.seed=0 \
    training.device="cuda:5" \
    exp_name=lift_tissue-40 \
    logging.mode=online &

CUDA_VISIBLE_DEVICES=5 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=peg_transfer-40 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/peg_transfer-40" \
    training.seed=0 \
    training.device="cuda:5" \
    exp_name=peg_transfer-40 \
    logging.mode=online &

CUDA_VISIBLE_DEVICES=5 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=liver_needle_handover-10 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/liver_needle_handover-10" \
    training.seed=0 \
    training.device="cuda:5" \
    exp_name=liver_needle_handover-10 \
    logging.mode=online &

# GPU 6
CUDA_VISIBLE_DEVICES=6 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=liver_needle_lift-40 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/liver_needle_lift-40" \
    training.seed=0 \
    training.device="cuda:6" \
    exp_name=liver_needle_lift-40 \
    logging.mode=online &

CUDA_VISIBLE_DEVICES=6 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=suture_pad-40 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/suture_pad-40" \
    training.seed=0 \
    training.device="cuda:6" \
    exp_name=suture_pad-40 \
    logging.mode=online &

CUDA_VISIBLE_DEVICES=6 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=peg_transfer-10 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/peg_transfer-10" \
    training.seed=0 \
    training.device="cuda:6" \
    exp_name=peg_transfer-10 \
    logging.mode=online &

# GPU 7
CUDA_VISIBLE_DEVICES=7 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=liver_needle_handover-40 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/liver_needle_handover-40" \
    training.seed=0 \
    training.device="cuda:7" \
    exp_name=liver_needle_handover-40 \
    logging.mode=online &

CUDA_VISIBLE_DEVICES=7 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=lift_tissue-30 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/lift_tissue-30" \
    training.seed=0 \
    training.device="cuda:7" \
    exp_name=lift_tissue-30 \
    logging.mode=online &

CUDA_VISIBLE_DEVICES=7 HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=liver_needle_handover-20 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/liver_needle_handover-20" \
    training.seed=0 \
    training.device="cuda:7" \
    exp_name=liver_needle_handover-20 \
    logging.mode=online &

wait
echo "All processes completed."
