#!/bin/bash
export HF_HOME=/lustre/fsw/portfolios/healthcareeng/users/nigeln/hf_home
export TRANSFORMERS_CACHE=/lustre/fsw/portfolios/healthcareeng/users/nigeln/hf_home

# GPU 0
HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=lift_tissue-50 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/lift_tissue-50" \
    training.seed=0 \
    training.device="cuda:0" \
    exp_name=lift_tissue-50-1 \
    logging.mode=online &

HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=liver_needle_lift-30 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/liver_needle_lift-30" \
    training.seed=0 \
    training.device="cuda:0" \
    exp_name=liver_needle_lift-30-1\
    logging.mode=online &

HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=peg_transfer-20 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/peg_transfer-20" \
    training.seed=0 \
    training.device="cuda:0" \
    exp_name=peg_transfer-20-1 \
    logging.mode=online &

# GPU 1
HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=liver_needle_lift-50 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/liver_needle_lift-50" \
    training.seed=0 \
    training.device="cuda:1" \
    exp_name=liver_needle_lift-50-1 \
    logging.mode=online &

HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=needle_handover-30 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/needle_handover-30" \
    training.seed=0 \
    training.device="cuda:1" \
    exp_name=needle_handover-30-1 \
    logging.mode=online &

HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=suture_pad-20 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/suture_pad-20" \
    training.seed=0 \
    training.device="cuda:1" \
    exp_name=suture_pad-20-1 \
    logging.mode=online &

# GPU 2
HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=needle_handover-50 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/needle_handover-50" \
    training.seed=0 \
    training.device="cuda:2" \
    exp_name=needle_handover-50-1 \
    logging.mode=online &

HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=peg_transfer-30 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/peg_transfer-30" \
    training.seed=0 \
    training.device="cuda:2" \
    exp_name=peg_transfer-30-1 \
    logging.mode=online &

HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=lift_tissue-10 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/lift_tissue-10" \
    training.seed=0 \
    training.device="cuda:2" \
    exp_name=lift_tissue-10-1 \
    logging.mode=online &

HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=suture_pad-10 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/suture_pad-10" \
    training.seed=0 \
    training.device="cuda:2" \
    exp_name=suture_pad-10 \
    logging.mode=online &

# GPU 3
HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=peg_transfer-50 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/peg_transfer-50" \
    training.seed=0 \
    training.device="cuda:3" \
    exp_name=peg_transfer-50-1 \
    logging.mode=online &

HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=suture_pad-30 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/suture_pad-30" \
    training.seed=0 \
    training.device="cuda:3" \
    exp_name=suture_pad-30-1 \
    logging.mode=online &

HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=liver_needle_lift-10 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/liver_needle_lift-10" \
    training.seed=0 \
    training.device="cuda:3" \
    exp_name=liver_needle_lift-10-1 \
    logging.mode=online &

# GPU 4
HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=suture_pad-50 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/suture_pad-50" \
    training.seed=0 \
    training.device="cuda:4" \
    exp_name=suture_pad-50-1 \
    logging.mode=online &

HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=lift_tissue-20 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/lift_tissue-20" \
    training.seed=0 \
    training.device="cuda:4" \
    exp_name=lift_tissue-20-1 \
    logging.mode=online &

HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=liver_needle_lift-20 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/liver_needle_lift-20" \
    training.seed=0 \
    training.device="cuda:4" \
    exp_name=liver_needle_lift-20-1 \
    logging.mode=online &

# GPU 5
HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=lift_tissue-40 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/lift_tissue-40" \
    training.seed=0 \
    training.device="cuda:5" \
    exp_name=lift_tissue-40-1 \
    logging.mode=online &

HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=peg_transfer-40 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/peg_transfer-40" \
    training.seed=0 \
    training.device="cuda:5" \
    exp_name=peg_transfer-40-1 \
    logging.mode=online &

HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=needle_handover-10 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/needle_handover-10" \
    training.seed=0 \
    training.device="cuda:5" \
    exp_name=needle_handover-10-1 \
    logging.mode=online &

# GPU 6
HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=liver_needle_lift-40 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/liver_needle_lift-40" \
    training.seed=0 \
    training.device="cuda:6" \
    exp_name=liver_needle_lift-40-1 \
    logging.mode=online &

HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=suture_pad-40 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/suture_pad-40" \
    training.seed=0 \
    training.device="cuda:6" \
    exp_name=suture_pad-40-1 \
    logging.mode=online &

HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=peg_transfer-10 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/peg_transfer-10" \
    training.seed=0 \
    training.device="cuda:6" \
    exp_name=peg_transfer-10-1 \
    logging.mode=online &

# GPU 7
HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=needle_handover-40 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/needle_handover-40" \
    training.seed=0 \
    training.device="cuda:7" \
    exp_name=needle_handover-40-1 \
    logging.mode=online &

HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=lift_tissue-30 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/lift_tissue-30" \
    training.seed=0 \
    training.device="cuda:7" \
    exp_name=lift_tissue-30-1 \
    logging.mode=online &

HYDRA_FULL_ERROR=1 python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
    task=needle_handover-20 \
    hydra.run.dir="/lustre/fsw/portfolios/healthcareeng/users/nigeln/diffusion-rgb-multi-cam/liver_nneedle_handovereedle_handover-20" \
    training.seed=0 \
    training.device="cuda:7" \
    exp_name=needle_handover-20-1 \
    logging.mode=online &

wait
echo "All processes completed."
