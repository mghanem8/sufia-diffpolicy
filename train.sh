export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

# tasks=("orbit_transfer_block" "orbit_dual_suture" "orbit_handover_needle" "orbit_needle" "orbit_tissue")
tasks=("orbit_needle_multi")
num_demos=(50)

for task in ${tasks[@]}; do
    for num_demo in ${num_demos[@]}; do
        exp_name="${task}_${num_demo}"
        sbatch slurm/run_rtx6000.sbatch python train.py --config-name="train_diffusion_unet_ddim_hybrid_workspace.yaml" \
            task=${task} \
            hydra.run.dir="/storage/home/hcoda1/7/mghanem8/codes/quest_v0/experiments/diffpolicy/${exp_name}" \
            training.seed=0 \
            training.device="cuda:0" \
            exp_name=${exp_name}_multi-cam\
            logging.mode=online \
            task.dataset.max_train_episodes=${num_demo}
    done
done