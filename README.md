# JPO:Juris Policy Optimization for Structured Legal Reasoning in Criminal Judgment Prediction

This repository contains the source code for the paper *JPO:Juris Policy Optimization for Structured Legal Reasoning in Criminal Judgment Prediction* submitted to EMNLP 2026. This project addresses the stringent requirements of structured legal reasoning in criminal judgment prediction tasks by proposing a reinforcement learning framework for large language models, aimed at guiding the model to generate judgment outcomes that are both accurate and supported by rigorous reasoning chains. Below, we provide a detailed guide on how to set up and use this project.

## Installation

1. Clone this repository to your local machine.
2. Install the required dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

### Step 1: Prepare SFT Data

Run the following script to prepare the SFT dataset. This script uses the vLLM framework and a teacher model to generate chain-of-thought data for each legal case:

```bash
python prepare_dataset/prepare_dataset_sft.py
```

### Step 2: Prepare RL Data

Run the following script to convert RL data into a format accepted by the VERL framework:

```bash
python prepare_dataset/prepare_dataset_rl.py
```

1. You can modify the global variables in these scripts to adjust parameters as needed.

2. `data/train_sft_sampled.json` and `data/train_rl_sampled.json` are sample files obtained by sampling 100 entries from the SFT training data and 50 entries from the RL training data, respectively, after running the data preprocessing program. They are provided as examples. Information such as dates, names, and locations has been redacted.

## Supervised Fine-Tuning (SFT)

1. Convert the chain-of-thought SFT data into a format accepted by the VERL framework:

   ```bash
   python sft/change_dataset_format_to_verl.py
   ```

2. Run the SFT training script. Below is an example script:

   ```bash
   #!/bin/bash
   set -x

   nproc_per_node=2
   save_path="models/qwen2_5_3b/sft"

   torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m verl.trainer.fsdp_sft_trainer \
       data.train_files=data/preprocessed_data/train_sft.parquet \
       data.val_files=data/preprocessed_data/test_sft.parquet \
       data.prompt_key=question \
       data.response_key=answer \
       optim.lr=2e-5 \
       data.micro_batch_size_per_gpu=4 \
       model.partial_pretrain=Qwen/Qwen2.5-3B-Instruct \
       trainer.default_local_dir=$save_path \
       trainer.project_name=qwen2_5_3b_sft \
       trainer.experiment_name=qwen2_5_3b_sft \
       trainer.logger='["console","tensorboard"]' \
       trainer.total_epochs=2 \
       data.max_length=3072 \
       model.enable_gradient_checkpointing=False \
       model.trust_remote_code=True \
       trainer.save_freq=400
   ```

## Reinforcement Learning (RL)

1. Build a Naive Bayes model for calculating logical consistency scores between facts and law articles:

   ```bash
   python rl/legal_naive_bayes.py
   ```

2. Compute the mean and standard deviation of sentencing terms for each accusation for calculating logical consistency scores between charges and sentencing terms:

   ```bash
   python rl/make_json_file_accusation_to_term_mean_std.py
   ```

3. Run the RL training script. The reward function is defined in `rl/reward.py`, including the implementation of sequence-level reward and token-level reward. Below is an example script:

   ```bash
   set -x

   python3 -m verl.trainer.main_ppo \
       algorithm.adv_estimator=grpo \
       data.train_files=data/preprocessed_data/train_rl.parquet \
       data.val_files=data/preproccessed_data/test_rl.parquet \
       data.train_batch_size=1024 \
       data.max_prompt_length=2048 \
       data.max_response_length=2048 \
       data.filter_overlong_prompts=True \
       data.truncation='error' \
       actor_rollout_ref.model.path=models/qwen2_5_3b/sft/huggingface \
       actor_rollout_ref.actor.optim.lr=1e-6 \
       actor_rollout_ref.model.use_remove_padding=False \
       actor_rollout_ref.actor.ppo_mini_batch_size=128 \
       actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
       actor_rollout_ref.actor.use_kl_loss=True \
       actor_rollout_ref.actor.entropy_coeff=0 \
       actor_rollout_ref.actor.kl_loss_coef=0.001 \
       actor_rollout_ref.actor.kl_loss_type=low_var_kl \
       actor_rollout_ref.model.enable_gradient_checkpointing=False \
       actor_rollout_ref.actor.fsdp_config.param_offload=False \
       actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
       actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
       actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
       actor_rollout_ref.rollout.name=vllm \
       actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
       actor_rollout_ref.rollout.n=5 \
       actor_rollout_ref.rollout.enable_chunked_prefill=False \
       actor_rollout_ref.rollout.enforce_eager=False \
       actor_rollout_ref.rollout.free_cache_engine=False \
       actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
       actor_rollout_ref.ref.fsdp_config.param_offload=True \
       algorithm.use_kl_in_reward=False \
       custom_reward_function.path=rl/reward.py \
       custom_reward_function.name=ljp_reward_fn_seq_level \
       trainer.critic_warmup=0 \
       trainer.logger='["console","tensorboard"]' \
       trainer.project_name='qwen2_5_3b_rl' \
       trainer.experiment_name='qwen2_5_3b_rl' \
       trainer.n_gpus_per_node=4 \
       trainer.nnodes=1 \
       trainer.save_freq=9 \
       trainer.test_freq=-1 \
       trainer.default_local_dir=/models/qwen2_5_3b/rl \
       trainer.total_epochs=4 $@
   ```

## Evaluation

1. Generate judgments for the test set using the trained model:

   ```bash
   python eval.py
   ```

   You can modify the global variables in the script to adjust hyperparameters.

2. Score the model's outputs:

   ```bash
   python acc_score.py
   ```