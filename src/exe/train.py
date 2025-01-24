# CUDA devices should be recognized first.
# isort: off
import set_cuda

# isort: on
import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

# isort: off
import path
import utils
from data import ComplexDataModule
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)
# torch.multiprocessing.set_sharing_strategy("file_system")


def run(
    model: torch.nn.Module,
    data: ComplexDataModule,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    train: bool,
):
    if train:
        model.train()
        loaders = data.train_dataloader()
    else:
        model.eval()
        loaders = data.val_dataloader()

    tasks = list(loaders.keys())
    total_batch_iteraion = 0
    for idx, batch in enumerate(tqdm(zip(*(loaders[task] for task in tasks)))):
        batch = dict(zip(tasks, batch))
        batch = {task: batch[task].to(device) for task in batch}

        if train:
            model.zero_grad()
            loss_total = model.training_step(batch)
            loss_total.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model.validation_step(batch)

        # wandb
        monitoring_value = dict()
        train_type = "train" if train else "test"
        for k1 in model.losses.keys():
            for k2 in model.losses[k1].keys():
                monitoring_value[f"{train_type}_{k1}_{k2}"] = model.losses[k1][k2][-1]
        wandb.log(monitoring_value)

        if train and idx > 4000:
            break
        elif not train and idx > 200:
            break
        total_batch_iteraion += 1


def init_wandb(config):
    # wandb/run-날짜시간-ID/files/config.yaml 형식으로 저장됨
    exp_dir = str(config.experiment_name)
    wandb_dir = "wandb"
    run_id = None
    if os.path.exists(f"{wandb_dir}/latest-run"):
        run_dirs = [
            d for d in os.listdir(f"{wandb_dir}/latest-run") if d.startswith("run-")
        ]
        for run_dir in run_dirs:
            run_id = run_dir.split("-")[-1].split(".")[0]  # run ID는 마지막 부분
    if run_id:
        wandb.init(project="pignet", id=run_id, resume=True)
    else:
        wandb.init(project="pignet", name=exp_dir)


@hydra.main(config_path="../config", config_name="config_train")
def main(config: DictConfig):
    init_wandb(config)
    logger = utils.initialize_logger(config.run.log_file)
    logger.info(f"Current working directory: {os.getcwd()}")

    os.makedirs(config.run.checkpoint_dir, exist_ok=True)
    os.makedirs(config.run.tensorboard_dir, exist_ok=True)

    # Set GPUs.
    gpu_idx = utils.cuda_visible_devices(config.run.ngpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the checkpoint if exists.
    if config.run.restart_file and os.path.isfile(config.run.restart_file):
        checkpoint = torch.load(config.run.restart_file, map_location=device)
        config = utils.merge_configs(checkpoint["config"], config)
        logger.info(f"Restart from: {os.path.realpath(config.run.restart_file)}")
    else:
        checkpoint = None

    logger.info(OmegaConf.to_yaml(config, resolve=True))
    logger.info(f"device: {repr(device)}, gpu_idx: {gpu_idx}")

    # Set a seed for reproducibility.
    if config.run.seed is not None:
        utils.seed(config.run.seed)
        logger.warning(
            "WARNING: Currently, manual seeding does not guarantee reproducibility!"
        )

    # Datamodule containing datasets & loaders
    data = ComplexDataModule(config)

    # Initialize the model.
    model, last_epoch = utils.initialize_state(
        device, checkpoint, config, data.num_features
    )

    # Initialize the optimizer.
    optimizer = model.configure_optimizers()
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Tell if processed data are used.
    for task in data.tasks:
        if dir_path := config.data[task].processed_data_dir:
            logger.info(
                f"Using processed data for '{task}' from: {os.path.realpath(dir_path)}"
            )

    # Tell data sizes.
    logger.info("Number of data: training | test")
    for task, (len_train, len_test) in data.size.items():
        msg = f"\t'{task}': {len_train} | {len_test}"
        if (n_samples := getattr(config.data[task], "n_samples", 0)) > 0:
            len_actual_train, len_actual_test = data.approximate_size(task)
            msg += f" Sampled {n_samples} per PDB"
            msg += f" -> Approximately {len_actual_train} | {len_actual_test}"
        logger.info(msg)

    # Tell model size.
    logger.info(f"Number of parameters: {model.size[0]}")
    logger.info("")

    writer = SummaryWriter(config.run.tensorboard_dir)

    for epoch in range(last_epoch + 1, config.run.num_epochs + 1):
        start_time = time.time()
        data.sample_keys()

        # Training
        model.reset_log()
        run(model, data, device, optimizer, True)
        train_losses = utils.get_losses(model)
        # Use "scoring" or, if unused, the first dataset for `get_stats`.
        task_name = "scoring"
        if task_name not in model.predictions:
            task_name = list(model.predictions.keys())[0]
        train_r, train_r2, train_tau = utils.get_stats(model, task_name)
        utils.write_predictions(model, config, True)

        # Test
        model.reset_log()
        run(model, data, device, optimizer, False)

        test_losses = utils.get_losses(model)
        test_r, test_r2, test_tau = utils.get_stats(model, task_name)
        utils.write_predictions(model, config, False)

        end_time = time.time()

        # Print the header line.
        if epoch == last_epoch + 1:
            logger.info(utils.get_log_line(train_losses, title=True))
        # Print the loss values.
        log_elements = [
            str(epoch),
            utils.get_log_line(train_losses),
            utils.get_log_line(test_losses),
            "{:.3f}".format(train_r),
            "{:.3f}".format(test_r),
            "{:.3f}".format(train_tau),
            "{:.3f}".format(test_tau),
            "{:.3f}".format(end_time - start_time),
        ]
        logger.info("\t".join(log_elements))

        # Write tensorboard
        # writer.add_scalars("training loss", train_losses, epoch)
        # writer.add_scalars("test loss", test_losses, epoch)
        # writer.add_scalar("R2/train", train_r2, epoch)
        # writer.add_scalar("R2/test", test_r2, epoch)
        # writer.add_scalar("R/train", train_r, epoch)
        # writer.add_scalar("R/test", test_r, epoch)
        # writer.add_scalar("tau/train", train_tau, epoch)
        # writer.add_scalar("tau/test", test_tau, epoch)

        # Save the state.
        if config.run.save_every:
            if epoch == 1 or epoch % config.run.save_every == 0:
                save_path = os.path.join(config.run.checkpoint_dir, f"save_{epoch}.pt")
                utils.save_state(save_path, epoch, model, optimizer)

    wandb.finish()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    main()
