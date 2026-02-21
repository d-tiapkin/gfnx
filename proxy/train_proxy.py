import os
from dataclasses import dataclass

import chex
import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from datasets.amp import AMPRewardProxyDataset
from datasets.base import RewardProxyDataset
from datasets.gfp import GFPRewardProxyDataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gfnx.networks.reward_models import EqxTransformerRewardModel


@dataclass
class RewardProxyTrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    val_each_epoch: int
    early_stop_tol: int
    task: str  # ["classification", "regression"]


def fit_model(
    rng_key: chex.PRNGKey,
    model: eqx.Module,
    dataset: RewardProxyDataset,
    config: RewardProxyTrainingConfig,
) -> chex.ArrayTree:
    train_data, train_score = dataset.train_set()
    val_data, val_score = dataset.test_set()

    train_score = train_score.squeeze()
    val_score = val_score.squeeze()

    train_size = train_data.shape[0]
    val_size = val_data.shape[0]

    batch_loss_fn = {
        "classification": optax.losses.sigmoid_binary_cross_entropy,
        "regression": optax.losses.squared_error,
    }[config.task]

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"Number of parameters : {param_count}")

    optimizer = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    best_loss = 1e10  # Some large number
    early_stop_count = 0
    best_model = model

    @eqx.filter_jit
    def update(
        model: eqx.Module,
        opt_state: optax.OptState,
        data: chex.Array,
        target: chex.Array,
        key: chex.PRNGKey,
    ):
        def loss_fn(model, data, target, keys):
            pred_score = jax.vmap(lambda x, key: model(x, enable_dropout=True, key=key))(
                data, keys
            ).squeeze()
            return batch_loss_fn(pred_score, target).mean()

        keys = jax.random.split(key, data.shape[0])
        loss, grad = eqx.filter_value_and_grad(loss_fn)(model, data, target, keys)
        updates, new_opt_state = optimizer.update(grad, opt_state, eqx.filter(model, eqx.is_array))
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    print("Start training")
    for epoch in tqdm(range(config.num_epochs)):
        # Shuffle dataset in the start of each epoch
        rng_key, shuffle_rng_key = jax.random.split(rng_key)
        shuffle_idx = jax.random.permutation(shuffle_rng_key, jnp.arange(train_size))
        train_data, train_score = (
            train_data[shuffle_idx],
            train_score[shuffle_idx],
        )
        train_loss = 0.0
        n_batches = 0
        # Training loop
        for idx in range(0, train_size, config.batch_size):
            batch_end_idx = min(train_size, idx + config.batch_size)
            batch_data, batch_target = (
                train_data[idx:batch_end_idx],
                train_score[idx:batch_end_idx],
            )
            rng_key, batch_key = jax.random.split(rng_key)
            model, opt_state, loss = update(model, opt_state, batch_data, batch_target, batch_key)
            train_loss += loss
            n_batches += 1

        print(
            f"Epoch {epoch + 1}/{config.num_epochs}",
            f"Train loss: {train_loss / n_batches}",
        )
        # Validation loop
        total_val_loss = 0.0
        total_val_batches = 0
        total_val_acc = 0.0
        for idx in range(0, val_size, config.batch_size):
            batch_end_idx = min(val_size, idx + config.batch_size)
            batch_data, batch_target = (
                val_data[idx:batch_end_idx],
                val_score[idx:batch_end_idx],
            )

            rng_key, batch_key = jax.random.split(rng_key)
            batch_keys = jax.random.split(batch_key, batch_data.shape[0])
            pred_score = jax.vmap(
                lambda x, key, model=model: model(x, enable_dropout=True, key=key)
            )(batch_data, batch_keys).squeeze()
            loss = batch_loss_fn(pred_score, batch_target).mean()
            if config.task == "classification":
                acc = jnp.mean(jnp.equal(pred_score > 0, batch_target))
            else:
                # Dummy value for regression task
                acc = -1.0

            total_val_loss += loss
            total_val_acc += acc
            total_val_batches += 1

        average_val_loss = total_val_loss / total_val_batches
        average_val_acc = total_val_acc / total_val_batches
        print(f"Validation loss: {average_val_loss}")
        if average_val_acc > 0:
            print(f"Accuracy: {average_val_acc}")

        if average_val_loss < best_loss:
            best_loss = average_val_loss
            best_model = model
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= config.early_stop_tol:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"Best loss: {best_loss}")
    return best_model


@hydra.main(config_path="configs/", config_name="amp")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    rng_key = jax.random.PRNGKey(cfg.seed)
    rng_key, init_key = jax.random.split(rng_key)

    if cfg.dataset.name == "amp":
        dataset = AMPRewardProxyDataset(**cfg.dataset.params)
    elif cfg.dataset.name == "gfp":
        dataset = GFPRewardProxyDataset(**cfg.dataset.params)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")

    network = EqxTransformerRewardModel(
        encoder_params={
            "pad_id": dataset.char_to_id["[PAD]"],
            **OmegaConf.to_container(cfg.network.encoder_params),
        },
        offset=dataset.offset,
        output_dim=cfg.network.output_dim,
        key=init_key,
    )
    train_cfg = RewardProxyTrainingConfig(
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        num_epochs=cfg.training.num_epochs,
        val_each_epoch=cfg.training.val_each_epoch,
        early_stop_tol=cfg.training.early_stop_tol,
        task=cfg.training.task,
    )

    best_model = fit_model(rng_key, network, dataset, train_cfg)

    path = cfg.save_path
    if not os.path.isabs(path):
        # Assume that the path is relative to the root of the project
        module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(module_path, path)
    path = ocp.test_utils.erase_and_create_empty(path)
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckptr.save(path / "model", args=ocp.args.StandardSave(best_model))
    ckptr.wait_until_finished()


if __name__ == "__main__":
    main()
