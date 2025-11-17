import functools

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import orbax.checkpoint as ocp

from gfnx.proxy.datasets.base import RewardProxyDataset
from jax_tqdm import loop_tqdm
from tqdm import tqdm

@chex.dataclass
class RewardProxyTrainingConfig():
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    val_each_epoch: int
    early_stop_tol: int
    # also task: str   # ["classification", "regression"] should be defined, but it is not included in the cfg file

def fit_model(
    rng_key: chex.PRNGKey,
    network: nn.Module,
    dataset: RewardProxyDataset,
    config: RewardProxyTrainingConfig,
    task: str
) -> chex.ArrayTree:
    train_data, train_score = dataset.train_set()
    val_data, val_score = dataset.test_set()

    train_score = train_score.squeeze()
    val_score = val_score.squeeze()

    train_size = train_data.shape[0]
    val_size = val_data.shape[0]

    rng_key, init_rng_key = jax.random.split(rng_key)
    params = network.init(init_rng_key, train_data[:1], training=False)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f'Number of parameters : {param_count}')

    optimizer = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    
    tr_state = TrainState.create(
        apply_fn=network.apply,
        tx=optimizer,
        params=params,
    )
    best_loss = 1e10    # Some large number
    early_stop_count = 0
    best_params = tr_state.params.copy()

    
    @functools.partial(jax.jit, static_argnums=(1,2))
    def epoch_train_step(rng_key : chex.PRNGKey, task: str, batch_size: int, tr_state : TrainState, data : chex.Array, target : chex.Array):
        max_steps = data.shape[0] // batch_size  # statis variable
        @chex.dataclass
        class LoopCarry:
            data: chex.Array
            target: chex.Array
            tr_state: TrainState
            rng_key: chex.PRNGKey
            total_train_loss: float = 0.0
            total_train_batches: int = 0

        def loop_body(idx : int, loop_carry: LoopCarry):
            rng_key, batch_rng_key = jax.random.split(loop_carry.rng_key)
            true_idx = idx * batch_size

            batch_data = jax.lax.dynamic_slice_in_dim(loop_carry.data, true_idx, batch_size)
            batch_target = jax.lax.dynamic_slice_in_dim(loop_carry.target, true_idx, batch_size)
            tr_state = loop_carry.tr_state
            total_train_loss = loop_carry.total_train_loss
            total_train_batches = loop_carry.total_train_batches

            def loss_fn(params, task, tr_state, batch_data, batch_target, rng_key):
                pred_score = tr_state.apply_fn(params, batch_data, training=True, rngs={"dropout": rng_key}).squeeze()
                if task == "classification":
                    loss = optax.losses.sigmoid_binary_cross_entropy(pred_score, batch_target).mean()
                elif task == "regression":
                    loss = optax.losses.squared_error(pred_score, batch_target).mean()
                else:
                    raise ValueError("Invalid task type") # It will not be raised, but just in case
                return loss

            grad_fn = jax.value_and_grad(loss_fn)   # loss_fn is defined above
            loss, grads = grad_fn(tr_state.params, task, tr_state, batch_data, batch_target, batch_rng_key)
            tr_state = tr_state.apply_gradients(grads=grads)

            total_train_loss = total_train_loss + loss
            total_train_batches = total_train_batches + 1
            return LoopCarry(
                data=loop_carry.data, 
                target=loop_carry.target, 
                tr_state=tr_state, 
                rng_key=rng_key,
                total_train_loss=total_train_loss, 
                total_train_batches=total_train_batches
            ) 
        
        final_loop_carry = jax.lax.fori_loop(0, max_steps, loop_body, LoopCarry(data=data, target=target, tr_state=tr_state, rng_key=rng_key))
        return final_loop_carry.tr_state, final_loop_carry.total_train_loss / final_loop_carry.total_train_batches

    print('Start training')
    for epoch in tqdm(range(config.num_epochs)):
        # Shuffle dataset in the start of each epoch
        rng_key, shuffle_rng_key = jax.random.split(rng_key)
        shuffle_idx = jax.random.permutation(shuffle_rng_key, jnp.arange(train_size))
        train_data, train_score = train_data[shuffle_idx], train_score[shuffle_idx]
        # Training loop
        tr_state, train_avg_loss = epoch_train_step(rng_key, task, config.batch_size, tr_state, train_data, train_score)
        print(f"Epoch {epoch+1}/{config.num_epochs}, Train loss: {train_avg_loss}")
        # Validation loop
        if epoch % config.val_each_epoch == 0 or epoch+1 == config.num_epochs:
            total_val_loss = 0.0
            total_val_batches = 0
            total_val_acc = 0.0
            for idx in tqdm(range(0, val_size, config.batch_size)):
                batch_end_idx = min(val_size, idx+config.batch_size)
                batch_data, batch_target = val_data[idx:batch_end_idx], val_score[idx:batch_end_idx]
                # TODO: Add validation loss calculation here
                pred_score = tr_state.apply_fn(tr_state.params, batch_data, training=False).squeeze()
                if task == "classification":
                    loss = optax.losses.sigmoid_binary_cross_entropy(pred_score, batch_target).mean()
                    acc = jnp.mean(jnp.equal(pred_score > 0, batch_target))
                elif task == "regression":
                    loss = optax.losses.squared_error(pred_score, batch_target).mean()
                    acc = 0.0   # Incorrect in this case
                else:
                    raise ValueError("Invalid task type") # It will not be raised, but just in case

                total_val_loss += loss
                total_val_acc += acc
                total_val_batches += 1
                
            average_val_loss = total_val_loss / total_val_batches
            if task == "classification":
                average_val_acc = total_val_acc / total_val_batches
                print(f"Validation loss: {average_val_loss}, acc: {average_val_acc}")
            else:
                print(f"Validation loss: {average_val_loss}")
            if average_val_loss < best_loss:
                best_loss = average_val_loss
                best_params = tr_state.params.copy()
                early_stop_count = 0 
            else:
                early_stop_count += 1
                if early_stop_count >= config.early_stop_tol:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    print(f"Best loss: {best_loss}")
    return best_params

import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
)
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.2'

@hydra.main(config_path="configs/", config_name="amp")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    dataset = instantiate(cfg.dataset)
    network = instantiate(cfg.network)
    train_cfg = RewardProxyTrainingConfig(
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        num_epochs=cfg.training.num_epochs,
        val_each_epoch=cfg.training.val_each_epoch,
        early_stop_tol=cfg.training.early_stop_tol,
    )
    task=cfg.training.task

    rng_key = jax.random.PRNGKey(cfg.seed)
    best_params = fit_model(rng_key, network, dataset, train_cfg, task)

    path = ocp.test_utils.erase_and_create_empty(cfg.save_path)
    orbax_checkpointer = ocp.StandardCheckpointer()
    orbax_checkpointer.save(path / 'final_model', best_params)

if __name__ == '__main__':
    main()