"""Facilities for training models.

:copyright: Copyright 2023-2024 by Matt Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from functools import partial, wraps
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax.tree_util as jtu
import numpy as np
import optax  # type: ignore
from equinox import field
from jax_cookbook.misc import mse
from jax_cookbook.progress import piter, progress_piter
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from numpy._core.numeric import True_
from tensorboardX import SummaryWriter  # type: ignore

from feedbax import is_type, loss
from feedbax._model import AbstractModel, ModelInput
from feedbax._tree import (
    filter_spec_leaves,
    tree_infer_batch_size,
    tree_set,
    tree_take,
)
from feedbax.intervene import AbstractIntervenor
from feedbax.loss import AbstractLoss, CompositeLoss, TermTree
from feedbax.misc import (
    BatchInfo,
    Timer,
    batched_outer,
    delete_contents,
    exponential_smoothing,
    is_none,
    unkwargkey,
)
from feedbax.state import StateT
from feedbax.task import AbstractTask, TaskTrialSpec

LOSS_FMT = ".2e"


logger = logging.getLogger(__name__)


WhereFunc: TypeAlias = Callable[[AbstractModel[StateT]], Any]


class TaskTrainerHistory(eqx.Module):
    """A record of training history over a call to a
    [`TaskTrainer`][feedbax.train.TaskTrainer] instance.

    Attributes:
        loss: The training losses.
        learning_rate: The optimizer's learning rate.
        model_parameters: The model's trainable parameters. Non-trainable
            leaves appear as `None`.
        trial_specs: The training trial specifications.
    """

    loss: TermTree[AbstractLoss] | Array
    loss_validation: TermTree[AbstractLoss] | Array
    learning_rate: Optional[Array] = None
    model_parameters: Optional[AbstractModel] = None
    trial_specs: dict[int, TaskTrialSpec] = field(default_factory=dict)


def get_model_parameters(model, where_train_spec):
    return eqx.filter(eqx.filter(model, where_train_spec), eqx.is_array)


def update_opt_state(new, old, where_train_spec):
    """After re-initializing the `opt_state`, keep the optimizer state for parameters still being trained."""
    opt_state_flat, treedef = jt.flatten(new, is_leaf=is_type(AbstractModel))
    opt_state_flat_old = jt.leaves(old, is_leaf=is_type(AbstractModel))
    opt_state_flat_new = [
        eqx.combine(
            # Replace the new, empty opt state with the old one, where relevant
            eqx.filter(old, where_train_spec),
            new,
        )
        if isinstance(new, AbstractModel)
        else new
        for new, old in zip(opt_state_flat, opt_state_flat_old)
    ]
    return jt.unflatten(treedef, opt_state_flat_new)


class TaskTrainer(eqx.Module):
    """Manages resources needed to train models, given task specifications."""

    optimizer: optax.GradientTransformation
    checkpointing: bool
    chkpt_dir: Path
    writer: Optional[SummaryWriter]
    model_update_funcs: PyTree[Callable]
    _use_tb: bool

    def __init__(
        self,
        optimizer: optax.GradientTransformation,
        checkpointing: bool = True,
        chkpt_dir: str | Path = "/tmp/feedbax-checkpoints",
        enable_tensorboard: bool = False,
        tensorboard_logdir: str | Path = "/tmp/feedbax-tensorboard",
        model_update_funcs: Sequence[Callable] = (),
    ):
        """
        Arguments:
            optimizer: The Optax optimizer to use for training.
            checkpointing: Whether to save model checkpoints during training.
            chkpt_dir: The directory in which to save model checkpoints.
            enable_tensorboard: Whether to keep logs for Tensorboard.
            tensorboard_logdir: The directory in which to save Tensorboard logs.
            model_update_funcs: At the end of each training step/batch, each of these
                functions is passed 1) the model, and 2) the model states for all
                trials in the batch, and returns a model update. These
                can be used for implementing state-dependent offline learning rules
                such as batch-averaged Hebbian learning.
        """
        self.optimizer = optimizer
        self.model_update_funcs = model_update_funcs

        self._use_tb = enable_tensorboard
        if self._use_tb:
            self.writer = SummaryWriter(str(tensorboard_logdir))
            # display loss terms in the same figure under "Custom Scalars"
            # layout = {
            #     "Loss terms": {
            #         "Training loss": ["Multiline", ["Loss/train"] + [f'Loss/train/{term}'
            #                                         for term in term_weights.keys()]],
            #         "Evaluation loss": ["Multiline", ["Loss/validation"] + [f'Loss/validation/{term}'
            #                                         for term in term_weights.keys()]],
            #     },
            # }
            # writer.add_custom_scalars(layout)
        else:
            self.writer = None

        self.checkpointing = checkpointing
        self.chkpt_dir = Path(chkpt_dir)
        if self.checkpointing:
            self.chkpt_dir.mkdir(exist_ok=True)

    @jax.named_scope("fbx.TaskTrainer")
    def __call__(
        self,
        task: AbstractTask,
        model: AbstractModel[StateT],
        n_batches: int,
        batch_size: int,
        where_train: WhereFunc | dict[int, WhereFunc],
        idx_start: int = 0,
        opt_state: Optional[optax.OptState] = None,
        loss_func: Optional[AbstractLoss] = None,
        ensembled: bool = False,
        ensemble_random_trials: bool = True,
        log_step: int = 100,
        state_reset_iterations: Optional[Int[Array, "_"]] = None,
        save_model_parameters: bool | Int[Array, "_"] = False,
        save_trial_specs: Optional[Int[Array, "_"]] = None,
        toggle_model_update_funcs: bool | PyTree[Int[Array, "_"]] = True,
        restore_checkpoint: bool = False,
        disable_progress: bool = False,
        batch_callbacks: Optional[Mapping[int, Sequence[Callable]]] = None,
        run_label: Optional[str] = None,
        verbose_progress: bool = True,
        loss_update_func: Optional[Callable[[AbstractLoss, TermTree, PyTree], AbstractLoss]] = None,
        loss_update_iterations: bool | Int[Array, "_"] = True,
        *,
        key: PRNGKeyArray,
    ):
        """Train a model on a fixed number of batches of task trials.

        !!! Warning
            Model checkpointing only saves model parameters, and not the task or other
            hyperparameters. That is, we assume that the model and task passed to this
            method are, aside from their trainable state, identical to those from the
            original training run. This is typically the case when
            `restore_checkpoint=True` is toggled immediately after the interruption of a
            training run, to resume it.

            Trying to load a checkpoint as a model at a later time may fail.
            Use [`feedbax.save`][feedbax.save] and
            [`feedbax.load`][feedbax.load] for longer-term storage.

        Arguments:
            task: The task to train the model on.
            model: The model—or, vmapped batch/ensemble of models—to train.
            n_batches: The number of batches of trials to train on.
            batch_size: The number of trials in each batch.
            where_train: Selects the parameter arrays from the model PyTree to be
                trained. May be a dict whose integer keys indicate which (global)
                iteration to start training that subset of model parameters.
            idx_start: Starting index for training iterations. Useful when we are
                restarting from a previous run and want to specify certain other
                arguments (e.g. `save_model_parameters`) in terms of the overall
                training iteration, rather than the iteration within the sub-run.
                Overridden if we restore from a checkpoint.
            opt_state: An appropriate optimizer state to start from. For example,
                when running multiple slightly different optimizations in a row, the
                `opt_state` of the previous run can be passed to maintain (say) the
                momentum and parameter scaling for the Adam optimizer.
            ensembled: Should be set to `True` if `model` is a vmapped ensemble
                of models that should be trained in parallel.
            ensemble_random_trials: If `False`, every model in an ensemble will
                be trained on the same batches of trials. Otherwise, a distinct batch
                will be generated for each model. Has no effect if `ensembled` is
                `False`.
            log_step: Interval at which to evaluate model on the validation set,
                print losses to the console, log to tensorboard (if enabled), and save
                checkpoints.
            state_reset_iterations: Indices of the batches on which to reset the optimizer
                state.
            save_model_parameters: Whether to return the entire history of the
                trainable leaves of the model (e.g. network weights) across training
                iterations, as part of the `TaskTrainerHistory` object. May also pass a
                1D array of batch numbers on which to keep history; parameters are
                logged at the start of the indicated batches.
            save_trial_specs: A 1D array of batch numbers for which to keep
                trial specifications, and return as part of the training history.
            toggle_model_update_funcs: Whether to enable the model update functions.
                May also pass a PyTree with the same structure as the `TaskTrainer`'s
                `model_update_funcs` attribute, where each leaf is a 1D array of batch
                numbers on which to enable the respective function. If the
                `model_update_funcs` attribute is empty, this argument is ignored.
            restore_checkpoint: Whether to attempt to restore from the last saved
                checkpoint in the checkpoint directory. Typically, this option is
                toggled to continue a long training run immediately after it was
                interrupted.
            disable_progress: If `True`, do not show a progress bar over training iterations.
            batch_callbacks: A mapping from batch number to a sequence of
                functions (without parameters) to be called immediately after the
                training step is performed for that batch. This can be used (say) for
                profiling parts of the training run.
            run_label: For labeling the progress bar, if it is enabled.
            loss_update_func: Function to update the loss function between iterations.
                Called with the current loss function, computed losses (aggregated over
                replicates if ensembled), and gradients (aggregated over replicates if
                ensembled); returns updated loss function. Can update loss weights,
                or any other aspect of the loss function structure.
            loss_update_iterations: Whether/when to apply loss updates. If `True`,
                update every iteration. If `False`, never update. If array of batch numbers,
                update only on those iterations.
            key: The random key.
        """
        idx_end = idx_start + n_batches

        if isinstance(where_train, dict):
            if 0 not in where_train:
                raise ValueError(
                    "If where_train is a dict, it must contain an entry for iteration 0"
                )
            # Convert global iterations to local iterations for this training run
            where_train_local = {
                k - idx_start: v for k, v in where_train.items() if idx_start <= k < idx_end
            }
            # Find the most recent update before or at the start of this run
            most_recent_update = max(k for k in where_train.keys() if k <= idx_start)
            where_train_func = where_train[most_recent_update]
        else:
            where_train_local = {}
            where_train_func = where_train

        where_train_spec = filter_spec_leaves(model, where_train_func)

        model_parameters = get_model_parameters(model, where_train_spec)

        if ensembled:
            # Infer the number of replicates from shape of trainable arrays
            n_replicates = tree_infer_batch_size(
                model, exclude=lambda x: isinstance(x, AbstractIntervenor)
            )
            init_opt_state = eqx.filter_vmap(self.optimizer.init)
        else:
            # Unlikely to be used for anything, due to ensembled operations being in
            # conditionals. Make the type checker happy.
            n_replicates = 1
            init_opt_state = self.optimizer.init

        if opt_state is None:
            opt_state = init_opt_state(model_parameters)
        # This should never fail.
        assert opt_state is not None, "Optax `init` method returned `None`!"

        if getattr(opt_state, "hyperparams", None) is None:
            logger.debug(
                "Optimizer not wrapped in `optax.inject_hyperparameters`; "
                "learning rate history will not be returned"
            )

        if loss_func is None:
            loss_func = task.loss_func

        if state_reset_iterations is None:
            state_reset_iterations = jnp.array([])

        if isinstance(save_model_parameters, Array):
            # Convert batch numbers to a full-length Boolean mask over training iterations
            save_model_parameters_batches = np.array(
                jnp.zeros(idx_end, dtype=bool).at[save_model_parameters].set(True)
            )
            save_steps = save_model_parameters[
                (save_model_parameters >= idx_start) & (save_model_parameters < idx_end)
            ]
            # Associate batch numbers with indices to preallocated history array
            save_model_parameters_idxs = dict(
                zip(
                    save_steps.tolist(),
                    range(len(save_steps)),
                )
            )
            save_model_parameters_idx_func = lambda x: save_model_parameters_idxs[int(x)]
        else:
            save_model_parameters_batches = np.full(idx_end, save_model_parameters, dtype=bool)
            save_model_parameters_idx_func = lambda x: x

        if isinstance(save_trial_specs, Array):
            save_trial_specs_batches = np.array(
                jnp.zeros(idx_end, dtype=bool).at[save_trial_specs].set(True)
            )
        else:
            save_trial_specs_batches = np.full(idx_end, save_trial_specs, dtype=bool)

        history = init_task_trainer_history(
            loss_func,
            idx_end,
            n_replicates,
            ensembled,
            ensemble_random_trials=ensemble_random_trials,
            start_batch=idx_start,
            task=task,
            save_model_parameters=save_model_parameters,
            save_trial_specs=save_trial_specs,
            batch_size=batch_size,
            model=model,
            where_train=where_train,
        )

        start_batch = idx_start  # except when we load a checkpoint

        if restore_checkpoint:
            # TODO: should also restore opt_state, learning_rates...
            chkpt_path, last_batch, model, opt_state, history = self._load_last_checkpoint(
                model, opt_state, history
            )
            start_batch = last_batch + 1
            if chkpt_path is not None:
                logger.info(f"Restored checkpoint {chkpt_path} from training step {last_batch}.")
            else:
                raise ValueError("restore_checkpoint is True, but no checkpoint found")
        elif self.checkpointing:
            # Delete old checkpoints if checkpointing is on.
            delete_contents(self.chkpt_dir)

        model_update_funcs_flat = jtu.tree_leaves(
            self.model_update_funcs,
            is_leaf=lambda x: isinstance(x, Callable),
        )
        if model_update_funcs_flat == [] or not isinstance(
            toggle_model_update_funcs, PyTree[Array]
        ):
            n_update_funcs = len(model_update_funcs_flat)
            model_update_funcs_mask = np.full(
                (idx_end, n_update_funcs), toggle_model_update_funcs, dtype=bool
            )
        else:
            if not jtu.tree_structure(toggle_model_update_funcs) == jtu.tree_structure(
                self.model_update_funcs, is_leaf=lambda x: isinstance(x, Callable)
            ):
                raise ValueError(
                    "The structure of `toggle_model_update_funcs` must match that of "
                    "the `TaskTrainer`'s attribute `model_update_funcs`."
                )
            # For 10,000 iterations and 10 update functions, this takes 100 kB.
            model_update_funcs_mask = np.stack(
                [
                    jnp.zeros(idx_end, dtype=bool).at[idxs].set(True)
                    for idxs in jtu.tree_leaves(toggle_model_update_funcs)
                ]
            ).T

        # Loss update mask
        if loss_update_func is None:
            loss_update_mask = np.full(idx_end, False, dtype=bool)
        elif isinstance(loss_update_iterations, Array):
            loss_update_mask = np.zeros(idx_end, dtype=bool)
            loss_update_mask[loss_update_iterations] = True
        else:
            loss_update_mask = np.full(idx_end, loss_update_iterations, dtype=bool)

        # Passing the flattened pytrees through `_train_step` gives a slight
        # performance improvement. See the docstring of `_train_step`.
        flat_model, treedef_model = jtu.tree_flatten(
            model,
            is_leaf=lambda x: isinstance(x, AbstractIntervenor),
        )
        flat_opt_state, treedef_opt_state = jtu.tree_flatten(opt_state)

        if ensembled:
            # We only vmap over axis 0 of the *array* components of the model.
            flat_model_arr_spec = jt.map(
                lambda x: eqx.if_array(0) if not isinstance(x, AbstractIntervenor) else None,
                flat_model,
                is_leaf=lambda x: isinstance(x, AbstractIntervenor),
            )

            if ensemble_random_trials:
                key_in_axis = 0
                trial_specs_out_axis = 0
            else:
                key_in_axis = None
                trial_specs_out_axis = None

            in_axes = (
                None,
                None,
                None,
                flat_model_arr_spec,
                None,
                0,
                None,
                None,
                None,
                key_in_axis,
            )
            out_axes = (0, trial_specs_out_axis, flat_model_arr_spec, 0, 0)

            train_step = eqx.filter_vmap(
                self._train_step,
                in_axes=in_axes,
                out_axes=out_axes,
            )

            # We can't simply flatten `model_array_spec` to get `flat_model_array_spec`,
            # even if we use `is_leaf=is_none`.
            # model_array_spec = jt.map(
            #     lambda x: eqx.if_array(0) if not isinstance(x, AbstractIntervenor) else None,
            #     model,
            #     is_leaf=lambda x: isinstance(x, AbstractIntervenor),
            # )
            # evaluate = eqx.filter_vmap(
            #     task.eval_with_loss, in_axes=(model_array_spec, 0)
            # )
            def evaluate(model, key):
                return task.eval_ensemble_with_loss(
                    model,
                    n_replicates,
                    key,
                    ensemble_random_trials=ensemble_random_trials,
                )

        else:
            train_step = self._train_step
            evaluate = task.eval_with_loss

        # Finish the JIT compilation before the first training iteration.
        # TODO: <https://jax.readthedocs.io/en/latest/aot.html>
        if not jax.config.jax_disable_jit:  # type: ignore
            timer = Timer()

            with timer:
                if ensembled and ensemble_random_trials:
                    key_compile = jr.split(key, n_replicates)
                else:
                    key_compile = key

                train_step(  # doesn't alter model or opt_state
                    task,
                    loss_func,
                    BatchInfo(
                        size=batch_size,
                        start=jnp.array(0),
                        current=jnp.array(0),
                        total=jnp.array(1),
                    ),
                    flat_model,
                    treedef_model,
                    flat_opt_state,
                    treedef_opt_state,
                    where_train_spec,
                    model_update_funcs_flat,
                    key_compile,
                )

            logger.info(f"Training step compiled in {timer.time:.2f} seconds.")

            with timer:
                evaluate(model, key)

            logger.info(f"Validation step compiled in {timer.time:.2f} seconds.")

        else:
            logger.debug("JIT globally disabled. Skipping pre-run compilation.")

        log_batches_mask = np.zeros(idx_end, dtype=bool)
        log_batches = np.linspace(
            idx_start, idx_end, n_batches // log_step, endpoint=False, dtype=int
        )
        # Could also use `np.in1d` to do this out-of-place, but it's slightly slower
        log_batches_mask[log_batches] = True
        log_batches_mask[-1] = True

        keys = jr.split(key, n_batches)
        with progress_piter(
            jnp.arange(start_batch, idx_end),
            description=run_label or "",
            completed=start_batch,
            total=idx_end,
        ) as (batches, update_pbar):
            # Assume 1 epoch (i.e. batch iterations only; no fixed dataset).
            for batch in batches:
                key_train, key_eval = jr.split(keys[batch], 2)

                batch_local = int(batch - idx_start)

                if batch_local in where_train_local:
                    where_train_func = where_train_local[batch_local]
                    where_train_spec = filter_spec_leaves(model, where_train_func)
                    model = jt.unflatten(treedef_model, flat_model)
                    opt_state_old = opt_state
                    opt_state_init = init_opt_state(get_model_parameters(model, where_train_spec))
                    # Keep the `opt_state` for any parameters that remain trained
                    opt_state = update_opt_state(opt_state_init, opt_state_old, where_train_spec)

                    flat_opt_state, treedef_opt_state = jt.flatten(opt_state)

                if batch in state_reset_iterations:
                    model = jtu.tree_unflatten(treedef_model, flat_model)
                    opt_state = init_opt_state(get_model_parameters(model, where_train_spec))
                    flat_opt_state, treedef_opt_state = jtu.tree_flatten(opt_state)

                # Save parameters at the start of batch
                if save_model_parameters_batches[batch]:
                    model = jtu.tree_unflatten(treedef_model, flat_model)
                    history = eqx.tree_at(
                        lambda history: history.model_parameters,
                        history,
                        tree_set(
                            history.model_parameters,
                            eqx.filter(model, where_train_spec),
                            save_model_parameters_idx_func(batch),
                        ),
                    )

                if ensembled and ensemble_random_trials:
                    key_train = jr.split(key_train, n_replicates)

                update_funcs_i = [
                    model_update_funcs_flat[i]
                    for i, b in enumerate(model_update_funcs_mask[batch])
                    if b
                ]

                batch_info = BatchInfo(
                    size=batch_size,
                    start=idx_start,
                    current=batch,
                    total=idx_end,
                )
                (losses, trial_specs, flat_model, flat_opt_state, grads) = train_step(
                    task,
                    loss_func,
                    batch_info,
                    flat_model,
                    treedef_model,
                    flat_opt_state,
                    treedef_opt_state,
                    where_train_spec,
                    update_funcs_i,
                    key_train,
                )

                update_pbar.subdescription(
                    f"training loss: {losses.total.mean().item():{LOSS_FMT}}"
                )

                if batch_callbacks is not None and batch in batch_callbacks:
                    for func in batch_callbacks[batch]:
                        func()

                if save_trial_specs_batches[batch]:
                    history = eqx.tree_at(
                        lambda history: history.trial_specs,
                        history,
                        history.trial_specs | {batch: trial_specs},
                    )

                history = eqx.tree_at(
                    lambda history: history.loss,
                    history,
                    tree_set(
                        history.loss,
                        losses.map(lambda arr: jnp.mean(arr, axis=-1)),  # agg over trials
                        batch - idx_start,
                    ),
                )

                if (hyperparams := getattr(opt_state, "hyperparams", None)) is not None:
                    # requires that the optimizer was wrapped in `optax.inject_hyperparameters`
                    history = eqx.tree_at(
                        lambda history: history.learning_rate,
                        history,
                        history.learning_rate.at[batch - idx_start].set(
                            hyperparams["learning_rate"]
                        ),
                    )

                # Update loss function if requested
                if loss_update_mask[batch]:
                    # Aggregate losses and gradients over replicates if ensembled
                    if ensembled:
                        losses_for_update = losses.map(lambda arr: jnp.mean(arr, axis=0))  # mean over replicates
                        grads_for_update = jt.map(lambda arr: jnp.mean(arr, axis=0) if eqx.is_array(arr) else arr, grads)
                    else:
                        losses_for_update = losses
                        grads_for_update = grads

                    loss_func = loss_update_func(loss_func, losses_for_update, grads_for_update)

                # tensorboard losses on every iteration
                if ensembled:
                    losses_mean = losses.map(jnp.mean)
                    # This will be appended to user-facing labels
                    # e.g. "mean training loss"
                    ensembled_str = "mean "
                else:
                    losses_mean = losses
                    ensembled_str = ""

                if self._use_tb and self.writer is not None:
                    self.writer.add_scalar(
                        f"loss/{ensembled_str}train", losses_mean.total.item(), batch
                    )
                    for loss_term_label, loss_term in losses_mean.flatten().items():
                        self.writer.add_scalar(
                            f"loss/{ensembled_str}train/{loss_term_label}",
                            loss_term.item(),
                            batch,
                        )

                if jnp.isnan(losses_mean.total):
                    model = jtu.tree_unflatten(treedef_model, flat_model)
                    # opt_state = jtu.tree_unflatten(treedef_opt_state, flat_opt_state)

                    # TODO: Should probably not return a checkpoint automatically unless the user
                    # has explicitly requested it.
                    msg = f"NaN loss at batch {batch}! "
                    if (
                        checkpoint := self._load_last_checkpoint(model, opt_state, history)
                    ) is not None:
                        _, last_batch, model, _, history = checkpoint
                        msg += f"Returning checkpoint from batch {last_batch}."
                    else:
                        msg += "No checkpoint found, returning model from last iteration."

                    logger.warning(msg)

                    return model, history, opt_state

                # Checkpoint and validate, occasionally
                if log_batches_mask[batch]:
                    model = jtu.tree_unflatten(treedef_model, flat_model)
                    opt_state = jtu.tree_unflatten(treedef_opt_state, flat_opt_state)

                    if self.checkpointing and batch > 0:
                        self._save_checkpoint(batch, model, opt_state, history)

                    # if ensembled and ensemble_random_trials:
                    #     key_eval = jr.split(key_eval, n_replicates)

                    states, losses_validation = evaluate(model, key_eval)

                    history = eqx.tree_at(
                        lambda history: history.loss_validation,
                        history,
                        tree_set(
                            history.loss_validation,
                            losses_validation.map(
                                lambda arr: jnp.mean(arr, axis=-1)  # agg over trials
                            ),
                            batch,
                        ),
                    )

                    if ensembled:
                        losses_validation_mean = jt.map(
                            lambda x: jnp.mean(x, axis=-1), losses_validation
                        )
                        # Only log a validation plot for the first replicate.
                        states_plot = tree_take(states, 0)
                    else:
                        losses_validation_mean = losses_validation
                        states_plot = states

                    if self._use_tb and self.writer is not None:
                        # TODO: Allow user to register other plots.
                        trial_specs = task.validation_trials
                        figs = task.validation_plots(states_plot, trial_specs=trial_specs)
                        for label, fig in figs.items():
                            self.writer.add_figure(f"validation/{label}", fig, batch)
                        self.writer.add_scalar(
                            f"loss/{ensembled_str}validation",
                            losses_validation_mean.total.item(),
                            batch,
                        )
                        for (
                            loss_term_label,
                            loss_term,
                        ) in losses_validation_mean.flatten().items():
                            self.writer.add_scalar(
                                f"loss/{ensembled_str}validation/{loss_term_label}",
                                loss_term.item(),
                                batch,
                            )

                    # TODO: https://stackoverflow.com/a/69145493

                    if not disable_progress:
                        # Header
                        if batch == n_batches - 1:
                            loss_str_head = f"Final training iteration ({batch}):"
                        else:
                            loss_str_head = f"Training iteration {batch}:"

                        # Base summary lines
                        log_lines = [
                            f"{loss_str_head}",
                            f"\t{ensembled_str.capitalize()}training loss: {losses_mean.total:{LOSS_FMT}}",
                        ]

                        # Optional detailed breakdown
                        if verbose_progress:
                            flat_train = losses_mean.flatten()
                            for k, v in flat_train.items():
                                log_lines.append(f"\t\t{k}: {v:{LOSS_FMT}}")

                        # Validation summary
                        log_lines.append(
                            f"\t{ensembled_str.capitalize()}validation loss: {losses_validation_mean.total:{LOSS_FMT}}"
                        )

                        # Combine and log
                        log_str = "\n".join(log_lines)
                        logger.info(log_str)

        logger.info(
            f"Completed training run on a total of {n_batches * batch_size:,} trials "
            f"{'per model' if ensembled else ''}."
        )

        model = jtu.tree_unflatten(treedef_model, flat_model)

        return model, history, opt_state

    @eqx.filter_jit
    @jax.named_scope("fbx.TaskTrainer._train_step")
    def _train_step(
        self,
        task: AbstractTask,
        loss_func: AbstractLoss,
        batch_info: BatchInfo,
        flat_model,
        treedef_model,
        flat_opt_state,
        treedef_opt_state,
        where_train_spec,  #! can't do AbstractModel[StateT[bool]]
        update_funcs,
        key: PRNGKeyArray,
    ):
        """Executes a single training step of the model.

        Note that the primary output of `loss_func_wrapped` is the scalar
        `losses.total`, which is discarded because `losses` is itself returned
        as the auxiliary of `loss_func_wrapped`. This is necessary because
        the gradient is computed with respect to the primary output, but we
        only need to store the original `TermTree` containing all loss terms.

        The wrapping calls to `tree_unflatten` and `tree_leaves`, and passing
        of flattened versions of `model` and `opt_state`, bring slight
        performance improvements because they cancel out the inverse tree
        operations that would be performed by default during JIT compilation.

        See https://docs.kidger.site/equinox/tricks/#low-overhead-training-loops

        TODO:
        - Use a wrapper to make the flatten/unflatten stuff less ugly.
        - Typing of arguments. Not sure it's possible to type flattened PyTrees
          appropriately...
        """
        key_trials, key_init, key_model = jr.split(key, 3)
        keys_trials = jr.split(key_trials, batch_info.size)
        keys_init = jr.split(key_init, batch_info.size)
        keys_model = jr.split(key_model, batch_info.size)

        trial_specs = eqx.filter_vmap(
            partial(
                task.get_train_trial_with_intervenor_params,
                batch_info=batch_info,
            )
        )(keys_trials)

        model = jtu.tree_unflatten(treedef_model, flat_model)

        init_states = eqx.filter_vmap(unkwargkey(model.init))(keys_init)

        for where_substate, init_substates in trial_specs.inits.items():
            init_states = eqx.tree_at(
                where_substate,
                init_states,
                init_substates,
            )

        init_states = eqx.filter_vmap(model.step.state_consistency_update)(init_states)

        diff_model, static_model = eqx.partition(model, where_train_spec)

        opt_state = jtu.tree_unflatten(treedef_opt_state, flat_opt_state)

        (_, (losses, states)), grads = eqx.filter_value_and_grad(
            grad_wrap_abstract_loss(loss_func), has_aux=True
        )(
            diff_model,
            static_model,
            trial_specs,
            init_states,
            keys_model,
        )

        updates, opt_state = self.optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        # For updates computed directly from the state, without loss gradient.
        for update_func in update_funcs:
            model = update_func(model, states)

        flat_model = jtu.tree_leaves(model, is_leaf=lambda x: isinstance(x, AbstractIntervenor))
        flat_opt_state = jtu.tree_leaves(opt_state)

        return losses, trial_specs, flat_model, flat_opt_state, grads

    def _save_checkpoint(
        self,
        batch: int,
        model: AbstractModel[StateT],
        opt_state: optax.OptState,
        history: TaskTrainerHistory,
    ):
        # TODO: Save `opt_state` after fixing issue with first training iteration
        eqx.tree_serialise_leaves(
            self.chkpt_dir / f"ckpt_{batch}.eqx",
            (model, opt_state, history),
        )
        with open(self.chkpt_dir / "last_batch.txt", "w") as f:
            f.write(str(batch))

    def _load_last_checkpoint(
        self,
        model: AbstractModel[StateT],
        opt_state: optax.OptState,
        history: TaskTrainerHistory,
    ) -> Tuple[Optional[Path], int, AbstractModel[StateT], optax.OptState, TaskTrainerHistory]:
        try:
            with open(self.chkpt_dir / "last_batch.txt", "r") as f:
                last_batch = int(f.read())
        except FileNotFoundError:
            return None, -1, model, opt_state, history

        chkpt_path = self.chkpt_dir / f"ckpt_{last_batch}.eqx"
        # TODO: Load `opt_state` after fixing issue with first training iteration
        model, _, history = eqx.tree_deserialise_leaves(
            chkpt_path,
            (model, opt_state, history),
        )
        return chkpt_path, last_batch, model, opt_state, history


def _get_trainable_params_superset(
    model: AbstractModel[StateT],
    where_train: WhereFunc | dict[int, WhereFunc],
) -> PyTree[bool]:
    """Get a boolean mask for all parameters that are trainable at any point."""
    if isinstance(where_train, dict):
        # Combine all where_train specs with logical OR
        specs = [filter_spec_leaves(model, where_func) for where_func in where_train.values()]
        return jt.map(
            lambda *xs: any(x for x in xs if x is not None),
            *specs,
            is_leaf=lambda x: x is None,
        )
    else:
        return filter_spec_leaves(model, where_train)


def init_task_trainer_history(
    loss_func: AbstractLoss,
    n_batches: int,
    n_replicates: int,
    ensembled: bool,
    ensemble_random_trials: bool = True,
    start_batch: int = 0,
    save_model_parameters: bool | Int[Array, "_"] = False,
    save_trial_specs: Optional[Int[Array, "_"]] = None,
    task: Optional[AbstractTask] = None,
    loss_func_validation: Optional[AbstractLoss] = None,
    batch_size: Optional[int] = None,
    model: Optional[AbstractModel[StateT]] = None,
    where_train: Optional[WhereFunc | dict[int, WhereFunc]] = None,
):
    if ensembled:
        batch_dims = (n_batches - start_batch, n_replicates)
    else:
        batch_dims = (n_batches - start_batch,)

    # def _empty_loss_history(loss_func):
    #     if isinstance(loss_func, CompositeLoss):
    #         loss_keys = loss_func.weights.keys()
    #         n_terms = len(loss_keys)
    #         loss_arrays = [jnp.empty(batch_dims) for _ in range(n_terms)]
    #         return TermTree(zip(loss_keys, loss_arrays))
    #     else:
    #         return jnp.empty(batch_dims)

    loss_history = loss_func.skeleton(batch_dims)

    # Give precedence to the task's loss function, for the validation loss PyTree
    # (In order to exclude model-specific loss terms that `AbstractTask` should not know about
    # during validation)
    if loss_func_validation is not None:
        loss_history_validation = loss_func_validation.skeleton(batch_dims)
    elif task is not None:
        loss_history_validation = task.loss_func.skeleton(batch_dims)
    else:
        loss_history_validation = loss_func.skeleton(batch_dims)

    if save_trial_specs is not None:
        assert task is not None
        assert batch_size is not None
        if len(save_trial_specs.shape) != 1:
            raise ValueError("If save_trial_specs is an array, it must be 1D")

        def get_batch_example(key):
            keys_trials = jr.split(key, batch_size)
            return jax.vmap(task.get_train_trial_with_intervenor_params)(keys_trials)

        if ensembled and ensemble_random_trials:
            trial_specs_example = jax.vmap(get_batch_example)(jr.split(jr.PRNGKey(0), n_replicates))
        else:
            trial_specs_example = get_batch_example(jr.PRNGKey(0))

        trial_spec_batches = [int(i + (n_batches if i < 0 else 0)) for i in save_trial_specs]
        trial_spec_history = {int(i): trial_specs_example for i in trial_spec_batches}
    else:
        trial_spec_history = {}

    if save_model_parameters is not False:
        assert model is not None
        assert where_train is not None
        where_train_spec = _get_trainable_params_superset(model, where_train)
        model_parameters = eqx.filter(eqx.filter(model, where_train_spec), eqx.is_array)

        if isinstance(save_model_parameters, Array):
            if len(save_model_parameters.shape) != 1:
                raise ValueError("If save_model_parameters is an array, it must be 1D")

            save_steps = save_model_parameters[
                (save_model_parameters >= start_batch) & (save_model_parameters < n_batches)
            ]

            n_save_steps = save_steps.shape[0]
            model_train_history = jt.map(
                lambda x: (jnp.full((n_save_steps,) + x.shape, 0.0) if eqx.is_array(x) else x),
                model_parameters,
            )
        else:
            model_train_history = jt.map(
                lambda x: jnp.full((n_batches - start_batch,) + x.shape, 0.0)
                if eqx.is_array(x)
                else x,
                model_parameters,
            )
    else:
        model_train_history = None

    return TaskTrainerHistory(
        loss=loss_history,
        loss_validation=loss_history_validation,
        learning_rate=jnp.empty(batch_dims),
        model_parameters=model_train_history,
        trial_specs=trial_spec_history,
    )


def grad_wrap_simple_loss_func(
    loss_func: Callable[[Array, Array], Float],
    nan_safe: bool = False,
):
    """Wraps a loss function taking output and target arrays, to one taking a model
    and its input, along with the target array.

    In particular, this is used to transform the a loss function representation of a
    loss function as a norm, to a function that plays nicely with `jax.grad`.

    If `nan_safe == True`, the wrapped function will replace NaNs in the inputs before
    performing the forward pass, to ensure that gradients remain finite. In that case,
    a NaN-safe loss function (like `nan_safe_mse`) should be used, so that the respective
    training examples are excluded from the aggregate loss.
    """

    if nan_safe:

        def _forward_pass(model, X):
            X_cleaned = jnp.nan_to_num(X, nan=0.0)
            return model(X_cleaned)
    else:

        def _forward_pass(model, X):
            return model(X)

    @wraps(loss_func)
    def wrapper(
        model,
        X,
        y,
    ) -> Tuple[float, TermTree[AbstractLoss]]:
        return loss_func(_forward_pass(model, X), y)

    return wrapper


class SimpleTrainer(eqx.Module):
    """For training on whole datasets over a fixed number of iterations.

    By default, uses SGD with a fixed learning rate of 1e-2, and MSE loss.

    For example, use this for training a linear regression or jPCA model.
    """

    loss_func: Callable[[eqx.Module, Array, Array], Float] = field(
        default=grad_wrap_simple_loss_func(mse),
    )
    optimizer: optax.GradientTransformation = field(
        default=optax.sgd(1e-2),
    )

    def __call__(self, model, X, y, n_iter=100, progress_bar=True):
        opt_state = self.optimizer.init(model)

        # @eqx.filter_jit
        def train_step(current_model, current_opt_state, x_batch, y_batch):
            loss_value, grads = jax.value_and_grad(self.loss_func)(current_model, x_batch, y_batch)
            updates, new_opt_state = self.optimizer.update(grads, current_opt_state)
            new_model = eqx.apply_updates(current_model, updates)
            return new_model, new_opt_state, loss_value

        if progress_bar:
            train_iter = piter(range(n_iter), description="Training")
        else:
            train_iter = range(n_iter)

        for _ in train_iter:
            model, opt_state, loss = train_step(model, opt_state, X, y)
            if progress_bar:
                train_iter.set_postfix(loss=f"{loss:.4f}")

        return model


def grad_wrap_abstract_loss(loss_func: AbstractLoss):
    """Wraps a task loss function taking state to a `grad`-able one taking a model.

    It is convenient to first define the loss function in terms of a
    mapping from states to scalars, because sometimes we want to evaluate the
    loss on states without re-evaluating the model itself. It also helps to
    separate the mathematical logic of the loss function from the training
    logic of passing a model as an argument to a `grad`-able function.

    Note that we are assuming that

      1) `TaskTrainer` will manage a `where_train_spec` on the trainable parameters.
         When `jax.grad` is applied to the wrapper, the gradient will be
         taken with respect to the first argument `diff_model` only, and the
         `where_train_spec` defines this split.
      2) Model modules will use a `target_state, init_state, key` signature.

    TODO:
    - Typing
    - This is specific to `TaskTrainer`. Could it be more general?
      Note that the vmap here works the same as in `Task.eval`; perhaps if this
      is a `TaskTrainer`-specific function, then `Task` could provide an interface
    """

    @wraps(loss_func)
    def wrapper(
        diff_model: AbstractModel[StateT],
        static_model: AbstractModel[StateT],  #! Type is technically not identical to `diff_model`
        trial_specs: TaskTrialSpec,
        init_states: StateT,  #! has a batch dimension
        keys: PRNGKeyArray,  # per trial
    ) -> Tuple[Array, Tuple[TermTree[AbstractLoss], StateT]]:
        model = eqx.combine(diff_model, static_model)

        # ? will `in_axes` ever change?
        states: StateT = eqx.filter_vmap(model)(
            ModelInput(trial_specs.inputs, trial_specs.intervene), init_states, keys
        )

        losses = loss_func(states, trial_specs, model)

        return losses.total, (losses, states)

    return wrapper


def mask_diagonal(array):
    """Set the diagonal of (the last two dimensions of) `array` to zero."""
    mask = 1 - jnp.eye(array.shape[-1])
    return array * mask


class HebbianGRUUpdate(eqx.Module):
    """DEPRECATED. Use `HebbianUpdate`.

    Hebbian update rule for the recurrent weights of a GRUCell.

    This specifically applies the Hebbian update to the candidate activation
    weights of the GRU, while leaving the update and reset weights unchanged.
    """

    scale: float = 0.01
    mode: Literal["default", "differential"] = "default"
    weight_type: Literal["candidate", "update", "reset"] = "candidate"

    def __call__(self, model: AbstractModel[StateT], states: StateT) -> AbstractModel[StateT]:
        x = states.net.hidden

        if self.mode == "default":
            # Hebbian learning rule
            dW = x[..., :, None] @ x[..., None, :]
        elif self.mode == "differential":
            dx = jnp.diff(x, axis=-2)  # diff over time
            dW = dx[..., :, None] @ dx[..., None, :]
        else:
            raise ValueError("invalid mode field value encountered for HebbianGRUUpdate")

        dW = self.scale * dW
        # Updates do not apply to self weights.
        dW = mask_diagonal(dW)

        # Sum over all batch dimensions (e.g. trials, time)
        dW_batch = jnp.mean(jnp.reshape(dW, (-1, dW.shape[-2], dW.shape[-1])), axis=0)

        # Build the update for the appropriate weights of the GRU.
        weight_hh = jnp.zeros_like(model.step.net.hidden.weight_hh)
        weight_idxs = {
            "reset": slice(0, weight_hh.shape[-2] // 3),
            "update": slice(weight_hh.shape[-2] // 3, 2 * weight_hh.shape[-2] // 3),
            "candidate": slice(2 * weight_hh.shape[-2] // 3, None),
        }[self.weight_type]
        weight_hh = weight_hh.at[..., weight_idxs, :].set(dW_batch)

        update = eqx.tree_at(
            lambda model: model.step.net.hidden.weight_hh,
            jt.map(lambda x: None, model),
            weight_hh,
            is_leaf=is_none,
        )

        return update


def hebb_rule(
    activity: Float[Array, "*batch time units"],
) -> Float[Array, "*batch time units units"]:
    """Standard Hebbian learning rule, unscaled."""
    return batched_outer(activity, activity)


def hebb_differential_rule(
    activity: Float[Array, "*batch time units"],
) -> Float[Array, "*batch time units units"]:
    """Differential Hebbian learning rule, unscaled."""
    d_activity = jnp.diff(activity, axis=-2)
    return batched_outer(d_activity, d_activity)


def hebb_covariance_rule(
    activity: Float[Array, "*batch time units"],
    alpha: float = 0.01,
    init_window_size: int = 1,
) -> Float[Array, "*batch time units units"]:
    """Hebbian learning rule based on the local covariance of the activity.

    Uses exponential smoothing to estimate the local mean of the activity.
    """
    ema = exponential_smoothing(activity, alpha, init_window_size, axis=-2)
    return hebb_rule(activity - ema)


def oja_term(
    activity: Float[Array, "*batch time units"],
    weights: Float[Array, "units units"],
) -> Float[Array, "*batch time units units"]:
    """Regularization term from Oja's rule."""
    return -(activity**2)[..., None, :] * weights


def _null_weight_rule(
    activity: Float[Array, "*batch time units"],
    weights: Float[Array, "units units"],
):
    return 0.0


class ActivityDependentWeightUpdate(eqx.Module):
    """Compute a weight update according to a learning rule."""

    scale: float = 0.01
    rule: Callable[[Float[Array, "*batch time units"]], Float[Array, "*batch time units units"]] = (
        hebb_rule
    )
    # Keep this separate so that all the simpler rules don't need to be functions of `weights`
    weight_dep_rule: Callable[
        [Float[Array, "*batch time units"], Float[Array, "units units"]],
        Float[Array, "*batch time units units"] | float,
    ] = _null_weight_rule
    weight_dep_scale: float = 1.0
    agg_func: Callable = jnp.mean

    def __call__(
        self,
        activity: Float[Array, "*batch time units"],
        weights: Float[Array, "units units"],
    ) -> Float[Array, "units units"]:
        dW = self.rule(activity) + self.weight_dep_scale * self.weight_dep_rule(activity, weights)
        # Updates do not apply to self weights.
        dW = mask_diagonal(dW)

        # Aggregate over any batch dimensions (e.g. trials, time)
        dW_batch = self.agg_func(jnp.reshape(dW, (-1, dW.shape[-2], dW.shape[-1])), axis=0)

        return self.scale * dW_batch
