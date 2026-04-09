"""
FSDP Worker with HDPO support - Clean inheritance-based implementation.

This provides a cleaner alternative to monkey patching by properly
inheriting and overriding the necessary components.

Supports both sync and async rollout modes.
"""
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker as OriginalFSDPWorker,
    AsyncActorRolloutRefWorker as OriginalAsyncFSDPWorker
)
from verl.single_controller.base.decorator import Dispatch, register


class _HDPOActorMixin:
    """
    Mixin to add HDPO actor support to FSDP workers.
    
    This mixin can be used with both sync and async workers.
    """
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_info(self):
        """Return actor information for debugging/verification."""
        info = {
            "worker_class": self.__class__.__name__,
            "has_actor": hasattr(self, 'actor'),
            "is_actor": getattr(self, '_is_actor', False),
        }
        if hasattr(self, 'actor'):
            info["actor_class"] = self.actor.__class__.__name__
            info["actor_module"] = self.actor.__class__.__module__
        return info
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize model with HDPO actor instead of standard DataParallelPPOActor."""
        # Import HDPO actor for the main actor, and original for ref policy
        from verl_tool.workers.hdpo_actor import HDPODataParallelPPOActor as DataParallelPPOActor
        from verl.workers.actor.dp_actor import DataParallelPPOActor as OriginalDataParallelPPOActor
        from verl.utils.import_utils import import_external_libs
        from verl.utils.fs import copy_to_local
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import FSDPEngineConfig
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
        from verl.utils.fsdp_utils import fsdp_version, offload_fsdp_model_to_cpu, offload_fsdp_optimizer
        from verl.utils.profiler import log_gpu_memory_usage
        from verl.utils.flops_counter import FlopsCounter
        from omegaconf import OmegaConf, open_dict
        import logging
        
        logger = logging.getLogger(__name__)

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            # This will use HDPODataParallelPPOActor because of our import above
            self.actor = DataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )
            print(f"[HDPO] Initialized {type(self.actor).__name__}", flush=True)

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            # Ref policy only needs compute_log_prob, no HDPO dual-loss needed
            self.ref_policy = OriginalDataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # we still need to save the intermediate checkpoint
            # so that we can resume the rollout from a certain step.
            from verl.utils.tracking import Tracking
            from verl.utils.fs import copy_local_path_from_hdfs
            if isinstance(self.config.actor_rollout_ref.rollout.log_dir, str) and Tracking.is_hdfs_path(
                self.config.actor_rollout_ref.rollout.log_dir
            ):
                local_path = copy_local_path_from_hdfs(hdfs_path=self.config.actor_rollout_ref.rollout.log_dir)
                self.checkpoint_manager = FSDPCheckpointManager(
                    model=self.actor_module_fsdp,
                    optimizer=None,
                    lr_scheduler=None,
                    processing_class=self.processor if self.processor is not None else self.tokenizer,
                    checkpoint_config=self.config.actor.checkpoint,
                    local_save_path=local_path,
                )
            else:
                local_path = self.config.actor_rollout_ref.rollout.log_dir
                self.checkpoint_manager = FSDPCheckpointManager(
                    model=self.actor_module_fsdp,
                    optimizer=None,
                    lr_scheduler=None,
                    processing_class=self.processor if self.processor is not None else self.tokenizer,
                    checkpoint_config=self.config.actor.checkpoint,
                    local_save_path=local_path,
                )


class HDPOActorRolloutRefWorker(_HDPOActorMixin, OriginalFSDPWorker):
    """
    Sync FSDP Worker with HDPO actor support.
    
    Use this for standard (non-async) rollout mode.
    """
    pass


class HDPOAsyncActorRolloutRefWorker(_HDPOActorMixin, OriginalAsyncFSDPWorker):
    """
    Async FSDP Worker with HDPO actor support.
    
    Use this for async rollout mode (rollout and reward computation pipelined).
    
    Note: Only the rollout is async. The actor update (update_policy) is still
    synchronous as it requires gradients and parameter updates.
    """
    pass
