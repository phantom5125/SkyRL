from skyrl_train.model_wrapper import HFModelWrapper

from skyrl_train.distributed.fsdp_utils import get_fsdp_wrap_policy
import ray


@ray.remote(num_gpus=1)
def task():
    MODEL_NAME = "Qwen/Qwen3-8B"
    model = HFModelWrapper(
        pretrain_or_model=MODEL_NAME,
        use_flash_attention_2=True,
        bf16=False,
        sequence_parallel_size=1,
        use_sample_packing=True,
    )
    wrap_policy = get_fsdp_wrap_policy(model.model, None, is_lora=False)
    print("wrap policy", wrap_policy)
    return wrap_policy


ray.get(task.remote())
