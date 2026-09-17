"""Use the official tokenizer, network, generation and checkpoint without training."""
import os
import sys
from common import ROOT, UPSTREAM

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_MODE'] = 'disabled'
sys.path.insert(0, str(UPSTREAM))


def config(name):
    import yaml
    from box import Box
    return Box(yaml.safe_load((UPSTREAM / 'configs' / name).read_text()))


def construct(cpu_audit=False):
    import torch
    from accelerate import init_empty_weights
    from src.tokenizer.spec import TokenizerConfig
    from src.tokenizer.parse import get_tokenizer
    from src.model.unirig_ar import UniRigAR
    # The official tokenizer references skeleton YAML paths relative to UniRig.
    os.chdir(UPSTREAM)
    tokenizer = get_tokenizer(TokenizerConfig.parse(config('tokenizer/tokenizer_parts_articulationxl_256.yaml')))
    cfg = config('model/unirig_ar_350m_1024_81920_float32.yaml')
    cfg.llm.pretrained_model_name_or_path = str(ROOT/'models/opt-350m')
    if cpu_audit:
        # Architecture-only meta audit; FlashAttention requires an active CUDA device.
        cfg.llm._attn_implementation = 'eager'
    # Keep deterministic nonpersistent Fourier buffers on CPU while parameters
    # are empty; plain torch.device('meta') would lose those non-checkpoint buffers.
    with init_empty_weights(include_buffers=False):
        model = UniRigAR(tokenizer=tokenizer, llm=cfg.llm, mesh_encoder=cfg.mesh_encoder)
    checkpoint = torch.load(ROOT/'models/unirig/model.ckpt', map_location='cpu', mmap=True, weights_only=True)
    state = checkpoint['state_dict']
    if not all(k.startswith('model.') for k in state):
        raise ValueError('Unexpected checkpoint state namespace')
    state = {k.removeprefix('model.'): v for k, v in state.items()}
    expected = model.state_dict()
    if set(expected) != set(state):
        raise ValueError(f'Checkpoint key mismatch: missing={set(expected)-set(state)}, extra={set(state)-set(expected)}')
    for key in expected:
        if expected[key].shape != state[key].shape:
            raise ValueError('Checkpoint shape mismatch: ' + key)
    audit = dict(tensor_keys=len(state), parameters=sum(p.numel() for p in model.parameters()), strict_shapes=True)
    if cpu_audit:
        return audit
    model.load_state_dict(state, strict=True, assign=True)
    model = model.to('cuda').eval()
    return model, audit


def generation_config():
    return dict(config('system/ar_inference_articulationxl.yaml').generate_kwargs)
