"""Call the released networks; use meta parameters only to avoid redundant weight allocation."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from accelerate import init_empty_weights
from shared import ROOT, read, write


def checkpoint_path(name):
    return ROOT / ('models/riganything/riganything_ckpt.pt' if name == 'riganything'
                   else 'models/puppeteer/puppeteer_skeleton_w_diverse_pose.pth')


def construct(name, cpu_audit=False):
    if name == 'riganything':
        sys.path.insert(0, str(ROOT/'third_party/RigAnything'))
        from easydict import EasyDict
        import yaml
        from model.ar_rig_diffusion import RigARDiffusion
        import model.utils_ar_transformer as attention
        config = EasyDict(yaml.safe_load((ROOT/'third_party/RigAnything/config.yaml').read_text()))
        # The upstream attention constructor hardcodes CUDA. Only change construction placement;
        # all original attention masks, KV buffers, positional encodings and forward methods remain.
        original_init = attention.CustomCausalAttention.__init__
        def initialize_on_cpu(instance, *args, **kwargs):
            kwargs['device'] = 'cpu'
            original_init(instance, *args, **kwargs)
        with patch.object(attention.CustomCausalAttention, '__init__', initialize_on_cpu):
            with init_empty_weights(include_buffers=False):
                model = RigARDiffusion(config, device='cpu')
        return model
    sys.path.insert(0, str(ROOT/'third_party/Puppeteer/skeleton'))
    from omegaconf import OmegaConf
    import skeleton_models.skeletongen as skeleton
    from third_partys.Michelangelo.michelangelo.utils.misc import instantiate_from_config
    config = OmegaConf.load(ROOT/'third_party/Puppeteer/skeleton/third_partys/Michelangelo/configs/shapevae-256.yaml').model
    arguments = SimpleNamespace(llm=str(ROOT/'models/opt-350m'), joint_token=True, seq_shuffle=True,
                                n_discrete_size=128, n_max_bones=100, batchsize_per_gpu=1, num_beams=1)
    # The full released skeleton checkpoint includes the point encoder. Strict matching below
    # requires that entire encoder; no randomly initialized or missing encoder weights are allowed.
    with ExitStack() as stack:
        stack.enter_context(patch.object(skeleton, 'load_model', lambda: instantiate_from_config(config, ckpt_path=None)))
        if cpu_audit:
            original = skeleton.AutoModelForCausalLM.from_config
            def cpu_transformer(*, config, **kwargs):
                config._attn_implementation = 'eager'
                return original(config=config, attn_implementation='eager')
            stack.enter_context(patch.object(skeleton.AutoModelForCausalLM, 'from_config', cpu_transformer))
        stack.enter_context(init_empty_weights(include_buffers=False))
        model = skeleton.SkeletonGPT(arguments)
    return model


def load_state(name):
    # Only the canonical, SHA-256-verified official checkpoint is accepted by preflight/run.
    torch.serialization.add_safe_globals([argparse.Namespace])
    package = torch.load(checkpoint_path(name), mmap=True, map_location='cpu', weights_only=True)
    return package, package['model']


def inspect(name):
    model = construct(name, cpu_audit=True)
    if name == 'puppeteer':
        import flash_attn
        from postprocess import puppeteer_postprocess
        puppeteer_postprocess()
        decoded = model.detokenize_joint_token(torch.tensor([[64, 64, 64, 0, 64, 96, 64, 1]]))
        torch.testing.assert_close(decoded.float(), torch.tensor([[[[0., 0., 0.], [0., .25, 0.]]]]))
    assert all(not buffer.is_meta for buffer in model.buffers()), 'Missing real constant/KV buffers'
    package, state = load_state(name)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = [key for key in expected.keys() & state.keys() if expected[key].shape != state[key].shape]
    if missing or unexpected or mismatched:
        raise ValueError(dict(missing=missing, unexpected=unexpected, mismatched=mismatched))
    encoder = [key for key in state if key.startswith('point_encoder.')]
    result = dict(status='MODEL_STRUCTURE_PASS', model=name, tensor_keys=len(state),
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  checkpoint_model_bytes=sum(t.numel()*t.element_size() for t in state.values()),
                  point_encoder_keys=len(encoder), all_parameter_shapes_match=True,
                  constant_buffers_materialized=True,
                  torch=torch.__version__, gpu_available=torch.cuda.is_available(), gpu_inference_run=False)
    if name == 'puppeteer' and not encoder:
        raise ValueError('Puppeteer checkpoint must contain the shape encoder')
    if name == 'puppeteer':
        result.update(tokenization_contract_pass=True, flash_attn=flash_attn.__version__)
    else:
        result['actual_diffusion_steps'] = model.diffloss.gen_diffusion.num_timesteps
    print(result, flush=True)
    return result


class Runtime:
    def __init__(self, name):
        if not torch.cuda.is_available():
            raise RuntimeError('GPU is not enabled; this command will not run inference on CPU')
        self.name = name
        torch.cuda.set_device(0)
        self.model = construct(name)
        self.package, state = load_state(name)
        self.model.load_state_dict(state, strict=True, assign=True)
        self.model.to('cuda').eval()
        if name == 'riganything':
            self.model.device = 'cuda'
            self.model.joint_index_pos_embedding = self.model.joint_index_pos_embedding.to('cuda')
            for block in self.model.transformer:
                block.attn.attention_mask = block.attn.attention_mask.to('cuda')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.dtype = torch.bfloat16 if name == 'riganything' else torch.float16
        self.raw_tokens = None
        if name == 'puppeteer':
            original_generate = self.model.transformer.generate
            def capture_tokens(*args, **kwargs):
                tokens = original_generate(*args, **kwargs)
                self.raw_tokens = tokens.detach().cpu().numpy()
                return tokens
            self.model.transformer.generate = capture_tokens

    @torch.no_grad()
    def predict(self, arrays, seed=12345):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        with torch.autocast('cuda', dtype=self.dtype):
            if self.name == 'riganything':
                batch = dict(pointcloud=torch.as_tensor(arrays['encoder_points'], device='cuda').unsqueeze(0),
                             normals=torch.as_tensor(arrays['normals'], device='cuda').unsqueeze(0),
                             scale=torch.ones(1, device='cuda'), center=torch.zeros(3, device='cuda'))
                result = self.model.generate_sequence(batch, create_visual=False, save_skeleton=False, compute_loss=False)['npz_dict']
                joints = np.asarray(result['joints']).reshape(-1, 3)
                parents = np.asarray(result['parents'], dtype=np.int64).reshape(-1)
                if parents[0] != 0 or any(not 0 <= p < i for i, p in enumerate(parents[1:], 1)):
                    raise ValueError('Invalid autoregressive RigAnything topology')
                bones = np.array([[p, i] for i, p in enumerate(parents) if i > 0], dtype=np.int64).reshape(-1, 2)
                return joints, bones, dict(parents=parents, input_skinning_weights=result['skinning_weights'])
            from postprocess import puppeteer_postprocess
            post = puppeteer_postprocess()
            data = np.concatenate([arrays['encoder_points'], arrays['normals']], axis=1).astype(np.float16)
            generated = self.model.generate(dict(pc_normal=torch.as_tensor(data, device='cuda').unsqueeze(0)))
            if generated is None or generated.ndim != 4 or not torch.isfinite(generated).all():
                raise ValueError('Puppeteer returned an invalid token sequence')
            if not np.any(self.raw_tokens == self.model.eos_token_id):
                raise ValueError('Puppeteer sequence reached the limit without EOS')
            raw_bones = generated[0].float().cpu().numpy()
            joints, bones = post.pred_joints_and_bones(raw_bones)
            joints, bones = post.merge_duplicate_joints_and_fix_bones(joints, bones)
            return joints, np.asarray(bones, dtype=np.int64).reshape(-1, 2), dict(raw_bone_coordinates=raw_bones, tokens=self.raw_tokens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['riganything', 'puppeteer'], required=True)
    parser.add_argument('--audit-output', type=Path, required=True)
    args = parser.parse_args()
    write(args.audit_output, inspect(args.model))
