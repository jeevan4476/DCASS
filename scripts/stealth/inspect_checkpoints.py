import sys, pathlib
sys.modules['pathlib._local'] = pathlib
import torch

# Inspect GAN checkpoint
gan_path = 'storage/models/gan_generator.pt'
try:
    ckpt = torch.load(gan_path, map_location='cpu', weights_only=False)
    print('=== GAN Generator Checkpoint ===')
    print(f'Keys: {list(ckpt.keys())}')
    if 'config' in ckpt:
        cfg = ckpt['config']
        if hasattr(cfg, '__dict__'):
            print(f'Config: {cfg.__dict__}')
        else:
            print(f'Config: {cfg}')
    if 'epoch' in ckpt:
        print(f'Epoch: {ckpt["epoch"]}')
    if 'global_step' in ckpt:
        print(f'Global step: {ckpt["global_step"]}')
    if 'generator_state' in ckpt:
        gen_state = ckpt['generator_state']
        print(f'Generator state dict keys: {list(gen_state.keys())}')
        for k, v in list(gen_state.items())[:5]:
            if hasattr(v, 'shape'):
                print(f'  {k}: {v.shape}')
    if 'warden_state' in ckpt:
        warden_state = ckpt['warden_state']
        print(f'Warden state dict keys: {list(warden_state.keys())}')
        for k, v in list(warden_state.items())[:5]:
            if hasattr(v, 'shape'):
                print(f'  {k}: {v.shape}')
except Exception as e:
    print(f'GAN Error: {e}')
    import traceback
    traceback.print_exc()

# Inspect RL checkpoint
rl_path = 'storage/models/rl_agent.pt'
try:
    ckpt = torch.load(rl_path, map_location='cpu', weights_only=False)
    print('\n=== RL Agent Checkpoint ===')
    print(f'Keys: {list(ckpt.keys())}')
    if 'config' in ckpt:
        cfg = ckpt['config']
        if hasattr(cfg, '__dict__'):
            print(f'Config: {cfg.__dict__}')
        else:
            print(f'Config: {cfg}')
    if 'actor_critic_state' in ckpt:
        ac_state = ckpt['actor_critic_state']
        print(f'Actor-Critic state dict keys: {list(ac_state.keys())}')
        for k, v in list(ac_state.items())[:5]:
            if hasattr(v, 'shape'):
                print(f'  {k}: {v.shape}')
    if 'optimizer_state' in ckpt:
        print(f'Optimizer state present')
except Exception as e:
    print(f'RL Error: {e}')
    import traceback
    traceback.print_exc()