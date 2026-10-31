"""Write the same-setup baseline configs (docs/BASELINES.md) from the MambaIRv2 ones.

Only the header, name, model_type, network_g, pretrain path, test padding and results tag change; data, batch, lr,
schedule, seed and test protocol are copied verbatim from
  options/train/mambairv2/train_MambaIRv2_SR_x2_500k.yml, train_MambaIRv2_SR_x{3,4}_ft250k.yml
  options/test/mambairv2/test_MambaIRv2_SR_x2_500k.yml, test_MambaIRv2_SR_x{3,4}_ft250k.yml
usage: python tests/make_baseline_configs.py   (run from the repo root)
"""
import re
import yaml

ROOT = '/speed-scratch/se_gham/GatedMambaIR'
MEAN = '[0.4488, 0.4371, 0.4040]'

# official classic-SR hyperparameters (source repo config named in the comment); {s} = scale
MODELS = {
    'SwinIR': dict(  # SwinIR main_test_swinir.py, task classical_sr, training_patch_size 64
        model_type='BaselineSRModel', test_pad='{multiple: 8, mode: flip}',
        net='''  type: SwinIR
  upscale: {s}
  in_chans: 3
  img_size: 64
  window_size: 8
  img_range: 1.
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 2
  upsampler: 'pixelshuffle'
  resi_connection: '1conv'
'''),
    'HAT': dict(  # HAT options/train/train_HAT_SRx2_from_scratch.yml
        model_type='BaselineSRModel', test_pad='{multiple: 16, mode: reflect}',
        net='''  type: HAT
  upscale: {s}
  in_chans: 3
  img_size: 64
  window_size: 16
  compress_ratio: 3
  squeeze_factor: 30
  conv_scale: 0.01
  overlap_ratio: 0.5
  img_range: 1.
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 2
  upsampler: 'pixelshuffle'
  resi_connection: '1conv'
'''),
    'MambaIR': dict(  # options/train/mambair/train_MambaIR_SR_x2.yml (official MambaIR config)
        model_type='MambaIRModel', test_pad=None,
        net='''  type: MambaIR
  upscale: {s}
  in_chans: 3
  img_size: 64
  img_range: 1.
  d_state: 16
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  mlp_ratio: 2
  upsampler: 'pixelshuffle'
  resi_connection: '1conv'
'''),
    'RCAN': dict(  # RCAN paper / BasicSR rcan_arch.py with the official 10 groups x 20 blocks
        model_type='BaselineSRModel', test_pad=None,
        net=f'''  type: RCAN
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  num_group: 10
  num_block: 20
  squeeze_factor: 16
  upscale: {{s}}
  res_scale: 1
  img_range: 255.
  rgb_mean: {MEAN}
'''),
    'DAT': dict(  # DAT options/Train/train_DAT_x2.yml (use_chop False at test)
        model_type='BaselineSRModel', test_pad=None,
        net='''  type: DAT
  upscale: {s}
  in_chans: 3
  img_size: 64
  img_range: 1.
  split_size: [8, 32]
  depth: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  expansion_factor: 4
  resi_connection: '1conv'
'''),
    'EDSR': dict(  # EDSR (large) / BasicSR edsr_arch.py: 256 feats, 32 blocks, res_scale 0.1
        model_type='BaselineSRModel', test_pad=None,
        net=f'''  type: EDSR
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 256
  num_block: 32
  upscale: {{s}}
  res_scale: 0.1
  img_range: 255.
  rgb_mean: {MEAN}
'''),
    'CAT_A': dict(  # CAT options/train/train_CAT_A_sr_x2.yml (CATModel pads with flips to max(split_size_0))
        model_type='BaselineSRModel', test_pad='{multiple: 4, mode: flip}',
        net='''  type: CAT
  upscale: {s}
  in_chans: 3
  img_size: 64
  split_size_0: [2, 2, 2, 4, 4, 4]
  split_size_1: [0, 0, 0, 0, 0, 0]
  img_range: 1.
  depth: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 4
  resi_connection: '1conv'
  block_name: 'CATB_axial'
  upsampler: 'pixelshuffle'
'''),
    'SAN': dict(  # SAN TrainCode/option.py defaults + README (20 groups x 10 blocks, 64 feats, reduction 16)
        model_type='BaselineSRModel', test_pad=None, unused=True,
        net='''  type: SAN
  upscale: {s}
  n_resgroups: 20
  n_resblocks: 10
  n_feats: 64
  reduction: 16
  res_scale: 1
  img_range: 255.
'''),
}


def swap(text, pattern, repl):
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M | re.S)
    assert n == 1, pattern
    return new


def make(model, m, s, kind):
    ft = s != 2
    stem = f'{model}_SR_x{s}_' + ('ft250k' if ft else '500k')
    if kind == 'train':
        src = f'options/train/mambairv2/train_MambaIRv2_SR_x{s}_' + ('ft250k' if ft else '500k') + '.yml'
    else:
        src = f'options/test/mambairv2/test_MambaIRv2_SR_x{s}_' + ('ft250k' if ft else '500k') + '.yml'
    t = open(src).read()
    if kind == 'train':
        head = (f'# Same-setup baseline {model} x{s} (docs/BASELINES.md): data, batch, lr, schedule and seed copied from\n'
                f'# {src}; only network_g / model_type / test padding change.\n')
        if ft:
            head += f'# Fine-tuned from this model\'s own x2 500k weights ({model}_SR_x2_500k), upsampler skipped (strict false).\n'
        t = head + t[t.index('name:'):]
    t = swap(t, r'^name: .*?$', f'name: {"test_" if kind == "test" else ""}{stem}')
    t = swap(t, r'^model_type: .*?$', f'model_type: {m["model_type"]}')
    if m.get('unused'):
        t = swap(t, r'^(num_gpu: .*?$)', r'\1\nfind_unused_parameters: true  # SAN keeps unused modules (conv_last, Nonlocal_CA.soca)')
    t = swap(t, r'^network_g:\n.*?\n\n', 'network_g:\n' + m['net'].replace('{s}', str(s)) + '\n')
    if kind == 'train':
        if ft:
            t = swap(t, r'^  pretrain_network_g: .*?$',
                     f'  pretrain_network_g: {ROOT}/experiments/{model}_SR_x2_500k/models/net_g_500000.pth')
    else:
        t = swap(t, r'^  pretrain_network_g: .*?$',
                 f'  pretrain_network_g: {ROOT}/experiments/{stem}/models/net_g_{250000 if ft else 500000}.pth')
        t = swap(t, r'^  tag: .*?$', f'  tag: baseline-sameSetup-{model}  # free-text label stored in docs/results.csv')
    if m['test_pad']:
        t = swap(t, r'^(val:\n)', r'\1' + f'  test_pad: {m["test_pad"]}  # official test-time padding, see BaselineSRModel\n')
    out = f'options/{kind}/baselines/{kind}_{stem}.yml'
    yaml.safe_load(t)
    open(out, 'w').write(t)
    return out


if __name__ == '__main__':
    import os
    os.makedirs('options/train/baselines', exist_ok=True)
    os.makedirs('options/test/baselines', exist_ok=True)
    for model, m in MODELS.items():
        for s in (2, 3, 4):
            for kind in ('train', 'test'):
                print(make(model, m, s, kind))
