import torch


# path = 'output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp/checkpoint0004.pth'

# missed_keys = [
#     "decoder.decoder.layers.0.self_attn.in_proj.weight", 
#     "decoder.decoder.layers.0.self_attn.in_proj.bias", 
#     "decoder.decoder.layers.0.update_attn.in_proj.weight", 
#     "decoder.decoder.layers.0.update_attn.in_proj.bias", 
#     "decoder.decoder.layers.1.self_attn.in_proj.weight", 
#     "decoder.decoder.layers.1.self_attn.in_proj.bias", 
#     "decoder.decoder.layers.1.update_attn.in_proj.weight", 
#     "decoder.decoder.layers.1.update_attn.in_proj.bias", 
#     "decoder.decoder.layers.2.self_attn.in_proj.weight", 
#     "decoder.decoder.layers.2.self_attn.in_proj.bias", 
#     "decoder.decoder.layers.2.update_attn.in_proj.weight", 
#     "decoder.decoder.layers.2.update_attn.in_proj.bias", 
#     "decoder.decoder.layers.3.self_attn.in_proj.weight", 
#     "decoder.decoder.layers.3.self_attn.in_proj.bias", 
#     "decoder.decoder.layers.3.update_attn.in_proj.weight", 
#     "decoder.decoder.layers.3.update_attn.in_proj.bias", 
#     "decoder.decoder.layers.4.self_attn.in_proj.weight", 
#     "decoder.decoder.layers.4.self_attn.in_proj.bias", 
#     "decoder.decoder.layers.4.update_attn.in_proj.weight", 
#     "decoder.decoder.layers.4.update_attn.in_proj.bias", 
#     "decoder.decoder.layers.5.self_attn.in_proj.weight", 
#     "decoder.decoder.layers.5.self_attn.in_proj.bias",
#     "decoder.decoder.layers.5.update_attn.in_proj.weight", 
#     "decoder.decoder.layers.5.update_attn.in_proj.bias"
#      ]

# weight = torch.load(path, map_location='cpu')

# for mk in missed_keys:
#     weight['model'][mk] = weight['model'][mk.replace('in_proj.weight', 'in_proj_weight').replace('in_proj.bias', 'in_proj_bias')]
#     weight['ema']['module'][mk] = weight['ema']['module'][mk.replace('in_proj.weight', 'in_proj_weight').replace('in_proj.bias', 'in_proj_bias')]

# torch.save(weight, path.replace('.pth', '_converted.pth'))








# missed keys in RT-DETR pretrained ckpt
path = 'output/rtdetr_r50vd_6x_coco_from_paddle.pth'
missed_keys = [
    'encoder.encoder.0.layers.0.self_attn.in_proj.weight', 
    'encoder.encoder.0.layers.0.self_attn.in_proj.bias', 
    'decoder.decoder.layers.0.self_attn.in_proj.weight', 
    'decoder.decoder.layers.0.self_attn.in_proj.bias', 
    # 'decoder.decoder.layers.0.update_attn.in_proj_weight', 
    # 'decoder.decoder.layers.0.update_attn.in_proj_bias', 
    # 'decoder.decoder.layers.0.update_attn.in_proj.weight', 
    # 'decoder.decoder.layers.0.update_attn.in_proj.bias', 
    # 'decoder.decoder.layers.0.update_attn.out_proj.weight', 
    # 'decoder.decoder.layers.0.update_attn.out_proj.bias', 
    # 'decoder.decoder.layers.0.norm4.weight', 
    # 'decoder.decoder.layers.0.norm4.bias', 
    'decoder.decoder.layers.1.self_attn.in_proj.weight', 
    'decoder.decoder.layers.1.self_attn.in_proj.bias', 
    # 'decoder.decoder.layers.1.update_attn.in_proj_weight', 
    # 'decoder.decoder.layers.1.update_attn.in_proj_bias', 
    # 'decoder.decoder.layers.1.update_attn.in_proj.weight', 
    # 'decoder.decoder.layers.1.update_attn.in_proj.bias', 
    # 'decoder.decoder.layers.1.update_attn.out_proj.weight', 
    # 'decoder.decoder.layers.1.update_attn.out_proj.bias', 
    # 'decoder.decoder.layers.1.norm4.weight', 
    # 'decoder.decoder.layers.1.norm4.bias', 
    'decoder.decoder.layers.2.self_attn.in_proj.weight', 
    'decoder.decoder.layers.2.self_attn.in_proj.bias', 
    # 'decoder.decoder.layers.2.update_attn.in_proj_weight', 
    # 'decoder.decoder.layers.2.update_attn.in_proj_bias', 
    # 'decoder.decoder.layers.2.update_attn.in_proj.weight', 
    # 'decoder.decoder.layers.2.update_attn.in_proj.bias', 
    # 'decoder.decoder.layers.2.update_attn.out_proj.weight', 
    # 'decoder.decoder.layers.2.update_attn.out_proj.bias', 
    # 'decoder.decoder.layers.2.norm4.weight', 
    # 'decoder.decoder.layers.2.norm4.bias', 
    'decoder.decoder.layers.3.self_attn.in_proj.weight', 
    'decoder.decoder.layers.3.self_attn.in_proj.bias', 
    # 'decoder.decoder.layers.3.update_attn.in_proj_weight', 
    # 'decoder.decoder.layers.3.update_attn.in_proj_bias', 
    # 'decoder.decoder.layers.3.update_attn.in_proj.weight', 
    # 'decoder.decoder.layers.3.update_attn.in_proj.bias', 
    # 'decoder.decoder.layers.3.update_attn.out_proj.weight', 
    # 'decoder.decoder.layers.3.update_attn.out_proj.bias', 
    # 'decoder.decoder.layers.3.norm4.weight', 
    # 'decoder.decoder.layers.3.norm4.bias', 
    'decoder.decoder.layers.4.self_attn.in_proj.weight', 
    'decoder.decoder.layers.4.self_attn.in_proj.bias', 
    # 'decoder.decoder.layers.4.update_attn.in_proj_weight', 
    # 'decoder.decoder.layers.4.update_attn.in_proj_bias', 
    # 'decoder.decoder.layers.4.update_attn.in_proj.weight', 
    # 'decoder.decoder.layers.4.update_attn.in_proj.bias', 
    # 'decoder.decoder.layers.4.update_attn.out_proj.weight', 
    # 'decoder.decoder.layers.4.update_attn.out_proj.bias', 
    # 'decoder.decoder.layers.4.norm4.weight', 
    # 'decoder.decoder.layers.4.norm4.bias', 
    'decoder.decoder.layers.5.self_attn.in_proj.weight', 
    'decoder.decoder.layers.5.self_attn.in_proj.bias', 
    # 'decoder.decoder.layers.5.update_attn.in_proj_weight', 
    # 'decoder.decoder.layers.5.update_attn.in_proj_bias',
    # 'decoder.decoder.layers.5.update_attn.in_proj.weight', 
    # 'decoder.decoder.layers.5.update_attn.in_proj.bias', 
    # 'decoder.decoder.layers.5.update_attn.out_proj.weight', 
    # 'decoder.decoder.layers.5.update_attn.out_proj.bias', 
    # 'decoder.decoder.layers.5.norm4.weight', 
    # 'decoder.decoder.layers.5.norm4.bias', 
    # 'track_embed.self_attn.in_proj_weight', 
    # 'track_embed.self_attn.in_proj_bias', 
    # 'track_embed.self_attn.in_proj.weight', 
    # 'track_embed.self_attn.in_proj.bias', 
    # 'track_embed.self_attn.out_proj.weight', 
    # 'track_embed.self_attn.out_proj.bias', 
    # 'track_embed.linear1.weight', 
    # 'track_embed.linear1.bias', 
    # 'track_embed.linear2.weight', 
    # 'track_embed.linear2.bias', 
    # 'track_embed.linear_feat1.weight', 
    # 'track_embed.linear_feat1.bias', 
    # 'track_embed.linear_feat2.weight', 
    # 'track_embed.linear_feat2.bias', 
    # 'track_embed.norm_feat.weight', 
    # 'track_embed.norm_feat.bias', 
    # 'track_embed.norm1.weight', 
    # 'track_embed.norm1.bias', 
    # 'track_embed.norm2.weight', 
    # 'track_embed.norm2.bias', 
    # 'track_embed.norm3.weight', 
    # 'track_embed.norm3.bias'
    ]


weight = torch.load(path, map_location='cpu')
# import pdb; pdb.set_trace()
for mk in missed_keys:
    # weight['model'][mk] = weight['model'][mk.replace('in_proj.weight', 'in_proj_weight').replace('in_proj.bias', 'in_proj_bias')]
    weight['ema']['module'][mk] = weight['ema']['module'][mk.replace('in_proj.weight', 'in_proj_weight').replace('in_proj.bias', 'in_proj_bias')]

torch.save(weight, path.replace('.pth', '_converted.pth'))