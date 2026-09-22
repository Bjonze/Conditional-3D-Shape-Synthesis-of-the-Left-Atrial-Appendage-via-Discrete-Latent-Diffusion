"""Top-level stage-2 model: descriptor-conditioned latent diffusion.

Ties together the three pieces named in ``configs/diffusion.yaml``:

* ``content_codec``  - the frozen VQ-GAN that decodes token grids to masks,
* ``condition_codec`` - the descriptor pass-through,
* ``transformer``    - the discrete diffusion process and its denoiser.

Adapted from VQ-Diffusion (Microsoft, MIT licence); the text/image specific
paths were replaced by descriptor conditioning over 3D volumes.
"""

import torch
from torch import nn
from torch.cuda.amp import autocast

from laa_ldm.utils.config import instantiate_from_config

__all__ = ["ConditionalLatentDiffusion"]


class ConditionalLatentDiffusion(nn.Module):
    """Descriptor-conditioned discrete latent diffusion over VQ-GAN codes.

    Args:
        content_info: key of the token grid in a batch (``indices``).
        condition_info: key of the descriptor vector in a batch (``ctx``).
        learnable_cf: use a learned empty embedding for classifier-free guidance.
        content_codec_config: config of the frozen VQ-GAN codec.
        condition_codec_config: config of the descriptor codec.
        diffusion_config: config of the diffusion transformer.
    """

    def __init__(
        self,
        *,
        content_info={'key': 'image'},
        condition_info={'key': 'ctx'},
        learnable_cf=True,
        content_codec_config,
        condition_codec_config,
        diffusion_config
    ):
        super().__init__()
        self.content_info = content_info
        self.condition_info = condition_info
        self.guidance_scale = 5.0
        self.learnable_cf = learnable_cf
        self.content_codec = instantiate_from_config(content_codec_config)
        self.condition_codec = instantiate_from_config(condition_codec_config)
        self.transformer = instantiate_from_config(diffusion_config)
        self.truncation_forward = False

    def parameters(self, recurse=True, name=None):
        """All parameters, or those of the named submodule (used by the solver)."""
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            names = name.split('+')
            params = []
            for n in names:
                try: # the parameters() method is not overwritten for some classes
                    params += getattr(self, name).parameters(recurse=recurse, name=name)
                except:
                    params += getattr(self, name).parameters(recurse=recurse)
            return params

    @property
    def device(self):
        return self.transformer.device

    def get_ema_model(self):
        return self.transformer

    def prepare_condition(self, batch, condition=None):
        """Descriptor vector -> ``{'condition_token': ..., 'condition_mask': ...}``."""
        cond_key = self.condition_info['key']
        cond = batch[cond_key] if condition is None else condition
        if torch.is_tensor(cond):
            cond = cond.to(self.device)
        cond = self.condition_codec.get_tokens(cond)
        cond_ = {}
        for k, v in cond.items():
            v = v.to(self.device) if torch.is_tensor(v) else v
            cond_['condition_' + k] = v
        return cond_

    @autocast(enabled=False)
    @torch.no_grad()
    def prepare_content(self, batch):
        """Token grid -> ``{'content_token': ...}``."""
        cont = batch[self.content_info['key']]
        if torch.is_tensor(cont):
            cont = cont.to(self.device)
        cont = self.content_codec.get_tokens(cont)
        cont_ = {}
        for k, v in cont.items():
            v = v.to(self.device) if torch.is_tensor(v) else v
            cont_['content_' + k] = v
        return cont_

    @autocast(enabled=False)
    def prepare_input(self, batch):
        """Assemble the transformer input from one dataloader batch."""
        input = self.prepare_condition(batch)
        input.update(self.prepare_content(batch))
        return input

    def p_sample_with_truncation(self, func, sample_type):
        truncation_rate = float(sample_type.replace('q', ''))
        def wrapper(*args, **kwards):
            out = func(*args, **kwards)
            import random
            if random.random() < truncation_rate:
                out = func(out, args[1], args[2], **kwards)
            return out
        return wrapper


    def predict_start_with_truncation(self, func, sample_type):
        """Wrap ``predict_start`` with top-r nucleus truncation.

        ``sample_type`` looks like ``"top0.86r"``: keep the smallest set of
        codes whose cumulative probability exceeds r, and floor the rest.
        """
        if sample_type[-1] != 'r':
            raise ValueError(f"Unsupported truncation {sample_type!r}; expected e.g. 'top0.86r'.")
        truncation_r = float(sample_type[:-1].replace('top', ''))

        def wrapper(*args, **kwards):
            out = func(*args, **kwards)
            # notice for different batches, out are same, we do it on out[0]
            temp, indices = torch.sort(out, 1, descending=True)
            temp1 = torch.exp(temp)
            temp2 = temp1.cumsum(dim=1)
            temp3 = temp2 < truncation_r
            new_temp = torch.full_like(temp3[:, 0:1, :], True)
            temp6 = torch.cat((new_temp, temp3), dim=1)
            temp3 = temp6[:, :-1, :]
            temp4 = temp3.gather(1, indices.argsort(1))
            temp5 = temp4.float() * out + (1 - temp4.float()) * (-70)
            return temp5

        return wrapper

    @torch.no_grad()
    def generate_content(
        self,
        *,
        batch,
        condition=None,
        filter_ratio = 0.5,
        temperature = 1.0,
        content_ratio = 0.0,
        replicate=1,
        sample_type="top0.85r",
    ):
        """Sample shapes for the descriptors in ``batch`` (or in ``condition``).

        Args:
            batch: dict with the descriptor vector under ``ctx``.
            filter_ratio: 0 starts from a fully masked grid; >0 starts from a
                partially noised version of the batch's own tokens.
            replicate: number of samples to draw per descriptor vector.
            sample_type: ``"top<r>r"`` nucleus truncation, optionally followed
                by ``",time<f>"`` to trade sampling steps for speed.

        Returns:
            ``{'content': occupancy volumes (B, 1, D, H, W)}``.
        """
        self.eval()
        if condition is None:
            condition = self.prepare_condition(batch=batch)
        else:
            condition = self.prepare_condition(batch=None, condition=condition)
        
        batch_size = len(batch['ctx']) * replicate

        if self.learnable_cf:
            cf_cond_emb = self.transformer.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            #batch['ctx'] = [''] * batch_size
            #if batch_size != 1:
            #    batch['ctx'] = torch.zeros_like(batch['ctx']).repeat(batch_size,1)
            #else: 
            batch['ctx'] = torch.zeros_like(batch['ctx'])
            cf_condition = self.prepare_condition(batch=batch)
            cf_cond_emb = self.transformer.condition_emb(cf_condition['condition_token']).float()
        
        def cf_predict_start(log_x_t, cond_emb, t):
            log_x_recon = self.transformer.predict_start(log_x_t, cond_emb, t)[:, :-1]
            if abs(self.guidance_scale - 1) < 1e-3:
                return torch.cat((log_x_recon, self.transformer.zero_vector), dim=1)
            cf_log_x_recon = self.transformer.predict_start(log_x_t, cf_cond_emb.type_as(cond_emb), t)[:, :-1]
            log_new_x_recon = cf_log_x_recon + self.guidance_scale * (log_x_recon - cf_log_x_recon)
            log_new_x_recon -= torch.logsumexp(log_new_x_recon, dim=1, keepdim=True)
            log_new_x_recon = log_new_x_recon.clamp(-70, 0)
            log_pred = torch.cat((log_new_x_recon, self.transformer.zero_vector), dim=1)
            return log_pred

        if replicate != 1:
            for k in condition.keys():
                if condition[k] is not None:
                    condition[k] = torch.cat([condition[k] for _ in range(replicate)], dim=0)
            
        content_token = None

        if len(sample_type.split(',')) > 1:
            if sample_type.split(',')[1][:1]=='q':
                self.transformer.p_sample = self.p_sample_with_truncation(self.transformer.p_sample, sample_type.split(',')[1])
        if sample_type.split(',')[0][:3] == "top" and self.truncation_forward == False:
            self.transformer.cf_predict_start = self.predict_start_with_truncation(cf_predict_start, sample_type.split(',')[0])
            self.truncation_forward = True

        if len(sample_type.split(',')) == 2 and sample_type.split(',')[1][:4]=='time' and int(float(sample_type.split(',')[1][4:])) >= 2:
            trans_out = self.transformer.sample_fast(condition_token=condition['condition_token'],
                                                condition_mask=condition.get('condition_mask', None),
                                                condition_embed=condition.get('condition_embed_token', None),
                                                content_token=content_token,
                                                filter_ratio=filter_ratio,
                                                temperature=temperature,
                                                return_logits=False,
                                                print_log=False,
                                                sample_type=sample_type,
                                                skip_step=int(float(sample_type.split(',')[1][4:])-1))

        else:
            if 'time' in sample_type and float(sample_type.split(',')[1][4:]) < 1:
                self.transformer.prior_ps = int(512 // self.transformer.num_timesteps * float(sample_type.split(',')[1][4:]))
                if self.transformer.prior_rule == 0:
                    self.transformer.prior_rule = 1
                self.transformer.update_n_sample()
            trans_out = self.transformer.sample(condition_token=condition['condition_token'],
                                            condition_mask=condition.get('condition_mask', None),
                                            condition_embed=condition.get('condition_embed_token', None),
                                            content_token=content_token,
                                            filter_ratio=filter_ratio,
                                            temperature=temperature,
                                            return_logits=False,
                                            print_log=False,
                                            sample_type=sample_type)


        content = self.content_codec.decode(trans_out['content_token'])
        self.train()
        return {'content': content}
    
    @torch.no_grad()
    def reconstruct(
        self,
        input
    ):
        if torch.is_tensor(input):
            input = input.to(self.device)
        cont = self.content_codec.get_tokens(input)
        cont_ = {}
        for k, v in cont.items():
            v = v.to(self.device) if torch.is_tensor(v) else v
            cont_['content_' + k] = v
        rec = self.content_codec.decode(cont_['content_token'])
        return rec

    @torch.no_grad()
    def sample(
        self,
        batch,
        temperature=1.,
        return_rec=True,
        filter_ratio=[0, 0.5, 1.0],
        content_ratio=[1],
        return_logits=False,
        sample_type="normal",
        **kwargs,
    ):
        """Training-time preview: reconstruct and re-sample one batch.

        Used by the solver to dump meshes during training.  Returns a dict of
        volumes keyed by the filter/content ratio they were produced with.
        """
        self.eval()
        condition = self.prepare_condition(batch)
        content = self.prepare_content(batch)

        content_samples = {'input_image': batch[self.content_info['key']]}
        if return_rec:
            content_samples['reconstruction_image'] = self.content_codec.decode(
                content['content_token'])

        for fr in filter_ratio:
            for cr in content_ratio:
                num_content_tokens = int(content['content_token'].shape[1] * cr)
                if num_content_tokens < 0:
                    continue
                content_token = content['content_token'][:, :num_content_tokens]
                trans_out = self.transformer.sample(
                    condition_token=condition['condition_token'],
                    condition_mask=condition.get('condition_mask', None),
                    condition_embed=condition.get('condition_embed_token', None),
                    content_token=content_token,
                    filter_ratio=fr,
                    temperature=temperature,
                    return_logits=return_logits,
                    content_logits=content.get('content_logits', None),
                    sample_type=sample_type,
                    **kwargs)

                key = 'cond1_cont{}_fr{}_image'.format(cr, fr)
                content_samples[key] = self.content_codec.decode(trans_out['content_token'])
                if return_logits:
                    content_samples['logits'] = trans_out['logits']

        self.train()
        output = {'condition': batch[self.condition_info['key']]}
        output.update(content_samples)
        return output

    def forward(self, batch, name='none', **kwargs):
        """Training forward: returns the loss and the predicted x0 logits."""
        input = self.prepare_input(batch)
        output = self.transformer(input, **kwargs)
        return output
