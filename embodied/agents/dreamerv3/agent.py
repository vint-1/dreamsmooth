import embodied
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
f32 = jnp.float32

import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    return 'check_types' not in record.getMessage()
logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj


@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.wm = WorldModel(obs_space, act_space, config, name='wm')
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.config, name='task_behavior')
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.config, name='expl_behavior')

  def policy_initial(self, batch_size):
    return (
        self.wm.initial(batch_size),
        self.task_behavior.initial(batch_size),
        self.expl_behavior.initial(batch_size))

  def train_initial(self, batch_size):
    return self.wm.initial(batch_size)

  def policy(self, obs, state, mode='train'):
    self.config.jax.jit and print('Tracing policy function.')
    obs = self.preprocess(obs)
    (prev_latent, prev_action), task_state, expl_state = state
    embed = self.wm.encoder(obs)
    latent = self.wm.rssm.obs_step(
        prev_latent, prev_action, embed, obs['is_first'])
    self.expl_behavior.policy(latent, expl_state)
    task_outs, task_state = self.task_behavior.policy(latent, task_state)
    expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
    outs = {'eval': task_outs, 'explore': expl_outs, 'train': task_outs}[mode]
    state = ((latent, outs['action']), task_state, expl_state)
    return outs, state

  def train(self, data, state):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    data = self.preprocess(data)
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    if self.config.run.is_train_behavior:
      _, mets = self.task_behavior.train(self.wm.imagine, start, context)
      metrics.update(mets)
      if self.config.expl_behavior != 'None':
        _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
        metrics.update({'expl_' + key: value for key, value in mets.items()})
    outs = {}
    return outs, state, metrics

  def report(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.wm.report(data))
    mets = self.task_behavior.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})
    if self.expl_behavior is not self.task_behavior:
      mets = self.expl_behavior.report(data)
      report.update({f'expl_{k}': v for k, v in mets.items()})
    return report

  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)
      obs[key] = value
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
    return obs


class WorldModel(nj.Module):

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.config = config
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
    self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')

    if config.rssm_type == 'rssm':
      self.rssm = nets.RSSM(**config.rssm, name='rssm')
    elif config.rssm_type == 'simple':
      kw = dict(config.rssm)
      [kw.pop(key) for key in ('impl', 'maskgit')]
      self.rssm = nets.SimpleRSSM(**kw, name='rssm')
    elif config.rssm_type == 'early':
      kw = dict(config.rssm)
      [kw.pop(key) for key in ('impl', 'maskgit')]
      self.rssm = nets.EarlyRSSM(**kw, name='rssm')
    else:
      raise NotImplementedError(config.rssm_type)

    self.heads = {
        'decoder': nets.MultiDecoder(shapes, **config.decoder, name='dec'),
        'reward': nets.MLP((), **config.reward_head, name='rew'),
        'cont': nets.MLP((), **config.cont_head, name='cont')}
    self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)
    scales = self.config.loss_scales.copy()
    image, vector = scales.pop('image'), scales.pop('vector')
    scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
    scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
    self.scales = scales

  def initial(self, batch_size):
    prev_latent = self.rssm.initial(batch_size)
    prev_action = jnp.zeros((batch_size, *self.act_space.shape))
    return prev_latent, prev_action

  def train(self, data, state):
    modules = [self.encoder, self.rssm, *self.heads.values()]
    mets, (state, outs, metrics) = self.opt(
        modules, self.loss, data, state, has_aux=True)
    metrics.update(mets)
    return state, outs, metrics

  def loss(self, data, state):
    embed = self.encoder(data)
    prev_latent, prev_action = state
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    post = self.rssm.observe(
        embed, prev_actions, data['is_first'], prev_latent)
    dists = {}
    feats = {**post, 'embed': embed}
    for name, head in self.heads.items():
      out = head(feats if name in self.config.grad_heads else sg(feats))
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)
    losses = {}
    if self.config.rssm_type == 'early':
      rssm_losses, prior = self.rssm.loss(
          post, prev_actions, **self.config.rssm_loss)
    else:
      rssm_losses, prior = self.rssm.loss(
          post, **self.config.rssm_loss)
    losses.update(rssm_losses)
    for key, dist in dists.items():
      loss = -dist.log_prob(data[key].astype(jnp.float32))
      assert loss.shape == embed.shape[:2], (key, loss.shape)
      losses[key] = loss
    scaled = {k: v * self.scales[k] for k, v in losses.items()}
    model_loss = sum(scaled.values())
    out = {'embed':  embed, 'post': post, 'prior': prior}
    out.update({f'{k}_loss': v for k, v in losses.items()})
    last_latent = {k: v[:, -1] for k, v in post.items()}
    last_action = data['action'][:, -1]
    state = last_latent, last_action
    metrics = self._metrics(data, dists, post, prior, losses, model_loss)
    return model_loss.mean(), (state, out, metrics)

  def imagine(self, policy, start, horizon, carry=None):
    if carry is None:
      policy = lambda s, c, f=policy: (f(s), {})
      carry = {}
    state_keys = list(self.rssm.initial(1).keys())
    state = {k: v for k, v in start.items() if k in state_keys}
    action, carry = policy(state, carry)
    keys = list(state.keys()) + list(action.keys()) + list(carry.keys())
    assert len(set(keys)) == len(keys), ('Colliding keys', keys)
    def step(prev, _):
      state, action, carry = prev
      state = self.rssm.img_step(state, action['action'])
      action, carry = policy(state, carry)
      return state, action, carry
    states, actions, carries = jaxutils.scan(
        step, jnp.arange(horizon), (state, action, carry),
        self.config.imag_unroll)
    states, actions, carries = tree_map(
        lambda traj, first: jnp.concatenate([first[None], traj], 0),
        (states, actions, carries), (state, action, carry))
    traj = {**states, **actions, **carries}
    if self.config.imag_cont == 'mode':
      cont = self.heads['cont'](traj).mode()
    elif self.config.imag_cont == 'mean':
      cont = self.heads['cont'](traj).mean()
    else:
      raise NotImplementedError(self.config.imag_cont)
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
    traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
    return traj

  def report(self, data): # This is where openl videos are generated
    MAX_VIDS = 10
    state = self.initial(len(data['is_first']))
    report = {}
    report.update(self.loss(data, state)[-1][-1])
    latent_state = self.rssm.observe(
        self.encoder(data)[:MAX_VIDS,:], data['action'][:MAX_VIDS,:],
        data['is_first'][:MAX_VIDS,:])
    # context = self.rssm.observe(
    #     self.encoder(data)[:MAX_VIDS, :5], data['action'][:MAX_VIDS, :5],
    #     data['is_first'][:MAX_VIDS, :5])
    context = {k: v[:, :5] for k, v in latent_state.items()}
    start = {k: v[:, -1] for k, v in context.items()}
    
    recon_full = self.heads['decoder'](latent_state)
    recon = self.heads['decoder'](context)
    imagined_state = self.rssm.imagine(data['action'][:MAX_VIDS, 5:], start)
    openl = self.heads['decoder'](imagined_state)
    # # TODO do prior and posterior predictions, and then compute KL for stochastic states here!
    # import ipdb; ipdb.set_trace()
    # reward predictions
    predicted_rew = self.heads['reward'](latent_state).mode()
    imagined_rew = self.heads['reward'](imagined_state).mode()
    openl_reward = jnp.concatenate([predicted_rew[:, :5], imagined_rew], 1)
    report['rewards_imagined'] = openl_reward
    report['rewards_predicted'] = predicted_rew

    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:MAX_VIDS].astype(jnp.float32)
      model_recon = recon_full[key].mode()
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = jnp.concatenate([truth, model_recon, model, error], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)
    return report

  def _metrics(self, data, dists, post, prior, losses, model_loss):
    entropy = lambda feat: self.rssm.get_dist(feat).entropy()
    metrics = {}
    metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()
    metrics['reward_max_data'] = jnp.abs(data['reward']).max()
    metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
    if 'reward' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
      metrics.update({f'reward_{k}': v for k, v in stats.items()})
    if 'cont' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
      metrics.update({f'cont_{k}': v for k, v in stats.items()})
    return metrics


class ImagActorCritic(nj.Module):

  def __init__(self, critics, scales, act_space, config):
    critics = {k: v for k, v in critics.items() if scales[k]}
    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}
    self.scales = scales
    self.act_space = act_space
    self.config = config
    disc = act_space.discrete
    self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
    self.actor = nets.MLP(
        name='actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}
    self.advnorm = jaxutils.Moments(**config.advnorm, name=f'advnorm')
    self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)

    if self.config.actent_norm:
        # shape = act_space.shape[:-1] if disc else act_space.shape
        self.actent = AutoAdapt((), **self.config.actent_norm_cfg, inverse=True, name='actent_norm')

  def initial(self, batch_size):
    return {}

  def policy(self, state, carry, sample=True):
    dist = self.actor(sg(state))
    action = dist.sample(seed=nj.rng()) if sample else dist.mode()
    return {'action': action}, carry

  def train(self, imagine, start, context):
    carry = self.initial(len(start['deter']))
    def loss(start):
      traj = imagine(self.policy, start, self.config.imag_horizon, carry)
      loss, metrics = self.loss(traj)
      return loss, (traj, metrics)
    mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
    metrics.update(mets)
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    return traj, metrics

  def loss(self, traj):
    metrics = {}
    advs = []
    # total = sum(self.scales[k] for k in self.critics)
    for key, critic in self.critics.items():
      rew, ret, base = critic.score(traj, self.actor)
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      # advs.append((normed_ret - normed_base) * self.scales[key] / total)
      advs.append((normed_ret - normed_base) * self.scales[key])
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()

    adv = jnp.stack(advs).sum(0)
    offset, invscale = self.advnorm(adv)
    normed_adv = (adv - offset) / invscale
    metrics.update(jaxutils.tensorstats(normed_adv, f'adv_normed'))

    policy = self.actor(sg(traj))
    logpi = policy.log_prob(sg(traj['action']))[:-1]
    loss = {'backprop': -normed_adv, 'reinforce': -logpi * sg(normed_adv)}[self.grad]

    ent = policy.entropy()[:-1]
    if self.config.actent_norm:
      lo = policy.minent
      hi = policy.maxent
      ent = (ent - lo) / (hi - lo)
      ent_loss, mets = self.actent(ent)
      loss += ent_loss  # sign inversed in `self.actent`
      for k, v in mets.items():
        metrics.update(jaxutils.tensorstats(v, f'actent_norm_{k}'))
    else:
      loss -= self.config.actent * ent

    loss *= sg(traj['weight'])[:-1]
    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, ent, adv))
    return loss.mean(), metrics

  def _metrics(self, traj, policy, logpi, ent, adv):
    metrics = {}
    ent = policy.entropy()[:-1]
    rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    rand = rand.mean(range(2, len(rand.shape)))
    act = traj['action']
    act = jnp.argmax(act, -1) if self.act_space.discrete else act
    metrics.update(jaxutils.tensorstats(act, 'action'))
    metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
    metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics


class VFunction(nj.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)

  def train(self, traj, actor):
    target = sg(self.score(traj, slow=self.config.slow_critic_target)[1])
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None, slow=False):
    rew = self.rewfn(traj)
    assert len(rew) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    if slow:
      value = self.slow(traj).mean()
    else:
      value = self.net(traj).mean()
    vals = [value[-1]]
    interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return rew, ret, value[:-1]


class AutoAdapt(nj.Module):

  def __init__(
      self, shape, impl, scale, target, min, max,
      vel=0.1, thres=0.1, inverse=False):
    self._shape = shape
    self._impl = impl
    self._target = target
    self._min = min
    self._max = max
    self._vel = vel
    self._inverse = inverse
    self._thres = thres
    self._scale = nj.Variable(jnp.ones, shape, f32, name='scale')

  def __call__(self, reg, update=True):
    update and self.update(reg)
    scale = self.scale()
    loss = scale * (-reg if self._inverse else reg)
    metrics = {
        'mean': reg.mean(), 'std': reg.std(),
        'scale_mean': scale.mean(), 'scale_std': scale.std()}
    return loss, metrics

  def scale(self):
    return sg(self._scale.read())

  def update(self, reg):
    avg = reg.mean(list(range(len(reg.shape) - len(self._shape))))
    scale = self._scale.read()
    below = avg < (1 / (1 + self._thres)) * self._target
    above = avg > (1 + self._thres) * self._target
    if self._inverse:
      below, above = above, below
    inside = ~below & ~above
    adjusted = (
        above.astype(f32) * scale * (1 + self._vel) +
        below.astype(f32) * scale / (1 + self._vel) +
        inside.astype(f32) * scale)
    self._scale.write(jnp.clip(adjusted, self._min, self._max))

