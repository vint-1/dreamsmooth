from functools import partial as bind

import embodied
import jax
import jax.numpy as jnp
import numpy as np
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
f32 = jnp.float32

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from . import agent
from . import jaxutils
from . import nets
from . import ninjax as nj


class DreamHigh(nj.Module):

  def __init__(self, wm, act_space, config):
    self.wm = wm
    self.config = config
    self.extr_reward = lambda s: wm.heads['reward'](s).mean()[1:]
    VF = agent.VFunction

    wconfig = config.update({
        'actor.inputs': self.config.worker_inputs,
        'critic.inputs': self.config.worker_inputs,
        'actent': self.config.worker_actent,
    })
    self.worker = agent.ImagActorCritic({
        'extr': VF(lambda s: s['reward_extr'], wconfig, name='extr_v'),
        'expl': VF(lambda s: s['reward_expl'], wconfig, name='expl_v'),
        'goal': VF(lambda s: s['reward_goal'], wconfig, name='goal_v'),
    }, config.worker_rews, act_space, wconfig, name='worker')

    skill_space = embodied.Space(np.int32, config.skill_shape)
    mconfig = config.update({
        'actor.inputs': self.config.manager_inputs,  # DreamHigh: manager only uses `deter`.
        'critic.inputs': self.config.manager_inputs,  # DreamHigh: manager only uses `deter`.
        'actent': self.config.manager_actent,
    })
    self.manager = agent.ImagActorCritic({
        'extr': VF(lambda s: s['reward_extr'], mconfig, name='extr_v'),
        'expl': VF(lambda s: s['reward_expl'], mconfig, name='expl_v'),
        'goal': VF(lambda s: s['reward_goal'], mconfig, name='goal_v'),
    }, config.manager_rews, skill_space, mconfig, name='manager')

    # DreamHigh: define high-level models.
    self.wm_high = nets.MLP(config.rssm.deter, name='wm_high', **config.wm_high)
    self.wm_high_opt = jaxutils.Optimizer(name='wm_high_opt', **config.wm_high_opt)
    self.rew_high = {
        'extr': nets.MLP((), **config.rew_high_head, name='extr_rew_high'),
        'expl': nets.MLP((), **config.rew_high_head, name='expl_rew_high'),
        'goal': nets.MLP((), **config.rew_high_head, name='goal_rew_high')}
    self.rew_high_opt = jaxutils.Optimizer(name='rew_high_opt', **config.rew_high_opt)
    self.cont_high = nets.MLP((), **config.cont_high_head, name='cont_high')
    self.cont_high_opt = jaxutils.Optimizer(name='cont_high_opt', **config.cont_high_opt)

    self.goal_shape = (config.rssm.deter,)
    self.goal_feat = nets.Input(['deter'])
    self.enc = nets.MLP(
        config.skill_shape, **config.goal_enc, dims='deter', name='enc')
    self.dec = nets.MLP(
        self.goal_shape, **config.goal_dec, dims='deter', name='dec')
    self.opt = jaxutils.Optimizer(name='goal_opt', **config.goal_opt)
    logits = jax.device_put(np.zeros(config.skill_shape))
    self.prior = tfd.Independent(
        jaxutils.OneHotDist(logits),
        len(config.skill_shape) - 1)

  def initial(self, batch_size):
    return {
        'step': jnp.zeros((batch_size,), jnp.int64),
        'skill': jnp.zeros((batch_size,) + self.config.skill_shape, f32),
        'goal': jnp.zeros((batch_size,) + self.goal_shape, f32),
    }

  def policy(self, latent, carry, imag=False):
    duration = self.config.train_skill_duration if imag else (
        self.config.env_skill_duration)
    skill = sg(jaxutils.switch(
        carry['step'] % duration == 0,
        self.manager.actor(latent).sample(seed=nj.rng()),
        carry['skill']))
    goal = sg(jaxutils.switch(
        carry['step'] % duration == 0,
        self.dec({**latent, 'skill': skill}).mode(),
        carry['goal']))
    dist = self.worker.actor(sg({**latent, 'goal': goal}))
    outs = {'action': dist.sample(seed=nj.rng())}
    carry = {'step': carry['step'] + 1, 'skill': skill, 'goal': goal}
    return outs, carry

  def train(self, imagine, start, data):
    metrics = {}
    metrics.update(self.train_vae(data))
    if self.config.director_jointly:
      metrics.update(self.train_jointly(imagine, start))
    else:
      raise NotImplementedError
    return None, metrics

  def train_jointly(self, imagine, start):
    start = start.copy()
    metrics = {}

    def wloss(start):
      traj = imagine(
          bind(self.policy, imag=True), start, self.config.imag_horizon,
          carry=self.initial(len(start['is_first'])))
      traj['reward_extr'] = self.extr_reward(traj)
      traj['reward_expl'] = self.expl_reward(traj)
      traj['reward_goal'] = self.goal_reward(traj)
      # traj, wtraj
      # start = {deter_0, stoch_0}
      # deter:    0, 1, ..., T
      # stoch:    0, 1, ..., T
      # action:   0, 1, ..., [T]   T'th action is not used
      # skill:    0, 1, ..., [T]
      # reward:      1, ..., T
      # cont:     0, 1, ..., T
      # weight:   0, 1, ..., T
      wtraj = self.split_traj(traj)
      mtraj = self.abstract_traj(traj)
      # mtraj
      # deter:    0, K, ..., T
      # stoch:    0, K, ..., T
      # action:   0, K, ..., [T]   this is skill, T'th skill is not used
      # reward:      K, ..., T
      # cont:     0, K, ..., T
      # weight:   0, K, ..., T
      wloss, wmets = self.worker.loss(wtraj)
      return wloss, (wtraj, mtraj, wmets)

    mets, (wtraj, mtraj, wmets) = self.worker.opt(
        self.worker.actor, wloss, start, has_aux=True)
    wmets.update(mets)
    for key, critic in self.worker.critics.items():
      mets = critic.train(wtraj, self.worker.actor)
      wmets.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    metrics.update({f'worker_{k}': v for k, v in wmets.items()})

    # DreamHigh: train high-level model, reward, cont functions.
    def wm_high_loss(traj):
      deter = traj['deter'][:-1]
      action = traj['action'][:-1]
      pred = self.wm_high(sg(dict(deter=deter, action=action)))
      wm_high_loss = -pred.log_prob(sg(traj["deter"][1:]))
      return wm_high_loss.mean()

    mets = self.wm_high_opt(
        self.wm_high, wm_high_loss, mtraj, has_aux=False)
    metrics.update({f'wm_high_{k}': v for k, v in mets.items()})

    def rew_high_loss(traj):
      deter = traj['deter'][1:]
      rew_high_losses = []
      for k, rew_fn in self.rew_high.items():
        pred = rew_fn(sg({'deter': deter}))
        rew_high_losses.append(-pred.log_prob(sg(traj[f"reward_{k}"])))
      rew_high_loss = jnp.stack(rew_high_losses).sum(0)
      return rew_high_loss.mean()

    mets = self.rew_high_opt(
        list(self.rew_high.values()), rew_high_loss, mtraj, has_aux=False)
    metrics.update({f'rew_high_{k}': v for k, v in mets.items()})

    def cont_high_loss(traj):
      deter = traj['deter'][1:]
      pred = self.cont_high(sg({'deter': deter}))
      cont_high_loss = -pred.log_prob(sg(traj["cont"][1:]))
      return cont_high_loss.mean()

    mets = self.cont_high_opt(
        self.cont_high, cont_high_loss, mtraj, has_aux=False)
    metrics.update({f'cont_high_{k}': v for k, v in mets.items()})

    # DreamHigh: generate high-level rollouts to train manager.
    K = self.config.imag_horizon_high
    policy_high = lambda x: self.manager.policy(x, {})[0]
    def step(prev, _):
      state, action = prev
      state = self.wm_high({**state, 'action': action['action']}).mode()
      state = {'deter': state.astype(jnp.float16)}
      action = policy_high(state)
      return state, action
    state = {'deter': start['deter']}
    action = policy_high(state)
    states, actions = jaxutils.scan(
        step, jnp.arange(K), (state, action), self.config.imag_unroll)
    states, actions = tree_map(
        lambda traj, first: jnp.concatenate([first[None], traj], 0),
        (states, actions), (state, action))
    htraj = {**states, **actions}
    H = self.config.train_skill_duration
    cont = self.cont_high(htraj).mode()
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
    htraj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
    discount = 1 - 1 / self.config.horizon
    htraj['weight'] = jnp.cumprod(discount * htraj['cont'], 0) / discount
    for k, rew_fn in self.rew_high.items():
      htraj[f'reward_{k}'] = rew_fn(htraj).mode()[1:]
    # Does it make sense to predict expl reward?
    # htraj[f'reward_expl'] = self.expl_reward(htraj) * H

    # htraj
    # start = {deter_0}
    # deter:    0, 1, ..., T
    # action:   0, 1, ..., [T]   T'th action is not used
    # reward:      1, ..., T
    # cont:     0, 1, ..., T
    # weight:   0, 1, ..., T

    # DreamHigh: train manager on high-level rollouts.
    mets, mmets = self.manager.opt(
        self.manager.actor, self.manager.loss, htraj, has_aux=True)
    mmets.update(mets)
    for key, critic in self.manager.critics.items():
      mets = critic.train(htraj, self.manager.actor)
      mmets.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    metrics.update({f'manager_{k}': v for k, v in mmets.items()})

    return metrics

  def train_vae(self, data):
    def loss(data):
      metrics = {}
      goal = self.goal_feat(data).astype(f32)
      enc = self.enc({**data, 'goal': goal})
      dec = self.dec({**data, 'skill': enc.sample(seed=nj.rng())})
      rec = -dec.log_prob(sg(goal))
      kl = tfd.kl_divergence(enc, self.prior)
      kl = jnp.maximum(self.config.goal_kl_free, kl)
      assert rec.shape == kl.shape, (rec.shape, kl.shape)
      metrics['goalkl_mean'] = kl.mean()
      metrics['goalkl_std'] = kl.std()
      metrics['goalrec_mean'] = rec.mean()
      metrics['goalrec_std'] = rec.std()
      loss = (rec + self.config.goal_kl_scale * kl).mean()
      return loss, metrics
    metrics, mets = self.opt([self.enc, self.dec], loss, data, has_aux=True)
    metrics.update(mets)
    return metrics

  def propose_goal(self, start, impl):
    if impl == 'replay':
      feat = self.goal_feat(start).astype(f32)
      target = jax.random.permutation(nj.rng(), feat).astype(f32)
      skill = self.enc({**start, 'goal': target}).sample(seed=nj.rng())
      return self.dec({**start, 'skill': skill}).mode()
    if impl == 'replay_direct':
      feat = self.goal_feat(start).astype(f32)
      return jax.random.permutation(nj.rng(), feat).astype(f32)
    if impl == 'manager':
      skill = self.manager.actor(start).sample(seed=nj.rng())
      return self.dec({**start, 'skill': skill}).mode()
    if impl == 'prior':
      skill = self.prior.sample(len(start['is_terminal']), seed=nj.rng())
      return self.dec({**start, 'skill': skill}).mode()
    raise NotImplementedError(impl)

  def goal_reward(self, traj):
    feat = self.goal_feat(traj).astype(f32)
    goal = sg(traj['goal'].astype(f32))
    gnorm = jnp.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
    fnorm = jnp.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
    norm = jnp.maximum(gnorm, fnorm)
    return jnp.einsum('...i,...i->...', goal / norm, feat / norm)[1:]

  def expl_reward(self, traj):
    feat = self.goal_feat(traj).astype(f32)
    enc = self.enc({**traj, 'goal': feat})
    dec = self.dec({**traj, 'skill': enc.sample(seed=nj.rng())})
    return ((dec.mode() - feat) ** 2).mean(-1)[1:]

  def split_traj(self, traj):
    traj = traj.copy()
    k = self.config.train_skill_duration
    for key, x in list(traj.items()):
      if key.startswith('reward_'):
        x = jnp.concatenate([0 * x[:1], x], 0)
      x = x[:-1]  # DreamHigh: Remove the last prediction to fit (B' x T').
      # T x B x ... -> B' x T' x B x ...
      x = x.reshape((x.shape[0] // k, k) + x.shape[1:])
      # B' x T' x B x ... -> T' x (B' B) x ...
      x = x.transpose((1, 0) + tuple(range(2, len(x.shape))))
      x = x.reshape((x.shape[0], -1, *x.shape[3:]))
      if key.startswith('reward_'):
        x = x[1:]
      traj[key] = x
    return traj

  def abstract_traj(self, traj):
    traj = traj.copy()
    traj['action'] = traj.pop('skill')
    # DreamHigh: `traj` has one more transition than Director.
    k = self.config.train_skill_duration
    assert traj['cont'].shape[0] % k == 1, (traj['cont'].shape[0], k)
    # reshape: T x B x ... -> T' x K x B x ...
    reshape = lambda x: x.reshape((x.shape[0] // k, k, *x.shape[1:]))
    w = jnp.cumprod(traj['cont'][:-1], 0)
    # T x B x ... -> T' x K x B x ... -> (T' + 1) x B x ...
    for key, x in list(traj.items()):
      if key.startswith('reward_'):
        x = reshape(x * w).mean(1)
      elif key == 'cont':
        x = jnp.concatenate([x[:1], reshape(x[1:]).prod(1)])
      else:
        x = jnp.concatenate([reshape(x[:-1])[:, 0], x[-1:]])
      traj[key] = x
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont']) / discount
    return traj

  def report(self, data):
    metrics = {}
    for impl in ('manager', 'prior', 'replay'):
      for key, video in self.report_worker(data, impl).items():
        metrics[f'impl_{impl}_{key}'] = video

    for key, video in self.report_manager(data).items():
      metrics[f'impl_high_{key}'] = video
    return metrics

  # DreamHigh: visualize high-level rollouts.
  def report_manager(self, data):
    # Prepare initial state.
    decoder = self.wm.heads['decoder']
    states = self.wm.rssm.observe(
        self.wm.encoder(data)[:6], data['action'][:6], data['is_first'][:6])
    start = {k: v[:, 4] for k, v in states.items()}
    start['is_terminal'] = data['is_terminal'][:6, 4]
    start['is_first'] = data['is_first'][:6, 4]
    # Decode into images.
    initial = decoder(start)

    traj = self.wm.imagine(
        bind(self.policy, imag=True), start, self.config.worker_report_horizon,
        carry=self.initial(len(start['is_first'])))

    deter = traj['deter'][0]
    action = traj['skill'][:-1:self.config.train_skill_duration]

    def step(prev, action):
      state = self.wm_high({**prev, 'action': action}).mode()
      state = {'deter': state.astype(jnp.float16)}
      return state
    state = {'deter': start['deter']}
    states = jaxutils.scan(step, action, state, self.config.imag_unroll)
    deter_high = states['deter']
    stoch_high = self.wm.rssm._prior(deter_high, sample=True)['stoch']
    # goal = self.dec(dict(deter=deter_high, stoch=stoch_high, skill=action)).mode()
    goal = traj['goal'][:-1:self.config.train_skill_duration]
    stoch_goal = self.wm.rssm._prior(goal, sample=True)['stoch']

    # Decode into images.
    rollout_high = decoder({'deter': deter_high, 'stoch': stoch_high})
    rollout_goal = decoder({'deter': goal, 'stoch': stoch_goal})
    rollout = decoder(traj)

    # Stich together into videos.
    videos = {}
    for k in rollout.keys():
      if k not in decoder.cnn_shapes:
        continue
      length = 1 + self.config.worker_report_horizon
      rows = []
      # (1) Initial states.
      rows.append(jnp.repeat(initial[k].mode()[:, None], length, 1))
      # (2) Low-level imaginary rollouts.
      rows.append(rollout[k].mode().transpose((1, 0, 2, 3, 4)))
      # (3) Goals used for (2).
      rows.append(jnp.concatenate([
          initial[k].mode()[:, None],
          jnp.repeat(rollout_goal[k].mode().transpose((1, 0, 2, 3, 4)), self.config.train_skill_duration, 1)], 1))
      # (4) High-level model predictions using skills in (3).
      rows.append(jnp.concatenate([
          initial[k].mode()[:, None],
          jnp.repeat(rollout_high[k].mode().transpose((1, 0, 2, 3, 4)), self.config.train_skill_duration, 1)], 1))
      videos[k] = jaxutils.video_grid(jnp.concatenate(rows, 2))
    return videos

  def report_worker(self, data, impl):
    # Prepare initial state.
    decoder = self.wm.heads['decoder']
    states = self.wm.rssm.observe(
        self.wm.encoder(data)[:6], data['action'][:6], data['is_first'][:6])
    start = {k: v[:, 4] for k, v in states.items()}
    start['is_terminal'] = data['is_terminal'][:6, 4]
    goal = self.propose_goal(start, impl)
    # Worker rollout.
    traj = self.wm.imagine(
        lambda s, c: self.worker.policy({**s, 'goal': goal}, c),
        start, self.config.worker_report_horizon, {})
    # Decode into images.
    initial = decoder(start)
    stoch = self.wm.rssm._prior(goal, sample=True)['stoch']
    target = decoder({'deter': goal, 'stoch': stoch})
    rollout = decoder(traj)
    # Stich together into videos.
    videos = {}
    for k in rollout.keys():
      if k not in decoder.cnn_shapes:
        continue
      length = 1 + self.config.worker_report_horizon
      rows = []
      rows.append(jnp.repeat(initial[k].mode()[:, None], length, 1))
      if target is not None:
        rows.append(jnp.repeat(target[k].mode()[:, None], length, 1))
      rows.append(rollout[k].mode().transpose((1, 0, 2, 3, 4)))
      videos[k] = jaxutils.video_grid(jnp.concatenate(rows, 2))
    return videos
