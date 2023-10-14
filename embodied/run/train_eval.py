import re

import embodied
import numpy as np
import matplotlib.pyplot as plt

def train_eval(
    agent, train_env, eval_env, train_replay, eval_replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
  step = logger.step
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(train_env.obs_space), sep='\n')
  print('Action space:', embodied.format(train_env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', train_env, ['step'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])

  nonzeros = set()
  def per_episode(ep, mode):
    if args.rew_smoothing_mode == "gaussian":
      embodied.gaussian_rewards(ep, float(args.rew_smoothing_amt))
    elif args.rew_smoothing_mode == "exponential":
      embodied.exponential_rewards(ep, float(args.rew_smoothing_amt))
    elif args.rew_smoothing_mode.startswith("uniform"):
      embodied.uniform_rewards(ep, int(args.rew_smoothing_amt), kernel_type=args.rew_smoothing_mode)
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    logger.add({
        'length': length, 'score': score,
        'reward_rate': (ep['reward'] - ep['reward'].min() >= 0.1).mean(),
    }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
    print(f'Episode has {length} steps and return {score:.1f}.')
    if mode == "eval":
      report = eval_report(ep, agent, args)
      logger.add(report, prefix="eval_episode")
    stats = {}
    for key in args.log_keys_video: # This is where train_stats and eval_stats appear
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
      if re.match(args.log_keys_last, key):
        stats[f'last_{key}'] = ep[key][-1].mean()
      if re.match(args.log_keys_success, key):
        stats[f'success_{key}'] = np.minimum(ep[key].max(0), 1).mean()
    if mode == 'train' or args.eval_eps > 1:
      metrics.add(stats, prefix=f'{mode}_stats')
    else:
      logger.add(stats, prefix="eval_episode")

  driver_train = embodied.Driver(train_env)
  driver_train.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
  driver_train.on_step(lambda tran, _: step.increment())
  driver_train.on_step(train_replay.add)
  driver_eval = embodied.Driver(eval_env)
  driver_eval.on_step(eval_replay.add)
  driver_eval.on_episode(lambda ep, worker: per_episode(ep, mode='eval'))

  random_agent = embodied.RandomAgent(train_env.act_space)
  print('Prefill train dataset.')
  while len(train_replay) < max(args.batch_steps, args.train_fill):
    driver_train(random_agent.policy, steps=100)
  print('Prefill eval dataset.')
  while len(eval_replay) < max(args.batch_steps, args.eval_fill):
    driver_eval(random_agent.policy, steps=100)

  dataset_train = agent.dataset(train_replay.dataset)
  dataset_eval = agent.dataset(eval_replay.dataset)
  state = [None]  # To be writable from train step function below.
  batch = [None]
  def train_step(tran, worker):
    for _ in range(should_train(step)):
      with timer.scope('dataset_train'):
        batch[0] = next(dataset_train)
      outs, state[0], mets = agent.train(batch[0], state[0])
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        train_replay.prioritize(outs['key'], outs['priority'])
    if should_log(step):
      logger.add(metrics.result())
      logger.add(agent.report(batch[0]), prefix='report')
      with timer.scope('dataset_eval'):
        eval_batch = next(dataset_eval)
      logger.add(agent.report(eval_batch), prefix='eval')
      logger.add(train_replay.stats, prefix='replay')
      logger.add(eval_replay.stats, prefix='eval_replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver_train.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we just saved.

  print('Start training loop.')
  policy_train = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  policy_eval = lambda *args: agent.policy(*args, mode='eval')
  while step < args.steps:
    if should_eval(step):
      print('Starting evaluation at step', int(step))
      driver_eval.reset()
      driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps))
    driver_train(policy_train, steps=100)
    if should_save(step):
      checkpoint.save()
  logger.write()
  logger.write()

def eval_report(ep, agent, args):
  # preprocess data
  data = {"rng": agent.rng.integers(2 ** 63 - 1)} # lol this is not legit
  ep_len = len(ep['reward']) - 1
  sequence_length = min(100, ep_len)
  num_sequences = ep_len // sequence_length
  max_index = (sequence_length * num_sequences) + 1
  for key, value in ep.items():
    data[key] = value[1:max_index].reshape([num_sequences, -1] + list(value.shape[1:])) 
  report = agent.report(data)

  filtered_report = {}
  # postprocess report
  for key, value in report.items():
    if re.match(args.log_keys_eval_report, key):
      filtered_report[key] = value
  # add plots for imagined and predicted rewards
  rew = ep['reward'][1:]
  rew_unsmoothed = ep['reward_raw'][1:] if 'reward_raw' in ep else None
  rew_predicted = report['rewards_predicted'].reshape((-1, 1))
  rew_imagined = report['rewards_imagined'].reshape((-1, 1))
  filtered_report['openl_rew_plot'] = create_rew_plot(rew, rew_predicted, rew_imagined, rew_unsmoothed)
  return filtered_report

def create_rew_plot(rew, rew_pred, rew_imagined, rew_unsmoothed = None):
  total_rew = np.sum(rew)
  total_pred = np.sum(rew_pred)
  fig1, ax1 = plt.subplots(nrows=1, ncols=1)
  ax1.plot(rew, label="Observed")
  ax1.plot(rew_pred, label="Predicted")
  ax1.plot(rew_imagined, label="Imagined")
  if rew_unsmoothed is not None:
    ax1.plot(rew_unsmoothed, label="Observed (unsmoothed)")
  ax1.set_title(
      f"Total Reward = {total_rew:.3f}, Predicted = {total_pred:.3f}"
  )
  ax1.set_xlabel("Timestep")
  ax1.set_ylabel("Reward")
  ax1.legend()
  return fig1