import re

import embodied
import numpy as np
import matplotlib.pyplot as plt

def train_supervised(
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
    metrics.add(stats, prefix=f'{mode}_stats')

  driver_train = embodied.Driver(train_env)
  driver_train.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
  driver_train.on_step(lambda tran, _: step.increment())
  # driver_train.on_step(train_replay.add)
  driver_eval = embodied.Driver(eval_env)
  # driver_eval.on_step(eval_replay.add)
  driver_eval.on_episode(lambda ep, worker: per_episode(generate_eval_ep(dataset_eval), mode='eval')) # ignore ep and use something from driver_train instead?
  # Basically we can do the same thing as for mvmwm

  # dont do prefill

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

def generate_eval_ep(eval_dataset):
  max_batch_size = 6
  data = next(eval_dataset)
  ep = {}
  for key, value in data.items():
    if key != "rng":
      ep[key] = value[:max_batch_size].reshape([-1] + list(value.shape[2:]))
  return ep

def eval_report(ep, agent, args):
  # preprocess data
  data = {"rng": agent.rng.integers(2 ** 63 - 1)}
  ep_len = len(ep['reward'])
  num_sequences = round(ep_len / 100)
  for key, value in ep.items():
    data[key] = value.reshape([num_sequences, -1] + list(value.shape[1:]))
    # data[key] = value[1:].reshape([num_sequences, -1] + list(value.shape[1:])) 
  report = agent.report(data)

  filtered_report = {}
  # postprocess report
  for key, value in report.items():
    if re.match(args.log_keys_eval_report, key):
      filtered_report[key] = value
  # add plots for imagined and predicted rewards
  rew = ep['reward']
  rew_predicted = report['rewards_predicted'].reshape((ep_len, -1))
  rew_imagined = report['rewards_imagined'].reshape((ep_len, -1))
  filtered_report['openl_rew_plot'] = create_rew_plot(rew, rew_predicted, rew_imagined)
  return filtered_report

# def eval_report(ep, agent, args):
#   # preprocess data
#   data = {"rng": agent.rng.integers(2 ** 63 - 1)}
#   for key, value in ep.items():
#     data[key] = value
#   # num_sequences = round(ep_len / 100)
#   # for key, value in ep.items():
#   #   data[key] = value[1:].reshape([num_sequences, -1] + list(value.shape[1:])) 
#   report = agent.report(data)

#   filtered_report = {}
#   # postprocess report
#   for key, value in report.items():
#     if re.match(args.log_keys_eval_report, key):
#       filtered_report[key] = value
#   # add plots for imagined and predicted rewards
#   rew = ep['reward']
#   ep_len = report['rewards_predicted'].shape[0] * report['rewards_predicted'].shape[1]
#   rew_predicted = report['rewards_predicted'].reshape((ep_len, -1))
#   rew_imagined = report['rewards_imagined'].reshape((ep_len, -1))
#   filtered_report['openl_rew_plot'] = create_rew_plot(rew, rew_predicted, rew_imagined)
#   return filtered_report

def create_rew_plot(rew, rew_pred, rew_imagined):
  total_rew = np.sum(rew)
  total_pred = np.sum(rew_pred)
  fig1, ax1 = plt.subplots(nrows=1, ncols=1)
  ax1.plot(rew, label="Observed")
  ax1.plot(rew_pred, label="Predicted")
  ax1.plot(rew_imagined, label="Imagined")
  ax1.set_title(
      f"Total Reward = {total_rew:.3f}, Predicted = {total_pred:.3f}"
  )
  ax1.set_xlabel("Timestep")
  ax1.set_ylabel("Reward")
  ax1.legend()
  return fig1