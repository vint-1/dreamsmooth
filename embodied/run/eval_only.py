import re

import embodied
import numpy as np
import matplotlib.pyplot as plt

def eval_only(agent, env, logger, args):
  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_log = embodied.when.Clock(args.log_every)
  step = logger.step
  metrics = embodied.Metrics()
  print('Observation space:', env.obs_space)
  print('Action space:', env.act_space)

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy'])
  timer.wrap('env', env, ['step'])
  timer.wrap('logger', logger, ['write'])

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

  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: per_episode(ep, "eval"))
  driver.on_step(lambda tran, _: step.increment())

  # checkpoint = embodied.Checkpoint()
  if args.from_checkpoint == "": # use default checkpoint from logdir
    checkpoint_path = logdir / 'checkpoint.ckpt'
  else:
    checkpoint_path = logdir.parent / args.from_checkpoint / 'checkpoint.ckpt'
  checkpoint = embodied.Checkpoint(checkpoint_path)
  checkpoint.agent = agent
  checkpoint.load("", keys=['agent'])

  print('Start evaluation loop.')
  policy = lambda *args: agent.policy(*args, mode='eval')
  while step < args.steps:
    driver(policy, steps=100)
    if should_log(step):
      logger.add(metrics.result())
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
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
  # add plots for trajectories
  trajectories = {}
  for key, value in ep.items():
    if re.match(args.log_keys_trajectory, key):
      trajectories[key] = np.minimum(value[1:], 1) - np.minimum(value[:-1], 1)
  filtered_report["trajectory_plot"] = create_trajectory_plot(trajectories)
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

def create_trajectory_plot(trajectories, title = "Trajectories", ylabel = "Trajectory Value"):
  # input: dict of trajectories
  fig1, ax1 = plt.subplots(nrows=1, ncols=1)
  for k, v in trajectories.items():
    ax1.plot(v, label=f"{k}")
  ax1.set_title("Trajectories")
  ax1.set_xlabel("Timestep")
  ax1.set_ylabel("Trajectory value")
  ax1.legend()
  return fig1