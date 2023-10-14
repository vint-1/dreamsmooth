import importlib
import os
import pathlib
import sys
import warnings
from functools import partial as bind

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied
from embodied import wrappers


def main(argv=None):
  from . import agent as agt

  parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
  config = embodied.Config(agt.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(agt.Agent.configs[name])
  config = embodied.Flags(config).parse(other)
  logdir = pathlib.Path(config.logdir).expanduser() / f"{config.task}_{config.method}_{config.seed}"
  logdir.mkdir(parents=True, exist_ok=True)
  config = config.update(logdir = logdir)
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  # print(config)

  logdir = embodied.Path(args.logdir)
  if args.script != 'parallel_env':
    logdir.mkdirs()
    config.save(logdir / 'config.yaml')
    step = embodied.Counter()
    logger = make_logger(parsed, logdir, step, config)

  cleanup = []
  try:

    if args.script == 'train':
      replay = make_replay(config, logdir / 'replay')
      env = wrapped_env(config, batch=True)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train(agent, env, replay, logger, args)

    elif args.script == 'train_save':
      replay = make_replay(config, logdir / 'replay')
      env = wrapped_env(config, batch=True)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_save(agent, env, replay, logger, args)

    elif args.script == 'train_eval':
      replay = make_replay(config, logdir / 'replay')
      eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      env = wrapped_env(config, batch=True)
      eval_env = wrapped_env(config, batch=True)
      cleanup += [env, eval_env]
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_eval(
          agent, env, eval_env, replay, eval_replay, logger, args)
    
    elif args.script == 'train_supervised':
      replay = make_replay(config, logdir / 'replay')
      eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      env = wrapped_env(config, batch=True)
      eval_env = wrapped_env(config, batch=True)
      cleanup += [env, eval_env]
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_supervised(
          agent, env, eval_env, replay, eval_replay, logger, args)

    elif args.script == 'train_holdout': # don't collect new eval episodes
      replay = make_replay(config, logdir / 'replay')
      if config.eval_dir:
        assert not config.train.eval_fill
        eval_replay = make_replay(config, config.eval_dir, is_eval=True)
      else:
        assert 0 < args.eval_fill <= config.replay_size // 10, args.eval_fill
        eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      env = wrapped_env(config, batch=True)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_holdout(
          agent, env, replay, eval_replay, logger, args)

    elif args.script == 'eval_only':
      env = wrapped_env(config, batch=True)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      suite, _ = config.task.split('_', 1)
      if suite != "crafter":
        embodied.run.eval_only(agent, env, logger, args)
      else:
        embodied.run.eval_only_crafter(agent, env, logger, args)

    elif args.script == 'parallel':
      assert config.run.actor_batch <= config.envs.amount, (
          config.run.actor_batch, config.envs.amount)
      ctor = bind(wrapped_env, config, batch=False)
      step = embodied.Counter()
      env = ctor()
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      env.close()
      replay = make_replay(config, logdir / 'replay', rate_limit=True)
      embodied.run.parallel(
          agent, replay, logger, ctor, config.envs.amount, args)

    elif args.script == 'parallel_agent':
      ctor = bind(wrapped_env, config, batch=False)
      step = embodied.Counter()
      env = ctor()
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      env.close()
      replay = make_replay(config, logdir / 'replay', rate_limit=True)
      embodied.run.parallel(agent, replay, logger, None, num_envs=0, args=args)

    elif args.script == 'parallel_env':
      ctor = bind(wrapped_env, config, batch=False)
      replica_id = args.env_replica
      if replica_id < 0:
        replica_id = int(os.environ['JOB_COMPLETION_INDEX'])
      embodied.run.parallel_env(replica_id, ctor, args)

    else:
      raise NotImplementedError(args.script)
  finally:
    for obj in cleanup:
      obj.close()


def make_logger(parsed, logdir, step, config):
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  outputs = [
      embodied.logger.TerminalOutput(config.filter),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
      # embodied.logger.TensorBoardOutput(logdir),
  ]
  if config.run.get("wandb", False):
    wandb_run_name = f"{config.task}.{config.method}.{config.seed}"
    outputs.append(embodied.logger.WandBOutput(wandb_run_name, config))
  logger = embodied.Logger(step, outputs, multiplier)
  return logger


def make_replay(
    config, directory=None, is_eval=False, rate_limit=False, **kwargs):
  assert (config.replay == 'uniform' or config.replay == 'prioritize_sparse') or not rate_limit
  length = config.eval_batch_length if is_eval else config.batch_length
  batch_size = config.eval_batch_size if is_eval else config.batch_size
  size = config.replay_size // 10 if is_eval else config.replay_size
  if config.replay == 'uniform' or config.replay == 'prioritize_sparse' or is_eval:
    kw = {'online': config.replay_online}
    if rate_limit and config.run.train_ratio > 0:
      kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
      kw['tolerance'] = 10 * batch_size
      kw['min_size'] = batch_size
    if config.replay == 'prioritize_sparse' and not is_eval:
      kw['priority_p'] = config.replay_priority_p
      kw['thresh'] = config.replay_priority_thresh
      replay = embodied.replay.PrioritizeSparseReward(length, size, directory, **kw)
    else:
      replay = embodied.replay.Uniform(length, size, directory, **kw)
  elif config.replay == 'reverb':
    replay = embodied.replay.Reverb(length, size, directory)
  elif config.replay == 'chunks':
    replay = embodied.replay.NaiveChunks(length, size, directory)
  else:
    raise NotImplementedError(config.replay)
  
  # for reward smoothing
  if config.run.rew_smoothing_amt > 0:
    if config.run.rew_smoothing_mode == "gaussian":
      replay = embodied.GaussianSmoothing(replay, sigma = config.run.rew_smoothing_amt)
    elif config.run.rew_smoothing_mode == "exponential":
      replay = embodied.ExponentialSmoothing(replay, alpha = config.run.rew_smoothing_amt)
    elif config.run.rew_smoothing_mode.startswith("uniform"):
      replay = embodied.UniformSmoothing(replay, delta = config.run.rew_smoothing_amt, kernel_type=config.run.rew_smoothing_mode)

  return replay


def wrapped_env(config, batch, **overrides):
  ctor = bind(make_env, config, **overrides)
  if batch and config.envs.parallel != 'none':
    ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
  if config.envs.restart:
    ctor = bind(wrappers.RestartOnException, ctor)
  if batch:
    envs = [ctor() for _ in range(config.envs.amount)]
    return embodied.BatchEnv(envs, config.envs.parallel)
  else:
    return ctor()


def make_env(config, **overrides):
  from embodied.envs import from_gym
  suite, task = config.task.split('_', 1)
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'atari100k': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
      'kitchen': 'embodied.envs.kitchen:Kitchen',
      'agx': 'embodied.envs.agx:Agx',
      'robodesk': 'embodied.envs.robodesk:RoboDeskMulti',
      'hand': 'embodied.envs.hand:Hand',
      # TODO
      'procgen': lambda task, **kw: from_gym.FromGym(
          f'procgen:procgen-{task}-v0', **kw),
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  env = ctor(task, **kwargs)
  return wrap_env(env, config)


def wrap_env(env, config):
  args = config.wrapper
  if args.resize:
    env = wrappers.ResizeImage(env, size=(args.resize, args.resize), keep_original=True)
  for name, space in env.act_space.items():
    if name == 'reset':
      continue
    elif space.discrete:
      env = wrappers.OneHotAction(env, name)
    elif args.discretize:
      env = wrappers.DiscretizeAction(env, name, args.discretize)
    else:
      env = wrappers.NormalizeAction(env, name)
  env = wrappers.ExpandScalars(env)
  if args.length:
    env = wrappers.TimeLimit(env, args.length, args.reset)
  if args.checks:
    env = wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = wrappers.ClipAction(env, name)
  return env


if __name__ == '__main__':
  main()
