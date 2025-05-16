from collections.abc import Iterable
import warnings

import gymnasium as gym
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit
import numpy as np

from .multiagentenv import MultiAgentEnv
from .wrappers import FlattenObservation
import envs.pretrained as pretrained  # noqa

try:
    from bsk_rl import GeneralSatelliteTasking
except ImportError:
    warnings.warn(
        "BSK-RL is not installed, so these environments will not be available! Please double-check the installation of Basilisk and BSK-RL library first (`pip show basilisk` and `pip show bsk-rl`)"
    )
from bsk_rl import sats, act, obs, scene, data, comm
from bsk_rl.sim import dyn, fsw

from bsk_rl.utils.orbital import random_orbit
from bsk_rl.utils.orbital import walker_delta_args
from munch import Munch

def make_BSK_Cluster_env(env_args, satellite_names, scenario):
    # Common orbital parameters for all satellites
    inclination = 50.0          # degrees, fixed for all satellites
    altitude = 500              # km, fixed for all satellites
    eccentricity = 0            # Circular orbit
    # Longitude of Ascending Node (Omega), fixed for all
    LAN = 0
    arg_periapsis = 0           # Argument of Periapsis (omega), fixed for all

    # True anomaly offsets for spacing satellites along the Cluster orbit
    true_anomaly_offsets = [225 - 0.0001*i for i in range(
        len(satellite_names))]  # degrees
    orbit_ls = []
    for offset in true_anomaly_offsets:
        orbit = random_orbit(
            i=inclination, alt=altitude, e=eccentricity, Omega=LAN, omega=arg_periapsis, f=offset
        )
        orbit_ls.append(orbit)

    if scenario == "ideal":
        battery_sizes = [1e6]*len(satellite_names)
        memory_sizes = [1e6]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = env_args.baud_rate
        instr_baud_rate = 500

    elif scenario == "limited_batt":
        battery_sizes = [50]*len(satellite_names)
        memory_sizes = [int(env_args.memory_size)]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = env_args.baud_rate
        instr_baud_rate = env_args.instr_baud_rate

    elif scenario == "limited_mem":
        battery_sizes = [env_args.battery_capacity]*len(satellite_names)
        memory_sizes = [5000]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = env_args.baud_rate
        instr_baud_rate = env_args.instr_baud_rate

    elif scenario == "limited_baud":
        battery_sizes = [env_args.battery_capacity]*len(satellite_names)
        memory_sizes = [env_args.memory_size]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = 0.5
        instr_baud_rate = env_args.instr_baud_rate

    elif scenario == "limited_img":
        battery_sizes = [int(env_args.battery_capacity)
                         ]*len(satellite_names)
        memory_sizes = [int(env_args.memory_size)]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = env_args.baud_rate
        instr_baud_rate = 125

    elif scenario == "limited_all":
        battery_sizes = [50]*len(satellite_names)
        memory_sizes = [5000]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = 0.5
        instr_baud_rate = 125

    elif scenario == "default":
        battery_sizes = [int(env_args.battery_capacity)
                         ]*len(satellite_names)
        memory_sizes = [int(env_args.memory_size)]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = env_args.baud_rate
        instr_baud_rate = env_args.instr_baud_rate

    elif scenario == "random_all":
        battery_sizes = [env_args.battery_capacity]*len(satellite_names)
        memory_sizes = [env_args.memory_size]*len(satellite_names)
        random_init_memory = True
        random_init_battery = True
        random_disturbance = True
        random_RW_speed = True
        baud_rate = env_args.baud_rate
        instr_baud_rate = env_args.instr_baud_rate

    elif scenario == "random_batt":
        battery_sizes = [env_args.battery_capacity]*len(satellite_names)
        memory_sizes = [env_args.memory_size]*len(satellite_names)
        random_init_memory = False
        random_init_battery = True
        random_disturbance = False
        random_RW_speed = False
        baud_rate = env_args.baud_rate
        instr_baud_rate = env_args.instr_baud_rate

    elif scenario == "random_mem":
        battery_sizes = [env_args.battery_capacity]*len(satellite_names)
        memory_sizes = [env_args.memory_size]*len(satellite_names)
        random_init_memory = True
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = env_args.baud_rate
        instr_baud_rate = env_args.instr_baud_rate

    elif scenario == "random_dist":
        battery_sizes = [env_args.battery_capacity]*len(satellite_names)
        memory_sizes = [env_args.memory_size]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = True
        random_RW_speed = False
        baud_rate = env_args.baud_rate
        instr_baud_rate = env_args.instr_baud_rate

    elif scenario == "random_rw":
        battery_sizes = [env_args.battery_capacity]*len(satellite_names)
        memory_sizes = [env_args.memory_size]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = True
        baud_rate = env_args.baud_rate
        instr_baud_rate = env_args.instr_baud_rate

    elif scenario == "hetero_batt":
        battery_sizes = [50, 100, 200, 400]
        memory_sizes = [int(env_args.memory_size)]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = env_args.baud_rate
        instr_baud_rate = env_args.instr_baud_rate

    elif scenario == "hetero_mem":
        battery_sizes = [env_args.battery_capacity]*len(satellite_names)
        memory_sizes = [5000, 10000, 250000, 500000]
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = env_args.baud_rate
        instr_baud_rate = env_args.instr_baud_rate

    else:
        print("Scenario name not available")
        NotImplementedError

    # Define four satellites in a "train" Cluster formation along the same orbit
    multiSat = []
    index = 0

    for orbit, battery_size, memory_size in zip(orbit_ls, battery_sizes, memory_sizes):
        sat_args = dict(
            # Power
            batteryStorageCapacity=battery_size * 3600,
            storedCharge_Init=int(battery_size * env_args.init_battery_level / 100 * 3600) if not random_init_battery else np.random.uniform(
                battery_size * 3600 * 0.4, battery_size * 3600 * 0.5),
            panelArea=1.0,
            panelEfficiency=20.0,
            basePowerDraw=-10.0,
            instrumentPowerDraw=-30,
            transmitterPowerDraw=-25,
            thrusterPowerDraw=-80,
            # Data Storage
            dataStorageCapacity=memory_size * 8e6,  # MB to bits,
            storageInit=int(memory_size *
                            env_args.init_memory_percent/100) * 8e6 if not random_init_memory else np.random.uniform(memory_size * 8e6 * 0.2, memory_size * 8e6 * 0.8),
            instrumentBaudRate=instr_baud_rate * 1e6,
            transmitterBaudRate=-1*baud_rate * 1e6,
            # Attitude
            imageAttErrorRequirement=0.1,
            imageRateErrorRequirement=0.1,
            disturbance_vector=lambda: np.random.normal(
                scale=0.0001, size=3) if random_disturbance else np.array([0.0, 0.0, 0.0]),
            maxWheelSpeed=6000.0,  # RPM
            wheelSpeeds=lambda: np.random.uniform(
                -3000, 3000, 3) if random_RW_speed else np.array([0.0, 0.0, 0.0]),
            desatAttitude="nadir",
            u_max=0.4,
            K1=0.25,
            K3=3.0,
            omega_max=0.1,
            servo_Ki=5.0,
            servo_P=150,
            # Orbital elements
            oe=orbit
        )

        class ImagingSatellite(sats.ImagingSatellite):
            observation_spec = [
                obs.SatProperties(
                    dict(prop="storage_level_fraction"),
                    dict(prop="battery_charge_fraction"),
                    dict(prop="wheel_speeds_fraction"),

                ),
                obs.Eclipse(norm=5700),
                obs.OpportunityProperties(
                    dict(prop="priority"),
                    dict(prop="opportunity_open", norm=5700.0),
                    n_ahead_observe=env_args.n_obs_image,
                ),
                obs.OpportunityProperties(
                    dict(prop="opportunity_open", norm=5700),
                    dict(prop="opportunity_close", norm=5700),
                    type="ground_station",
                    n_ahead_observe=1,
                ),
                obs.Time(),
            ]
            action_spec = [act.Image(n_ahead_image=env_args.n_act_image),
                           act.Downlink(duration=20.0),
                           act.Desat(duration=20.0),
                           act.Charge(duration=20.0),
                           ]
            dyn_type = dyn.ManyGroundStationFullFeaturedDynModel
            fsw_type = fsw.SteeringImagerFSWModel

        sat = ImagingSatellite(f"EO-{index}", sat_args)
        multiSat.append(sat)
        index += 1

    duration = env_args.orbit_num * 5700.0  # About 2 orbits

    env = GeneralSatelliteTasking(
        satellites=multiSat,
        scenario=scene.UniformTargets(env_args.uniform_targets),
        rewarder=data.UniqueImageReward(),
        time_limit=duration,
        # Note that dyn must inherit from LOSCommunication
        communicator=comm.LOSCommunication(),
        log_level="WARNING",
        terminate_on_time_limit=True,
        failure_penalty=env_args.failure_penalty,
        vizard_dir="./tmp_cluster/vizard" if env_args.use_render else None,
        vizard_settings=dict(showLocationLabels=-
                             1) if env_args.use_render else None,
    )
    return env


def make_BSK_Walker_env(env_args, satellite_names, scenario):
    # Define four satellites in walker delta orbits
    sat_arg_randomizer = walker_delta_args(
        altitude=500.0, inc=50.0, n_planes=env_args.n_satellites, randomize_lan=False, randomize_true_anomaly=False)

    if scenario == "ideal":
        battery_sizes = [1e6]*len(satellite_names)
        memory_size = 1e6
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = env_args.baud_rate
        instr_baud_rate = 500

    elif scenario == "limited":
        battery_sizes = [int(env_args.battery_capacity/4)
                         ]*len(satellite_names)
        memory_size = int(env_args.memory_size/20)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = env_args.baud_rate
        instr_baud_rate = env_args.instr_baud_rate

    elif scenario == "default":
        battery_sizes = [int(env_args.battery_capacity)
                         ]*len(satellite_names)
        memory_size = int(env_args.memory_size)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = env_args.baud_rate
        instr_baud_rate = env_args.instr_baud_rate

    elif scenario == "random":
        battery_sizes = [env_args.battery_capacity]*len(satellite_names)
        memory_size = env_args.memory_size
        random_init_memory = True
        random_init_battery = True
        random_disturbance = True
        random_RW_speed = True
        baud_rate = env_args.baud_rate
        instr_baud_rate = env_args.instr_baud_rate
        
    elif scenario == "limited_all":
        battery_sizes = [50]*len(satellite_names)
        memory_sizes = [5000]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = 0.5
        instr_baud_rate = 125

    else:
        print("Scenario name not available")
        NotImplementedError

    # Define four satellites in a "train" Cluster formation along the same orbit
    multiSat = []
    index = 0
    for battery_size, memory_size in zip(battery_sizes, memory_sizes):
        sat_args = dict(
            # Power
            batteryStorageCapacity=battery_size * 3600,
            storedCharge_Init=int(battery_size * env_args.init_battery_level / 100 * 3600) if not random_init_battery else np.random.uniform(
                battery_size * 3600 * 0.4, battery_size * 3600 * 0.5),
            panelArea=1.0,
            panelEfficiency=20.0,
            basePowerDraw=-10.0,
            instrumentPowerDraw=-30.0,
            transmitterPowerDraw=-25.0,
            thrusterPowerDraw=-80.0,
            # Data Storage
            dataStorageCapacity=memory_size * 8e6,  # MB to bits,
            storageInit=int(memory_size *
                            env_args.init_memory_percent/100) * 8e6 if not random_init_memory else np.random.uniform(memory_size * 8e6 * 0.2, memory_size * 8e6 * 0.8),
            instrumentBaudRate=instr_baud_rate * 1e6,
            transmitterBaudRate=-1*baud_rate * 1e6,
            # Attitude
            imageAttErrorRequirement=0.1,
            imageRateErrorRequirement=0.1,
            disturbance_vector=lambda: np.random.normal(
                scale=0.0001, size=3) if random_disturbance else np.array([0.0, 0.0, 0.0]),
            maxWheelSpeed=6000.0,  # RPM
            wheelSpeeds=lambda: np.random.uniform(
                -3000, 3000, 3) if random_RW_speed else np.array([0.0, 0.0, 0.0]),
            desatAttitude="nadir",
            u_max=0.4,
            K1=0.25,
            K3=3.0,
            omega_max=0.1,
            servo_Ki=5.0,
            servo_P=150,
        )

        class ImagingSatellite(sats.ImagingSatellite):
            observation_spec = [
                obs.SatProperties(
                    dict(prop="storage_level_fraction"),
                    dict(prop="battery_charge_fraction"),
                    dict(prop="wheel_speeds_fraction"),

                ),
                obs.Eclipse(),
                obs.OpportunityProperties(
                    dict(prop="priority"),
                    dict(prop="opportunity_open", norm=5700.0),
                    n_ahead_observe=env_args.n_obs_image,
                ),
                obs.OpportunityProperties(
                    dict(prop="opportunity_open", norm=5700),
                    dict(prop="opportunity_close", norm=5700),
                    type="ground_station",
                    n_ahead_observe=1,
                ),
                obs.Time(),
            ]
            action_spec = [act.Image(n_ahead_image=env_args.n_act_image),
                           act.Downlink(duration=20.0),
                           act.Desat(duration=20.0),
                           act.Charge(duration=20.0),
                           ]
            dyn_type = dyn.ManyGroundStationFullFeaturedDynModel
            fsw_type = fsw.SteeringImagerFSWModel

        sat = ImagingSatellite(f"EO-{index}", sat_args)
        multiSat.append(sat)
        index += 1

    duration = env_args.orbit_num * 5700.0  # About 2 orbits

    env = GeneralSatelliteTasking(
        satellites=multiSat,
        scenario=scene.UniformTargets(env_args.uniform_targets),
        rewarder=data.UniqueImageReward(),
        time_limit=duration,
        # Note that dyn must inherit from LOSCommunication
        communicator=comm.LOSCommunication(),
        sat_arg_randomizer=sat_arg_randomizer,
        log_level="WARNING",
        terminate_on_time_limit=True,
        failure_penalty=env_args.failure_penalty,
        vizard_dir="./tmp_cluster/vizard" if env_args.use_render else None,
        vizard_settings=dict(showLocationLabels=-
                             1) if env_args.use_render else None,
    )
    return env


class BSKWrapper(MultiAgentEnv):
    def __init__(
        self,
        key,
        time_limit,
        pretrained_wrapper,
        seed,
        common_reward,
        reward_scalarisation,
        **kwargs,
    ):
        env_args=Munch.fromDict(kwargs)
        self.satellite_names = []
        for i in range(env_args.n_satellites):
            self.satellite_names.append(f"Satellite{i}")
        
        bsk_scenario = f"{key}".split("-")
        
        if bsk_scenario[0] == "cluster":
            self._env = make_BSK_Cluster_env(env_args,self.satellite_names,bsk_scenario[1])
            print("Running BSK-ENV with cluster scenario")
        elif bsk_scenario[0] == "walker":
            self._env = make_BSK_Walker_env(env_args,self.satellite_names,bsk_scenario[1])
            print("Running BSK-ENV with walker-delta scenario")
        else:
            print("Scenario name not available")
            NotImplementedError
            
        self._env = TimeLimit(self._env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        self.n_agents = len(self.satellite_names)
        self.episode_limit = time_limit
        self._obs = None
        self._info = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = seed
        try:
            self._env.unwrapped.seed(self._seed)
        except:
            self._env.reset(seed=self._seed)

        # self.common_reward = common_reward
        # if self.common_reward:
        #     if reward_scalarisation == "sum":
        #         self.reward_agg_fn = lambda rewards: sum(rewards)
        #     elif reward_scalarisation == "mean":
        #         self.reward_agg_fn = lambda rewards: sum(rewards) / len(rewards)
        #     else:
        #         raise ValueError(
        #             f"Invalid reward_scalarisation: {reward_scalarisation} (only support 'sum' or 'mean')"
        #         )

    def _pad_observation(self, obs):
        return [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in obs
        ]

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        actions = [int(a) for a in actions]
        obs, reward, done, truncated, self._info = self._env.step(actions)
        self._obs = self._pad_observation(obs)
        self._info = {}

        # if self.common_reward and isinstance(reward, Iterable):
        #     reward = float(self.reward_agg_fn(reward))
        # elif not self.common_reward and not isinstance(reward, Iterable):
        #     warnings.warn(
        #         "common_reward is False but received scalar reward from the environment, returning reward as is"
        #     )

        if isinstance(done, Iterable):
            done = all(done)
        return self._obs, reward, done, truncated, self._info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self._obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        raise self._obs[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """Returns the shape of the state"""
        if hasattr(self._env.unwrapped, "state_size"):
            return self._env.unwrapped.state_size
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        obs, info = self._env.reset(seed=seed, options=options)
        self._obs = self._pad_observation(obs)
        return self._obs, info

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        return self._env.unwrapped.seed(seed)

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
