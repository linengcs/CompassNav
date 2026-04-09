import gzip
import json
import logging
import math
import os
import random
import requests
import traceback
import habitat_sim
import csv
import pandas as pd
import numpy as np

from PIL import Image
from simWrapper import PolarAction, SimWrapper
from agent import *
from utils import *

class Env:
    """
    Base class for creating an environment for embodied navigation tasks.
    This class defines the setup, logging, running, and evaluation of episodes.
    """

    task = 'Not defined'

    def __init__(self, cfg: dict):
        """
        Initializes the environment with the provided configuration.

        Args:
            cfg (dict): Configuration dictionary containing environment, simulation, and agent settings.
        """
        self.cfg = cfg['env_cfg']
        self.sim_cfg = cfg['sim_cfg']
        if self.cfg['name'] == 'default':
            self.cfg['name'] = f'default_{random.randint(0, 1000)}'
        self._initialize_logging(cfg)
        self._initialize_agent(cfg)
        self.outer_run_name = self.task + '_' + self.cfg['name']
        self.inner_run_name = f'{self.cfg["instance"]}_of_{self.cfg["instances"]}'
        self.curr_run_name = "Not initialized"
        self.path_calculator = habitat_sim.MultiGoalShortestPath()
        self.simWrapper: SimWrapper = None
        self.num_episodes = 0
        self.num_episodes_success = False
        self._initialize_experiment()

    def _initialize_agent(self, cfg: dict):
        """Initializes the agent for the environment."""
        PolarAction.default = PolarAction(cfg['agent_cfg']['default_action'], 0, 'default')
        cfg['agent_cfg']['sensor_cfg'] = cfg['sim_cfg']['sensor_cfg']
        agent_cls = globals()[cfg['agent_cls']]
        self.agent: Agent = agent_cls(cfg['agent_cfg']) 

    def _initialize_logging(self, cfg: dict):
        """
        Initializes logging for the environment.

        Args:
            cfg (dict): Configuration dictionary containing logging settings.
        """
        self.log_file = f'logs/{cfg["task"]}_{self.cfg["name"]}/{self.cfg["instance"]}_of_{self.cfg["instances"]}.txt'
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if self.cfg['parallel']:
            logging.basicConfig(
                filename=self.log_file,
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s'
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s'
            )

    def _initialize_experiment(self):
        """
        Abstract method for setting up the environment and initializing all required variables.
        Should be implemented in derived classes.
        """
        raise NotImplementedError

    def run_experiment(self):
        """
        Runs the experiment by iterating over episodes.
        """
        print("Start run env") 
        instance_size = math.ceil(self.num_episodes / self.cfg['instances'])
        start_ndx = self.cfg['instance'] * instance_size
        for episode_ndx in range(start_ndx, min(start_ndx + self.cfg['num_episodes'], 3000)):
            self.wandb_log_data = {
                'episode_ndx': episode_ndx,
                'instance': self.inner_run_name,
                'total_episodes': self.cfg['instances'] * self.cfg['num_episodes'],
                'task': self.task,
                'task_data': {},
                'spl': 0,
                'goal_reached': False
            }
            try:
                
                self._run_episode(episode_ndx)
            except Exception as e:
                log_exception(e)
                self.simWrapper.reset()


    def _run_episode(self, episode_ndx: int):
        """
        Runs a single episode.

        Args:
            episode_ndx (int): The index of the episode to run.
        """
        print(f"Start run episode{episode_ndx}") 
        obs = self._initialize_episode(episode_ndx)

        logging.info(f'\n===================STARTING RUN: {self.curr_run_name} ===================\n')
        for _ in range(self.cfg['max_steps']):
            try:
                if hasattr(self, 'habitat_steps') and self.habitat_steps >= 500:
                    logging.info(f"Episode terminated: Reached Habitat max steps limit (500)")
                    break

                agent_action = self._step_env(obs)
                if agent_action is None:
                    break
                obs = self.simWrapper.step(agent_action)

            except Exception as e:
                log_exception(e)

            finally:
                self.step += 1
        self._post_episode()

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode. This method should be implemented in derived classes.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        print(f"Start init env episode{episode_ndx}") 
        self.step = 0
        self.habitat_steps = 0 
        self.init_pos = None
        self.df = pd.DataFrame({})
        self.agent_distance_traveled = 0
        self.prev_agent_position = None
        print(f"finish init env episode{episode_ndx}") 

    def _calculate_habitat_steps(self, action: PolarAction):
        if action.type == 'stop':
            return 1  

        step_count = 0

        if action.r > 0:
            forward_steps = math.ceil(action.r / 0.25)
            step_count += forward_steps

        if abs(action.theta) > 0:
            degrees = abs(math.degrees(action.theta))
            turn_steps = math.ceil(degrees / 30)
            step_count += turn_steps
        
        return max(1, step_count)
    
    def _step_env(self, obs: dict):
        """
        Takes a step in the environment. This method should be implemented in derived classes.

        Args:
            obs (dict): The current observation. Contains agent state and sensor observations.

        Returns:
            PolarAction: The next action to be taken by the agent.
        """
        logging.info(f'Step {self.step}')
        agent_state = obs['agent_state']
        if self.prev_agent_position is not None:
            self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)

            if hasattr(self, 'last_agent_action') and self.last_agent_action:
                habitat_step_delta = self._calculate_habitat_steps(self.last_agent_action)
                self.habitat_steps += habitat_step_delta
                logging.info(f"Habitat steps count: {self.habitat_steps}/500")

        self.prev_agent_position = agent_state.position
        obs['habitat_steps'] = self.habitat_steps

        return None

    def _post_episode(self):
        """
        Called after the episode is complete, saves the dataframe log, and resets the environment.
        Sends a request to the aggregator server if parallel is set to True.
        """
        self.df.to_pickle(f'logs/{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/df_results.pkl')
        self.simWrapper.reset()
        self.agent.reset()
        if self.cfg['parallel']:
            try:
                self.wandb_log_data['spend'] = self.agent.get_spend()
                self.wandb_log_data['default_rate'] = len(self.df[self.df['success'] == 0]) / len(self.df)
                response = requests.post(f'http://localhost:{self.cfg["port"]}/log', json=self.wandb_log_data)
                if response.status_code != 200:
                    logging.error(f"Failed to send metrics: {response.text}")
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    logging.error(f"Frame {frame.filename} line {frame.lineno}")
                logging.error(e)

        logging.info('\n===================RUN COMPLETE===================\n')
        if self.cfg['log_freq'] == 1:
            create_gif(
                f'logs/{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}',
            )
        with open(f'logs/{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/results.txt', 'w') as file:
            file.write(f'success: {self.num_episodes_success}\n\n')
            self.num_episodes_success = False

    def _log(self, images: dict, step_metadata: dict, logging_data: dict):
        """
        Appends the step metadata to the dataframe, and saves the images and general metadata to disk.

        Args:
            images (dict): Images generated during the step.
            step_metadata (dict): Metadata for the current step.
            logging_data (dict): General logging data.
        """
        self.df = pd.concat([self.df, pd.DataFrame([step_metadata])], ignore_index=True)

        if self.step % self.cfg['log_freq'] == 0 or step_metadata['success'] == 0:
            path = f'logs/{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/step{self.step}'
            if not step_metadata['success']:
                path += '_ERROR'
            os.makedirs(path, exist_ok=True)
            for name, im in images.items():
                im = Image.fromarray(im[:, :, 0:3], mode='RGB')
                im.save(f'{path}/{name}.png')
            with open(f'{path}/details.txt', 'w') as file:
                if step_metadata['success']:
                    for k, v in logging_data.items():
                        file.write(f'{k}\n{v}\n\n')

    def _calculate_metrics(self, agent_state: habitat_sim.AgentState, agent_action: PolarAction, geodesic_path: int, max_steps: int):
        """
        Calculates the navigation metrics at a given step.

        Args:
            agent_state: The state of the agent.
            agent_action: The action taken by the agent.
            geodesic_path: The shortest path to the goal.
            max_steps (int): Maximum steps allowed for the episode.

        Returns:
            dict: A dictionary containing calculated metrics.
        """
        metrics = {}
        self.path_calculator.requested_start = agent_state.position
        metrics['distance_to_goal'] = self.simWrapper.get_path(self.path_calculator)

        view_euclidean_distance = float('inf')
        if hasattr(self, 'current_episode'):
            if isinstance(self.current_episode, dict) and 'view_positions' in self.current_episode:
                distances = []
                for pos in self.current_episode['view_positions']:
                    distances.append(np.linalg.norm(np.array(agent_state.position) - np.array(pos)))
                if distances:
                    view_euclidean_distance = min(distances)
                    metrics['view_euclidean_distance'] = view_euclidean_distance
        
        euclidean_distance = float('inf')
        if hasattr(self, 'current_episode') and 'object_positions' in self.current_episode:
            if self.current_episode['object_positions']:
                euclidean_distances = []
                for pos in self.current_episode['object_positions']:
                    if isinstance(pos, (list, np.ndarray)):
                        euclidean_distances.append(np.linalg.norm(np.array(agent_state.position) - np.array(pos)))
                
                if euclidean_distances:
                    euclidean_distance = min(euclidean_distances)
                    metrics['euclidean_distance'] = euclidean_distance
        
        logging.info(f"nav_distance: {metrics['distance_to_goal']}m, view_distance: {view_euclidean_distance}m")

        metrics['spl'] = 0
        metrics['goal_reached'] = False
        metrics['done'] = False
        metrics['finish_status'] = 'running'
        metrics['habitat_steps'] = self.habitat_steps
        if agent_action is PolarAction.stop or self.step + 1 == max_steps or self.habitat_steps >= 500:
            metrics['done'] = True

            if self.habitat_steps >= 500:
                metrics['finish_status'] = 'habitat_max_steps'
                logging.info(f"Reached Habitat max steps limit (500)")
            
            logging.info(f"Stop action triggered:")
            logging.info(f"navigational distance: {metrics['distance_to_goal']}m")
            logging.info(f"Linear distance: {euclidean_distance}m")
            logging.info(f"Success threshold: {self.cfg['success_threshold']}")
            logging.info(f"Agent position: {agent_state.position}")

            effective_distance = min(metrics['distance_to_goal'], view_euclidean_distance)
            if effective_distance < self.cfg['success_threshold']:
                metrics['finish_status'] = 'success'
                metrics['goal_reached'] = True
                self.num_episodes_success = True
                metrics['spl'] = geodesic_path / max(geodesic_path, self.agent_distance_traveled)
                self.wandb_log_data.update({
                    'spl': metrics['spl'],
                    'goal_reached': metrics['goal_reached']
                })
            else:
                if agent_action is PolarAction.stop:
                    metrics['finish_status'] = 'fp'
                else:
                    metrics['finish_status'] = 'max_steps'
        
        logging.info(f"spl: {metrics['spl']}; goal_reached: {metrics['goal_reached']}")
        
        return metrics

class ObjectNavEnv(Env):
    """
    Environment for the ObjectNav task, extending the base Env class.
    This class defines the setup, initialization, and running of ObjectNav episodes.
    """

    task = 'ObjectNav'

    def _initialize_experiment(self):
        """
        Initializes the experiment by setting up the dataset split, scene configuration, and goals.
        """
        print("Start initialize env") 
        self.all_episodes = []
        self.sim_cfg['scene_config'] = "/habitat/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        self.goals = {}

        for f in sorted(os.listdir(f'/habitat/datasets/objectnav_hm3d_v2/{self.cfg["split"]}/content')):
            with gzip.open(f'/habitat/datasets/objectnav_hm3d_v2/{self.cfg["split"]}/content/{f}', 'rt') as gz:
                js = json.load(gz)
                hsh = f.split('.')[0]
                self.goals[hsh] = js['goals_by_category']
                self.all_episodes += js['episodes']
        self.num_episodes = len(self.all_episodes)
        print(f"episode: {self.num_episodes}") 
    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode for the ObjectNav task.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        print("start initialize objnav episode")
        super()._initialize_episode(episode_ndx)
        episode = self.all_episodes[episode_ndx]
        f = episode['scene_id'].split('/')[1:]
        self.sim_cfg['scene_id'] = f[1][2:5]
        self.sim_cfg['scene_path'] = f'/habitat/scene_datasets/hm3d/{self.cfg["split"]}/{f[1]}/{f[2]}'
        print("start initialize sim")
        self.simWrapper = SimWrapper(self.sim_cfg)
        print("finish initialize sim")
        goals = self.goals[f[1][6:]]
        #goals = self.goals[f[0]]
        all_objects = goals[f'{f[-1]}_{episode["object_category"]}']
        view_positions = []
        for obj in all_objects:
            for vp in obj['view_points']:
                view_positions.append(vp['agent_state']['position'])
        self.path_calculator.requested_ends = np.array(view_positions, dtype=np.float32)
        logging.info(f'RUNNING EPISODE {episode_ndx} with {episode["object_category"]} and {len(all_objects)} instances. GEODESIC DISTANCE: {episode["info"]["geodesic_distance"]}')
        if episode['object_category'] == 'tv_monitor':
            episode['object_category'] = 'television, tv screen or tv monitor'
        self.current_episode = {
            'object': episode['object_category'],
            'shortest_path': episode['info']['geodesic_distance'],
            'object_positions': [a['position'] for a in all_objects],
            'view_positions': view_positions,
            'history': [],
            'prev_obs': []
        }
        self.init_pos = np.array(episode['start_position'])
        self.simWrapper.set_state(pos=self.init_pos, quat=episode['start_rotation'])
        self.curr_run_name = f"{episode_ndx}_{self.simWrapper.scene_id}"

        obs = self.simWrapper.step(PolarAction.null)
        return obs

    def _step_env(self, obs: dict):
        """
        Takes a step in the environment for the ObjectNav task.

        Args:
            obs (dict): The current observation.

        Returns:
            list: The next action to be taken by the agent.
        """
        super()._step_env(obs)
        obs['goal'] = self.current_episode['object']
        obs['history'] = self.current_episode['history']
        obs['prev_obs'] = self.current_episode['prev_obs']
        agent_state = obs['agent_state']
        self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)
        self.prev_agent_position = agent_state.position
        agent_action, metadata = self.agent.step(obs)
        self.last_agent_action = agent_action
        if metadata['step_metadata']['success'] != 0:
            self.current_episode['history'].append(metadata['new_history'])
        self.current_episode['prev_obs'].append(metadata['images']['color_sensor'])
        self.current_episode['prev_obs'] = self.current_episode['prev_obs'][-5:]
        step_metadata = metadata['step_metadata']
        logging_data = metadata['logging_data']
        images = metadata['images']

        metrics = self._calculate_metrics(agent_state, agent_action, self.current_episode['shortest_path'], self.cfg['max_steps'])
        step_metadata.update(metrics)

        self._log(images, step_metadata, logging_data)

        if metrics['done']:
            agent_action = None

        return agent_action
