REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .parallel_runner_continual import ContinualParallelRunner
REGISTRY["parallel_continual"] = ContinualParallelRunner

from .episode_runner_continual import ContinualEpisodeRunner
REGISTRY["episode_continual"] = ContinualEpisodeRunner