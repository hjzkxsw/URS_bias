import os
import re
from typing import List, Optional
from logging import getLogger

from problem.ProblemSet import ProblemSet

logger = getLogger(__name__)


class OracleDict:
    """
    Store default and scale-specific oracle-solver settings.

    `scale=None` means the default oracle for that problem, which is generally used when scale is 100. 
    A scale-specific oracle takes priority over the default oracle when both are available.

    Example:
        oracle_dict = OracleDict()
        oracle_dict.get("cvrp", 100)  # "hgs"
        oracle_dict.set("cvrp", 1000, "pyvrp")
        oracle_dict.get("cvrp", 1000)  # "pyvrp"
    """

    def __init__(self):
        self.default_oracles = {}
        self.scale_oracles = {}
        self._setup_default_oracles()

    def _setup_default_oracles(self):
        """
        Build the default oracle-solver mapping for every supported problem.
        """
        asymmetric_list = ProblemSet.get(name="asymmetric_list")  # 55
        for problem in asymmetric_list:
            if problem in ["atsp"]:
                self.default_oracles[problem] = "lkh"
            elif problem in ["acvrp"]:
                self.default_oracles[problem] = "pyvrp"
            elif problem in ["aspctsp"]:
                self.default_oracles[problem] = None  # no oracle solution for aspctsp
            else:
                self.default_oracles[problem] = "ortools"

        mdvrpmix_list = ProblemSet.get(name="mdvrpmix_list")  # 24
        for problem in mdvrpmix_list:
            self.default_oracles[problem] = "pyvrp"

        vrpmix_list = ProblemSet.get(name="vrpmix_list")  # 24
        for problem in vrpmix_list:
            if problem in ["cvrpl", "ocvrp"]:
                self.default_oracles[problem] = "lkh"
            elif problem in ["cvrptw"]:
                self.default_oracles[problem] = "pyvrp"
            elif problem in ["cvrp"]:
                self.default_oracles[problem] = "hgs"
            else:
                self.default_oracles[problem] = "ortools"

        # ['pdcvrp', 'opdcvrp']; asymmetric PDCVRP variants are already included above.
        for problem in ProblemSet.get(included=["pd", "cvrp"]):
            self.default_oracles[problem] = "ortools"

        # For pctsp, spctsp and op, the data and oracle value are directly obtained from AM (https://arxiv.org/abs/1803.08475)
        # For pdtsp, the data and oracle value are directly obtained from N2S (https://arxiv.org/abs/2204.11399)
        # For tsp, the solutions are already included in the data files, we use the dataset which is provided by LEHD 
        # (https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD)
        self.default_oracles["pctsp"] = "ils"
        self.default_oracles["spctsp"] = "ils"
        self.default_oracles["tsp"] = "lkh"
        self.default_oracles["op"] = "compass"
        self.default_oracles["pdtsp"] = "lkh"

    def set(self, problem_name, scale, oracle_name):
        """
        Add or modify an oracle setting.

        Args:
            problem_name: Problem name, such as "cvrp".
            scale: Data scale. Use None to modify the default oracle.
            oracle_name: Oracle keyword, or None when no oracle solution exists.
        """
        problem_name = str(problem_name).lower()

        if scale is None:
            self.default_oracles[problem_name] = oracle_name
            return

        scale = int(scale)
        self.scale_oracles.setdefault(problem_name, {})[scale] = oracle_name

    def get(self, problem_name, scale):
        """
        Get the oracle for one problem and scale.

        Scale-specific settings take priority. If the scale is not configured,
        the problem default is returned. Unknown problems return None.
        """
        problem_name = str(problem_name).lower()

        if scale is not None:
            scale = int(scale)
            scale_oracles = self.scale_oracles.get(problem_name, {})
            if scale in scale_oracles:
                return scale_oracles[scale]

        return self.default_oracles.get(problem_name)

    def update(self, overrides):
        """
        Apply oracle overrides from a plain dictionary.

        Examples:
            {"cvrp": "pyvrp"} changes the default oracle for cvrp.
            {"cvrp": {1000: "ortools"}} changes only cvrp at scale 1000.
        """
        for problem_name, oracle_setting in overrides.items():
            if isinstance(oracle_setting, dict):
                for scale, oracle_name in oracle_setting.items():
                    if scale == "default":
                        scale = None
                    self.set(problem_name, scale, oracle_name)
            else:
                self.set(problem_name, None, oracle_setting)

    def all_oracles(self):
        """
        Merge default_oracles and scale_oracles.
        Return all oracle settings, including entries whose oracle is None.

        Problems without scale overrides are returned as `problem: oracle`.
        Problems with scale overrides are returned as `problem: {None: default,
        scale: oracle}`.
        """
        oracle_dict = dict(self.default_oracles)

        for problem_name, scale_oracles in self.scale_oracles.items():
            default_oracle = oracle_dict.get(problem_name) # from default_oracles to get the default oracle
            oracle_dict[problem_name] = {None: default_oracle} # Default oracle used when no scale is specified 
            oracle_dict[problem_name].update(scale_oracles)

        return oracle_dict


class DataFinder:
    """
    Find data and solution paths under one data directory.

    The finder keeps `data_dir` is derived oracle keywords as data state, so repeated lookups do not need to rebuild those settings.
    Note that oracle settings are kept as state.

    Example:
        finder = DataFinder("../data")
        paths = finder.get("cvrp", 100)
        paths["data_path"]  # "../data/cvrp/cvrp100_uniform.pkl"
        paths["solution_path"]  # "../data/cvrp/hgs_cvrp100_uniform.pkl"
    Find data and solution paths for one problem at one scale.

    The problem directory is required. If the requested data file cannot be
    found, this method returns None. The solution file is optional: when
    missing, the method prints a warning and returns None for `solution_file`
    and `solution_path`.

    Args:
        data_dir: Root data directory containing one subdirectory per problem.
        problem_name: Problem name and subdirectory, such as "cvrp".
        scale: data scale, such as 100, 1000, 2000, etc.
        oracle_dict: Optional `OracleDict` or plain dict override.

    Returns:
        dict: A dictionary with problem metadata and matched filenames/paths:
            `problem_name`, `scale`, `oracle`, `data_file`,
            `data_path`, `solution_file`, and `solution_path`.
    """

    def __init__(self, data_dir, oracle_dict=None):
        """
        Args:
            data_dir: Root data directory containing one subdirectory per problem.
            oracle_dict: Optional `OracleDict` or plain dict override. Plain dict
                values can be either oracle names or scale-to-oracle mappings.
        """
        self.data_dir = os.fspath(data_dir)
        if oracle_dict is None:
            self.oracle_dict = OracleDict()
        elif isinstance(oracle_dict, OracleDict):
            self.oracle_dict = oracle_dict
        elif isinstance(oracle_dict, dict):
            self.oracle_dict = OracleDict()
            self.oracle_dict.update(oracle_dict)
        else:
            raise TypeError("oracle_dict must be None, OracleDict, or dict")

        self.supported_suffixes = (".pkl", ".pt", ".txt")

    @property
    def known_oracle_keywords(self):
        """
        Return current non-empty oracle keywords.

        This is computed dynamically so later calls to `self.oracle_dict.set()`
        are reflected when filtering data files.
        """
        return tuple(sorted({
            str(oracle_name).lower()
            for oracle_name in self._iter_oracle_values(self.oracle_dict.all_oracles())
            if oracle_name is not None
        }))

    @staticmethod
    def _iter_oracle_values(oracle_config):
        """
        Yield oracle values from nested oracle dictionaries.
        """
        if isinstance(oracle_config, dict):
            for value in oracle_config.values():
                yield from DataFinder._iter_oracle_values(value)
            return

        yield oracle_config

    def _contains_problem_scale(self, filename, problem_name, scale):
        """
        Check whether `filename` contains the requested problem and scale token.

        The final negative lookahead prevents scale 100 from matching scale 1000.

        Example:
            self._contains_problem_scale("cvrptw1000_C200.pt", "cvrptw", 1000)
            # True
        """
        pattern = rf"{re.escape(problem_name.lower())}[_-]*{int(scale)}(?!\d)"
        return re.search(pattern, filename.lower()) is not None


    def _list_supported_files(self, problem_dir, append_suffixes: Optional[List[str]] = None):
        """
        List supported files directly under one problem directory.

        The result contains filenames only, not full paths, and is sorted for
        deterministic selection. Note that `filename` must be a supported dataset/solution file.
        
        """
        supported_suffixes = self.supported_suffixes
        if append_suffixes is not None:
            supported_suffixes = set(supported_suffixes + tuple(append_suffixes))

        return [
            filename
            for filename in sorted(os.listdir(problem_dir))
            if os.path.isfile(os.path.join(problem_dir, filename))
            and os.path.splitext(filename)[1].lower() in supported_suffixes
        ]

    def _find_data_file(self, files, problem_name, scale):
        """
        Find the data dataset file for one problem and scale.

        data files must contain the problem-scale token and must not contain
        known oracle keywords. Returns None when no matching data exists.
        """
        oracle_keywords = self.known_oracle_keywords
        candidates = [
            filename
            for filename in files
            if self._contains_problem_scale(filename, problem_name, scale)
            and not any(keyword in filename.lower() for keyword in oracle_keywords)
        ]
        if not candidates:
            return None

        return sorted(candidates)[0]

    @staticmethod
    def _solution_rank(filename):
        """
        Rank solution files so larger time limits are preferred.

        Lower tuple values are better. Files with a time suffix such as "12000s"
        are preferred over files without one, and the largest time value wins.
        """
        time_limits = [int(value) for value in re.findall(r"(\d+)s", filename.lower())]
        has_time_rank = 0 if time_limits else 1
        best_time_limit = -max(time_limits) if time_limits else 0
        return has_time_rank, best_time_limit, filename.lower()

    def _find_solution_file(self, files, problem_name, scale, oracle_name, data_file=None):
        """
        Find the solution file for one problem, scale, and oracle keyword.

        Matching is strict substring matching on the oracle keyword after
        converting both strings to lowercase. No oracle aliases are supported.
        """
        if oracle_name is None:
            return None

        oracle_name = str(oracle_name).lower()
        candidates = [
            filename
            for filename in files
            if filename != data_file
            and self._contains_problem_scale(filename, problem_name, scale)
            and oracle_name in filename.lower()
        ]
        if not candidates:
            return None

        return min(candidates, key=self._solution_rank)

    def get(self, problem_name, scale):
        """
        Find data and solution paths for one problem at one scale.

        The data file is required. If it cannot be found, this method returns None. 
        The solution file is optional: when missing, the method
        prints a warning and returns None for `solution_file` and `solution_path`.
        
        Args:
            problem_name: Problem name, which is used as the subdirectory, such as "cvrp".
            scale: data scale, such as 100, 1000, 2000, etc.

        Returns:
            dict: A dictionary with problem metadata and matched filenames/paths:
                `problem_name`, `scale`, `oracle`, `data_file`,
                `data_path`, `solution_file`, and `solution_path`.
        """
        problem_name = str(problem_name).lower()
        scale = int(scale)

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"data_dir does not exist: {self.data_dir}")

        # The problem directory must exist to find any data or solution files
        problem_dir = os.path.join(self.data_dir, problem_name)
        if not os.path.isdir(problem_dir):
            raise FileNotFoundError(f"problem directory does not exist: {problem_dir}")

        # data files must exist, but solution files may be missing.
        files = self._list_supported_files(problem_dir)
        data_file = self._find_data_file(files, problem_name, scale)
        if data_file is None:
            return None  # not found, return None to indicate missing data file

        oracle_name = self.oracle_dict.get(problem_name, scale)
        solution_file = self._find_solution_file(
            files,
            problem_name,
            scale,
            oracle_name,
            data_file=data_file,
        )

        if solution_file is None:
            if oracle_name is not None:
                logger.warning(
                    f"The solution file not found: problem={problem_name}, "
                    f"scale={scale}, oracle={oracle_name}"
                )
            else:
                logger.info(
                    f"No oracle configured: problem={problem_name}, "
                    f"scale={scale}"
                )

        return {
            "problem_name": problem_name,
            "scale": scale,
            "oracle": oracle_name,
            "data_file": data_file,
            "data_path": os.path.join(problem_dir, data_file),
            "solution_file": solution_file,
            "solution_path": None if solution_file is None else os.path.join(problem_dir, solution_file),
        }
            
