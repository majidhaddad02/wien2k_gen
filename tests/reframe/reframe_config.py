"""
Minimal ReFrame Configuration for WIEN2kGen CI/CD Testing.

Place this file at tests/reframe/reframe_config.py and invoke with:
    reframe -C tests/reframe/reframe_config.py -c tests/reframe/ -r
"""

site_configuration = {
    "systems": [
        {
            "name": "generic",
            "descr": "Generic system for CI/CD pipeline testing",
            "hostnames": [".*"],
            "modules_system": "nomod",
            "partitions": [
                {
                    "name": "default",
                    "descr": "Default local partition",
                    "scheduler": "local",
                    "launcher": "local",
                    "environs": [
                        {
                            "name": "default",
                            "cc": "gcc",
                            "cxx": "g++",
                            "ftn": "gfortran",
                        },
                    ],
                    "max_jobs": 4,
                    "access": [],
                    "resources": [],
                    "container_platforms": [],
                }
            ],
        }
    ],
    "environments": [
        {
            "name": "default",
            "modules": [],
            "variables": {},
            "features": [],
            "extras": {},
            "target_systems": ["generic"],
            "prepare_cmds": [],
        }
    ],
    "logging": [
        {
            "level": "debug",
            "handlers": [
                {
                    "type": "stream",
                    "name": "stdout",
                    "level": "info",
                    "format": "%(message)s",
                },
                {
                    "type": "file",
                    "name": "reframe.log",
                    "level": "debug",
                    "format": "[%(asctime)s] %(levelname)s: %(check_info)s: %(message)s",
                    "append": False,
                },
            ],
            "handlers_perflog": [
                {
                    "type": "filelog",
                    "prefix": "%(check_system)s/%(check_partition)s",
                    "level": "info",
                    "format": (
                        "%(check_job_completion_time)s|"
                        "%(check_name)s|"
                        "%(check_display_name)s|"
                        "%(check_system)s|"
                        "%(check_partition)s|"
                        "%(check_environ)s|"
                        "%(check_jobid)s|"
                        "%(check_result)s|"
                        "%(check_perfvalues)s"
                    ),
                    "append": True,
                }
            ],
        }
    ],
    "general": [
        {
            "check_search_path": ["tests/reframe/"],
            "stage_dir_prefix": "./stage",
            "output_dir_prefix": "./output",
            "purge": False,
            "unload_modules": False,
            "check_search_recursive": False,
            "ignore_check_conflicts": False,
        }
    ],
}

# Optional: SLURM-based systems can be added for HPC center deployment
# by uncommenting and customising the block below:

# site_configuration["systems"].append({
#     "name": "hpc-cluster",
#     "descr": "Production HPC cluster",
#     "hostnames": [r"login\d+", r"node\d+"],
#     "modules_system": "lmod",
#     "partitions": [
#         {
#             "name": "cpu",
#             "descr": "CPU compute partition",
#             "scheduler": "slurm",
#             "launcher": "srun",
#             "access": ["--partition=cpu", "--qos=normal"],
#             "environs": ["gnu", "intel"],
#             "max_jobs": 100,
#             "resources": [],
#         }
#     ],
# })
