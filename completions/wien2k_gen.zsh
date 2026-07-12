#compdef wien2k_gen

_wien2k_gen() {
    local curcontext="$curcontext" state line
    typeset -A opt_args

    _arguments -C \
        '(- :)'{-v,--verbose}'[Increase verbosity (-v, -vv)]' \
        '(- :)'{-q,--quiet}'[Suppress console output]' \
        '(- :)--json[Output results in JSON format]' \
        '(- :)--config=[Path to custom config file]:_files' \
        '(- :)--backend=[Override auto-detected backend]:(wien2k qe vasp cp2k)' \
        '(- :)--log-file=[Redirect logs to file]:_files' \
        '(- :)--version[Show version]' \
        '(- :)--plain[Use plain output (no Rich formatting)]' \
        '(- :)--no-color[Disable colored output]' \
        '1:subcommand:->subcmds' \
        '*::args:->args'

    case $state in
        (subcmds)
            local -a subcmds
            subcmds=(
                'generate:Generate parallel configuration files'
                'submit:Submit job to scheduler'
                'benchmark:Run empirical or synthetic benchmarks'
                'diagnostics:Collect system and environment health report'
                'hardware:Show hardware info and parallelization recommendations'
                'analyze:Parse SCF logs and generate performance reports'
                'tui:Launch interactive terminal UI'
                'monitor:Monitor SCF convergence in real-time'
                'run:Execute a WIEN2k workflow from YAML'
                'workflow:Generate workflow YAML templates'
                'diagnose:Diagnose SCF convergence issues'
                'optimize:Bayesian auto-tuning of RKMAX, k-points, mixing'
                'screen:High-throughput screening via Materials Project'
                'predict:Predict SCF convergence time before calculation'
                'converge:Run automated convergence tests'
                'advise:Get intelligent optimization advice (Roofline, Amdahl, NUMA)'
                'history:Query execution history database'
                'analyze-bands:Extract band structure and DOS data'
            )
            _describe -t subcmds 'subcommand' subcmds
            ;;
        (args)
            case $line[1] in
                generate)
                    _arguments \
                        '--nodes=[Number of nodes]' \
                        '--cores=[Total cores]' \
                        '--omp=[OpenMP threads per rank]' \
                        '--mode=[Parallel execution mode]:(mpi hybrid kpoint)' \
                        '--target=[Optimization target]:(time memory balanced cost)' \
                        '--max-cores=[Hard limit on total cores]' \
                        '--reserve-os-cores=[Reserve N cores for OS]' \
                        '--memory-limit=[Hard memory limit per node (GB)]' \
                        '--dry-run[Generate config without writing to disk]' \
                        '--export=[Export config summary to path]:_files' \
                        '--overwrite[Overwrite existing .machines without prompt]' \
                        '(-S --scheduler)'{-S,--scheduler}'=[Target scheduler]:(slurm pbs lsf sge auto)' \
                        '--gpu[Enable GPU-aware configuration]' \
                        '--gpu-mixed-precision[Enable FP32/FP16 mixed precision]' \
                        '--manual[Open .machines in editor for manual review]'
                    ;;
                submit)
                    _arguments \
                        '(-S --scheduler)'{-S,--scheduler}'=[Target scheduler]:(slurm pbs lsf sge auto)' \
                        '--partition=[Scheduler partition/queue]' \
                        '--nodes=[Number of nodes]' \
                        '--ntasks=[Total tasks (0 = auto)]' \
                        '--time=[Walltime (HH:MM:SS)]' \
                        '--mem=[Memory per node]' \
                        '--job-name=[Job identifier]' \
                        '--dependency=[Job dependency (e.g., afterok:123)]' \
                        '--dry-run[Generate script only, do not submit]' \
                        '--export=[Export script to path]:_files'
                    ;;
                benchmark)
                    _arguments \
                        '--type=[Benchmark type]:(real synthetic)' \
                        '--max-cores=[Maximum cores for scaling suite]' \
                        '--walltime=[Max runtime per run]' \
                        '--output=[Save results to JSON path]:_files' \
                        '--skip-cleanup[Retain temporary benchmark directories]' \
                        '(-S --scheduler)'{-S,--scheduler}'=[Target scheduler]:(slurm pbs lsf auto)'
                    ;;
                diagnostics)
                    _arguments \
                        '--export=[Save diagnostic report to path]:_files' \
                        '--full[Include verbose library and interconnect checks]'
                    ;;
                hardware)
                    _arguments \
                        '(-r --recommend)'{-r,--recommend}'[Show NUMA/hybrid/IO optimization advice]' \
                        '--case=[Case name for problem-specific recommendations]'
                    ;;
                analyze)
                    _arguments \
                        '--log=[Path to SCF/output log file]:_files' \
                        '--code=[Force DFT code parser]:(wien2k vasp qe)' \
                        '--export=[Export analysis report to JSON]:_files'
                    ;;
                tui)
                    _arguments '--compact[Enable compact UI layout]'
                    ;;
                monitor)
                    _arguments \
                        '--interval=[Polling interval in seconds (default: 2)]' \
                        '--output=[SCF output file to parse]:_files' \
                        '1:case:'
                    ;;
                run)
                    _arguments \
                        '--auto-retry[Auto-retry on convergence failure]' \
                        '--no-retry[Disable auto-retry]' \
                        '--max-retries=[Maximum retry attempts (default: 3)]' \
                        '--poll=[Job polling interval in seconds (default: 5)]' \
                        '1:workflow_file:_files -g "*.yml *.yaml(-.)"'
                    ;;
                workflow)
                    _arguments \
                        '--case=[Case name]' \
                        '--steps=[Comma-separated workflow steps]' \
                        '--output=[Output YAML path]:_files' \
                        '1:action:(create list visualize)'
                    ;;
                diagnose)
                    _arguments \
                        '--log=[Path to .scf or .output file]:_files' \
                        '1:case:'
                    ;;
                optimize)
                    _arguments \
                        '--case=[WIEN2k case name]' \
                        '--budget=[Max DFT runs (default: 10)]' \
                        '--target=[Target metric]' \
                        '--simulated[Use simulated objective for testing]' \
                        '(-v --verbose)'{-v,--verbose}'[Show iteration details]'
                    ;;
                screen)
                    _arguments \
                        '--formula=[Chemical formula (e.g., ABO3)]' \
                        '--elements=[Comma-separated elements (e.g., Ti,O,Zr)]' \
                        '--mp-id=[Single Materials Project ID]' \
                        '--max=[Max materials (default: 50)]' \
                        '--api-key=[Materials Project API key]' \
                        '--output=[Output directory]:_directories'
                    ;;
                predict)
                    _arguments \
                        '--case=[WIEN2k case name]' \
                        '--struct=[Path to .struct file]:_files' \
                        '--no-history[Skip ML training from history]'
                    ;;
                converge)
                    _arguments \
                        '--case=[Case name]' \
                        '--mode=[Parameter to converge]:(kpoints rkmax both)' \
                        '--tolerance=[Energy tolerance in Ry (default: 0.001)]' \
                        '--kpoints=[Space-separated k-point grids]' \
                        '--rkmax=[Comma-separated RKmax values]'
                    ;;
                advise)
                    _arguments \
                        '--case=[WIEN2k case name for problem-aware advice]' \
                        '--nmat=[Override matrix size]' \
                        '--kpoints=[Override k-point count]' \
                        '--cores=[Target total cores]' \
                        '--target=[Optimization goal]:(time energy cost balanced)' \
                        '--plain[Show advice in simple language (non-expert mode)]' \
                        '--json[Export advice as JSON]'
                    ;;
                history)
                    _arguments \
                        '--list[List past runs]' \
                        '--show=[Show details of run ID]' \
                        '--similar-to=[Find similar past cases by case path]' \
                        '--limit=[Max results to display]'
                    ;;
                analyze-bands)
                    _arguments \
                        '--case=[Case name]' \
                        '--output=[Output file for band data (JSON)]:_files' \
                        '--dos[Also parse DOS data]'
                    ;;
            esac
            ;;
    esac
}

_wien2k_gen
