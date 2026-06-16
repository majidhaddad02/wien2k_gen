#compdef wien2k_gen

_wien2k_gen() {
    local curcontext="$curcontext" state line
    typeset -A opt_args

    _arguments -C \
        '(- :)'{-v,--verbose}'[Increase verbosity]' \
        '(- :)'{-q,--quiet}'[Suppress console output]' \
        '(- :)--json[Output results in JSON]' \
        '(- :)--config=[Config path]:_files' \
        '(- :)--backend=[Backend]:(wien2k vasp qe cp2k)' \
        '(- :)--log-file=[Log path]:_files' \
        '1:subcommand:->subcmds' \
        '*::args:->args'

    case $state in
        (subcmds)
            local -a subcmds
            subcmds=(
                'generate:Generate parallel config'
                'submit:Submit job to SLURM'
                'benchmark:Run benchmarks'
                'diagnostics:Check system health'
                'analyze:Parse SCF logs'
                'tui:Launch interactive UI'
            )
            _describe -t subcmds 'subcommand' subcmds
            ;;
        (args)
            case $line[1] in
                generate)
                    _arguments \
                        '--nodes=[Nodes]' \
                        '--cores=[Total Cores]' \
                        '--omp=[OMP threads]' \
                        '--mode=[Mode]:(mpi hybrid kpoint)' \
                        '--dry-run' \
                        '--export=[Export path]:_files' \
                        '--overwrite'
                    ;;
                submit)
                    _arguments \
                        '--partition=[Queue]' \
                        '--nodes=[Nodes]' \
                        '--ntasks=[Tasks]' \
                        '--time=[Walltime]' \
                        '--mem=[Memory]' \
                        '--job-name=[Job Name]' \
                        '--dependency=[Dependency]' \
                        '--dry-run' \
                        '--export=[Export path]:_files'
                    ;;
                benchmark)
                    _arguments \
                        '--type=[Type]:(real synthetic)' \
                        '--max-cores=[Max Cores]' \
                        '--walltime=[Walltime]' \
                        '--output=[JSON path]:_files' \
                        '--skip-cleanup'
                    ;;
                diagnostics)
                    _arguments \
                        '--export=[Report path]:_files' \
                        '--full'
                    ;;
                analyze)
                    _arguments \
                        '--log=[Log file]:_files' \
                        '--code=[Code]:(wien2k vasp qe)' \
                        '--export=[JSON path]:_files'
                    ;;
                tui)
                    _arguments '--compact'
                    ;;
            esac
            ;;
    esac
}

_wien2k_gen