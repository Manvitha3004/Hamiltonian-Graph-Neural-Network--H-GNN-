"""
Unified Scaling Law Discovery Runner.

This script provides a single entry point for all scaling law experiments.

Usage:
    python run_discovery.py --all          # Run all experiments
    python run_discovery.py --quick        # Quick subset for testing
    python run_discovery.py --epsilon       # Only epsilon threshold
    python run_discovery.py --n-scaling     # Only N scaling
    python run_discovery.py --lyapunov      # Only Lyapunov analysis
    python run_discovery.py --dt            # Only dt validation
    python run_discovery.py --failure       # Only failure modes
    python run_discovery.py --cross         # Cross-parameter sweep
    python run_discovery.py --plot          # Generate plots from results
"""

import argparse
import time
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Scaling Law Discovery for Hamiltonian Correction Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_discovery.py --all              Run complete scaling law analysis
  python run_discovery.py --quick            Quick test run (fewer configs)
  python run_discovery.py --epsilon --n-scaling   Run epsilon and N scaling
  python run_discovery.py --plot             Generate plots from existing results
        """
    )
    
    # Experiment selection
    parser.add_argument('--all', action='store_true', 
                        help='Run all experiments')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with reduced configurations')
    parser.add_argument('--epsilon', action='store_true',
                        help='Run epsilon threshold sweep')
    parser.add_argument('--n-scaling', action='store_true',
                        help='Run system size scaling')
    parser.add_argument('--lyapunov', action='store_true',
                        help='Run Lyapunov analysis')
    parser.add_argument('--dt', action='store_true',
                        help='Run dt backward error validation')
    parser.add_argument('--failure', action='store_true',
                        help='Run failure mode characterization')
    parser.add_argument('--cross', action='store_true',
                        help='Run cross-parameter sweep')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots from existing results')
    
    # Configuration
    parser.add_argument('--output', type=str, default='results/scaling_laws',
                        help='Output directory')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of seeds per configuration')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    print("=" * 70)
    print("SCALING LAW DISCOVERY FOR H-GNN")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Seeds per config: {args.seeds}")
    print("=" * 70 + "\n")
    
    # Determine which experiments to run
    run_any = args.epsilon or args.n_scaling or args.lyapunov or args.dt or args.failure or args.cross
    
    if args.plot:
        from h_gnn.experiments.scaling_visualization import generate_all_plots
        generate_all_plots(results_dir=output_dir, output_dir=output_dir / "figures")
        return
    
    if args.all or not run_any:
        # Run all experiments
        from h_gnn.experiments.scaling_law_discovery import run_all_scaling_experiments
        
        if args.quick:
            # Quick mode - reduced configurations
            run_all_scaling_experiments(
                output_dir=output_dir,
                run_epsilon_sweep=True,
                run_n_scaling=True,
                run_lyapunov=True,
                run_dt_validation=True,
                run_failure_modes=True,
            )
        else:
            run_all_scaling_experiments(output_dir=output_dir)
    
    else:
        # Run selected experiments
        if args.epsilon:
            print("\n>>> EPSILON THRESHOLD SWEEP <<<\n")
            from h_gnn.experiments.scaling_law_discovery import learnability_threshold_sweep
            
            if args.quick:
                learnability_threshold_sweep(
                    epsilon_range=(1.0, 100.0),
                    n_points=8,
                    n_seeds=2,
                    output_dir=output_dir,
                )
            else:
                learnability_threshold_sweep(
                    epsilon_range=(0.1, 300.0),
                    n_points=20,
                    n_seeds=args.seeds,
                    output_dir=output_dir,
                )
        
        if args.n_scaling:
            print("\n>>> SYSTEM SIZE SCALING <<<\n")
            from h_gnn.experiments.scaling_law_discovery import system_size_scaling
            
            if args.quick:
                system_size_scaling(
                    n_values=[10, 20, 40],
                    n_seeds=2,
                    output_dir=output_dir,
                )
            else:
                system_size_scaling(
                    n_values=[10, 20, 40, 80, 160],
                    n_seeds=args.seeds,
                    output_dir=output_dir,
                )
        
        if args.lyapunov:
            print("\n>>> LYAPUNOV ANALYSIS <<<\n")
            from h_gnn.experiments.scaling_law_discovery import chaos_analysis
            chaos_analysis(output_dir=output_dir)
        
        if args.dt:
            print("\n>>> DT BACKWARD ERROR VALIDATION <<<\n")
            from h_gnn.experiments.scaling_law_discovery import backward_error_dt_validation
            
            if args.quick:
                backward_error_dt_validation(
                    dt_values=[30.0, 60.0],
                    n_seeds=2,
                    output_dir=output_dir,
                )
            else:
                backward_error_dt_validation(
                    dt_values=[15.0, 30.0, 60.0, 120.0],
                    n_seeds=args.seeds,
                    output_dir=output_dir,
                )
        
        if args.failure:
            print("\n>>> FAILURE MODE CHARACTERIZATION <<<\n")
            from h_gnn.experiments.scaling_law_discovery import characterize_failure_modes
            characterize_failure_modes(output_dir=output_dir)
        
        if args.cross:
            print("\n>>> CROSS-PARAMETER SWEEP <<<\n")
            from h_gnn.experiments.cross_parameter_analysis import cross_parameter_sweep
            
            if args.quick:
                cross_parameter_sweep(
                    n_values=[5, 10],
                    t_values=[20, 50],
                    dt_values=[30.0, 60.0],
                    epsilon_values=[1.0, 10.0, 50.0],
                    n_seeds=1,
                    output_dir=output_dir,
                )
            else:
                cross_parameter_sweep(
                    n_values=[5, 10, 20],
                    t_values=[20, 50, 100],
                    dt_values=[30.0, 60.0, 120.0],
                    epsilon_values=[1.0, 5.0, 25.0, 100.0],
                    n_seeds=2,
                    output_dir=output_dir,
                )
    
    # Generate plots
    print("\n>>> GENERATING PLOTS <<<\n")
    try:
        from h_gnn.experiments.scaling_visualization import generate_all_plots
        generate_all_plots(results_dir=output_dir, output_dir=output_dir / "figures")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"COMPLETED in {elapsed/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
