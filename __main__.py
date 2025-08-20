#!/usr/bin/env python3
"""
Main entry point for the Comprehensive Reporting and Feedback System
"""

import sys
import argparse
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from integrated_system import IntegratedReportingSystem, IntegratedReportingConfig, run_demo_scenario

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Reporting and Feedback System for Supervisor Agent'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (JSON)'
    )
    
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run demo scenario'
    )
    
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='Generate comprehensive report and exit'
    )
    
    parser.add_argument(
        '--dashboard-only',
        action='store_true',
        help='Start dashboard only (no background processing)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='reporting_output',
        help='Output directory for reports and logs'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5000,
        help='Dashboard port (default: 5000)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        config = IntegratedReportingConfig(**config_data)
    else:
        config = IntegratedReportingConfig(
            base_output_dir=args.output_dir,
            dashboard_port=args.port,
            background_processing=not args.dashboard_only
        )
    
    # Create system
    system = IntegratedReportingSystem(config)
    
    if args.demo:
        print("Running demo scenario...")
        run_demo_scenario(system)
        return
    
    if args.report:
        print("Generating comprehensive report...")
        reports = system.generate_comprehensive_report()
        print(f"Reports generated: {reports}")
        return
    
    # Start the system
    print("Starting Comprehensive Reporting and Feedback System...")
    print(f"Dashboard: http://localhost:{config.dashboard_port}")
    print(f"Output directory: {config.base_output_dir}")
    print("Press Ctrl+C to stop")
    
    try:
        system.start()
        
        # Keep running
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping system...")
        system.stop()
        print("System stopped.")

if __name__ == '__main__':
    main()
