# Enhanced gem5 Script for Detailed Cycle-by-Cycle Pipeline Analysis with Stage Metrics

import argparse
import sys
import os
import time

import m5
from m5.defines import buildEnv
from m5.objects import *
from m5.util import addToPath, fatal, warn

addToPath("../../")

from common import Options, Simulation, CacheConfig, CpuConfig, ObjectList, MemConfig
from common.FileSystemConfig import config_filesystem
from common.Caches import *

def get_processes(args):
    """Interprets provided args and returns a list of processes."""
    multiprocesses = []
    workloads = args.cmd.split(";")

    idx = 0
    for wrkld in workloads:
        process = Process(pid=100 + idx)
        process.executable = wrkld
        process.cwd = os.getcwd()
        process.cmd = [wrkld]
        multiprocesses.append(process)
        idx += 1

    if args.smt:
        assert args.cpu_type == "DerivO3CPU"
        return multiprocesses, idx
    else:
        return multiprocesses, 1

# Parse command line arguments
parser = argparse.ArgumentParser()
Options.addCommonOptions(parser)
Options.addSEOptions(parser)
args = parser.parse_args()

# Get process workloads and number of threads
multiprocesses, numThreads = get_processes(args)

# Set CPU class to DerivO3CPU for out-of-order execution and set memory mode to 'timing'
(CPUClass, test_mem_mode, FutureClass) = Simulation.setCPUClass(args)
if CPUClass == DerivO3CPU:
    test_mem_mode = 'timing'

# Configure the system
np = args.num_cpus
system = System(
    cpu=[CPUClass(cpu_id=i) for i in range(np)],
    mem_mode=test_mem_mode,
    mem_ranges=[AddrRange('8192MB')],
    cache_line_size=args.cacheline_size
)

# Create clock domains for the system and CPUs
system.voltage_domain = VoltageDomain(voltage=args.sys_voltage)
system.clk_domain = SrcClockDomain(clock=args.sys_clock, voltage_domain=system.voltage_domain)
system.cpu_voltage_domain = VoltageDomain()
system.cpu_clk_domain = SrcClockDomain(clock=args.cpu_clock, voltage_domain=system.cpu_voltage_domain)

# Assign clock domains to CPUs
for cpu in system.cpu:
    cpu.clk_domain = system.cpu_clk_domain

# Adding a static branch predictor to the DerivO3CPU
for cpu in system.cpu:
    cpu.branchPred = BranchPredictor(predictor="StaticPred")

# Configuring the CPU to allow issuing multiple instructions per cycle
for cpu in system.cpu:
    cpu.issueWidth = 4  # Set the issue width to 4 for superscalar execution



# Configure workloads for CPUs
for i in range(np):
    if len(multiprocesses) == 1:
        system.cpu[i].workload = multiprocesses[0]
    else:
        system.cpu[i].workload = multiprocesses[i]
    system.cpu[i].createThreads()

# Enable Simultaneous Multithreading (SMT)
system.cpu[0].workload = multiprocesses
system.cpu[0].numThreads = 2  # Enable two threads for each CPU


# Enable detailed pipeline statistics collection for DerivO3CPU
if CPUClass == DerivO3CPU:
    for cpu in system.cpu:
        cpu.fetchWidth = 4
        cpu.decodeWidth = 4
        cpu.issueWidth = 4
        cpu.executeWidth = 4
        cpu.commitWidth = 4
        # Use the detailed execution tracing to gather cycle-by-cycle information
        cpu.tracer = ExeTracer()
        # Enable cycle-level tracing of the pipeline stages
        cpu.progressInterval = 1  # Dump stats every cycle

# Set up cache and memory configurations
MemClass = Simulation.setMemClass(args)
system.membus = SystemXBar()
system.system_port = system.membus.cpu_side_ports
CacheConfig.config_cache(args, system)
MemConfig.config_mem(args, system)
config_filesystem(system, args)

# Initialize the workload
system.workload = SEWorkload.init_compatible(multiprocesses[0].executable)

# Set up the root of the simulation
root = Root(full_system=False, system=system)

# Instantiate the simulation
m5.instantiate()

# Set up cycle-by-cycle simulation to dump stats for every cycle
cycle_interval = 10  # Dump stats for each cycle
total_cycles = 100  # Total number of cycles to simulate

# Run the simulation in intervals to enable detailed analysis
current_cycle = 0
while current_cycle < total_cycles:
    # Simulate for one cycle at a time
    m5.simulate(cycle_interval)

    # Dump statistics after each cycle
    m5.stats.dump()

    # Optionally reset stats to measure the next cycle separately
    m5.stats.reset()

    # Update the current cycle count
    current_cycle += cycle_interval

    # Save stats file with a unique timestamp after each cycle
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    m5out_dir = "m5out"
    stats_file_path = os.path.join(m5out_dir, "stats.txt")
    new_stats_filename = os.path.join(m5out_dir, f"stats_{timestamp}_cycle_{current_cycle}.txt")

    if os.path.exists(stats_file_path):
        os.rename(stats_file_path, new_stats_filename)
        print(f"Stats file saved as {new_stats_filename}")
    else:
        print("stats.txt not found in m5out.")

# Final stats dump after the entire simulation
m5.stats.dump()

# End the simulation
m5.simulate()

print("Detailed cycle-by-cycle simulation complete.")
