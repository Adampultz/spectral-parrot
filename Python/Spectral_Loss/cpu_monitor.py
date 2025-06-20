# cpu_monitor.py
"""
Standalone CPU and memory monitoring module for audio performance analysis.
Usage: from cpu_monitor import CPUMonitor
"""

import psutil
import time
import threading
from collections import deque
import cProfile
import pstats
import io


class CPUMonitor:
    """
    Real-time CPU usage monitor for measuring optimization impact.
    """
    
    def __init__(self, sample_interval=0.1):
        self.sample_interval = sample_interval
        self.is_monitoring = False
        self.cpu_history = deque(maxlen=1000)  # Keep last 100 seconds at 0.1s intervals
        self.memory_history = deque(maxlen=1000)
        self.monitor_thread = None
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Start CPU monitoring in background thread."""
        if self.is_monitoring:
            print("CPU monitoring already running")
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("CPU monitoring started")
        
    def stop_monitoring(self):
        """Stop CPU monitoring."""
        if not self.is_monitoring:
            print("CPU monitoring not running")
            return
            
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print("CPU monitoring stopped")
        
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                # Get CPU usage (% of one core)
                cpu_percent = self.process.cpu_percent()
                
                # Get memory usage (MB)
                memory_mb = self.process.memory_info().rss / (1024 * 1024)
                
                # Store with timestamp
                timestamp = time.time()
                self.cpu_history.append((timestamp, cpu_percent))
                self.memory_history.append((timestamp, memory_mb))
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                print(f"CPU monitoring error: {e}")
                break
                
    def get_current_stats(self):
        """Get current CPU and memory usage."""
        if not self.cpu_history:
            return None
            
        latest_cpu = self.cpu_history[-1][1]
        latest_memory = self.memory_history[-1][1]
        
        return {
            'cpu_percent': latest_cpu,
            'memory_mb': latest_memory,
            'timestamp': time.time()
        }
        
    def get_average_stats(self, seconds=10):
        """Get average CPU usage over last N seconds."""
        if not self.cpu_history:
            return None
            
        cutoff_time = time.time() - seconds
        recent_cpu = [cpu for timestamp, cpu in self.cpu_history if timestamp > cutoff_time]
        recent_memory = [mem for timestamp, mem in self.memory_history if timestamp > cutoff_time]
        
        if not recent_cpu:
            return None
            
        return {
            'avg_cpu_percent': sum(recent_cpu) / len(recent_cpu),
            'max_cpu_percent': max(recent_cpu),
            'min_cpu_percent': min(recent_cpu),
            'avg_memory_mb': sum(recent_memory) / len(recent_memory),
            'max_memory_mb': max(recent_memory),
            'min_memory_mb': min(recent_memory),
            'sample_count': len(recent_cpu)
        }
        
    def print_live_stats(self):
        """Print current stats (call periodically)."""
        current = self.get_current_stats()
        avg_10s = self.get_average_stats(10)
        
        if current and avg_10s:
            print(f"\rCPU: {current['cpu_percent']:5.1f}% | "
                  f"Mem: {current['memory_mb']:6.1f}MB | "
                  f"10s Avg: {avg_10s['avg_cpu_percent']:5.1f}% CPU, {avg_10s['avg_memory_mb']:6.1f}MB RAM   ", 
                  end='')
        elif current:
            print(f"\rCPU: {current['cpu_percent']:5.1f}% | Mem: {current['memory_mb']:6.1f}MB   ", end='')
        else:
            print(f"\rNo monitoring data available   ", end='')
            
    def print_summary_stats(self, duration=30):
        """Print a summary of stats over the specified duration."""
        stats = self.get_average_stats(duration)
        if not stats:
            print("No monitoring data available")
            return
            
        print(f"\n--- CPU Monitor Summary (last {duration}s) ---")
        print(f"CPU Usage    - Avg: {stats['avg_cpu_percent']:5.1f}%  Min: {stats['min_cpu_percent']:5.1f}%  Max: {stats['max_cpu_percent']:5.1f}%")
        print(f"Memory Usage - Avg: {stats['avg_memory_mb']:6.1f}MB Min: {stats['min_memory_mb']:6.1f}MB Max: {stats['max_memory_mb']:6.1f}MB")
        print(f"Sample Count: {stats['sample_count']}")
        
    def reset_history(self):
        """Clear monitoring history."""
        self.cpu_history.clear()
        self.memory_history.clear()
        print("CPU monitoring history cleared")


class PerformanceProfiler:
    """
    Code profiler for detailed performance analysis.
    """
    
    def __init__(self):
        self.profiler = None
        self.is_profiling = False
        
    def start_profiling(self):
        """Start code profiling."""
        if self.is_profiling:
            print("Profiling already running")
            return
            
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.is_profiling = True
        print("Performance profiling started")
        
    def stop_profiling(self):
        """Stop profiling and return stats."""
        if not self.is_profiling:
            print("Profiling not running")
            return None
            
        self.profiler.disable()
        self.is_profiling = False
        print("Performance profiling stopped")
        return self.profiler
        
    def print_top_functions(self, top_n=15):
        """Print the most time-consuming functions."""
        if not self.profiler:
            print("No profiling data available")
            return
            
        # Create string buffer for stats
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(top_n)
        
        print("\n" + "="*80)
        print(f"TOP {top_n} TIME-CONSUMING FUNCTIONS:")
        print("="*80)
        print(s.getvalue())


class AudioPerformanceMonitor:
    """
    Combined CPU monitoring and profiling for audio systems.
    Simple interface for performance testing.
    """
    
    def __init__(self):
        self.cpu_monitor = CPUMonitor(sample_interval=0.1)
        self.profiler = PerformanceProfiler()
        
    def start_monitoring(self):
        """Start CPU monitoring."""
        self.cpu_monitor.start_monitoring()
        
    def stop_monitoring(self):
        """Stop CPU monitoring."""
        self.cpu_monitor.stop_monitoring()
        
    def start_profiling(self):
        """Start code profiling."""
        self.profiler.start_profiling()
        
    def stop_profiling(self):
        """Stop profiling and show results."""
        self.profiler.stop_profiling()
        self.profiler.print_top_functions(10)
        
    def print_current_stats(self):
        """Print current CPU/memory usage."""
        stats = self.cpu_monitor.get_current_stats()
        if stats:
            print(f"Current: {stats['cpu_percent']:5.1f}% CPU, {stats['memory_mb']:6.1f}MB memory")
        else:
            print("No monitoring data available. Start monitoring first.")
            
    def run_performance_test(self, duration=30):
        """
        Run a performance test for specified duration.
        
        Args:
            duration: Test duration in seconds
        """
        print(f"\nRunning {duration}s performance test...")
        print("Press Ctrl+C to stop early")
        print("-" * 50)
        
        # Start monitoring
        was_monitoring = self.cpu_monitor.is_monitoring
        if not was_monitoring:
            self.start_monitoring()
            time.sleep(0.5)  # Let it collect initial data
        
        # Clear history for clean test
        self.cpu_monitor.reset_history()
        
        start_time = time.time()
        
        try:
            # Run test loop
            while time.time() - start_time < duration:
                self.cpu_monitor.print_live_stats()
                time.sleep(1)
                
        except KeyboardInterrupt:
            actual_duration = time.time() - start_time
            print(f"\nTest stopped early after {actual_duration:.1f}s")
            duration = actual_duration
            
        # Print results
        print(f"\n")
        self.cpu_monitor.print_summary_stats(duration)
        
        # Stop monitoring if we started it
        if not was_monitoring:
            self.stop_monitoring()
            
    def compare_before_after(self, test_duration=20):
        """
        Helper for before/after optimization comparison.
        """
        print("\n" + "="*60)
        print("BEFORE/AFTER OPTIMIZATION COMPARISON")
        print("="*60)
        
        input("\nPress Enter when ready to start BEFORE test...")
        print("\nRunning BEFORE optimization test:")
        self.run_performance_test(test_duration)
        before_stats = self.cpu_monitor.get_average_stats(test_duration)
        
        input("\nPress Enter after implementing optimization for AFTER test...")
        print("\nRunning AFTER optimization test:")
        self.run_performance_test(test_duration)
        after_stats = self.cpu_monitor.get_average_stats(test_duration)
        
        # Compare results
        if before_stats and after_stats:
            print("\n" + "="*60)
            print("COMPARISON RESULTS:")
            print("="*60)
            
            cpu_improvement = before_stats['avg_cpu_percent'] - after_stats['avg_cpu_percent']
            cpu_improvement_pct = (cpu_improvement / before_stats['avg_cpu_percent']) * 100
            
            memory_change = after_stats['avg_memory_mb'] - before_stats['avg_memory_mb']
            
            print(f"CPU Usage:")
            print(f"  Before: {before_stats['avg_cpu_percent']:5.1f}% (peak: {before_stats['max_cpu_percent']:5.1f}%)")
            print(f"  After:  {after_stats['avg_cpu_percent']:5.1f}% (peak: {after_stats['max_cpu_percent']:5.1f}%)")
            print(f"  Change: {cpu_improvement:+5.1f}% ({cpu_improvement_pct:+5.1f}%)")
            
            print(f"\nMemory Usage:")
            print(f"  Before: {before_stats['avg_memory_mb']:6.1f}MB")
            print(f"  After:  {after_stats['avg_memory_mb']:6.1f}MB") 
            print(f"  Change: {memory_change:+6.1f}MB")
            
            if cpu_improvement > 0:
                print(f"\n✅ Optimization successful! CPU usage reduced by {cpu_improvement_pct:.1f}%")
            else:
                print(f"\n⚠️  CPU usage increased by {abs(cpu_improvement_pct):.1f}%")


# Simple command line interface for testing
if __name__ == "__main__":
    monitor = AudioPerformanceMonitor()
    
    print("Audio Performance Monitor")
    print("Commands: [start] [stop] [test] [current] [profile] [quit]")
    
    try:
        while True:
            cmd = input("\n> ").lower().strip()
            
            if cmd == 'start':
                monitor.start_monitoring()
            elif cmd == 'stop':
                monitor.stop_monitoring()
            elif cmd == 'test':
                duration = input("Test duration (default 30s): ").strip()
                duration = int(duration) if duration else 30
                monitor.run_performance_test(duration)
            elif cmd == 'current':
                monitor.print_current_stats()
            elif cmd == 'profile':
                if monitor.profiler.is_profiling:
                    monitor.stop_profiling()
                else:
                    monitor.start_profiling()
            elif cmd == 'compare':
                monitor.compare_before_after()
            elif cmd in ['quit', 'q', 'exit']:
                break
            else:
                print("Commands: start, stop, test, current, profile, compare, quit")
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        monitor.stop_monitoring()