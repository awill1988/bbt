use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[derive(Debug, Clone, Default)]
pub struct ResourceStats {
    pub cpu_percent: f32,
    pub gpu_percent: f32,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub gpu_memory_used_mb: u64,
    pub gpu_memory_total_mb: u64,
}

impl ResourceStats {
    pub fn format_combined(&self) -> String {
        format!(
            "cpu:{:>5.0}% gpu:{:>4.0}% mem:{:>5.1}g vram:{:>5.1}g",
            self.cpu_percent,
            self.gpu_percent,
            self.memory_used_mb as f32 / 1024.0,
            self.gpu_memory_used_mb as f32 / 1024.0,
        )
    }
}

pub struct ResourceMonitor {
    stats: Arc<std::sync::Mutex<ResourceStats>>,
    running: Arc<AtomicBool>,
    thread_handle: Option<thread::JoinHandle<()>>,
}

impl ResourceMonitor {
    pub fn new() -> Self {
        let stats = Arc::new(std::sync::Mutex::new(ResourceStats::default()));
        let running = Arc::new(AtomicBool::new(true));

        let stats_clone = Arc::clone(&stats);
        let running_clone = Arc::clone(&running);

        let thread_handle = thread::spawn(move || {
            let pid = std::process::id();
            let mut prev_cpu_time = 0u64;
            let mut prev_timestamp = std::time::Instant::now();

            while running_clone.load(Ordering::Relaxed) {
                // sample cpu usage
                let cpu_percent = sample_cpu_usage(pid, &mut prev_cpu_time, &mut prev_timestamp);

                // sample memory usage
                let (memory_used_mb, memory_total_mb) = sample_memory_usage(pid);

                // sample gpu usage (nvidia only)
                let (gpu_percent, gpu_memory_used_mb, gpu_memory_total_mb) = sample_gpu_usage();

                if let Ok(mut guard) = stats_clone.lock() {
                    guard.cpu_percent = cpu_percent;
                    guard.memory_used_mb = memory_used_mb;
                    guard.memory_total_mb = memory_total_mb;
                    guard.gpu_percent = gpu_percent;
                    guard.gpu_memory_used_mb = gpu_memory_used_mb;
                    guard.gpu_memory_total_mb = gpu_memory_total_mb;
                }

                thread::sleep(Duration::from_millis(500)); // sample every 500ms
            }
        });

        Self {
            stats,
            running,
            thread_handle: Some(thread_handle),
        }
    }

    pub fn get_stats(&self) -> ResourceStats {
        self.stats
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ResourceMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}

fn sample_cpu_usage(pid: u32, prev_cpu_time: &mut u64, prev_timestamp: &mut std::time::Instant) -> f32 {
    #[cfg(target_os = "linux")]
    {
        let stat_path = format!("/proc/{}/stat", pid);
        if let Ok(contents) = std::fs::read_to_string(&stat_path) {
            let fields: Vec<&str> = contents.split_whitespace().collect();
            if fields.len() > 14 {
                // utime + stime (fields 14 and 15, 0-indexed: 13 and 14)
                let utime: u64 = fields.get(13).and_then(|s| s.parse().ok()).unwrap_or(0);
                let stime: u64 = fields.get(14).and_then(|s| s.parse().ok()).unwrap_or(0);
                let total_cpu_time = utime + stime;

                let now = std::time::Instant::now();
                let elapsed = now.duration_since(*prev_timestamp).as_secs_f32();

                if elapsed > 0.0 && *prev_cpu_time > 0 {
                    let delta_cpu = total_cpu_time.saturating_sub(*prev_cpu_time) as f32;
                    // cpu time is in clock ticks (typically 100 per second)
                    let clock_ticks_per_sec = 100.0;
                    let cpu_percent = (delta_cpu / clock_ticks_per_sec / elapsed) * 100.0;

                    *prev_cpu_time = total_cpu_time;
                    *prev_timestamp = now;

                    return cpu_percent.min(100.0 * num_cpus::get() as f32);
                }

                *prev_cpu_time = total_cpu_time;
                *prev_timestamp = now;
            }
        }
        0.0
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = (pid, prev_cpu_time, prev_timestamp);
        0.0
    }
}

fn sample_memory_usage(pid: u32) -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        let status_path = format!("/proc/{}/status", pid);
        let mut rss_kb = 0u64;

        if let Ok(contents) = std::fs::read_to_string(&status_path) {
            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        rss_kb = value.parse().unwrap_or(0);
                        break;
                    }
                }
            }
        }

        let memory_used_mb = rss_kb / 1024;

        // get total system memory
        let mut memory_total_mb = 0u64;
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        let total_kb: u64 = value.parse().unwrap_or(0);
                        memory_total_mb = total_kb / 1024;
                        break;
                    }
                }
            }
        }

        (memory_used_mb, memory_total_mb)
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        (0, 0)
    }
}

fn sample_gpu_usage() -> (f32, u64, u64) {
    let nvidia_smi_paths = vec![
        "/usr/bin/nvidia-smi",
        "/usr/lib/wsl/lib/nvidia-smi",
    ];

    for nvidia_smi_path in nvidia_smi_paths {
        let output = Command::new(nvidia_smi_path)
            .args(&[
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                if let Ok(stdout) = String::from_utf8(output.stdout) {
                    let line = stdout.lines().next().unwrap_or("");
                    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

                    if parts.len() == 3 {
                        let gpu_percent: f32 = parts[0].parse().unwrap_or(0.0);
                        let memory_used_mb: u64 = parts[1].parse().unwrap_or(0);
                        let memory_total_mb: u64 = parts[2].parse().unwrap_or(0);

                        return (gpu_percent, memory_used_mb, memory_total_mb);
                    }
                }
            }
        }
    }

    (0.0, 0, 0)
}
