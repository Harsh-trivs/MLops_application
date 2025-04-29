import time
import logging
from subprocess import Popen, PIPE
from prometheus_client import start_http_server, Gauge, Counter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metric definitions
cpu_metrics = Gauge('cpu_avg_percent', 'Average CPU usage percentage', ['mode'])

# Disk I/O metrics
io_read_rate = Gauge('io_read_rate_kbps', 'Disk read rate in KB/s', ['device'])
io_write_rate = Gauge('io_write_rate_kbps', 'Disk write rate in KB/s', ['device'])
io_tps = Gauge('io_tps', 'Disk transactions per second', ['device'])
io_read_bytes_total = Counter('io_read_bytes_total', 'Total bytes read from disk', ['device'])
io_write_bytes_total = Counter('io_write_bytes_total', 'Total bytes written to disk', ['device'])

# Memory metrics
meminfo_metrics = {}

# Load average metrics
load_avg_1m = Gauge('load_avg_1m', 'Load average over 1 minute')
load_avg_5m = Gauge('load_avg_5m', 'Load average over 5 minutes')
load_avg_15m = Gauge('load_avg_15m', 'Load average over 15 minutes')

# Network metrics
net_rx_bytes = Counter('network_receive_bytes_total', 'Total bytes received', ['interface'])
net_tx_bytes = Counter('network_transmit_bytes_total', 'Total bytes transmitted', ['interface'])
net_rx_packets = Counter('network_receive_packets_total', 'Total packets received', ['interface'])
net_tx_packets = Counter('network_transmit_packets_total', 'Total packets transmitted', ['interface'])

# Helper functions
def is_iostat_available():
    """Check if iostat is available in the system"""
    try:
        process = Popen(['which', 'iostat'], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        return process.returncode == 0
    except Exception:
        return False

def sanitize_metric_name(name):
    """Sanitize metric name by replacing invalid characters"""
    name = name.replace('(', '_').replace(')', '_').replace(' ', '_')
    return name.lower()

# Collectors
def collect_iostat_metrics():
    if not is_iostat_available():
        logger.warning("iostat command not found. Please install sysstat package.")
        return
    
    try:
        process = Popen(['iostat', '-d', '-k'], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error(f"iostat command failed: {stderr.decode('utf-8')}")
            return
        
        output = stdout.decode('utf-8')
        lines = output.strip().split('\n')

        disk_section = False
        for i, line in enumerate(lines):
            if line.startswith('Device'):
                disk_section = True
                continue

            if disk_section and line.strip():
                parts = line.split()
                if len(parts) >= 4:
                    device = parts[0]
                    tps = float(parts[1])
                    read_kbps = float(parts[2])
                    write_kbps = float(parts[3])

                    io_tps.labels(device=device).set(tps)
                    io_read_rate.labels(device=device).set(read_kbps)
                    io_write_rate.labels(device=device).set(write_kbps)

                    # Increment counters assuming 1-second collection interval
                    io_read_bytes_total.labels(device=device).inc(read_kbps * 1024)
                    io_write_bytes_total.labels(device=device).inc(write_kbps * 1024)

        logger.info("Collected iostat metrics successfully")
    
    except Exception as e:
        logger.error(f"Error collecting iostat metrics: {str(e)}")

def collect_cpu_metrics():
    """Collect CPU usage from iostat"""
    try:
        process = Popen(['iostat', '-c'], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error(f"iostat CPU command failed: {stderr.decode('utf-8')}")
            return
        
        output = stdout.decode('utf-8')
        lines = output.strip().split('\n')

        for i, line in enumerate(lines):
            if line.startswith('avg-cpu'):
                header_line = line
                data_line = lines[i+1]
                headers = [h.strip('%') for h in header_line.split()[1:]]
                values = data_line.split()

                for header, value in zip(headers, values):
                    cpu_metrics.labels(mode=header.lower()).set(float(value))
                break

        logger.info("Collected CPU metrics successfully")
    
    except Exception as e:
        logger.error(f"Error collecting CPU metrics: {str(e)}")

def collect_meminfo_metrics():
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()

        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip().split()[0]

                metric_name = 'meminfo_' + sanitize_metric_name(key)

                if metric_name not in meminfo_metrics:
                    meminfo_metrics[metric_name] = Gauge(
                        metric_name,
                        f'Memory information field {key}'
                    )
                
                try:
                    meminfo_metrics[metric_name].set(float(value))
                except ValueError:
                    logger.warning(f"Invalid meminfo value: {line}")

        logger.info("Collected meminfo metrics successfully")
    
    except Exception as e:
        logger.error(f"Error collecting meminfo metrics: {str(e)}")

def collect_loadavg_metrics():
    try:
        with open('/proc/loadavg', 'r') as f:
            parts = f.read().split()

        load_avg_1m.set(float(parts[0]))
        load_avg_5m.set(float(parts[1]))
        load_avg_15m.set(float(parts[2]))

        logger.info("Collected loadavg metrics successfully")
    
    except Exception as e:
        logger.error(f"Error collecting load average: {str(e)}")

def collect_network_metrics():
    try:
        with open('/proc/net/dev', 'r') as f:
            lines = f.readlines()

        for line in lines[2:]:
            if ':' not in line:
                continue
            iface, data = line.split(':', 1)
            iface = iface.strip()
            fields = data.split()

            rx_bytes = int(fields[0])
            rx_packets = int(fields[1])
            tx_bytes = int(fields[8])
            tx_packets = int(fields[9])

            net_rx_bytes.labels(interface=iface).inc(rx_bytes)
            net_tx_bytes.labels(interface=iface).inc(tx_bytes)
            net_rx_packets.labels(interface=iface).inc(rx_packets)
            net_tx_packets.labels(interface=iface).inc(tx_packets)

        logger.info("Collected network metrics successfully")
    
    except Exception as e:
        logger.error(f"Error collecting network metrics: {str(e)}")

# Main loop
def main():
    start_http_server(18000)
    logger.info("Prometheus metrics server started at :18000")

    while True:
        collect_cpu_metrics()
        collect_iostat_metrics()
        collect_meminfo_metrics()
        collect_loadavg_metrics()
        collect_network_metrics()
        time.sleep(1)

if __name__ == '__main__':
    main()
