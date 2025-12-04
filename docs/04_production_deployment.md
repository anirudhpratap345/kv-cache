# Production Deployment Checklist & Best Practices

## Phase 1: Development (Week 1-2)

### Local Setup
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Start local Redis: `docker run -p 6379:6379 redis:7-alpine`
- [ ] Run basic KV cache test: `python -c "from src.redis_impl.distributed_kv_cache import DistributedKVCache; ..."`
- [ ] Verify vLLM integration: `python src/vllm_integration_test.py`
- [ ] Run benchmarks locally: `python src/benchmarks/benchmark_suite.py`

### Development Checklist
```python
# 1. Verify prefix hashing consistency
from src.core.prefix_matching import compute_prefix_hash

prefix1 = "Compare Next.js vs Remix"
hash1_a = compute_prefix_hash(prefix1)
hash1_b = compute_prefix_hash(prefix1)
assert hash1_a == hash1_b  # Must be deterministic

# 2. Test serialization round-trip
import torch
from src.core.tensor_serialization import TensorSerializer

k_tensor = torch.randn(1, 32, 2048, 64, device="cuda")
serialized = TensorSerializer.serialize(k_tensor, precision="float16", compress=True)
deserialized = TensorSerializer.deserialize(serialized, target_device="cuda")
assert torch.allclose(k_tensor.to(torch.float16), deserialized.to(torch.float16), atol=0.01)

# 3. Test cache hit/miss tracking
from src.core.base_kv_cache import LocalKVCache

cache = LocalKVCache()
cache.cache_kv("test", 0, k_tensor, k_tensor)
hit = cache.get_cached_kv("test", 0) is not None
miss = cache.get_cached_kv("test", 1) is None
assert hit and miss
```

### Load Testing
```python
# src/benchmarks/load_test.py
import concurrent.futures
import time

def load_test_redis(num_concurrent=100, num_requests=1000):
    """Simulate production load."""
    from src.redis_impl.distributed_kv_cache import DistributedKVCache
    import torch
    
    cache = DistributedKVCache()
    
    def worker():
        for i in range(num_requests // num_concurrent):
            prefix = f"prefix_{i}"
            k = torch.randn(1, 32, 128, 64)
            v = torch.randn(1, 32, 128, 64)
            cache.cache_kv(prefix, 0, k, v)
            kv = cache.get_cached_kv(prefix, 0)
            assert kv is not None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        start = time.time()
        futures = [executor.submit(worker) for _ in range(num_concurrent)]
        concurrent.futures.wait(futures)
        elapsed = time.time() - start
    
    print(f"Completed {num_requests} requests in {elapsed:.2f}s")
    print(f"Throughput: {num_requests/elapsed:.0f} requests/sec")
```

---

## Phase 2: Staging (Week 3-4)

### AWS Infrastructure Setup

#### Option A: ElastiCache Redis (Recommended for < 1000 GPUs)

```bash
# 1. Create ElastiCache Redis cluster
aws elasticache create-replication-group \
  --replication-group-description "KV Cache for LLM" \
  --engine redis \
  --engine-version 7.0 \
  --cache-node-type cache.r6g.2xlarge \
  --num-cache-clusters 3 \
  --automatic-failover-enabled \
  --multi-az-enabled \
  --preferred-maintenance-window "sun:05:00-sun:06:00" \
  --notification-topic-arn arn:aws:sns:region:account:alerting \
  --tags Key=Environment,Value=Production Key=Service,Value=llm-serving

# 2. Create security group
aws ec2 create-security-group \
  --group-name kv-cache-sg \
  --description "Security group for KV cache"

aws ec2 authorize-security-group-ingress \
  --group-name kv-cache-sg \
  --protocol tcp \
  --port 6379 \
  --source-security-group-id sg-gpu-servers

# 3. Get endpoint
aws elasticache describe-replication-groups \
  --replication-group-id kv-cache-prod \
  --query 'ReplicationGroups[0].PrimaryEndpoint'
```

#### Option B: Self-Hosted Redis (More Control)

```bash
# 1. Launch EC2 instances (r6i.4xlarge = 512 GB RAM)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type r6i.4xlarge \
  --key-name my-key \
  --security-group-ids sg-kv-cache \
  --count 3 \
  --block-device-mappings DeviceName=/dev/sda1,Ebs={VolumeSize=1000,VolumeType=gp3}

# 2. Install Redis Cluster
for node in node1 node2 node3; do
  ssh -i my-key.pem ubuntu@$node <<'EOF'
    sudo apt-get update
    sudo apt-get install -y redis-server
    sudo systemctl enable redis-server
    
    # Configure Redis Cluster
    echo "cluster-enabled yes" | sudo tee -a /etc/redis/redis.conf
    echo "cluster-config-file /var/lib/redis/nodes.conf" | sudo tee -a /etc/redis/redis.conf
    echo "cluster-node-timeout 5000" | sudo tee -a /etc/redis/redis.conf
    echo "maxmemory 450gb" | sudo tee -a /etc/redis/redis.conf
    
    sudo systemctl restart redis-server
EOF
done

# 3. Create cluster
redis-cli --cluster create \
  node1:6379 node2:6379 node3:6379 \
  --cluster-replicas 1
```

### Configuration

```python
# config/kv_cache_config.py

import os

class KVCacheConfig:
    """Production configuration."""
    
    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
    REDIS_SSL = os.getenv("REDIS_SSL", "false") == "true"
    
    # Cache behavior
    PRECISION = os.getenv("CACHE_PRECISION", "float16")  # float32, float16, bfloat16
    COMPRESS = os.getenv("CACHE_COMPRESS", "true") == "true"
    TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 86400))  # 24 hours
    MAX_CACHE_SIZE_GB = float(os.getenv("CACHE_MAX_SIZE_GB", 100.0))
    
    # Monitoring
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true") == "true"
    METRICS_PORT = int(os.getenv("METRICS_PORT", 9090))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "json"  # For ELK stack integration
    
    # Fault tolerance
    RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", 3))
    RETRY_BACKOFF_MS = int(os.getenv("RETRY_BACKOFF_MS", 100))
    FALLBACK_TO_COMPUTATION = os.getenv("FALLBACK_TO_COMPUTATION", "true") == "true"

# Environment file: .env.prod
"""
REDIS_HOST=kv-cache-prod.xxxxx.ng.0001.use1.cache.amazonaws.com
REDIS_PORT=6379
CACHE_PRECISION=float16
CACHE_COMPRESS=true
CACHE_TTL_SECONDS=86400
ENABLE_METRICS=true
LOG_LEVEL=INFO
"""
```

### Monitoring & Observability

#### Prometheus Metrics

```python
# src/monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge
import time

# Cache metrics
cache_hits = Counter('kv_cache_hits_total', 'Total cache hits')
cache_misses = Counter('kv_cache_misses_total', 'Total cache misses')
cache_evictions = Counter('kv_cache_evictions_total', 'Total evictions')
cache_size_bytes = Gauge('kv_cache_size_bytes', 'Current cache size')

cache_latency = Histogram(
    'kv_cache_latency_seconds',
    'Cache operation latency',
    buckets=(0.001, 0.01, 0.1, 1.0)
)

redis_latency = Histogram(
    'redis_operation_latency_seconds',
    'Redis operation latency',
    buckets=(0.001, 0.01, 0.1, 1.0)
)

hit_rate = Gauge('kv_cache_hit_rate', 'Cache hit rate percentage')

# Usage
def tracked_cache_get(cache, prefix, layer):
    """Wrapper with metrics."""
    start = time.time()
    try:
        kv = cache.get_cached_kv(prefix, layer)
        elapsed = time.time() - start
        
        if kv:
            cache_hits.inc()
        else:
            cache_misses.inc()
        
        cache_latency.observe(elapsed)
        return kv
    except Exception as e:
        cache_misses.inc()
        raise
```

#### Logging

```python
# config/logging_config.py

import logging
import logging.handlers
import json
import sys

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

def setup_logging():
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # File handler (for ElasticSearch/ELK)
    file_handler = logging.handlers.RotatingFileHandler(
        'kv_cache.log',
        maxBytes=1000000,
        backupCount=10,
    )
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)

# Initialize on startup
setup_logging()
logger = logging.getLogger(__name__)
```

### Health Checks

```python
# src/health_check.py

from src.redis_impl.distributed_kv_cache import DistributedKVCache
import time

def health_check_redis(cache: DistributedKVCache):
    """Check Redis connectivity and performance."""
    health = {
        "timestamp": time.time(),
        "status": "healthy",
        "checks": {}
    }
    
    # 1. Connectivity check
    try:
        result = cache.health_check()
        health["checks"]["redis_connectivity"] = result["status"]
        health["checks"]["redis_latency_ms"] = result.get("latency_ms", -1)
    except Exception as e:
        health["status"] = "unhealthy"
        health["checks"]["redis_connectivity"] = "failed"
        health["checks"]["error"] = str(e)
    
    # 2. Memory check
    try:
        memory = cache.get_memory_usage()
        percent_used = memory.get("percent_used", 0)
        health["checks"]["memory_percent_used"] = percent_used
        
        if percent_used > 90:
            health["status"] = "degraded"
            health["checks"]["memory_warning"] = "Cache > 90% full"
    except Exception as e:
        health["checks"]["memory_check"] = "failed"
    
    # 3. Performance check (write + read)
    try:
        import torch
        start = time.time()
        
        k = torch.randn(1, 32, 128, 64)
        v = torch.randn(1, 32, 128, 64)
        cache.cache_kv("health_check", 0, k, v)
        
        kv = cache.get_cached_kv("health_check", 0)
        assert kv is not None
        
        elapsed = time.time() - start
        health["checks"]["round_trip_latency_ms"] = elapsed * 1000
        
        if elapsed > 1.0:  # 1 second is too slow
            health["status"] = "degraded"
            health["checks"]["latency_warning"] = "Slow round-trip"
    except Exception as e:
        health["status"] = "unhealthy"
        health["checks"]["performance_check"] = "failed"
    
    return health
```

### Alerting

```python
# config/alerts.yaml (Prometheus)

groups:
  - name: kv_cache_alerts
    rules:
      - alert: HighCacheMissRate
        expr: kv_cache_hit_rate < 60
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Cache hit rate below 60%"
          description: "Hit rate is {{ $value }}%"
      
      - alert: CacheFull
        expr: kv_cache_size_bytes > kv_cache_max_bytes * 0.9
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Cache is > 90% full"
      
      - alert: RedisUnavailable
        expr: redis_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis is unavailable"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, kv_cache_latency_seconds) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Cache p95 latency > 1s"
```

---

## Phase 3: Production Deployment

### Blue-Green Deployment

```bash
# 1. Deploy to green environment
kubectl apply -f kv-cache-green.yaml

# 2. Run smoke tests
python scripts/smoke_tests.py --namespace kv-cache-green

# 3. Switch traffic
kubectl patch service kv-cache-lb \
  -p '{"spec":{"selector":{"environment":"green"}}}'

# 4. Monitor for issues (30 min)
watch 'kubectl logs -f deployment/kv-cache-green'

# 5. If all good, clean up blue
kubectl delete -f kv-cache-blue.yaml
```

### Scaling Policy

```yaml
# kv-cache-scaling.yaml (Kubernetes)

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: kv-cache-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: kv-cache-server
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    - type: Pods
      pods:
        metric:
          name: kv_cache_hit_rate
        target:
          type: AverageValue
          averageValue: "0.85"  # Target 85% hit rate
```

### Disaster Recovery

#### Backup & Restore

```bash
# Automated daily backup
*/2 * * * * redis-cli BGSAVE

# Store in S3
aws s3 sync /var/lib/redis/ s3://kv-cache-backups/daily/

# Restore from backup
aws s3 cp s3://kv-cache-backups/daily/dump.rdb ./
redis-cli SHUTDOWN
cp dump.rdb /var/lib/redis/
redis-server /etc/redis/redis.conf
```

#### Failover Testing

```bash
#!/bin/bash
# scripts/test_failover.sh

set -e

echo "Testing failover..."

# 1. Get current primary
PRIMARY=$(redis-cli ROLE | head -1)
echo "Primary: $PRIMARY"

# 2. Simulate primary failure
echo "Killing primary..."
redis-cli -p 6379 SHUTDOWN NOSAVE

# 3. Verify failover happens automatically
sleep 5
NEW_PRIMARY=$(redis-cli -p 6380 ROLE | head -1)
echo "New primary: $NEW_PRIMARY"

if [ "$NEW_PRIMARY" != "$PRIMARY" ]; then
    echo "✓ Failover successful"
else
    echo "✗ Failover failed"
    exit 1
fi
```

---

## Operational Playbooks

### Playbook 1: Cache Hit Rate Too Low (< 50%)

```markdown
## Problem: Cache hit rate dropped below 50%

### Root Causes (in order of likelihood)
1. User prompts became more diverse (not repeatable)
2. TTL too short (entries expiring)
3. Redis ran out of memory (evicting too much)
4. Bug in prefix hashing

### Investigation
```bash
# 1. Check Redis memory
redis-cli INFO memory | grep used_memory_human

# 2. Check TTL settings
redis-cli CONFIG GET maxmemory-policy

# 3. Look at cache hit rate time series
# (In Grafana): kv_cache_hit_rate over last 24h
```

### Fix
```python
# Likely fix: Increase Redis memory or TTL
cache = DistributedKVCache(
    max_cache_size_gb=500,  # Was 100
    ttl_seconds=172800,  # Was 86400 (now 48 hours)
)
```
```

### Playbook 2: High Latency Spikes

```markdown
## Problem: Cache latency spiking to > 500ms

### Root Causes
1. Redis under memory pressure (swapping to disk)
2. Network congestion
3. GC pause in Redis
4. GPU waiting for cache lookup completion

### Investigation
```bash
# 1. Check Redis CPU and memory
redis-cli INFO stats | grep instantaneous_

# 2. Check network latency to Redis
ping -c 100 redis-host | tail -1

# 3. Check slow log
redis-cli SLOWLOG GET 10

# 4. Monitor GPU wait times
nvidia-smi dmon  # Look for memory waits
```

### Fix
```python
# Add exponential backoff + fallback
def get_kv_with_fallback(cache, prefix, layer, model, tokenizer):
    for attempt in range(3):
        try:
            kv = cache.get_cached_kv(prefix, layer)
            if kv:
                return kv
        except TimeoutError:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
    
    # Fallback to computation
    return compute_kv_on_gpu(prefix, model, tokenizer)
```
```

---

## Maintenance Schedule

| Task | Frequency | Owner |
|------|-----------|-------|
| Monitor dashboards | Every 4 hours | On-call |
| Review cache hit rate | Daily | Platform team |
| Rotate Redis AOF | Weekly | DevOps |
| Backup Redis | Daily | Automated |
| Update Redis version | Monthly | DevOps (during maintenance window) |
| Capacity planning | Quarterly | Infrastructure |
| Disaster recovery drill | Quarterly | DevOps |
| Performance review | Monthly | Platform team |

---

## Cost Optimization

### Monitor Real Costs

```python
# Calculate actual ROI

monthly_compute_cost_without_cache = 50000  # 100 GPUs × 10/hour × 730 hours

monthly_cache_cost = 5000  # Redis + storage

hit_rate = 0.85
compute_reduction = 1 - (1 - hit_rate) ** 1.5  # Super-linear with compound effects
monthly_compute_savings = monthly_compute_cost_without_cache * compute_reduction

net_monthly_benefit = monthly_compute_savings - monthly_cache_cost

print(f"Monthly savings: ${net_monthly_benefit:,.0f}")
print(f"Annual savings: ${net_monthly_benefit * 12:,.0f}")
```

### Cost Reduction Opportunities

1. **Tune TTL aggressively** - Don't cache for 24h if access pattern is 4h
2. **Compression always ON** - 50% network bandwidth savings
3. **Size prefixes appropriately** - Don't cache full 32K context if only 4K is reused
4. **Use DragonflyDB** - 30% cheaper than Redis with same speed
5. **Schedule batch jobs** - Use cheaper off-peak GPU hours

---

**Next steps:**
1. Follow Phase 1 checklist this week
2. Set up staging environment
3. Run production benchmarks
4. Deploy to small prod subset (5% of traffic)
5. Monitor for 1 week
6. Scale to full prod if all good
