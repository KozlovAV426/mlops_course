## Отчет по конфигурации модели для тритона

### OS и версия
Ubuntu 20.04.2 LTS

### Модель CPU
AMD® Ryzen 5 4500u with radeon graphics × 6

### Количество vCPU и RAM при котором собирались метрики
6vCpu, 7,1 GIB

### Описание решаемой задачи
Классификация MNIST

### Model repository

```commandline
tree -a
.
└── onnx-mnist
    ├── 1
    │   ├── .gitkeep
    │   └── model.onnx
    └── config.pbtxt

2 directories, 3 files

```
### Эксперименты

#### Сначала с dynamic batching {} instance_group [{ count: 1, kind: KIND_CPU }]

```commandline
perf_analyzer -m onnx-mnist -u localhost:9000 --concurrency-range 8:8 --shape input:1,1,28,28
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Latency limit: 0 msec
  Concurrency limit: 8 concurrent requests
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 8
  Client:
    Request count: 51663
    Throughput: 2867.45 infer/sec
    Avg latency: 2785 usec (standard deviation 1722 usec)
    p50 latency: 2518 usec
    p90 latency: 3964 usec
    p95 latency: 5750 usec
    p99 latency: 10304 usec
    Avg HTTP time: 2774 usec (send/recv 91 usec + response wait 2683 usec)
  Server:
    Inference count: 51666
    Execution count: 51666
    Successful request count: 51666
    Avg request latency: 2233 usec (overhead 94 usec + queue 1963 usec + compute input 20 usec + compute infer 141 usec + compute output 13 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 8, throughput: 2867.45 infer/sec, latency 2785 usec

```
#### С dynamic_batching: { max_queue_delay_microseconds: 1000 } instance_group [{ count: 1, kind: KIND_CPU }]

```commandline
perf_analyzer -m onnx-mnist -u localhost:9000 --concurrency-range 8:8 --shape input:1,1,28,28
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Latency limit: 0 msec
  Concurrency limit: 8 concurrent requests
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 8
  Client:
    Request count: 51303
    Throughput: 2848.07 infer/sec
    Avg latency: 2804 usec (standard deviation 2219 usec)
    p50 latency: 2186 usec
    p90 latency: 5005 usec
    p95 latency: 7473 usec
    p99 latency: 11864 usec
    Avg HTTP time: 2794 usec (send/recv 87 usec + response wait 2707 usec)
  Server:
    Inference count: 51303
    Execution count: 51303
    Successful request count: 51303
    Avg request latency: 2098 usec (overhead 103 usec + queue 1835 usec + compute input 17 usec + compute infer 130 usec + compute output 12 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 8, throughput: 2848.07 infer/sec, latency 2804 usec

```

#### С dynamic_batching: { max_queue_delay_microseconds: 2000 } instance_group [{ count: 1, kind: KIND_CPU }]

```commandline
perf_analyzer -m onnx-mnist -u localhost:9000 --concurrency-range 8:8 --shape input:1,1,28,28
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Latency limit: 0 msec
  Concurrency limit: 8 concurrent requests
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 8
  Client:
    Request count: 48186
    Throughput: 2675.02 infer/sec
    Avg latency: 2985 usec (standard deviation 2530 usec)
    p50 latency: 2193 usec
    p90 latency: 5927 usec
    p95 latency: 8437 usec
    p99 latency: 12877 usec
    Avg HTTP time: 2973 usec (send/recv 100 usec + response wait 2873 usec)
  Server:
    Inference count: 48188
    Execution count: 48188
    Successful request count: 48188
    Avg request latency: 2238 usec (overhead 78 usec + queue 1977 usec + compute input 15 usec + compute infer 152 usec + compute output 15 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 8, throughput: 2675.02 infer/sec, latency 2985 usec

```

#### С dynamic_batching: { max_queue_delay_microseconds: 1000 } instance_group [{ count: 2, kind: KIND_CPU }]
```commandline
perf_analyzer -m onnx-mnist -u localhost:9000 --concurrency-range 8:8 --shape input:1,1,28,28
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Latency limit: 0 msec
  Concurrency limit: 8 concurrent requests
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 8
  Client:
    Request count: 44513
    Throughput: 2471.81 infer/sec
    Avg latency: 3231 usec (standard deviation 2756 usec)
    p50 latency: 2309 usec
    p90 latency: 6776 usec
    p95 latency: 8899 usec
    p99 latency: 13367 usec
    Avg HTTP time: 3219 usec (send/recv 99 usec + response wait 3120 usec)
  Server:
    Inference count: 44515
    Execution count: 44515
    Successful request count: 44515
    Avg request latency: 2561 usec (overhead 284 usec + queue 1992 usec + compute input 17 usec + compute infer 244 usec + compute output 23 usec)

[WARNING] Perf Analyzer is not able to keep up with the desired load. The results may not be accurate.
Inferences/Second vs. Client Average Batch Latency
Concurrency: 8, throughput: 2471.81 infer/sec, latency 3231 usec

```

Увеличение `instance_group` до 2 не повлияло на throughput, поэтому оптимальным оказывается лишь настройка
`dynamic_batching: { max_queue_delay_microseconds: 1000 }`

### Код
Код по конвертации находится в `commands.py` в функции `train`.

Клиент для тритона с тестом находися в `mlops_course/triton`
