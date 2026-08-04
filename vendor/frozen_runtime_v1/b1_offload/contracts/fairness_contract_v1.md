# Fairness Contract v1

Within a scenario every method must share dataset/source observations, task manifest/hash, arrivals, deadlines, values, requirements, contact trace/hash, capacity multiplier definition, compute hardware, storage/queue capacities, battery and charging, initial SOC, thermal model/coefficients/thresholds/initial state, slot/horizon/order, common evaluator, and external resource trace.

Methods may naturally differ in model/product selection, compute/generation timing, downlink scheduling, queue trajectories, actual SOC/thermal state, actual traffic, closures, utility, and latency.

Method override files may contain only method decisions. Experiment files may select experiment ID, seeds, capacities, methods, output, reporting, and smoke/full mode. Dataset, tasks, contacts, environment resources, products, and evaluator cannot be overridden.

Future multi-satellite comparisons must share satellite count and per-satellite hardware/battery/thermal limits, source satellite, ISL topology/rate, and charge ISL TX, RX, waiting, and remote compute. Offloading cannot be free or instantaneous, and a task earns closure once.
