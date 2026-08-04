# Queue Behavior Contract v1

The downlink queue enforces a 5,120 MB byte cap and a 4,096 MB policy-safe threshold; there is no item-count cap. Duplicate task/product pairs are suppressed. Overflow evicts lowest-priority items. Expired entries are explicitly removed, partial bytes persist across slots and contacts, and fully transmitted items are removed. Scheduling iterates eligible sorted items, so a non-fitting item does not impose universal head-of-line blocking.

PROP and B1 share the physical queue capacity but use different method scheduling semantics. B1 full-frame fit/non-preemption is a method rule. Raw storage, separate product storage, compute queue size, and generation-request numeric capacity are not implemented as independently enforced resources and remain `UNKNOWN` or inactive as listed in the CSV.
