# Aborted implementation attempt

The initial registered runner stopped before producing any candidate prediction packet or metric: pandas 3 returned a read-only label array, and converting negative labels in place raised ValueError. All 84 scheduled tasks aborted with this same issue. No results informed any recipe changes. The versioned replacement is ../other_combiner_v2; it requests a writable NumPy copy. All seven model definitions and statistical settings are unchanged. Original protocol and source remain here.
