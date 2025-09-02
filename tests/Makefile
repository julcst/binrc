BIN := build/biNRC

# Only take config JSONs (no "_" before .json)
JSONS := $(shell find . -type f -name '*.json' ! -name '*_*.json')

.PHONY: all clean

all: $(JSONS:.json=.run)

# Rule: for each ./foo.json create ./foo.run
%.run: %.json ../$(BIN)
	@echo ">>> Checking outputs for $<"
	@needs_run=false; \
	for out in $(basename $<)_*.hdr $(basename $<)_*.hdr.json; do \
		if [ ! -e $$out ] || [ $< -nt $$out ]; then \
			needs_run=true; \
		fi; \
	done; \
	if $$needs_run; then \
		echo ">>> Running $(BIN) $<"; \
		(cd .. && $(BIN) tests/$<); \
	else \
		echo ">>> Up to date: $<"; \
	fi

clean:
	@echo ">>> Cleaning generated HDRs and JSON outputs"
	@for cfg in $(JSONS); do \
		base=$${cfg%.json}; \
		rm -f $$base_*.hdr $$base_*.hdr.json $$base.run; \
	done