BIN := ./build/biNRC
JSONS := $(shell find . -name '*.json')

.PHONY: all clean

all: $(JSONS:.json=.run)

# "Run targets": one per .json
%.run: %.json $(BIN)
	@echo ">>> Checking outputs for $<"
	@needs_run=false; \
	for out in $(basename $<)_*.hdr $(basename $<)_*.hdr.json; do \
		if [ ! -e $$out ] || [ $< -nt $$out ]; then \
			needs_run=true; \
		fi; \
	done; \
	if $$needs_run; then \
		echo ">>> Running $(BIN) $<"; \
		$(BIN) $<; \
	else \
		echo ">>> Up to date: $<"; \
	fi

# Clean only generated outputs
clean:
	@echo ">>> Cleaning generated HDRs and JSON outputs"
	@for cfg in $(JSONS); do \
		base=$${cfg%.json}; \
		rm -f $$base_*.hdr $$base_*.hdr.json; \
	done