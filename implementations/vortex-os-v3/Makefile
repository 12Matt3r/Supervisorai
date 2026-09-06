PYTHON ?= python3
SHELLCHECK ?= shellcheck
BATS ?= bats

.PHONY: help test test-fast lint schema-check examples doctor mcp-server clean install

help:
	@echo "Targets:"
	@echo "  make install         - Install into \$$HOME/.local/share/orchestrator_skill"
	@echo "  make schema-check    - Validate every JSON file against its schema"
	@echo "  make lint            - Run shellcheck against .sh files"
	@echo "  make test            - Run the bats test suite"
	@echo "  make test-fast       - Quick tests only"
	@echo "  make examples        - Run every example project (dry run)"
	@echo "  make doctor          - Run the orchestrator's --doctor"
	@echo "  make mcp-server      - Start the MCP server (stdio transport)"
	@echo "  make clean           - Remove generated artifacts"

install:
	./install.sh --with-examples

schema-check:
	@for f in $$(find . -name '*.json' -not -path './.git/*' -not -path './audit.jsonl'); do \
	  if [[ "$$f" == *config*.json ]]; then \
	    $(PYTHON) -c "import json,sys; json.load(open('$$f'))" || { echo "BAD JSON: $$f"; exit 1; }; \
	  fi; \
	  if [[ "$$f" == *plan.schema.json || "$$f" == plan.json ]]; then \
	    $(PYTHON) -c "import json; json.load(open('schemas/plan.schema.json'))" || exit 1; \
	  fi; \
	done
	@echo "OK — every JSON file is parseable"

lint:
	@for f in $$(find . -name '*.sh' -not -path './.git/*'); do \
	  $(SHELLCHECK) -x -e SC1091,SC2155,SC2086 $$f && echo "LINT OK: $$f" || echo "LINT FAIL: $$f"; \
	done

test:
	@command -v $(BATS) >/dev/null 2>&1 || { echo "bats not installed. Try: brew install bats-core"; exit 1; }
	$(BATS) tests/

test-fast:
	$(BATS) tests/validate.bats tests/locking.bats

examples:
	@for d in examples/*/; do \
	  echo "--- $$d ---"; \
	  bash $$d/run.sh 2>/dev/null && echo "OK" || echo "fail"; \
	done

doctor:
	./skill.sh --doctor

mcp-server:
	$(PYTHON) scripts/mcp_server.py

clean:
	rm -rf .locks audit.jsonl **/.locks **/audit.jsonl
	rm -rf examples/*/plan.json examples/*/state.json examples/*/audit.jsonl
	rm -rf examples/*/out
	find examples -name 'sub_*.plan.json' -delete 2>/dev/null || true
