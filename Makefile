# SPDX-License-Identifier: Apache-2.0
# demo_I2S — always-on KWS from an I2S microphone on Apollo510 EVB.
# Needs on PATH: nsx (pip install -r requirements.txt), arm-none-eabi-gcc,
# cmake, ninja, and SEGGER J-Link for flash / dashboard. See README.md.
.PHONY: help doctor lock sync configure build flash dashboard demo replay capture test test-core test-i2s clean

BUILD   := build/apollo510_evb
BIN     := $(BUILD)/kws_i2s.bin
DEVICE  ?= AP510NFA-CBR
JLINK   ?= $(shell command -v JLinkExe || echo /opt/SEGGER/JLink/JLinkExe)
PORT    ?= 8765
# Inside the mlModelOptimization monorepo the devcontainer can supply nsx:
#   make build NSX='tools/in_container.sh nsx'
NSX     ?= nsx

help: ## List targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

doctor: ## Check that nsx finds the toolchain and J-Link
	$(NSX) doctor

lock: ## Re-resolve nsx.yml into nsx.lock (only after editing nsx.yml; needs network)
	$(NSX) lock --app-dir .

sync: ## Download the pinned modules into modules/ (~290 MB, needs network)
	$(NSX) sync --app-dir .

configure: ## Fetch modules if absent, generate cmake/nsx, boards/ and the build tree
	@test -e modules || $(MAKE) sync
	$(NSX) configure --app-dir .

build: ## Build the firmware -> build/apollo510_evb/kws_i2s.bin
	@test -f $(BUILD)/build.ninja || $(MAKE) configure
	$(NSX) build --app-dir .

flash: ## Flash build/apollo510_evb/kws_i2s.bin from the host
	@test -f $(BIN) || { echo "$(BIN) missing: run 'make build'"; exit 1; }
	@sed 's#@BIN@#$(abspath $(BIN))#' host/flash.jlink.in > $(BUILD)/flash_host.jlink
	$(JLINK) -nogui 1 -device $(DEVICE) -if SWD -speed 4000 -autoconnect 1 \
	  -commandfile $(BUILD)/flash_host.jlink

dashboard: ## Live dashboard on http://127.0.0.1:$(PORT)/ (resets the board)
	python3 host/kws_dashboard.py --reset --port $(PORT)

demo: build flash dashboard ## Build, flash, open the dashboard

replay: ## Dashboard on a recorded SWO log, no board
	python3 host/kws_dashboard.py --replay host/fixtures/kws_i2s_swo.txt --port $(PORT)

capture: ## Record the hop PCM the model receives (needs -DKWS_I2S_RTT_PCM=ON)
	python3 host/rtt_pcm_dump.py --duration 8 --out capture/hop_pcm.wav

test-core: ## Host GoogleTest for the kws_core copy (first run fetches GoogleTest)
	cmake -S kws_core -B kws_core/build
	cmake --build kws_core/build -j
	ctest --test-dir kws_core/build --output-on-failure

test-i2s: ## I2S conditioning + dashboard parser
	cmake -S tests -B build-host
	cmake --build build-host -j
	ctest --test-dir build-host --output-on-failure

test: test-core test-i2s ## All host tests (no EVB)

clean: ## Remove build trees
	rm -rf build build-host kws_core/build
