# Chess-to-Music Makefile
# Variables
PYTHON = python3
PGN_DIR = data
OPENINGS_DIR = openings
VENV_DIR = venv

# Default target
.PHONY: help
help:
	@echo "Chess-to-Music Project Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install     Install dependencies"
	@echo "  venv        Create virtual environment"
	@echo "  setup       Create venv and install dependencies"
	@echo ""
	@echo "Running:"
	@echo "  run PGN=<file> [CONFIG=<file>]  Convert PGN file to MIDI"
	@echo "  demo                            Run demo with sample game"
	@echo "  to-mp3 MIDI=<file>              Convert MIDI to MP3"
	@echo ""
	@echo "Development:"
	@echo "  clean       Clean generated files"
	@echo "  clean-all   Clean all generated files including audio"
	@echo "  list-games  List available PGN files"
	@echo "  list-midi   List generated MIDI files"
	@echo ""
	@echo "Examples:"
	@echo "  make run PGN=data/game14-gukesh-ding.pgn"
	@echo "  make run PGN=data/game.pgn CONFIG=custom_config.yaml"
	@echo "  make to-mp3 MIDI=data/game.mid"
	@echo "  make demo"

# Installation targets
.PHONY: install
install:
	$(PYTHON) -m pip install -r requirements.txt

.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created. Activate with: source $(VENV_DIR)/bin/activate"

.PHONY: setup
setup: venv
	$(VENV_DIR)/bin/pip install -r requirements.txt
	@echo "Setup complete. Activate venv with: source $(VENV_DIR)/bin/activate"

# Running targets
.PHONY: run
run:
ifndef PGN
	@echo "Error: Please specify PGN file. Usage: make run PGN=<path_to_pgn_file> [CONFIG=<config_file>]"
	@echo "Example: make run PGN=data/game.pgn"
	@echo "Example: make run PGN=data/game.pgn CONFIG=custom_config.yaml"
	@exit 1
endif
ifdef CONFIG
	$(PYTHON) c2m.py $(PGN) --config $(CONFIG)
else
	$(PYTHON) c2m.py $(PGN)
endif

.PHONY: demo
demo:
	$(PYTHON) c2m.py data/game.pgn

.PHONY: to-mp3
to-mp3:
ifndef MIDI
	@echo "Error: Please specify MIDI file. Usage: make to-mp3 MIDI=<path_to_midi_file>"
	@echo "Example: make to-mp3 MIDI=data/game.mid"
	@exit 1
endif
	@echo "Converting $(MIDI) to MP3..."
	@BASE_NAME=$$(basename "$(MIDI)" .mid); \
	DIR_NAME=$$(dirname "$(MIDI)"); \
	WAV_FILE="$$DIR_NAME/$$BASE_NAME.wav"; \
	MP3_FILE="$$DIR_NAME/$$BASE_NAME.mp3"; \
	timidity "$(MIDI)" -Ow -o "$$WAV_FILE" && \
	lame "$$WAV_FILE" "$$MP3_FILE" && \
	rm "$$WAV_FILE" && \
	echo "Created $$MP3_FILE"

# Utility targets
.PHONY: list-games
list-games:
	@echo "Available PGN files:"
	@find $(PGN_DIR) -name "*.pgn" -type f | sort

.PHONY: list-midi
list-midi:
	@echo "Generated MIDI files:"
	@find $(PGN_DIR) -name "*_music.mid" -type f | sort

.PHONY: clean
clean:
	@echo "Cleaning generated MIDI files..."
	@find $(PGN_DIR) -name "*_music.mid" -type f -delete
	@find . -name "*.pyc" -type f -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

.PHONY: clean-all
clean-all: clean
	@echo "Cleaning all generated audio files..."
	@find $(PGN_DIR) -name "*.wav" -type f -delete
	@find $(PGN_DIR) -name "*.mp3" -type f -delete
	@find $(PGN_DIR) -name "*.mid" -type f -delete

# Validation targets
.PHONY: check-deps
check-deps:
	@echo "Checking dependencies..."
	@$(PYTHON) -c "import chess, mido, yaml; print('All dependencies available')" || echo "Missing dependencies. Run 'make install'"

.PHONY: validate-openings
validate-openings:
	@echo "Validating opening database files..."
	@for file in $(OPENINGS_DIR)/*.tsv; do \
		if [ -f "$$file" ]; then \
			echo "✓ $$file exists"; \
		else \
			echo "✗ $$file missing"; \
		fi; \
	done