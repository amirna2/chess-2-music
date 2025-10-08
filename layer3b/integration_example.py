#!/usr/bin/env python3
"""
Complete integration example for Layer 3b gesture system.

Demonstrates the unified gesture generation architecture with GestureCoordinator
as the primary interface. This example mirrors the integration pattern shown in
the layer3b_implementation.md specification.
"""

import numpy as np
from synth_engine import SubtractiveSynth
from layer3b import GestureCoordinator, ARCHETYPES


def main():
    """
    Complete integration example showing:
    1. Coordinator initialization
    2. Gesture generation for multiple archetypes
    3. Audio analysis and validation
    4. Timeline integration (simulated)
    """
    print("=" * 80)
    print("Layer 3b: Unified Gesture System Integration Example")
    print("=" * 80)

    # Configuration
    sample_rate = 88200  # 2x 44.1kHz for anti-aliasing
    seed = 42

    # Initialize RNG and synthesis engine
    print("\nInitializing synthesis engine...")
    rng = np.random.default_rng(seed=seed)
    synth_engine = SubtractiveSynth(sample_rate=sample_rate, rng=rng)
    print(f"  ✓ SubtractiveSynth initialized at {sample_rate} Hz")

    # Initialize gesture coordinator
    print("\nInitializing gesture coordinator...")
    coordinator = GestureCoordinator(rng, synth_engine=synth_engine)
    archetypes = coordinator.get_available_archetypes()
    print(f"  ✓ GestureCoordinator initialized")
    print(f"  Available archetypes: {', '.join(archetypes)}")

    # Simulate section context (from game analysis)
    # In real usage, this would come from chess game analysis
    section_context = {
        'tension': 0.7,      # High tension (critical position)
        'entropy': 0.5,      # Moderate complexity
        'scale': 'D_DORIAN', # Musical scale for section
        'key': 'D'           # Musical key
    }

    print(f"\nSection context:")
    print(f"  Tension: {section_context['tension']:.2f}")
    print(f"  Entropy: {section_context['entropy']:.2f}")
    print(f"  Scale: {section_context['scale']}")
    print(f"  Key: {section_context['key']}")

    # Simulate moment events (from move tagging)
    # In real usage, these would come from emotional_moment_tagger
    moments = [
        {
            'event_type': 'BLUNDER',
            'timestamp': 12.5,
            'move_number': 8,
            'quality': 'poor',
            'eval_drop': 3.5
        },
        {
            'event_type': 'BRILLIANT',
            'timestamp': 45.2,
            'move_number': 23,
            'quality': 'excellent',
            'eval_gain': 2.8
        },
        {
            'event_type': 'TIME_PRESSURE',
            'timestamp': 78.9,
            'move_number': 35,
            'time_left': 15.0  # 15 seconds remaining
        },
        {
            'event_type': 'TACTICAL_SEQUENCE',
            'timestamp': 102.3,
            'move_number': 42,
            'sequence_length': 5,
            'forcing_moves': True
        }
    ]

    print(f"\nProcessing {len(moments)} moment events...")
    print("=" * 80)

    # Generate gestures for each moment
    timeline_buffer = []  # Simulated timeline buffer

    for i, moment in enumerate(moments, 1):
        archetype = moment['event_type']

        print(f"\n{i}. {archetype} @ {moment['timestamp']:.1f}s (move {moment['move_number']})")
        print("-" * 80)

        # Compute expected duration (useful for timeline planning)
        duration = coordinator.compute_archetype_duration(archetype, section_context)
        print(f"   Expected duration: {duration:.2f}s")

        # Get archetype configuration (useful for debugging/introspection)
        config = coordinator.get_archetype_config(archetype)
        print(f"   Configuration:")
        print(f"     - Duration base: {config['duration_base']:.2f}s")
        print(f"     - Pitch type: {config['pitch']['type']}")
        print(f"     - Harmony type: {config['harmony']['type']}")
        print(f"     - Filter type: {config['filter']['type']}")
        print(f"     - Target RMS: {config.get('rms_target', -18.0):.1f} dBFS")
        print(f"     - Peak limit: {config.get('peak_limit', 0.8):.2f}")

        # Generate gesture audio
        print(f"   Generating audio...")
        audio = coordinator.generate_gesture(
            archetype_name=archetype,
            moment_event=moment,
            section_context=section_context,
            sample_rate=sample_rate
        )

        # Analyze generated audio
        actual_duration = len(audio) / sample_rate
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)
        dynamic_range = 20 * np.log10(peak / (rms + 1e-10))

        print(f"   ✓ Generated successfully")
        print(f"   Audio analysis:")
        print(f"     - Actual duration: {actual_duration:.2f}s")
        print(f"     - Samples: {len(audio):,}")
        print(f"     - Peak: {peak:.4f}")
        print(f"     - RMS: {rms:.4f} ({rms_db:.1f} dBFS)")
        print(f"     - Dynamic range: {dynamic_range:.1f} dB")

        # Validate audio quality
        target_rms_db = config.get('rms_target', -18.0)
        peak_limit = config.get('peak_limit', 0.8)

        rms_error = abs(rms_db - target_rms_db)
        if rms_error > 1.0:  # Allow 1dB tolerance
            print(f"     ⚠ RMS deviation: {rms_error:.1f}dB from target")

        if peak > peak_limit + 0.05:  # Allow 5% tolerance
            print(f"     ⚠ Peak {peak:.4f} exceeds limit {peak_limit:.2f}")
        else:
            print(f"     ✓ Peak within limit")

        # Simulate timeline integration
        # In real usage, this would be inserted into the Layer 3b mix buffer
        timeline_buffer.append({
            'archetype': archetype,
            'timestamp': moment['timestamp'],
            'audio': audio,
            'duration': actual_duration
        })

    # Simulate timeline placement
    print("\n" + "=" * 80)
    print("Timeline Summary:")
    print("=" * 80)
    print(f"\nTotal gestures generated: {len(timeline_buffer)}")
    print(f"Timeline span: 0.0s → {max(m['timestamp'] for m in moments):.1f}s")

    total_audio_duration = sum(item['duration'] for item in timeline_buffer)
    print(f"Total audio generated: {total_audio_duration:.2f}s")

    print("\nTimeline contents:")
    for item in timeline_buffer:
        end_time = item['timestamp'] + item['duration']
        print(f"  [{item['timestamp']:6.1f}s → {end_time:6.1f}s] "
              f"{item['archetype']:20s} ({item['duration']:.2f}s)")

    # Summary statistics
    print("\n" + "=" * 80)
    print("Integration Test Summary:")
    print("=" * 80)

    archetype_counts = {}
    for item in timeline_buffer:
        arch = item['archetype']
        archetype_counts[arch] = archetype_counts.get(arch, 0) + 1

    print("\nGestures by archetype:")
    for archetype, count in sorted(archetype_counts.items()):
        print(f"  {archetype:20s}: {count}")

    print("\nConfiguration-driven architecture benefits:")
    print("  ✓ All archetypes share same synthesis pipeline")
    print("  ✓ No duplicate code across gesture types")
    print("  ✓ Easy to add new archetypes (config dict only)")
    print("  ✓ Testable: Pure functions for curve generation")
    print("  ✓ Debuggable: Can inspect/plot curves independently")
    print("  ✓ Consistent with synth_composer/patterns/ architecture")

    print("\n" + "=" * 80)
    print("Integration example complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
