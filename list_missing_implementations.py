#!/usr/bin/env python3
"""
List all missing Layer3b curve generator implementations.

Scans all 31 archetypes and reports which curve generators still need to be implemented.
"""

from layer3b.archetype_configs import ARCHETYPES

# Currently implemented types (keep this updated!)
IMPLEMENTED_PITCH = {
    'exponential_gliss', 'ascending_spread', 'oscillating_tremor', 'cellular_sequence',
    'weak_parabolic', 'slow_drift', 'impact_transient', 'final_descent'
}

IMPLEMENTED_HARMONY = {
    'cluster_to_interval', 'unison_to_chord', 'dense_cluster', 'harmonic_stack',
    'minimal_dyad', 'collision_cluster', 'shifting_voices', 'resolving_to_root'
}

IMPLEMENTED_FILTER = {
    'bandpass_to_lowpass_choke', 'lowpass_to_highpass_open', 'bandpass_sweep', 'rhythmic_gate',
    'gentle_bandpass', 'gradual_sweep', 'impact_spike', 'closing_focus'
}

IMPLEMENTED_ENVELOPE = {
    'sudden_short_tail', 'gradual_sustained', 'gated_pulse'
}


def main():
    print("=" * 70)
    print("Layer3b Missing Implementations Report")
    print("=" * 70)

    # Collect all used types
    all_pitch = set()
    all_harmony = set()
    all_filter = set()
    all_envelope = set()

    for archetype_name, config in ARCHETYPES.items():
        all_pitch.add(config['pitch']['type'])
        all_harmony.add(config['harmony']['type'])
        all_filter.add(config['filter']['type'])
        all_envelope.add(config['envelope']['type'])

    # Calculate missing
    missing_pitch = all_pitch - IMPLEMENTED_PITCH
    missing_harmony = all_harmony - IMPLEMENTED_HARMONY
    missing_filter = all_filter - IMPLEMENTED_FILTER
    missing_envelope = all_envelope - IMPLEMENTED_ENVELOPE

    # Statistics
    total_pitch = len(all_pitch)
    total_harmony = len(all_harmony)
    total_filter = len(all_filter)
    total_envelope = len(all_envelope)

    impl_pitch = len(IMPLEMENTED_PITCH & all_pitch)
    impl_harmony = len(IMPLEMENTED_HARMONY & all_harmony)
    impl_filter = len(IMPLEMENTED_FILTER & all_filter)
    impl_envelope = len(IMPLEMENTED_ENVELOPE & all_envelope)

    print(f"\n📊 Implementation Statistics:")
    print(f"  Pitch:    {impl_pitch}/{total_pitch} implemented ({impl_pitch/total_pitch*100:.1f}%)")
    print(f"  Harmony:  {impl_harmony}/{total_harmony} implemented ({impl_harmony/total_harmony*100:.1f}%)")
    print(f"  Filter:   {impl_filter}/{total_filter} implemented ({impl_filter/total_filter*100:.1f}%)")
    print(f"  Envelope: {impl_envelope}/{total_envelope} implemented ({impl_envelope/total_envelope*100:.1f}%)")

    total_impl = impl_pitch + impl_harmony + impl_filter + impl_envelope
    total_needed = total_pitch + total_harmony + total_filter + total_envelope
    print(f"\n  Overall:  {total_impl}/{total_needed} implemented ({total_impl/total_needed*100:.1f}%)")

    # List missing implementations
    if missing_pitch:
        print(f"\n⚠ Missing Pitch Types ({len(missing_pitch)}):")
        for ptype in sorted(missing_pitch):
            # Find which archetypes use this
            users = [name for name, cfg in ARCHETYPES.items() if cfg['pitch']['type'] == ptype]
            print(f"  • {ptype}")
            print(f"    Used by: {', '.join(users)}")

    if missing_harmony:
        print(f"\n⚠ Missing Harmony Types ({len(missing_harmony)}):")
        for htype in sorted(missing_harmony):
            users = [name for name, cfg in ARCHETYPES.items() if cfg['harmony']['type'] == htype]
            print(f"  • {htype}")
            print(f"    Used by: {', '.join(users)}")

    if missing_filter:
        print(f"\n⚠ Missing Filter Types ({len(missing_filter)}):")
        for ftype in sorted(missing_filter):
            users = [name for name, cfg in ARCHETYPES.items() if cfg['filter']['type'] == ftype]
            print(f"  • {ftype}")
            print(f"    Used by: {', '.join(users)}")

    if missing_envelope:
        print(f"\n⚠ Missing Envelope Types ({len(missing_envelope)}):")
        for etype in sorted(missing_envelope):
            users = [name for name, cfg in ARCHETYPES.items() if cfg['envelope']['type'] == etype]
            print(f"  • {etype}")
            print(f"    Used by: {', '.join(users)}")

    # Fully implemented archetypes
    fully_impl = []
    for name, cfg in ARCHETYPES.items():
        if (cfg['pitch']['type'] in IMPLEMENTED_PITCH and
            cfg['harmony']['type'] in IMPLEMENTED_HARMONY and
            cfg['filter']['type'] in IMPLEMENTED_FILTER and
            cfg['envelope']['type'] in IMPLEMENTED_ENVELOPE):
            fully_impl.append(name)

    print(f"\n✓ Fully Implemented Archetypes ({len(fully_impl)}/31):")
    for name in sorted(fully_impl):
        print(f"  • {name}")

    # Priority recommendations
    if missing_pitch or missing_harmony or missing_filter or missing_envelope:
        print(f"\n📝 Implementation Priority Recommendations:")
        print(f"\n  High priority (implement these first for maximum coverage):")

        # Find most-used missing types
        pitch_usage = {}
        for ptype in missing_pitch:
            pitch_usage[ptype] = len([1 for cfg in ARCHETYPES.values() if cfg['pitch']['type'] == ptype])

        harmony_usage = {}
        for htype in missing_harmony:
            harmony_usage[htype] = len([1 for cfg in ARCHETYPES.values() if cfg['harmony']['type'] == htype])

        filter_usage = {}
        for ftype in missing_filter:
            filter_usage[ftype] = len([1 for cfg in ARCHETYPES.values() if cfg['filter']['type'] == ftype])

        # Top 3 of each
        if pitch_usage:
            top_pitch = sorted(pitch_usage.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"\n  Pitch (top 3):")
            for ptype, count in top_pitch:
                print(f"    • {ptype} (used by {count} archetypes)")

        if harmony_usage:
            top_harmony = sorted(harmony_usage.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"\n  Harmony (top 3):")
            for htype, count in top_harmony:
                print(f"    • {htype} (used by {count} archetypes)")

        if filter_usage:
            top_filter = sorted(filter_usage.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"\n  Filter (top 3):")
            for ftype, count in top_filter:
                print(f"    • {ftype} (used by {count} archetypes)")

    print("\n" + "=" * 70)

    if not any([missing_pitch, missing_harmony, missing_filter, missing_envelope]):
        print("🎉 All implementations complete!")
        return 0
    else:
        total_missing = len(missing_pitch) + len(missing_harmony) + len(missing_filter) + len(missing_envelope)
        print(f"💡 {total_missing} implementations remaining")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
