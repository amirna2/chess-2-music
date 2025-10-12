#!/usr/bin/env python3
"""
Layer3b Spectromorphological Analysis Tool

Generates detailed JSON analysis showing how each chess moment from tags JSON
will be converted to synthesized audio through Layer3b gesture system.

Usage:
    python3 analyze_layer3b_moments.py tags-game1.json output-analysis.json

Output includes:
- Archetype mapping for each moment
- Spectromorphological characteristics
- Synthesis parameters (pitch, harmony, filter, envelope, texture)
- Phase timeline breakdown
- Duration calculations
"""

import json
import sys
from typing import Dict, Any, List
from layer3b.archetype_configs import ARCHETYPES


def analyze_moment(moment: Dict[str, Any], section_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a single moment and generate detailed spectromorphological breakdown.

    Args:
        moment: Moment dict with keys: ply, second, score, type, side
        section_context: Section dict with keys: name, narrative, tension

    Returns:
        Detailed analysis dict with archetype info and synthesis parameters
    """
    moment_type = moment.get('type', 'MOVE')

    # Get archetype configuration
    if moment_type not in ARCHETYPES:
        return {
            'moment': moment,
            'error': f"Unknown moment type: {moment_type}",
            'archetype': None
        }

    archetype = ARCHETYPES[moment_type]

    # Compute estimated duration
    # Use section tension for context (we don't have entropy in tags JSON yet)
    tension = section_context.get('tension', 0.5)
    entropy = 0.5  # Default since tags JSON doesn't include entropy

    duration_base = archetype['duration_base']
    duration_tension = archetype.get('duration_tension_scale', 0.0) * tension
    duration_entropy = archetype.get('duration_entropy_scale', 0.0) * entropy
    estimated_duration = max(0.5, min(10.0, duration_base + duration_tension + duration_entropy))

    # Build detailed analysis
    analysis = {
        'moment_info': {
            'ply': moment.get('ply'),
            'timestamp_seconds': moment.get('second'),
            'move_side': moment.get('side'),
            'score': moment.get('score'),
            'moment_type': moment_type
        },
        'section_context': {
            'section_name': section_context.get('name'),
            'narrative': section_context.get('narrative'),
            'tension': tension
        },
        'archetype': {
            'name': moment_type,
            'duration_seconds': round(estimated_duration, 2),
            'morphology': archetype.get('morphology', {}),
            'peak_limit': archetype.get('peak_limit', 0.8),
            'rms_target_db': archetype.get('rms_target', -18.0)
        },
        'synthesis_parameters': {
            'pitch': {
                'type': archetype['pitch']['type'],
                'description': _describe_pitch_trajectory(archetype['pitch'])
            },
            'harmony': {
                'type': archetype['harmony']['type'],
                'num_voices': archetype['harmony'].get('num_voices',
                                                       archetype['harmony'].get('num_voices_base', 1)),
                'description': _describe_harmony(archetype['harmony'])
            },
            'filter': {
                'type': archetype['filter']['type'],
                'description': _describe_filter(archetype['filter'])
            },
            'envelope': {
                'type': archetype['envelope']['type'],
                'description': _describe_envelope(archetype['envelope'])
            },
            'texture': {
                'noise_ratio': archetype['texture']['noise_ratio_base'],
                'noise_type': archetype['texture']['noise_type'],
                'shimmer': archetype['texture'].get('shimmer_enable', False)
            }
        },
        'phase_structure': archetype['phases'],
        'spectromorphology': {
            'archetype': archetype.get('morphology', {}).get('spectromorphological_archetype', 'Unknown'),
            'gesture_class': archetype.get('morphology', {}).get('gesture_class', 'Unknown'),
            'motion_type': archetype.get('morphology', {}).get('motion_type', 'Unknown')
        }
    }

    return analysis


def _describe_pitch_trajectory(pitch_config: Dict[str, Any]) -> str:
    """Generate human-readable description of pitch trajectory."""
    ptype = pitch_config['type']

    descriptions = {
        'stable': 'Stable center frequency with minimal drift',
        'exponential_gliss': f"Exponential glissando dropping {pitch_config.get('octave_drop_base', 2)} octaves",
        'ascending_spread': f"Ascending spread rising {pitch_config.get('octave_rise', 2)} octaves",
        'oscillating_tremor': f"Tremor oscillating at {pitch_config.get('tremor_rate_base_hz', 8)}Hz",
        'parabolic': 'Parabolic arc (rise and fall)',
        'slow_drift': f"Slow drift {pitch_config.get('drift_direction', 'ascending')} {pitch_config.get('drift_semitones', 7)} semitones",
        'converging_steps': f"Converging steps over {pitch_config.get('num_steps', 6)} discrete pitches",
        'sustained_drone': f"Sustained drone at {pitch_config.get('center_freq_base', 220)}Hz with micro-drift",
        'weak_parabolic': f"Gentle parabolic rise {pitch_config.get('rise_semitones', 3)}st, fall {pitch_config.get('fall_semitones', 4)}st",
        'wandering_spiral': f"Wandering spiral at {pitch_config.get('spiral_rate_hz', 3.5)}Hz",
        'stable_rise': f"Stable rise {pitch_config.get('octave_rise', 0.9)} octaves",
        'aggressive_rise': f"Aggressive rise {pitch_config.get('octave_rise', 1.9)} octaves",
        'reciprocal_pair': f"Reciprocal pair: {pitch_config.get('freq_1', 330)}Hz ↔ {pitch_config.get('freq_2', 660)}Hz",
        'dual_iteration': f"Dual iteration at {pitch_config.get('iteration_rate_hz', 10)}Hz",
        'focused_center': f"Focused center at {pitch_config.get('center_freq_base', 550)}Hz",
        'impact_transient': f"Impact transient at {pitch_config.get('strike_freq_base', 660)}Hz",
        'divergent_split': f"Divergent split ±{pitch_config.get('divergence_semitones', 13)} semitones",
        'gentle_descent': f"{pitch_config.get('start_freq', 550)}Hz → {pitch_config.get('end_freq', 440)}Hz",
        'chaotic_iteration': f"Chaotic iteration at {pitch_config.get('iteration_rate_hz', 17)}Hz, chaos {pitch_config.get('chaos_amount', 0.85)}",
        'dual_descent': f"Dual descent converging to {pitch_config.get('convergence_point', 330)}Hz",
        'final_descent': f"Final descent to {pitch_config.get('end_freq', 110)}Hz ({pitch_config.get('descent_curve', 'logarithmic')})"
    }

    return descriptions.get(ptype, f"Type: {ptype}")


def _describe_harmony(harmony_config: Dict[str, Any]) -> str:
    """Generate human-readable description of harmony."""
    htype = harmony_config['type']

    descriptions = {
        'simple_unison': f"{harmony_config.get('num_voices', 2)} voices in unison",
        'cluster_to_interval': f"Cluster ({harmony_config.get('num_voices_base', 3)} voices) → {harmony_config.get('resolve_interval_type', 'interval')}",
        'unison_to_chord': f"Unison → {harmony_config.get('chord_type', 'chord')} ({harmony_config.get('num_voices', 4)} voices)",
        'dense_cluster': f"Dense microtonal cluster ({harmony_config.get('num_voices', 5)} voices)",
        'converging_cluster': f"Converging cluster {harmony_config.get('num_voices_start', 6)} → {harmony_config.get('num_voices_end', 1)} voices",
        'minimal_dyad': f"Minimal dyad ({harmony_config.get('interval_semitones', 3)} semitones)",
        'shifting_cluster': f"Shifting cluster ({harmony_config.get('num_voices', 3)} voices, detune randomness {harmony_config.get('detune_randomness', 0.6)})",
        'controlled_expansion': f"Controlled expansion ({harmony_config.get('num_voices', 4)} voices, {harmony_config.get('interval_type', 'perfect_fifths')})",
        'dense_agglomeration': f"Dense agglomeration ({harmony_config.get('num_voices', 7)} voices, density {harmony_config.get('cluster_density', 0.85)})",
        'dual_voices': f"Dual voices (mirror motion: {harmony_config.get('mirror_motion', True)})",
        'grounded_dyad': f"Grounded dyad with bass emphasis",
        'balanced_spectrum': f"Balanced spectrum ({harmony_config.get('num_voices', 5)} voices, {harmony_config.get('interval_balance', 'equal_spacing')})",
        'flowing_voices': f"Flowing voices ({harmony_config.get('num_voices', 3)} voices, {harmony_config.get('voice_motion', 'arced')} motion)",
        'expanding_formation': f"Expanding formation {harmony_config.get('num_voices_start', 1)} → {harmony_config.get('num_voices_end', 4)} voices",
        'forceful_stack': f"Forceful stack ({harmony_config.get('num_voices', 4)} voices, power chord)",
        'reinforced_pair': f"Reinforced pair ({harmony_config.get('num_voices', 4)} voices, coupling {harmony_config.get('coupling_strength', 0.82)})",
        'centered_cluster': f"Centered cluster ({harmony_config.get('num_voices', 5)} voices, {harmony_config.get('spectral_focus', 'midrange')})",
        'collision_cluster': f"Collision cluster ({harmony_config.get('num_voices', 3)} voices, density {harmony_config.get('impact_density', 0.9)})",
        'fragmented_voices': f"Fragmented voices ({harmony_config.get('num_voices', 4)} voices, divergence {harmony_config.get('divergence_factor', 0.72)})",
        'resolving_dyad': f"Resolving dyad (interval {harmony_config.get('resolution_interval', 5)})",
        'granular_cluster': f"Granular cluster ({harmony_config.get('num_voices', 7)} voices, micro-detune, chaos {harmony_config.get('chaos_factor', 0.8)})",
        'thinning_voices': f"Thinning voices {harmony_config.get('num_voices_start', 6)} → {harmony_config.get('num_voices_end', 2)}",
        'transforming_chord': f"Transforming chord {harmony_config.get('start_voices', 2)} → {harmony_config.get('end_voices', 5)} voices",
        'resolving_cluster': f"Resolving cluster {harmony_config.get('num_voices_start', 5)} → {harmony_config.get('num_voices_end', 1)}",
        'resolving_to_root': f"Resolving to root {harmony_config.get('num_voices_start', 5)} → 1 ({harmony_config.get('resolution_chord', 'tonic')})"
    }

    return descriptions.get(htype, f"Type: {htype}")


def _describe_filter(filter_config: Dict[str, Any]) -> str:
    """Generate human-readable description of filter."""
    ftype = filter_config['type']

    descriptions = {
        'simple_lowpass': f"Simple lowpass {filter_config.get('lp_cutoff_base', 1200)}Hz",
        'bandpass_to_lowpass_choke': f"Bandpass {filter_config.get('bp_center_base', 1000)}Hz → lowpass choke {filter_config.get('lp_cutoff_base', 100)}Hz",
        'lowpass_to_highpass_open': f"Lowpass {filter_config.get('lp_cutoff_start', 300)}Hz → highpass {filter_config.get('hp_cutoff_end', 3000)}Hz (opening)",
        'bandpass_sweep': f"Bandpass sweep {filter_config.get('bp_center_start', 500)}Hz → {filter_config.get('bp_center_end', 2000)}Hz",
        'gradual_sweep': f"Gradual sweep {filter_config.get('lp_start', 800)}Hz → {filter_config.get('lp_end', 1900)}Hz ({filter_config.get('sweep_curve', 'linear')})",
        'focusing_narrowband': f"Focusing narrowband {filter_config.get('bp_start_width', 1400)}Hz → {filter_config.get('bp_end_width', 250)}Hz @ {filter_config.get('center_freq', 880)}Hz",
        'static_bandpass': f"Static bandpass @ {filter_config.get('bp_center', 600)}Hz, Q={filter_config.get('resonance', 0.65)}",
        'gentle_bandpass': f"Gentle bandpass @ {filter_config.get('bp_center_base', 950)}Hz",
        'controlled_opening': f"Controlled opening {filter_config.get('lp_start', 450)}Hz → {filter_config.get('lp_end', 2300)}Hz",
        'aggressive_open': f"Aggressive open LP {filter_config.get('lp_start', 550)}Hz → HP {filter_config.get('hp_end', 3700)}Hz",
        'symmetric_sweep': f"Symmetric sweep @ {filter_config.get('bp_center', 1000)}Hz ±{filter_config.get('bp_bandwidth', 500)}Hz",
        'bass_emphasis': f"Bass emphasis LP {filter_config.get('lp_cutoff', 900)}Hz with low boost",
        'broadband_stable': f"Broadband stable LP {filter_config.get('lp_cutoff', 2600)}Hz / HP {filter_config.get('hp_cutoff', 220)}Hz",
        'curved_sweep': f"Curved sweep {filter_config.get('bp_center_start', 850)}Hz → {filter_config.get('bp_center_peak', 1700)}Hz → {filter_config.get('bp_center_end', 850)}Hz",
        'gradual_opening': f"Gradual opening {filter_config.get('lp_start', 520)}Hz → {filter_config.get('lp_end', 2100)}Hz ({filter_config.get('opening_curve', 'sigmoid')})",
        'forceful_open': f"Forceful open LP {filter_config.get('lp_start', 620)}Hz → HP {filter_config.get('hp_end', 2900)}Hz",
        'dual_resonance': f"Dual resonance @ {filter_config.get('bp_center_1', 850)}Hz & {filter_config.get('bp_center_2', 1250)}Hz",
        'focused_bandpass': f"Focused bandpass @ {filter_config.get('bp_center', 1100)}Hz, BW {filter_config.get('bp_bandwidth', 400)}Hz",
        'impact_spike': f"Impact spike @ {filter_config.get('bp_center', 1500)}Hz, Q={filter_config.get('impact_resonance', 0.75)}",
        'split_trajectories': f"Split trajectories: {filter_config.get('bp_1_path_start', 800)}→{filter_config.get('bp_1_path_end', 420)}Hz & {filter_config.get('bp_2_path_start', 800)}→{filter_config.get('bp_2_path_end', 2200)}Hz",
        'gentle_close': f"Gentle close {filter_config.get('lp_start', 1600)}Hz → {filter_config.get('lp_end', 850)}Hz",
        'turbulent_sweep': f"Turbulent sweep {filter_config.get('bp_range_low', 650)}–{filter_config.get('bp_range_high', 2700)}Hz (chaos {filter_config.get('chaos_modulation', 0.75)})",
        'narrowing_spectrum': f"Narrowing spectrum {filter_config.get('bp_start_width', 1600)}Hz → {filter_config.get('bp_end_width', 450)}Hz",
        'brightening_burst': f"Brightening burst LP {filter_config.get('lp_start', 420)}Hz → HP {filter_config.get('hp_end', 4200)}Hz",
        'terminal_focus': f"Terminal focus {filter_config.get('bp_start_width', 1300)}Hz → {filter_config.get('bp_end_width', 200)}Hz",
        'closing_focus': f"Closing focus {filter_config.get('lp_start', 2100)}Hz → {filter_config.get('lp_end', 320)}Hz",
        'erratic_sweep': f"Erratic sweep {filter_config.get('bp_center_range_low', 600)}–{filter_config.get('bp_center_range_high', 2100)}Hz (irregularity {filter_config.get('sweep_irregularity', 0.7)})"
    }

    return descriptions.get(ftype, f"Type: {ftype}")


def _describe_envelope(envelope_config: Dict[str, Any]) -> str:
    """Generate human-readable description of envelope."""
    etype = envelope_config['type']

    descriptions = {
        'sudden_short_tail': f"Sudden attack ({envelope_config.get('attack_ms_base', 5)}ms) + short tail ({envelope_config.get('decay_curve', 'exponential')})",
        'gradual_sustained': f"Gradual attack ({envelope_config.get('attack_ms', 50)}ms) + sustained ({envelope_config.get('sustain_phase_ratio', 0.5)*100}% duration)",
        'gated_pulse': f"Gated pulses @ {envelope_config.get('pulse_rate_hz', 10)}Hz",
        'stepped_convergence': f"Stepped convergence ({envelope_config.get('num_steps', 6)} steps, final accent ×{envelope_config.get('final_accent_mult', 1.4)})",
        'plateau_sustained': f"Plateau sustained (attack {envelope_config.get('attack_ms', 250)}ms, sustain {envelope_config.get('sustain_phase_ratio', 0.68)*100}%)",
        'symmetric_dual': f"Symmetric dual (attack {envelope_config.get('attack_ms', 12)}ms, sustain {envelope_config.get('sustain_phase_ratio', 0.32)*100}%)",
        'terminal_strike': f"Terminal strike (attack {envelope_config.get('attack_ms', 8)}ms, final accent)"
    }

    return descriptions.get(etype, f"Type: {etype}")


def analyze_game(tags_json_path: str) -> Dict[str, Any]:
    """
    Analyze entire game and generate comprehensive Layer3b synthesis report.

    Args:
        tags_json_path: Path to tags JSON file

    Returns:
        Complete analysis dict with all moments analyzed
    """
    with open(tags_json_path, 'r') as f:
        tags_data = json.load(f)

    analysis = {
        'game_metadata': {
            'total_plies': tags_data.get('total_plies'),
            'duration_seconds': tags_data.get('duration_seconds'),
            'result': tags_data.get('game_result'),
            'overall_narrative': tags_data.get('overall_narrative'),
            'eco_code': tags_data.get('eco')
        },
        'sections': [],
        'statistics': {
            'total_moments': 0,
            'moments_by_type': {},
            'moments_by_section': {},
            'archetypes_used': set(),
            'missing_implementations': set()
        }
    }

    # Analyze each section
    for section in tags_data.get('sections', []):
        section_analysis = {
            'section_info': {
                'name': section.get('name'),
                'start_ply': section.get('start_ply'),
                'end_ply': section.get('end_ply'),
                'duration': section.get('duration'),
                'narrative': section.get('narrative'),
                'tension': section.get('tension')
            },
            'moments': []
        }

        section_context = {
            'name': section.get('name'),
            'narrative': section.get('narrative'),
            'tension': section.get('tension', 0.5)
        }

        # Analyze each moment in section
        for moment in section.get('key_moments', []):
            moment_analysis = analyze_moment(moment, section_context)
            section_analysis['moments'].append(moment_analysis)

            # Update statistics
            analysis['statistics']['total_moments'] += 1
            moment_type = moment.get('type', 'MOVE')
            analysis['statistics']['moments_by_type'][moment_type] = \
                analysis['statistics']['moments_by_type'].get(moment_type, 0) + 1

            if moment_type in ARCHETYPES:
                analysis['statistics']['archetypes_used'].add(moment_type)

                # Check for missing implementations
                pitch_type = ARCHETYPES[moment_type]['pitch']['type']
                harmony_type = ARCHETYPES[moment_type]['harmony']['type']
                filter_type = ARCHETYPES[moment_type]['filter']['type']

                # These are the currently implemented types
                implemented_pitch = {'exponential_gliss', 'ascending_spread', 'oscillating_tremor', 'cellular_sequence',
                                    'weak_parabolic', 'slow_drift', 'impact_transient', 'final_descent'}
                implemented_harmony = {'cluster_to_interval', 'unison_to_chord', 'dense_cluster', 'harmonic_stack',
                                      'minimal_dyad', 'collision_cluster', 'shifting_voices', 'resolving_to_root'}
                implemented_filter = {'bandpass_to_lowpass_choke', 'lowpass_to_highpass_open', 'bandpass_sweep', 'rhythmic_gate',
                                     'gentle_bandpass', 'gradual_sweep', 'impact_spike', 'closing_focus'}

                if pitch_type not in implemented_pitch:
                    analysis['statistics']['missing_implementations'].add(f"pitch:{pitch_type}")
                if harmony_type not in implemented_harmony:
                    analysis['statistics']['missing_implementations'].add(f"harmony:{harmony_type}")
                if filter_type not in implemented_filter:
                    analysis['statistics']['missing_implementations'].add(f"filter:{filter_type}")

        section_name = section.get('name', 'Unknown')
        analysis['statistics']['moments_by_section'][section_name] = len(section.get('key_moments', []))
        analysis['sections'].append(section_analysis)

    # Convert sets to sorted lists for JSON serialization
    analysis['statistics']['archetypes_used'] = sorted(list(analysis['statistics']['archetypes_used']))
    analysis['statistics']['missing_implementations'] = sorted(list(analysis['statistics']['missing_implementations']))

    return analysis


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_layer3b_moments.py <tags-json> [output-json]")
        print("\nExample:")
        print("  python3 analyze_layer3b_moments.py tags-game1.json layer3b-analysis.json")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'layer3b-analysis.json'

    print(f"Analyzing Layer3b moments from: {input_path}")

    # Generate analysis
    analysis = analyze_game(input_path)

    # Write to output file
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    # Print summary
    print(f"\n✓ Analysis complete: {output_path}")
    print(f"\nSummary:")
    print(f"  Total moments: {analysis['statistics']['total_moments']}")
    print(f"  Unique archetypes used: {len(analysis['statistics']['archetypes_used'])}")
    print(f"  Missing implementations: {len(analysis['statistics']['missing_implementations'])}")

    if analysis['statistics']['missing_implementations']:
        print(f"\n⚠ Missing curve generator implementations:")
        for impl in analysis['statistics']['missing_implementations']:
            print(f"    - {impl}")

    print(f"\nMoments by type:")
    for mtype, count in sorted(analysis['statistics']['moments_by_type'].items()):
        print(f"  {mtype}: {count}")


if __name__ == '__main__':
    main()
