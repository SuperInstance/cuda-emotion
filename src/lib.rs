/*!
# cuda-emotion

Emotional state engine for agents.

Emotions aren't decoration. They're fast heuristics that modulate
behavior before deliberation catches up. Fear makes you cautious.
Joy makes you exploratory. Frustration makes you change strategy.

This crate maps neurochemical states + experiences to emotional states
that influence decision-making:
- Core emotions (Ekman's 6 + extensions)
- Emotional modulation of parameters (risk tolerance, speed, exploration)
- Emotional memory (emotional coloring of episodes)
- Emotional contagion (fleet-level mood propagation)
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core emotions
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Emotion {
    Joy,
    Trust,
    Fear,
    Surprise,
    Sadness,
    Disgust,
    Anger,
    Anticipation,
    Curiosity,
    Frustration,
    Calm,
}

impl Emotion {
    pub fn valence(&self) -> f64 {
        match self {
            Emotion::Joy | Emotion::Trust | Emotion::Anticipation | Emotion::Curiosity | Emotion::Calm => 1.0,
            Emotion::Surprise => 0.0,
            Emotion::Fear | Emotion::Sadness | Emotion::Disgust | Emotion::Anger | Emotion::Frustration => -1.0,
        }
    }

    pub fn arousal(&self) -> f64 {
        match self {
            Emotion::Fear | Emotion::Anger | Emotion::Surprise | Emotion::Frustration => 0.9,
            Emotion::Joy | Emotion::Anticipation | Emotion::Curiosity => 0.6,
            Emotion::Disgust | Emotion::Sadness => 0.3,
            Emotion::Trust | Emotion::Calm => 0.2,
        }
    }

    /// How this emotion affects risk tolerance
    pub fn risk_modulation(&self) -> f64 {
        match self {
            Emotion::Fear | Emotion::Sadness | Emotion::Frustration => -0.3,
            Emotion::Joy | Emotion::Trust | Emotion::Anticipation | Emotion::Calm => 0.2,
            Emotion::Anger => 0.1,
            Emotion::Surprise | Emotion::Curiosity => 0.0,
            Emotion::Disgust => -0.2,
        }
    }

    /// How this emotion affects exploration vs exploitation
    pub fn exploration_modulation(&self) -> f64 {
        match self {
            Emotion::Curiosity | Emotion::Anticipation | Emotion::Joy => 0.3,
            Emotion::Fear | Emotion::Anger => -0.2,
            Emotion::Frustration => 0.1, // frustration drives change
            Emotion::Calm | Emotion::Trust | Emotion::Sadness => 0.0,
            Emotion::Surprise | Emotion::Disgust => -0.1,
        }
    }
}

/// An emotional state at a point in time
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmotionalState {
    pub primary: Emotion,
    pub intensity: f64,        // [0, 1]
    pub secondary: Option<Emotion>,
    pub secondary_intensity: f64,
    pub timestamp: u64,
    pub cause: String,
    pub duration_estimate_ms: u64,
}

impl EmotionalState {
    pub fn new(emotion: Emotion, intensity: f64, cause: &str) -> Self {
        EmotionalState { primary: emotion, intensity: intensity.clamp(0.0, 1.0), secondary: None, secondary_intensity: 0.0, timestamp: now(), cause: cause.to_string(), duration_estimate_ms: 5000 }
    }

    pub fn with_secondary(mut self, emotion: Emotion, intensity: f64) -> Self {
        self.secondary = Some(emotion);
        self.secondary_intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Net valence = weighted sum of primary + secondary
    pub fn net_valence(&self) -> f64 {
        let primary = self.primary.valence() * self.intensity;
        let secondary = self.secondary.map(|e| e.valence() * self.secondary_intensity).unwrap_or(0.0);
        primary + secondary * 0.5
    }

    /// Net arousal
    pub fn net_arousal(&self) -> f64 {
        let primary = self.primary.arousal() * self.intensity;
        let secondary = self.secondary.map(|e| e.arousal() * self.secondary_intensity).unwrap_or(0.0);
        (primary + secondary * 0.5).clamp(0.0, 1.0)
    }

    /// Is this state positive?
    pub fn is_positive(&self) -> bool { self.net_valence() > 0.0 }
}

/// Emotional modulation — how current emotions change behavior parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmotionalModulation {
    pub risk_tolerance: f64,      // base 0.5, modified by emotions
    pub exploration_rate: f64,    // base 0.3
    pub speed_factor: f64,        // base 1.0
    pub patience: f64,            // base 0.5
    pub cooperation_willingness: f64, // base 0.5
}

impl EmotionalModulation {
    pub fn new() -> Self { EmotionalModulation { risk_tolerance: 0.5, exploration_rate: 0.3, speed_factor: 1.0, patience: 0.5, cooperation_willingness: 0.5 } }

    /// Apply emotional modulation
    pub fn apply(&mut self, state: &EmotionalState) {
        self.risk_tolerance = (self.risk_tolerance + state.primary.risk_modulation() * state.intensity * 0.3).clamp(0.0, 1.0);
        self.exploration_rate = (self.exploration_rate + state.primary.exploration_modulation() * state.intensity * 0.2).clamp(0.0, 1.0);
        self.speed_factor = (self.speed_factor + state.net_arousal() * 0.2 - 0.1).clamp(0.3, 2.0);
        // Frustration decreases patience
        if state.primary == Emotion::Frustration {
            self.patience = (self.patience - 0.1 * state.intensity).max(0.1);
        }
        // Joy/Calm increase cooperation
        if state.primary == Emotion::Joy || state.primary == Emotion::Calm || state.primary == Emotion::Trust {
            self.cooperation_willingness = (self.cooperation_willingness + 0.1 * state.intensity).min(1.0);
        }
        // Anger decreases cooperation
        if state.primary == Emotion::Anger {
            self.cooperation_willingness = (self.cooperation_willingness - 0.2 * state.intensity).max(0.1);
        }
    }
}

/// Emotional memory — color episodes with emotional context
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmotionalMemory {
    pub entries: Vec<EmotionalEpisode>,
    pub max_entries: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmotionalEpisode {
    pub episode_id: String,
    pub emotion: Emotion,
    pub intensity: f64,
    pub valence: f64,
    pub timestamp: u64,
    pub context: String,
}

impl EmotionalMemory {
    pub fn new() -> Self { EmotionalMemory { entries: vec![], max_entries: 200 } }

    pub fn record(&mut self, episode: EmotionalEpisode) {
        if self.entries.len() >= self.max_entries { self.entries.remove(0); }
        self.entries.push(episode);
    }

    /// Emotional coloring — how much positive/negative emotional history
    pub fn emotional_balance(&self) -> f64 {
        if self.entries.is_empty() { return 0.0; }
        let sum: f64 = self.entries.iter().map(|e| e.valence * e.intensity).sum();
        let total_intensity: f64 = self.entries.iter().map(|e| e.intensity).sum();
        if total_intensity < 0.001 { return 0.0; }
        sum / total_intensity
    }

    /// Mood — aggregated emotional state from recent history
    pub fn current_mood(&self, n_recent: usize) -> EmotionalState {
        let recent: Vec<_> = self.entries.iter().rev().take(n_recent).collect();
        if recent.is_empty() { return EmotionalState::new(Emotion::Calm, 0.3, "neutral"); }

        // Count emotion frequencies weighted by recency
        let mut scores: HashMap<Emotion, f64> = HashMap::new();
        for (i, entry) in recent.iter().enumerate() {
            let recency_weight = 1.0 / (i as f64 + 1.0); // more recent = higher weight
            *scores.entry(entry.emotion).or_insert(0.0) += entry.intensity * recency_weight;
        }

        let (emotion, score) = scores.into_iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap_or((Emotion::Calm, 0.3));
        let intensity = (score / recent.len() as f64).clamp(0.1, 1.0);
        EmotionalState::new(emotion, intensity, "aggregated from memory")
    }
}

/// Emotional contagion — fleet-level mood propagation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmotionalContagion {
    pub fleet_mood: f64,         // aggregate valence [-1, 1]
    pub susceptibility: f64,     // how much this agent catches moods
    pub influence: f64,          // how much this agent influences others
    pub decay_rate: f64,
}

impl EmotionalContagion {
    pub fn new() -> Self { EmotionalContagion { fleet_mood: 0.0, susceptibility: 0.3, influence: 0.5, decay_rate: 0.1 } }

    /// Receive mood from fleet
    pub fn receive(&mut self, external_mood: f64) {
        self.fleet_mood = self.fleet_mood * (1.0 - self.susceptibility) + external_mood * self.susceptibility;
    }

    /// Get influence signal to broadcast
    pub fn broadcast_mood(&self, own_mood: f64) -> f64 {
        own_mood * self.influence
    }

    /// Decay fleet mood toward neutral
    pub fn decay(&mut self) {
        self.fleet_mood *= (1.0 - self.decay_rate);
    }
}

/// The emotion engine — central controller
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmotionEngine {
    pub current_state: EmotionalState,
    pub modulation: EmotionalModulation,
    pub memory: EmotionalMemory,
    pub contagion: EmotionalContagion,
    pub emotion_threshold: f64, // minimum intensity to change primary emotion
}

impl EmotionEngine {
    pub fn new() -> Self { EmotionEngine { current_state: EmotionalState::new(Emotion::Calm, 0.3, "initial"), modulation: EmotionalModulation::new(), memory: EmotionalMemory::new(), contagion: EmotionalContagion::new(), emotion_threshold: 0.3 } }

    /// Process an event and update emotional state
    pub fn process_event(&mut self, event_type: &str, outcome: f64) -> EmotionalState {
        let emotion = self.map_event_to_emotion(event_type, outcome);
        let intensity = outcome.abs().clamp(0.1, 1.0);
        let new_state = EmotionalState::new(emotion, intensity, event_type);

        // Only change primary if new emotion is strong enough
        if intensity > self.emotion_threshold {
            self.current_state = new_state;
        }

        // Apply modulation
        self.modulation.apply(&self.current_state);

        // Record to emotional memory
        self.memory.record(EmotionalEpisode {
            episode_id: format!("ep_{}", now()),
            emotion, intensity, valence: emotion.valence() * intensity,
            timestamp: now(), context: event_type.to_string(),
        });

        self.current_state.clone()
    }

    fn map_event_to_emotion(&self, event_type: &str, outcome: f64) -> Emotion {
        match event_type {
            "success" | "achievement" | "reward" => Emotion::Joy,
            "cooperation" | "help" => Emotion::Trust,
            "danger" | "threat" | "pain" => Emotion::Fear,
            "unexpected" | "novelty" => if outcome > 0.0 { Emotion::Curiosity } else { Emotion::Surprise },
            "loss" | "failure" | "miss" => Emotion::Sadness,
            "rejection" | "harm" | "violation" => Emotion::Disgust,
            "blocked" | "obstacle" | "conflict" => Emotion::Frustration,
            "injustice" | "betrayal" => Emotion::Anger,
            "goal_set" | "plan" => Emotion::Anticipation,
            "rest" | "safe" | "routine" => Emotion::Calm,
            _ => if outcome > 0.3 { Emotion::Joy } else if outcome < -0.3 { Emotion::Sadness } else { Emotion::Calm },
        }
    }

    /// Get current modulation values
    pub fn get_modulation(&self) -> &EmotionalModulation { &self.modulation }

    /// Full state summary
    pub fn summary(&self) -> String {
        format!("emotion={:?} intensity={:.2} valence={:.2} arousal={:.2} risk={:.2} explore={:.2} fleet_mood={:.2}",
            self.current_state.primary, self.current_state.intensity,
            self.current_state.net_valence(), self.current_state.net_arousal(),
            self.modulation.risk_tolerance, self.modulation.exploration_rate,
            self.contagion.fleet_mood)
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_valence() {
        assert!(Emotion::Joy.valence() > 0.0);
        assert!(Emotion::Fear.valence() < 0.0);
        assert_eq!(Emotion::Surprise.valence(), 0.0);
    }

    #[test]
    fn test_emotional_state_net() {
        let state = EmotionalState::new(Emotion::Joy, 0.8, "test").with_secondary(Emotion::Fear, 0.3);
        assert!(state.is_positive());
    }

    #[test]
    fn test_modulation_from_emotion() {
        let mut modul = EmotionalModulation::new();
        let state = EmotionalState::new(Emotion::Fear, 0.8, "danger");
        modul.apply(&state);
        assert!(modul.risk_tolerance < 0.5); // fear reduces risk
    }

    #[test]
    fn test_emotion_engine_success() {
        let mut engine = EmotionEngine::new();
        let state = engine.process_event("success", 0.9);
        assert_eq!(state.primary, Emotion::Joy);
    }

    #[test]
    fn test_emotion_engine_frustration() {
        let mut engine = EmotionEngine::new();
        engine.process_event("blocked", 0.8);
        assert!(engine.modulation.patience < 0.5);
    }

    #[test]
    fn test_emotional_memory_balance() {
        let mut mem = EmotionalMemory::new();
        mem.record(EmotionalEpisode { episode_id: "1".into(), emotion: Emotion::Joy, intensity: 0.8, valence: 0.8, timestamp: 0, context: "win".into() });
        mem.record(EmotionalEpisode { episode_id: "2".into(), emotion: Emotion::Sadness, intensity: 0.3, valence: -0.3, timestamp: 0, context: "loss".into() });
        let balance = mem.emotional_balance();
        assert!(balance > 0.0); // more joy than sadness
    }

    #[test]
    fn test_emotional_mood() {
        let mut mem = EmotionalMemory::new();
        for _ in 0..5 { mem.record(EmotionalEpisode { episode_id: "x".into(), emotion: Emotion::Joy, intensity: 0.7, valence: 0.7, timestamp: 0, context: "y".into() }); }
        let mood = mem.current_mood(5);
        assert_eq!(mood.primary, Emotion::Joy);
    }

    #[test]
    fn test_contagion_receive() {
        let mut contagion = EmotionalContagion::new();
        contagion.susceptibility = 0.5;
        contagion.receive(0.8);
        assert!(contagion.fleet_mood > 0.0);
    }

    #[test]
    fn test_contagion_decay() {
        let mut contagion = EmotionalContagion::new();
        contagion.fleet_mood = 0.8;
        contagion.decay();
        assert!(contagion.fleet_mood < 0.8);
    }

    #[test]
    fn test_summary() {
        let engine = EmotionEngine::new();
        let s = engine.summary();
        assert!(s.contains("Calm"));
    }

    #[test]
    fn test_anger_reduces_cooperation() {
        let mut modul = EmotionalModulation::new();
        let state = EmotionalState::new(Emotion::Anger, 0.9, "betrayal");
        modul.apply(&state);
        assert!(modul.cooperation_willingness < 0.5);
    }

    #[test]
    fn test_secondary_emotion() {
        let state = EmotionalState::new(Emotion::Fear, 0.7, "threat").with_secondary(Emotion::Anger, 0.5);
        assert!(state.secondary.is_some());
        assert!(state.net_valence() < 0.0); // both negative
    }
}
