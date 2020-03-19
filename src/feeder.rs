use statfeed::Statfeed;
use soundsym::{Sound, SoundSequence};

use std::rc::Rc;
use std::sync::Arc;
use std::f64;

pub struct Feeder {
    pub sf: Statfeed<Arc<Sound>>,
    pub phonemes_per_line: usize,
    current_line: usize
}

impl Feeder {
    pub fn new(sounds: Vec<Arc<Sound>>, phonemes_per_line: usize, lines: usize) -> Feeder {
        let n = phonemes_per_line * lines;
        let max_power = sounds.iter().map(|s| s.max_power())
            .fold(0., |acc, x| x.max(acc)); 
        let max_con = sounds.iter().map(|s| s.pitch_confidence())
            .fold(0., |acc, x| x.max(acc)); 
        let weights = (0..lines).fold(Vec::<Vec<f64>>::new(), |mut weights, line| {
            let phase = (line as f64 / lines as f64) * f64::consts::PI;
            let power_target = 0.5 * (1. + (phase * 2. + f64::consts::PI).cos()); // in [0, 1]
            let pitch_target = 0.5 * (1. + (phase + f64::consts::PI).cos()); // in [0, 1]
            let w: Vec<f64> = sounds.iter().map(|s| {
                let pitch_confidence = s.pitch_confidence();
                let power = s.max_power();
                // linear normalization of power and pitch confidence
                let normed_power = (s.max_power() / max_power); // in [0, 1]
                let normed_pitch = (s.pitch_confidence() / max_con); // in [0, 1]
                (2. / ((normed_power - power_target).powi(2) 
                    + (normed_pitch - pitch_target).powi(2))).sqrt()
            }).collect();
            for _ in 0..phonemes_per_line {
                weights.push(w.clone());
            }
            weights
        });

        let mut feeder = Feeder {
            sf: Statfeed::<Arc<Sound>>::new(sounds, n),
            phonemes_per_line: phonemes_per_line,
            current_line: 0
        };

        feeder.sf.weights = weights;
        for hg in feeder.sf.heterogeneities.iter_mut() {
            *hg = 100.;
        }
        feeder.sf.populate_choices();
        feeder
    }
}

impl Iterator for Feeder {
    type Item = SoundSequence;

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.phonemes_per_line * self.current_line;
        self.current_line += 1;
        let end = self.phonemes_per_line * self.current_line;
        if end < self.sf.choices.len() {
            Some(SoundSequence::new(self.sf.choices[start..end].iter().map(|s| s.clone()).collect()))
        } else {
            None
        }
    }
}

#[inline]
fn calc_weight(power: f64, phase: f64) -> f64 {
    (phase.cos() + 1.) * 0.5 * power
}
