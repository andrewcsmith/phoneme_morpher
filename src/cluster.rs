use soundsym::*;
use vox_box::*;
use vox_box::periodic::{Autocorrelate, Pitch, HasPitch};
use vox_box::polynomial::Polynomial;
use vox_box::spectrum::{LPC, Resonance, ToResonance};
use vox_box::waves::{Normalize, Filter, WindowType, Windower, Resample};
use num::complex::Complex;
use rusty_machine::prelude::*;
use rusty_machine::linalg::utils;
use rusty_machine::linalg::matrix::MatrixSliceMut;
use rusty_machine::learning::k_means::KMeansClassifier;
use rusty_machine::learning::nnet::{BaseNeuralNet, NeuralNet, Criterion};
use rusty_machine::learning::optim::grad_desc::StochasticGD;
use rusty_machine::learning::optim::OptimAlgorithm;

use std::i32;
use std::rc::Rc;
use std::path::Path;
use std::fs::File;
use std::io::Write;
use std::error::Error;
use std::collections::VecDeque;

const NFORMANTS: usize = 4;
pub const SIZE: usize = NCOEFFS + NFORMANTS + 1;

/// Panics if any value is NaN or +/-Infinity
fn max_f64(v: &[f64]) -> f64 {
    v.iter()
        .max_by_key(|d| (*d * i32::MAX as f64) as i32)
        .map(|s| *s as f64 / i32::MAX as f64).unwrap()
}

/// Data layout:
///
/// 0..NCOEFFS: the mfcc coefficients
/// NCOEFFS..(NCOEFFS+NFORMANTS): pitch strength, 3 formant frequencies
pub fn sound_feature_vector(sound: &Sound) -> Vec<f64> {
    let mut features = Vec::<f64>::with_capacity(SIZE);
    features.extend_from_slice(&sound.mean_mfccs().0);
    let formants = sound.mean_formants(&mut vec![0f64; sound.samples().len()]);
    features.extend_from_slice(&formants[..]);
    features.push(sound.max_power());
    features
}

fn get_feature_matrix(sounds: &[Rc<Sound>]) -> Matrix<f64> {
    let features: Vec<f64> = sounds.iter().fold(
        Vec::with_capacity(sounds.len() * SIZE), 
        |mut acc, sound| {
            acc.extend_from_slice(&sound_feature_vector(&sound));
            acc
        });

    Matrix::new(sounds.len(), SIZE, features)
}

fn normalize_feature_matrix(mut feature_matrix: &mut Matrix<f64>) -> Vec<(f64, f64)> {
    let mut min = Matrix::new(1, SIZE, vec![0f64; SIZE]);
    let mut max = Matrix::new(1, SIZE, vec![0f64; SIZE]);
    for row in feature_matrix.iter_rows() {
        for (idx, elem) in row.iter().enumerate() {
            if *elem < min[[0, idx]] { min[[0, idx]] = *elem; }
            if *elem > max[[0, idx]] { max[[0, idx]] = *elem; }
        }
    }

    let means = feature_matrix.mean(0);

    let range = {
        let mut range = Matrix::new(1, SIZE, vec![0f64; SIZE]);
        for (idx, (n, x)) in min.data().iter().zip(max.data().iter()).enumerate() {
            range[[0, idx]] = x - n;
        }
        range
    };

    for row_idx in 0..feature_matrix.rows() {
        let mut row = MatrixSliceMut::from_matrix(&mut feature_matrix, [row_idx, 0], 1, SIZE);
        // row -= &min;
        for (idx, v) in row.iter_mut().enumerate() {
            *v -= means[idx];
            *v /= range[[0, idx]];
        }
    }

    let mut out = Vec::<(f64, f64)>::with_capacity(SIZE);

    for idx in 0..SIZE {
        out.push((means[idx], range[[0, idx]]));
    }

    out
}

/// Performs K-Means clustering on a slice of Sounds, renaming in-place
pub fn cluster_sounds(sounds: &mut [Rc<Sound>], nclusters: usize) {
    let mut feature_matrix = get_feature_matrix(sounds);
    normalize_feature_matrix(&mut feature_matrix);

    let mut model = KMeansClassifier::new(nclusters);
    model.train(&feature_matrix);
    let res = model.predict(&feature_matrix);

    for (idx, cluster) in res.data().iter().enumerate() {
        let name = format!("{:02}-{:04}", cluster, idx);
        Rc::get_mut(&mut sounds[idx])
            .expect("Could not change the name of the sound!")
            .name = Some(name);
    }
}

pub fn train_nn<'a, T: Criterion, A: OptimAlgorithm<BaseNeuralNet<'a, T>>>(nn: &mut NeuralNet<'a, T, A>, sounds: &[Rc<Sound>], labels: &[String], dict: &[char]) -> Vec<(f64, f64)> {
    // Need to have one row for each Sound
    let rows = sounds.len();
    let cols = dict.len();
    let mut targets = Matrix::new(rows, cols, vec![0.; rows * cols]);
    for (l_idx, l) in labels.iter().enumerate() {
        for (c_idx, c) in dict.iter().enumerate() {
            if l.contains(*c) { 
                targets[[l_idx, c_idx]] = 1.; 
            }
        }
    }

    let mut feature_matrix = get_feature_matrix(&sounds);
    let out = normalize_feature_matrix(&mut feature_matrix);
    matrix_to_csv(&feature_matrix, &Path::new("features.csv"));
    nn.train(&feature_matrix, &targets);
    for idx in 0..(nn.layers()-1) {
        matrix_to_csv(&nn.get_net_weights(idx), &Path::new(&format!("weights_{}.csv", idx)));
    }
    out
}

/// Predicts which char is contained by the given Sound
pub fn predict_nn<'a, T: Criterion, A: OptimAlgorithm<BaseNeuralNet<'a, T>>>(nn: &NeuralNet<'a, T, A>, sound: Rc<Sound>, dict: &[char], scales: Option<Vec<(f64, f64)>>) -> char {
    let mut feature_matrix = get_feature_matrix(&[sound]);

    println!("{}", &feature_matrix);
    if let Some(ref pairs) = scales {
        println!("Scaling data!");
        for row in feature_matrix.iter_rows_mut() {
            for (idx, r) in row.iter_mut().enumerate() {
                *r -= pairs[idx].0;
                *r /= pairs[idx].1;
            }
        }
        println!("{}", &feature_matrix);
    }

    let res: Matrix<f64> = nn.predict(&feature_matrix);
    println!("weights: {:?}", res);
    let (idx, val) = utils::argmax(&res.data()[..]);
    dict[idx]
}

fn matrix_to_csv(matrix: &Matrix<f64>, out: &Path) -> Result<(), Box<Error>> {
    let mut file = try!(File::create(out));
    for row in matrix.iter_rows() {
        let strings: Vec<String> = row.iter().map(|r| r.to_string()).collect();
        try!(write!(&mut file, "{}\n", strings.join(", ")));
    }
    Ok(())
}

#[test]
fn test_matrix_to_csv() {
    let matrix = Matrix::new(3, 4, vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
    let path = Path::new("test.csv");
    matrix_to_csv(&matrix, path);
}
