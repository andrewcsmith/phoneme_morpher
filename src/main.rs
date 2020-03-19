extern crate regex;
extern crate soundsym;
extern crate hound;
extern crate getopts;
extern crate interactor;
extern crate statfeed;
extern crate vox_box;
extern crate num;
extern crate rusty_machine;
extern crate rulinalg;

use std::error::Error;
use std::path::Path;
use std::str::from_utf8;
use std::env;
use std::io;
use std::fs::File;
use std::io::{Write, BufReader, BufRead, BufWriter};
use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::collections::HashSet;

use soundsym::*;
use getopts::{Options, Matches};
use regex::Regex;
use interactor::*;
use rulinalg::matrix::Matrix;
// use rusty_machine::prelude::*;
use rusty_machine::learning::nnet::{NeuralNet, BCECriterion, MSECriterion};
use rusty_machine::learning::optim::grad_desc::{GradientDesc, StochasticGD};
use rusty_machine::learning::optim::fmincg::ConjugateGD;

mod feeder;
mod cluster;
mod parser;

use feeder::Feeder;

const HELP_TEXT: &'static str = "
o: open sound file
partition threshold, depth: partition active sound with threshold and trie depth and add to dictionary
d: get info about dictionary
clear: clear all sounds from dictionary
labels: print labels to stdout
cluster ngroups: cluster sounds into ngroups clusters
load 'dir': load dictionary from dir
save 'dir': save dictionary to dir
read 'file': read file and extract labels
split x, y: extract from segment x to y
feeder line_length, nlines: create a sequence of lines, getting louder
morph x: morph the split sequence
q: quit
?: help
";

fn print_help() {
    io::stdout().write(HELP_TEXT.as_bytes());
}

pub enum Input {
    O,
    Q,
    D,
    P,
    Help,
    Clear,
    Labels,
    Train(String),
    Params(f64, f64, usize),
    Predict(usize),
    Cluster(usize),
    Load(String),
    Save(String),
    Read(String),
    Partition(usize, usize),
    Feeder(usize, usize),
    Split(usize, usize),
    Morph(f64),
    None
}

fn main() {
    match run() {
        Ok(_) => { },
        Err(e) => { println!("Error: {}", e); }
    }
}

fn run() -> Result<(), Box<Error>> {
    let matches = try!(parse_opts());
    print_help();

    let mut current_sound: Option<Rc<Sound>> = None;
    let mut dictionary = Rc::new(RefCell::new(SoundDictionary::new()));
    let mut sequence: Option<SoundSequence> = None;
    let mut labels = Vec::<String>::new();
    let mut possible_characters = Vec::<char>::new();
    // let mut nn: Option<NeuralNet<_, _>> = None;
    let mut scales: Option<Vec<(f64, f64)>> = None;
    let mut gdparams: (f64, f64, usize) = (0.1, 0.1, 1000);

    'main: loop {
        match try!(parser::parse_input()) {
            Input::O => {
                match prompt_filename() {
                    Ok(sound) => { 
                        print_sound_info(&sound);
                        current_sound = Some(Rc::new(sound));
                    },
                    Err(e) => {
                        println!("Encountered an error: {}", e);
                    }
                }
            },
            Input::Cluster(nclusters) => {
                cluster_sounds(&mut dictionary.borrow_mut().sounds[..], nclusters);
                println!("Clustered.")
            },
            Input::Labels => {
                println!("{:?}", possible_characters);
                // for label in labels.iter() {
                //     println!("{}", &label);
                //     for c in label.chars() {
                //         if !possible_characters.contains(&c) {
                //             possible_characters.push(c);
                //         }
                //     }
                // }
            }
            Input::Params(alpha, mu, iters) => {
                gdparams = (alpha, mu, iters);
            }
            Input::Train(path) => {
                println!("Not implemented.");
                // TODO: Implement
                // let training_set = SoundDictionary::from_path(&Path::new(&path)).unwrap();
                // let txt = format!("{}.txt", path);
                // let labels = read_labels(&Path::new(&txt)).unwrap();
                //
                // for label in labels.iter() {
                //     for c in label.chars() {
                //         if !possible_characters.contains(&c) {
                //             possible_characters.push(c);
                //         }
                //     }
                // }
                //
                // let hidden_layer_size = (cluster::SIZE + possible_characters.len()) / 2;
                // let layers = vec![cluster::SIZE, hidden_layer_size, possible_characters.len()];
                // let mut net = NeuralNet::new(Cow::Owned(layers), MSECriterion, StochasticGD::new(gdparams.0, gdparams.1, gdparams.2));
                // println!("Training commencing.");
                // scales = Some(cluster::train_nn(&mut net, &training_set.sounds, &labels, &possible_characters[..]));
                // nn = Some(net);
                // println!("I know Kung Fu.");
            }
            Input::Predict(idx) => {
                println!("Not implemented.");
                // TODO: Implement
                // match nn {
                //     Some(ref net) => {
                //         if let Some(sound) = dictionary.borrow().sounds.get(idx) {
                //             let prediction = cluster::predict_nn(net, sound.clone(), &possible_characters[..], scales.clone());
                //             println!("{}", prediction);
                //         }
                //     }
                //     None => { println!("No net initialized. Call train.") }
                // }
            }
            Input::Partition(threshold, depth) => {
                match current_sound {
                    Some(ref sound) => {
                        partition_sound(sound, &mut dictionary.borrow_mut(), threshold, depth);
                    }
                    None => {
                        println!("No sound loaded. Call O to open a sound.");
                    }
                }
            }
            Input::D => {
                let nsounds = dictionary.borrow().sounds.len();
                println!("Sounds: {}", nsounds);
                for sound in dictionary.borrow().sounds.iter() {
                    let name = sound.name.as_ref().map(|s| String::as_str(s)).unwrap_or("");
                    let con = sound.pitch_confidence();
                    let pow = sound.max_power();
                    println!("{}\tcon: {:.4}\tpow: {:.4}", name, con, pow);
                }
            }
            Input::Clear => {
                dictionary = Rc::new(RefCell::new(SoundDictionary::new()));
            }
            Input::Feeder(length, lines) => {
                let sounds: Vec<Arc<Sound>> = dictionary.borrow().sounds.iter().map(|s| s.clone()).collect();
                let mut feeder = Feeder::new(sounds, length, lines);
                let out_file = File::create("phoneme_poem.txt").expect("File not successfully created");
                let mut writer = BufWriter::new(out_file);
                for (idx, line) in feeder.enumerate() {
                    let fmt_line: String = line.sounds().iter().cloned()
                        .map(|s| s.name.clone().unwrap_or("".to_string()))
                        .collect::<Vec<String>>().join("\t");
                    write!(&mut writer, "{}\t{}\n", idx+1, fmt_line).expect("Writing unsuccessful");
                    // println!("{}", line);
                    line.to_sound().write_file(&Path::new(&format!("fed/line_{:03}.wav", idx)));
                }
                println!("Finished feeding.");
            }
            Input::Load(dir) => {
                println!("loading dictionary from {}", dir);
                match SoundDictionary::from_path(&Path::new(&dir)) {
                    Ok(dict) => {
                        println!("dictionary loaded");
                        dictionary = Rc::new(RefCell::new(dict));
                    },
                    Err(e) => {
                        println!("Could not load dictionary: {}", e.description());
                    }
                }
            }
            Input::Save(dir) => {
                println!("Saving dictionary to {}", dir);
                for (idx, sound) in dictionary.borrow().sounds.iter().enumerate() {
                    match sound.write_file(&Path::new(&format!("{}/{}_{:05}.wav", dir, sound.name.as_ref().map(|s| String::as_ref(&s)).unwrap_or(""), idx))) {
                        Err(e) => { println!("Error: {}", e.description()); }
                        _ => { }
                    }
                }
            }
            Input::Read(path) => {
                let path = Path::new(&path);
                if path.exists() {
                    println!("Reading labels from {}", path.display());
                    match read_labels(&path) {
                        Ok(l) => { 
                            for (idx, s) in dictionary.borrow_mut().sounds.iter_mut().enumerate() {
                                let mut new_sound = Arc::make_mut(s);
                                new_sound.name = Some(l[idx].clone());
                            }
                            labels.extend_from_slice(&l[..]); 
                        },
                        Err(e) => { println!("Error reading labels: {}", e.description()); }
                    }
                } else {
                    println!("{} does not exist", path.display());
                }
            }
            Input::Split(start, end) => {
                sequence = split_sound((start, end), &dictionary.borrow());
            },
            Input::Morph(factor) => {
                match sequence {
                    Some(ref seq) => {
                        match morph_sequence(seq, &dictionary.borrow(), factor) {
                            Ok(s) => { 
                                s.to_sound().write_file(&Path::new("tmp.wav"));
                            },
                            Err(_) => {}
                        }
                    },
                    None => {
                        println!("No active sequence. Call split x, y to create a sequence");
                        print_help();
                    }
                }
            }
            Input::Help => { 
                print_help();
            },
            Input::Q => break 'main,
            Input::None | _ => { },
        }
    }
    Ok(())
}

fn parse_opts() -> Result<Matches, Box<Error>> {
    let args: Vec<String> = env::args().collect();
    let mut opts = Options::new();
    let matches = try!(opts.parse(&args[1..]));
    Ok(matches)
}

fn prompt_filename() -> Result<Sound, Box<Error>> {
    println!("Please find the filename");
    let chosen_file = pick_file(|| default_menu_cmd(), 
                                std::env::current_dir().unwrap()).unwrap();
    println!("Opening file {}", chosen_file.to_str().unwrap_or(""));
    let mut sound = Sound::from_path(&chosen_file.as_path());
    // sound.as_mut().map(|ref mut s| s.preload_pitch_confidence());
    sound
}

fn print_sound_info(sound: &Sound) { 
    println!("{}", sound.name.clone().unwrap_or(String::new()));
    println!("nsamples: {}", sound.samples().len());
    println!("max_power: {}", sound.max_power());
}

fn partition_sound(sound: &Sound, dictionary: &mut SoundDictionary, threshold: usize, depth: usize) {
    println!("Partitioning {}", sound.name.clone().unwrap_or("no name".to_string()));
    let partitioner = { 
        let mut partitioner = Partitioner::new(Cow::Borrowed(sound))
            .threshold(threshold).depth(depth);
        partitioner.train().expect("Could not train partitioner");
        partitioner
    };

    let rows = sound.mfccs().len() / soundsym::NCOEFFS;
    let cols = soundsym::NCOEFFS;
    let data = Matrix::new(rows, cols, sound.mfccs().clone());
    let predictions = partitioner.predict(&data).unwrap();
    let splits = partitioner.partition(predictions).unwrap();
    println!("Found {} splits", splits.len());
    dictionary.add_segments(&sound, &splits[..]);
}

fn split_sound(bounds: (usize, usize), dictionary: &SoundDictionary) -> Option<SoundSequence> {
    let sounds = dictionary.sounds[bounds.0..bounds.1]
        .iter().map(|s| s.clone()).collect();
    let seq = SoundSequence::new(sounds);
    seq.to_sound().write_file(&Path::new("tmp.wav"));
    Some(seq)
}

fn morph_sequence(seq: &SoundSequence, dictionary: &SoundDictionary, factor: f64) -> Result<SoundSequence, String> {
    let distances = vec![factor; seq.sounds().len()];
    seq.morph_to(&distances[..], dictionary)
}

fn cluster_sounds(sounds: &mut [Arc<Sound>], nclusters: usize) {
    cluster::cluster_sounds(sounds, nclusters);
}

fn read_labels(path: &Path) -> Result<Vec<String>, Box<Error>> {
    let file = try!(File::open(path));
    let file = BufReader::new(file);
    let mut labels = Vec::<String>::new();
    let mut lines = file.lines();
    while let Some(Ok(line)) = lines.next() {
        labels.push(line.trim().to_string());
    }
    println!("{:?}", labels);
    Ok(labels)
}

