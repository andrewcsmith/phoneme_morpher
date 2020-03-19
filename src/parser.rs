use super::*;
use std::io;
use std::error::Error;

// TODO: macro-ify this
pub fn parse_input() -> Result<Input, Box<Error>> {
    let q = Regex::new(r"^[qQ]\s+$")?;
    let o = Regex::new(r"^[oO]\s+$")?;
    let p = Regex::new(r"^[pP]\s+$")?;
    let d = Regex::new(r"^[dD]\s+$")?;
    let clear = Regex::new(r"^clear\s+$")?;
    let labels = Regex::new(r"^labels\s+$")?;
    let cluster = Regex::new(r"^cluster (\d+)\s+$")?;
    let predict = Regex::new(r"^predict (\d+)\s+$")?;
    let partition = Regex::new(r"^partition (\d+), (\d+)")?;
    let split = Regex::new(r"^split (\d+), (\d+)")?;
    let morph = Regex::new(r"^morph (\d+(\.\d+)?)")?;
    let params = Regex::new(r"^params (\d+(\.\d+)?), (\d+(\.\d+)?), (\d+)\s+$")?;
    let feeder = Regex::new(r"^feeder (\d+), (\d+)")?;
    let load = Regex::new(r"^load '([\w/]+)'\s+$")?;
    let train = Regex::new(r"^train '([\w/]+)'\s+$")?;
    let save = Regex::new(r"^save '([\w/]+)'\s+$")?;
    let read = Regex::new(r"^read '([\w/\.]+)'\s+$")?;
    let help = Regex::new(r"^\?")?;

    let mut buf = String::new();
    try!(io::stdin().read_line(&mut buf));

    if o.is_match(&buf) { return Ok(Input::O) } 
    if q.is_match(&buf) { return Ok(Input::Q) } 
    if p.is_match(&buf) { return Ok(Input::P) } 
    if d.is_match(&buf) { return Ok(Input::D) } 
    if clear.is_match(&buf) { return Ok(Input::Clear) }
    if help.is_match(&buf) { return Ok(Input::Help) } 
    if labels.is_match(&buf) { return Ok(Input::Labels) }

    if let Some(cap) = cluster.captures(&buf) {
        return Ok(Input::Cluster(try!(cap.at(1).unwrap().parse::<usize>())))
    }

    if let Some(cap) = params.captures(&buf) {
        return Ok(Input::Params(
                try!(cap.at(1).unwrap().parse::<f64>()),
                try!(cap.at(3).unwrap().parse::<f64>()),
                try!(cap.at(5).unwrap().parse::<usize>())
                ))
    }

    if let Some(cap) = predict.captures(&buf) {
        return Ok(Input::Predict(try!(cap.at(1).unwrap().parse::<usize>())))
    }

    if let Some(cap) = train.captures(&buf) {
        return Ok(Input::Train(cap.at(1).unwrap().to_string()))
    }

    if let Some(cap) = load.captures(&buf) {
        return Ok(Input::Load(cap.at(1).unwrap().to_string()))
    }

    if let Some(cap) = save.captures(&buf) {
        return Ok(Input::Save(cap.at(1).unwrap().to_string()))
    }

    if let Some(cap) = read.captures(&buf) {
        return Ok(Input::Read(cap.at(1).unwrap().to_string()))
    }

    if let Some(cap) = split.captures(&buf) {
        return Ok(Input::Split(
                try!(cap.at(1).unwrap().parse::<usize>()), 
                try!(cap.at(2).unwrap().parse::<usize>())))
    }

    if let Some(cap) = partition.captures(&buf) {
        return Ok(Input::Partition(
                try!(cap.at(1).unwrap().parse::<usize>()),
                try!(cap.at(2).unwrap().parse::<usize>())))
    }

    if let Some(cap) = feeder.captures(&buf) {
        return Ok(Input::Feeder(
                try!(cap.at(1).unwrap().parse::<usize>()), 
                try!(cap.at(2).unwrap().parse::<usize>())))
    }

    if let Some(cap) = morph.captures(&buf) {
        return Ok(Input::Morph(
                try!(cap.at(1).unwrap().parse::<f64>())))
    }

    println!("Could not understand.");
    Ok(Input::None)
}

