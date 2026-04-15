use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

// ─── Channels ────────────────────────────────────────────────────────────────
//
// The pipeline has four concurrent TASKS, each running on its own thread,
// communicating via std::sync::mpsc channels:
//
//   [Task 1: Loader] ──(raw bytes)──► [Task 2: Decoder] ──(Image)──►
//   [Task 3: Classifier] ──(result)──► [Task 4: Aggregator]
//
// Each task does different work, which is what makes this TASK parallel
// rather than data parallel.
//
// ─────────────────────────────────────────────────────────────────────────────

use std::sync::mpsc::{self, SyncSender, Receiver};

// ─── Image ───────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Image {
    pub fn from_raw(width: usize, height: usize, raw: &[u8]) -> Self {
        assert_eq!(raw.len(), width * height);
        Image {
            width,
            height,
            data: raw.iter().map(|&p| p as f32 / 255.0).collect(),
        }
    }

    pub fn zeros(width: usize, height: usize) -> Self {
        Image { width, height, data: vec![0.0; width * height] }
    }

    #[inline] pub fn get(&self, r: usize, c: usize) -> f32 { self.data[r * self.width + c] }
    #[inline] pub fn set(&mut self, r: usize, c: usize, v: f32) { self.data[r * self.width + c] = v; }

    pub fn resize(&self, nw: usize, nh: usize) -> Image {
        let mut out = Image::zeros(nw, nh);
        for r in 0..nh {
            for c in 0..nw {
                let sr = (r * self.height / nh).min(self.height - 1);
                let sc = (c * self.width  / nw).min(self.width  - 1);
                out.set(r, c, self.get(sr, sc));
            }
        }
        out
    }

    pub fn mean(&self) -> f32 { self.data.iter().sum::<f32>() / self.data.len() as f32 }

    pub fn std_dev(&self) -> f32 {
        let m = self.mean();
        (self.data.iter().map(|&p| (p - m).powi(2)).sum::<f32>() / self.data.len() as f32).sqrt()
    }
}

// ─── Metric ──────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Metric { SSD, NCC, MAD }

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Metric::SSD => write!(f,"SSD"), Metric::NCC => write!(f,"NCC"), Metric::MAD => write!(f,"MAD") }
    }
}

pub fn compute_score(img: &Image, tmpl: &Image, metric: Metric) -> f32 {
    match metric {
        Metric::SSD => img.data.iter().zip(&tmpl.data).map(|(a,b)| (a-b).powi(2)).sum(),
        Metric::MAD => img.data.iter().zip(&tmpl.data).map(|(a,b)| (a-b).abs()).sum::<f32>() / img.data.len() as f32,
        Metric::NCC => {
            let (im, tm) = (img.mean(), tmpl.mean());
            let (is, ts) = (img.std_dev().max(1e-8), tmpl.std_dev().max(1e-8));
            let n = img.data.len() as f32;
            img.data.iter().zip(&tmpl.data).map(|(a,b)| ((a-im)/is)*((b-tm)/ts)).sum::<f32>() / n
        }
    }
}

// ─── Classifier ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Template { pub label: u8, pub image: Image }

pub struct TemplateClassifier {
    pub templates: Vec<Template>,
    pub metric: Metric,
    pub canonical_size: (usize, usize),
}

impl TemplateClassifier {
    pub fn new(metric: Metric, size: (usize, usize)) -> Self {
        Self { templates: Vec::new(), metric, canonical_size: size }
    }

    pub fn add_template(&mut self, label: u8, image: Image) {
        let (w, h) = self.canonical_size;
        self.templates.push(Template { label, image: image.resize(w, h) });
    }

    pub fn classify(&self, query: &Image) -> u8 {
        let (w, h) = self.canonical_size;
        let q = query.resize(w, h);
        let mut best_label = 0u8;
        let mut best = match self.metric { Metric::NCC => f32::NEG_INFINITY, _ => f32::INFINITY };

        for tmpl in &self.templates {
            let s = compute_score(&q, &tmpl.image, self.metric);
            let better = match self.metric { Metric::NCC => s > best, _ => s < best };
            if better { best = s; best_label = tmpl.label; }
        }
        best_label
    }
}

// ─── MNIST I/O ───────────────────────────────────────────────────────────────

fn read_u32_be(r: &mut impl Read) -> u32 {
    let mut b = [0u8; 4]; r.read_exact(&mut b).unwrap(); u32::from_be_bytes(b)
}

/// Returns (count, rows, cols, flat_raw_bytes).
/// We deliberately return *raw bytes* rather than decoded Images so that
/// decoding can happen on a separate task thread.
pub fn load_raw_images(path: &str) -> (usize, usize, usize, Vec<u8>) {
    let mut r = BufReader::new(File::open(path).expect("Cannot open image file"));
    assert_eq!(read_u32_be(&mut r), 0x0803, "Bad image magic");
    let count = read_u32_be(&mut r) as usize;
    let rows  = read_u32_be(&mut r) as usize;
    let cols  = read_u32_be(&mut r) as usize;
    let mut raw = vec![0u8; count * rows * cols];
    r.read_exact(&mut raw).unwrap();
    (count, rows, cols, raw)
}

pub fn load_mnist_labels(path: &str) -> Vec<u8> {
    let mut r = BufReader::new(File::open(path).expect("Cannot open label file"));
    assert_eq!(read_u32_be(&mut r), 0x0801, "Bad label magic");
    let count = read_u32_be(&mut r) as usize;
    let mut labels = vec![0u8; count];
    r.read_exact(&mut labels).unwrap();
    labels
}

// ─── Mean Template Builder ───────────────────────────────────────────────────

pub fn build_mean_templates(images: &[Image], labels: &[u8], num_classes: usize) -> Vec<(u8, Image)> {
    let (w, h) = (images[0].width, images[0].height);
    let mut accum  = vec![vec![0f64; w * h]; num_classes];
    let mut counts = vec![0usize; num_classes];

    for (img, &lbl) in images.iter().zip(labels) {
        counts[lbl as usize] += 1;
        for (a, &p) in accum[lbl as usize].iter_mut().zip(&img.data) { *a += p as f64; }
    }

    (0..num_classes).map(|c| {
        let n = counts[c] as f64;
        let data = accum[c].iter().map(|&x| (x / n) as f32).collect();
        (c as u8, Image { width: w, height: h, data })
    }).collect()
}

// ─── Pipeline Messages ───────────────────────────────────────────────────────

/// Raw bytes for a single image + its ground-truth label.
struct RawSample { pixels: Vec<u8>, width: usize, height: usize, label: u8 }

/// Decoded, normalised image ready for classification.
struct DecodedSample { image: Image, label: u8 }

/// Final result from the classifier.
struct ClassifiedSample { predicted: u8, actual: u8 }

// ─── MAIN ────────────────────────────────────────────────────────────────────

fn main() {
    // ── Load data (done on the main thread before the pipeline starts) ────
    println!("Loading MNIST data...");
    let (count, rows, cols, raw_pixels) = load_raw_images("data/train-images.idx3-ubyte");
    let train_labels = load_mnist_labels("data/train-labels.idx1-ubyte");

    // Decode training images to build templates (not part of the timed pipeline)
    let train_images: Vec<Image> = (0..count).map(|i| {
        let start = i * rows * cols;
        Image::from_raw(cols, rows, &raw_pixels[start..start + rows * cols])
    }).collect();

    let templates = build_mean_templates(&train_images, &train_labels, 10);
    let mut clf = TemplateClassifier::new(Metric::NCC, (28, 28));
    for (label, img) in templates { clf.add_template(label, img); }
    let clf = Arc::new(clf);

    // Load raw test bytes — the pipeline will decode + classify them
    let (test_count, test_rows, test_cols, test_raw) = load_raw_images("data/t10k-images.idx3-ubyte");
    let test_labels = load_mnist_labels("data/t10k-labels.idx1-ubyte");

    println!("Starting task-parallel pipeline ({} test samples)...\n", test_count);

    // ── Channel setup ─────────────────────────────────────────────────────
    //
    // Bounded channels apply back-pressure so fast tasks don't run far ahead
    // of slow ones, keeping memory usage under control.
    //
    let (raw_tx,     raw_rx):     (SyncSender<RawSample>,     Receiver<RawSample>)     = mpsc::sync_channel(256);
    let (decoded_tx, decoded_rx): (SyncSender<DecodedSample>, Receiver<DecodedSample>) = mpsc::sync_channel(256);
    let (result_tx,  result_rx):  (SyncSender<ClassifiedSample>, Receiver<ClassifiedSample>) = mpsc::sync_channel(256);

    let start = Instant::now();

    // ══════════════════════════════════════════════════════════════════════
    // TASK 1 — LOADER
    // Reads raw bytes from the in-memory buffer and sends one RawSample per
    // image down the pipeline.  In a real system this would do disk I/O
    // (e.g. reading individual JPEG files) while other tasks work in parallel.
    // ══════════════════════════════════════════════════════════════════════
    let loader = thread::Builder::new().name("loader".into()).spawn(move || {
        let pixel_size = test_rows * test_cols;
        for i in 0..test_count {
            let start  = i * pixel_size;
            let pixels = test_raw[start..start + pixel_size].to_vec();
            let label  = test_labels[i];

            raw_tx.send(RawSample { pixels, width: test_cols, height: test_rows, label })
                  .expect("loader: send failed");
        }
        // Dropping raw_tx closes the channel, signalling the decoder to stop.
        println!("[loader]     finished sending {} raw samples", test_count);
    }).unwrap();

    // ══════════════════════════════════════════════════════════════════════
    // TASK 2 — DECODER
    // Converts raw u8 bytes → normalised f32 Image structs.
    // Different work to Task 1 (format conversion, not I/O).
    // ══════════════════════════════════════════════════════════════════════
    let decoder = thread::Builder::new().name("decoder".into()).spawn(move || {
        let mut decoded_count = 0usize;
        for sample in raw_rx {
            let image = Image::from_raw(sample.width, sample.height, &sample.pixels);
            decoded_tx.send(DecodedSample { image, label: sample.label })
                      .expect("decoder: send failed");
            decoded_count += 1;
        }
        println!("[decoder]    finished decoding {} images", decoded_count);
    }).unwrap();

    // ══════════════════════════════════════════════════════════════════════
    // TASK 3 — CLASSIFIER
    // Runs template matching on each decoded image.
    // Different work again — NCC scoring against 10 templates.
    // ══════════════════════════════════════════════════════════════════════
    let clf_ref = Arc::clone(&clf);
    let classifier = thread::Builder::new().name("classifier".into()).spawn(move || {
        let mut classified_count = 0usize;
        for sample in decoded_rx {
            let predicted = clf_ref.classify(&sample.image);
            result_tx.send(ClassifiedSample { predicted, actual: sample.label })
                     .expect("classifier: send failed");
            classified_count += 1;
        }
        println!("[classifier] finished classifying {} images", classified_count);
    }).unwrap();

    // ══════════════════════════════════════════════════════════════════════
    // TASK 4 — AGGREGATOR
    // Collects results and computes accuracy + per-class breakdown.
    // Different work again — pure bookkeeping, no image processing.
    // ══════════════════════════════════════════════════════════════════════
    let aggregator = thread::Builder::new().name("aggregator".into()).spawn(move || {
        let mut correct = 0usize;
        let mut total   = 0usize;
        let mut per_class_correct = HashMap::<u8, usize>::new();
        let mut per_class_total   = HashMap::<u8, usize>::new();

        for result in result_rx {
            total += 1;
            *per_class_total.entry(result.actual).or_insert(0) += 1;
            if result.predicted == result.actual {
                correct += 1;
                *per_class_correct.entry(result.actual).or_insert(0) += 1;
            }
        }

        println!("[aggregator] finished accumulating {} results\n", total);
        (correct, total, per_class_correct, per_class_total)
    }).unwrap();

    // ── Join all tasks ────────────────────────────────────────────────────
    loader.join().expect("loader panicked");
    decoder.join().expect("decoder panicked");
    classifier.join().expect("classifier panicked");
    let (correct, total, per_class_correct, per_class_total) =
        aggregator.join().expect("aggregator panicked");

    let duration = start.elapsed();

    // ── Results ───────────────────────────────────────────────────────────
    println!("=== Task-Parallel Pipeline Results ===");
    println!("Total samples  : {}", total);
    println!("Correct        : {}", correct);
    println!("Accuracy       : {:.2}%", 100.0 * correct as f32 / total as f32);
    println!("Time           : {:.3}s", duration.as_secs_f64());
    println!("Throughput     : {:.0} images/sec\n", total as f64 / duration.as_secs_f64());

    println!("Per-class accuracy:");
    let mut classes: Vec<u8> = per_class_total.keys().cloned().collect();
    classes.sort();
    for cls in classes {
        let c = per_class_correct.get(&cls).cloned().unwrap_or(0);
        let t = per_class_total[&cls];
        println!("  Digit {} : {:4}/{} = {:.1}%", cls, c, t, 100.0 * c as f32 / t as f32);
    }
}