use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{Read, BufReader};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

// ─── Image Representation ───────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Image {
    pub fn from_raw(width: usize, height: usize, raw: Vec<u8>) -> Self {
        assert_eq!(raw.len(), width * height);
        let data = raw.iter().map(|&p| p as f32 / 255.0).collect();
        Image { width, height, data }
    }

    pub fn zeros(width: usize, height: usize) -> Self {
        Image { width, height, data: vec![0.0; width * height] }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.width + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        self.data[r * self.width + c] = v;
    }

    pub fn resize(&self, new_w: usize, new_h: usize) -> Image {
        let mut out = Image::zeros(new_w, new_h);
        for r in 0..new_h {
            for c in 0..new_w {
                let sr = (r * self.height / new_h).min(self.height - 1);
                let sc = (c * self.width / new_w).min(self.width - 1);
                out.set(r, c, self.get(sr, sc));
            }
        }
        out
    }

    pub fn mean(&self) -> f32 {
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    pub fn std_dev(&self) -> f32 {
        let m = self.mean();
        let var = self.data.iter()
            .map(|&p| (p - m).powi(2))
            .sum::<f32>() / self.data.len() as f32;
        var.sqrt()
    }
}

// ─── Metrics ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Metric {
    SSD,
    NCC,
    MAD,
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Metric::SSD => write!(f, "SSD"),
            Metric::NCC => write!(f, "NCC"),
            Metric::MAD => write!(f, "MAD"),
        }
    }
}

pub fn compute_score(img: &Image, tmpl: &Image, metric: Metric) -> f32 {
    match metric {
        Metric::SSD => img.data.iter().zip(&tmpl.data)
            .map(|(a, b)| (a - b).powi(2)).sum(),

        Metric::MAD => img.data.iter().zip(&tmpl.data)
            .map(|(a, b)| (a - b).abs()).sum::<f32>() / img.data.len() as f32,

        Metric::NCC => {
            let im = img.mean();
            let tm = tmpl.mean();
            let is = img.std_dev().max(1e-8);
            let ts = tmpl.std_dev().max(1e-8);

            let n = img.data.len() as f32;

            img.data.iter().zip(&tmpl.data)
                .map(|(a, b)| ((a - im) / is) * ((b - tm) / ts))
                .sum::<f32>() / n
        }
    }
}

// ─── Classifier ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Template {
    pub label: u8,
    pub image: Image,
}

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
        self.templates.push(Template {
            label,
            image: image.resize(w, h),
        });
    }

    pub fn classify(&self, query: &Image) -> u8 {
        let (w, h) = self.canonical_size;
        let q = query.resize(w, h);

        let mut best_label = 0;
        let mut best_score = match self.metric {
            Metric::NCC => f32::NEG_INFINITY,
            _ => f32::INFINITY,
        };

        for tmpl in &self.templates {
            let score = compute_score(&q, &tmpl.image, self.metric);

            let better = match self.metric {
                Metric::NCC => score > best_score,
                _ => score < best_score,
            };

            if better {
                best_score = score;
                best_label = tmpl.label;
            }
        }

        best_label
    }
}

// ─── MNIST Loading ──────────────────────────────────────────────────────────

fn read_u32_be(r: &mut impl Read) -> u32 {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).unwrap();
    u32::from_be_bytes(buf)
}

pub fn load_mnist_images(path: &str) -> Vec<Image> {
    let mut r = BufReader::new(File::open(path).unwrap());

    assert_eq!(read_u32_be(&mut r), 0x0803);
    let count = read_u32_be(&mut r) as usize;
    let rows = read_u32_be(&mut r) as usize;
    let cols = read_u32_be(&mut r) as usize;

    (0..count).map(|_| {
        let mut raw = vec![0u8; rows * cols];
        r.read_exact(&mut raw).unwrap();
        Image::from_raw(cols, rows, raw)
    }).collect()
}

pub fn load_mnist_labels(path: &str) -> Vec<u8> {
    let mut r = BufReader::new(File::open(path).unwrap());

    assert_eq!(read_u32_be(&mut r), 0x0801);
    let count = read_u32_be(&mut r) as usize;

    let mut labels = vec![0u8; count];
    r.read_exact(&mut labels).unwrap();
    labels
}

// ─── Mean Templates ─────────────────────────────────────────────────────────

pub fn build_mean_templates(
    images: &[Image],
    labels: &[u8],
    num_classes: usize,
) -> Vec<(u8, Image)> {
    let (w, h) = (images[0].width, images[0].height);
    let mut accum = vec![vec![0f64; w * h]; num_classes];
    let mut counts = vec![0usize; num_classes];

    for (img, &label) in images.iter().zip(labels) {
        counts[label as usize] += 1;
        for (a, &p) in accum[label as usize].iter_mut().zip(&img.data) {
            *a += p as f64;
        }
    }

    (0..num_classes).map(|c| {
        let n = counts[c] as f64;
        let data = accum[c].iter().map(|&x| (x / n) as f32).collect();
        (c as u8, Image { width: w, height: h, data })
    }).collect()
}

// ─── MAIN ───────────────────────────────────────────────────────────────────

fn main() {
    let train_images = load_mnist_images("data/train-images.idx3-ubyte");
    let train_labels = load_mnist_labels("data/train-labels.idx1-ubyte");
    let test_images  = load_mnist_images("data/t10k-images.idx3-ubyte");
    let test_labels  = load_mnist_labels("data/t10k-labels.idx1-ubyte");

    let templates = build_mean_templates(&train_images, &train_labels, 10);

    let mut clf = TemplateClassifier::new(Metric::NCC, (28, 28));
    for (label, img) in templates {
        clf.add_template(label, img);
    }

    let clf = Arc::new(clf);

    let pool = ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build()
        .unwrap();

    let total = test_images.len();

    // ─── Evaluation (timed) ────────────────────────────────────────────────

    let start = Instant::now();

    let correct = pool.install(|| {
        test_images
            .par_iter()
            .zip(test_labels.par_iter())
            .map(|(img, &label)| {
                if clf.classify(img) == label { 1 } else { 0 }
            })
            .sum::<usize>()
    });

    let duration = start.elapsed();

    // ─── Metrics ───────────────────────────────────────────────────────────

    let accuracy = correct as f32 / total as f32;
    let throughput = total as f64 / duration.as_secs_f64();

    println!("\n=== Performance Metrics ===");
    println!("Threads used      : {}", pool.current_num_threads());
    println!("Total samples     : {}", total);
    println!("Correct           : {}", correct);
    println!("Accuracy          : {:.2}%", accuracy * 100.0);
    println!("Execution time    : {:.3} seconds", duration.as_secs_f64());
    println!("Throughput        : {:.2} images/sec", throughput);
}